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
 * \file matmul_allto_all.h
 * \brief
 */

#ifndef MATMUL_ALLTO_ALL_H
#define MATMUL_ALLTO_ALL_H

using namespace AscendC;

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "adv_api/hccl/hccl.h"
#include "kernel_tiling/kernel_tiling.h"
#include "matmul_allto_all_moe_distribute_base.h"
#include "matmul_allto_all_util_arch22.h"

#include "../../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/tla_catlass.hpp"
#include "../../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/arch/tla_arch_arch.hpp"
#include "../../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/layout/tla_layout_layout.hpp"
#include "../../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/gemm/block/tla_gemm_block_mmad.hpp"
#include "../../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/gemm/block/tla_gemm_block_swizzle.hpp"
#include "../../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/gemm/tla_gemm_dispatch_policy.hpp"
#include "../../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/gemm/tla_gemm_gemm_type.hpp"
#include "../../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/tla_gemm_coord.hpp"
#include "../../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/epilogue/tile/tla_epilogue_copy_gm_to_ub.hpp"
#include "matmul_allto_all_block_epilogue_dequant.hpp"
#include "../../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/epilogue/tla_epilogue_dispatch_policy.hpp"
#include "../../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/epilogue/tile/tla_epilogue_tile_broadcast_mul.hpp"
#include "matmul_allto_all_tile_broadcast_add.hpp"
#include "../../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/epilogue/tile/tla_epilogue_tile_broadcast_one_blk.hpp"
#include "../../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/epilogue/tile/tla_epilogue_tile_swizzle.hpp"
#include "../../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/epilogue/tile/tla_epilogue_tile_copy.hpp"

#include "matmul_allto_all_quant_matmul.hpp"
#include "matmul.hpp"

#include "../../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/tla_status.hpp"
using namespace Catlass;

namespace Mc2Kernel {

// MMA2A : MatmulAlltoAll
#define TemplateMMA2AClass \
    typename AType, typename BType, typename BiasType, typename pertokenScaleType, typename scaleType, typename cType, \
        bool hasBias, bool TB
#define TemplateMMA2AFunc AType, BType, BiasType, pertokenScaleType, scaleType, cType, hasBias, TB

using namespace AscendC;
template <TemplateMMA2AClass>
class MatmulAlltoAll {
public:
    __aicore__ inline MatmulAlltoAll(){};
    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR pertokenScaleGM, GM_ADDR scaleGM,
                                GM_ADDR cGM, GM_ADDR workspaceGM, GM_ADDR tilingGM);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CatlassMatmul();
    __aicore__ inline void QuantCatlassMatmul();

private:
    GM_ADDR aGM_;
    GM_ADDR bGM_;
    GM_ADDR cGM_;
    GM_ADDR scaleGM_;
    GM_ADDR pertokenScaleGM_;
    GM_ADDR biasGM_;
    GM_ADDR workspaceGM_;

    int32_t aligned_a;
    int32_t aligned_b;
    int32_t commCount;

    int32_t peer_mem_m;

    int32_t m_align;
    int64_t k_align;
    int32_t n_align;

    uint32_t rank_size{0};
    uint32_t rank{0};

    __gm__ cType *gm_peer_mem;
    CommBase commUtil;

    bool TA;
    static constexpr bool has_bias = hasBias;
};

template <TemplateMMA2AClass>
__aicore__ inline void MatmulAlltoAll<TemplateMMA2AFunc>::Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM,
                                                               GM_ADDR pertokenScaleGM, GM_ADDR scaleGM, GM_ADDR cGM,
                                                               GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(MatmulAlltoAllTilingData);
    auto tiling = (__gm__ MatmulAlltoAllTilingData *)tilingGM;
    GET_TILING_DATA(tilingData, tilingGM);

    rank_size = tilingData.matmulAlltoAllInfo.worldSize;
    aGM_ = aGM;
    bGM_ = bGM;
    cGM_ = cGM;
    scaleGM_ = scaleGM;
    pertokenScaleGM_ = pertokenScaleGM;
    biasGM_ = biasGM;
    workspaceGM_ = GetUserWorkspace(workspaceGM);

    commUtil.SetArgs(&rank, rank_size, tilingData);
    gm_peer_mem = reinterpret_cast<__gm__ cType *>(commUtil.buff[rank]); // 注意注意，quant是aiv使用，非quant是aic使用
}

template <TemplateMMA2AClass>
__aicore__ inline void MatmulAlltoAll<TemplateMMA2AFunc>::CatlassMatmul()
{
    if ASCEND_IS_AIC {
        using ArchTag = Arch::AtlasA2;

        constexpr bool ENABLE_UNIT_FLAG = false;
        constexpr bool ENABLE_SHUFFLE_K = false;
        using ElementA = cType;
        using ElementB = cType;
        using ElementC = cType;
        using ElementBias = BiasType;
        using LayoutA = layout::RowMajor;
        // 支持B转置场景
        using LayoutB = typename std::conditional<TB, layout::ColumnMajor, layout::RowMajor>::type;
        using LayoutBias = layout::VectorLayout;

        using LayoutC = layout::RowMajor;
        LayoutA layoutA{static_cast<uint32_t>(commUtil.m), static_cast<uint32_t>(commUtil.k)};
        LayoutB layoutB{static_cast<uint32_t>(commUtil.k), static_cast<uint32_t>(commUtil.n)};
        LayoutC layoutC{static_cast<uint32_t>(commUtil.m * rank_size), static_cast<uint32_t>(commUtil.n / rank_size)};
        LayoutBias layoutBias{static_cast<uint32_t>(commUtil.n)};

        using LayoutPaddingA = std::conditional_t<std::is_same_v<LayoutA, layout::RowMajor>, layout::PaddingRowMajor,
                                                  layout::PaddingColumnMajor>;

        using DispatchPolicy = std::conditional_t<has_bias, Gemm::MmadAtlasA2PingpongBias<ENABLE_UNIT_FLAG>,
                                                  Gemm::MmadAtlasA2Preload<ENABLE_UNIT_FLAG, ENABLE_SHUFFLE_K>>;

        using AType_ = Gemm::GemmType<ElementA, LayoutA>;
        using BType_ = Gemm::GemmType<ElementB, LayoutB>;
        using CType_ = Gemm::GemmType<ElementC, LayoutC>;
        using BiasType_ =
            std::conditional_t<std::is_same_v<BiasType, void>, void, Gemm::GemmType<ElementBias, LayoutBias>>;

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

        using BlockEpilogue = void;
        using BlockScheduler30 = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;
        GemmCoord processSize{static_cast<uint32_t>(commUtil.m), static_cast<uint32_t>(commUtil.n),
                              static_cast<uint32_t>(commUtil.k)};

        if (commUtil.m0 == 128) {
            using L1TileShape = GemmShape<128, 256, 256>;
            using L0TileShape = GemmShape<128, 256, 64>;
            using BlockMmadOpt = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType_, BType_,
                                                        CType_, BiasType_, TileCopy>;
            using MatmulKernel =
                Gemm::Kernel::MatmulAlltoAllKernel<void, void, BlockMmadOpt, BlockEpilogue, BlockScheduler30, has_bias>;
            MatmulKernel matmul_op;
            typename MatmulKernel::Params params{processSize,
                                                 reinterpret_cast<GM_ADDR>(aGM_),
                                                 layoutA,
                                                 reinterpret_cast<GM_ADDR>(bGM_),
                                                 layoutB,
                                                 reinterpret_cast<GM_ADDR>(biasGM_),
                                                 reinterpret_cast<GM_ADDR>(gm_peer_mem),
                                                 layoutC,
                                                 commUtil.p_value,
                                                 static_cast<int32_t>(rank_size),
                                                 MAX_BLOCK_COUNT,
                                                 TB};
            matmul_op(params);
        } else {
            using L1TileShape = GemmShape<256, 128, 256>;
            using L0TileShape = GemmShape<256, 128, 64>;
            using BlockMmadOpt = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType_, BType_,
                                                        CType_, BiasType_, TileCopy>;
            using MatmulKernel =
                Gemm::Kernel::MatmulAlltoAllKernel<void, void, BlockMmadOpt, BlockEpilogue, BlockScheduler30, has_bias>;
            MatmulKernel matmul_op;
            typename MatmulKernel::Params params{processSize,
                                                 reinterpret_cast<GM_ADDR>(aGM_),
                                                 layoutA,
                                                 reinterpret_cast<GM_ADDR>(bGM_),
                                                 layoutB,
                                                 reinterpret_cast<GM_ADDR>(biasGM_),
                                                 reinterpret_cast<GM_ADDR>(gm_peer_mem),
                                                 layoutC,
                                                 commUtil.p_value,
                                                 static_cast<int32_t>(rank_size),
                                                 MAX_BLOCK_COUNT,
                                                 TB};
            matmul_op(params);
        }
    }
}

template <TemplateMMA2AClass>
__aicore__ inline void MatmulAlltoAll<TemplateMMA2AFunc>::QuantCatlassMatmul()
{
    using ArchTag = Arch::AtlasA2;

    constexpr bool ENABLE_UNIT_FLAG = false;
    constexpr bool ENABLE_SHUFFLE_K = false;
    using ElementA = AType;
    using ElementB = BType;
    using ElementC = int32_t;
    using LayoutA = layout::RowMajor;
    // 支持B转置场景
    using LayoutB = typename std::conditional<TB, layout::ColumnMajor, layout::RowMajor>::type;
    using LayoutD = layout::RowMajor;
    LayoutA layoutA{static_cast<uint32_t>(commUtil.m), static_cast<uint32_t>(commUtil.k)};
    LayoutB layoutB{static_cast<uint32_t>(commUtil.k), static_cast<uint32_t>(commUtil.n)};
    LayoutD layoutD{static_cast<uint32_t>(commUtil.m * rank_size), static_cast<uint32_t>(commUtil.n / rank_size)};

    constexpr uint32_t preloadStages = 1;
    constexpr uint32_t l1Stages = 2;
    constexpr uint32_t l0AStages = 2;
    constexpr uint32_t l0BStages = 2;
    constexpr uint32_t l0CStages = 1;
    constexpr bool enableUnitFlag = false;
    constexpr bool enableShuffleK = true;
    using DispatchPolicy = Gemm::MmadAtlasA2PreloadAsyncWithCallback<preloadStages, l1Stages, l0AStages, l0BStages,
                                                                     l0CStages, enableUnitFlag, enableShuffleK>;

    using AType_ = Gemm::GemmType<ElementA, LayoutA>;
    using BType_ = Gemm::GemmType<ElementB, LayoutB>;
    using CType_ = Gemm::GemmType<ElementC, layout::RowMajor>;

    constexpr uint32_t ubStages = 2;
    constexpr uint32_t workspaceStages = 2;
    using EpilogueDispatchPolicy = Epilogue::EpilogueAtlasA2PerTokenDequant<ubStages>;
    using ScaleType = Gemm::GemmType<scaleType, layout::VectorLayout>;
    using PerTokenScaleType = Gemm::GemmType<pertokenScaleType, layout::VectorLayout>;
    using BiasType_ = Gemm::GemmType<BiasType, layout::VectorLayout>;
    using DType = Gemm::GemmType<cType, layout::RowMajor>;
    layout::VectorLayout layoutScale{static_cast<uint32_t>(commUtil.n)};
    layout::VectorLayout layoutPerTokenScale{static_cast<uint32_t>(commUtil.m)};
    layout::VectorLayout layoutBias{static_cast<uint32_t>(commUtil.n)};

    using RowBroadcastMulType = Gemm::GemmType<float, layout::RowMajor>;
    using RowBroadcastAddType = Gemm::GemmType<float, layout::RowMajor>;
    using BroadcastOneBlkType = Gemm::GemmType<float, layout::RowMajor>;
    using OneBlkColumnBroadcastMulType = Gemm::GemmType<float, layout::RowMajor>;

    struct TileCopyDequant
        : public Catlass::Epilogue::Tile::TileCopy<ArchTag, CType_, ScaleType, PerTokenScaleType, DType> {
        using Base = Catlass::Epilogue::Tile::TileCopy<ArchTag, CType_, ScaleType, PerTokenScaleType, DType>;
        using ElementC = typename Base::ElementC;
        using ElementScale = typename Base::ElementX;
        using ElementPerTokenScale = typename Base::ElementY;
        using ElementBias = typename BiasType_::Element;
        using ElementD = typename Base::ElementD;

        using CopyGmToUbC = typename Base::CopyGmToUbC;
        using CopyGmToUbScale = typename Base::CopyGmToUbX;
        using CopyGmToUbPerTokenScale = typename Base::CopyGmToUbY;
        using CopyGmToUbBias = Catlass::Epilogue::Tile::CopyGm2Ub<ArchTag, BiasType_>;
        using CopyUbToGmD = typename Base::CopyUbToGmD;
    };

    using EpilogueTileScheduler = Epilogue::Tile::EpilogueHorizontalTileSwizzle;
    using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;
    GemmCoord problemShape{static_cast<uint32_t>(commUtil.m), static_cast<uint32_t>(commUtil.n),
                           static_cast<uint32_t>(commUtil.k)};

    if (commUtil.m0 == 128) {
        using L1TileShape = GemmShape<128, 256, 512>;
        using L0TileShape = GemmShape<128, 256, 128>;
        using EpilogueTileShape = MatrixShape<32, 256>;
        using TileRowBroadcastMul =
            Epilogue::Tile::TileRowBroadcastMul<ArchTag, RowBroadcastMulType, EpilogueTileShape>;

        using TileRowBroadcastAdd =
            Epilogue::Tile::TileRowBroadcastAdd<ArchTag, RowBroadcastAddType, EpilogueTileShape>;

        using TileBroadcastOneBlk =
            Epilogue::Tile::TileBroadcastOneBlk<ArchTag, BroadcastOneBlkType, EpilogueTileShape::ROW>;
        using TileOneBlkColumnBroadcastMul =
            Epilogue::Tile::TileOneBlkColumnBroadcastMul<ArchTag, OneBlkColumnBroadcastMulType, EpilogueTileShape>;

        using QuantBlockEpilogue =
            Epilogue::Block::BlockEpilogue<EpilogueDispatchPolicy, CType_, ScaleType, PerTokenScaleType, BiasType_,
                                           DType, TileRowBroadcastMul, TileRowBroadcastAdd, TileBroadcastOneBlk,
                                           TileOneBlkColumnBroadcastMul, TileCopyDequant, EpilogueTileScheduler>;

        // kernel level
        using BlockMmadOpt = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType_, BType_, CType_>;
        using QuantMatmulKernel =
            Gemm::Kernel::QuantMatmulAllToAllKernel<BlockMmadOpt, QuantBlockEpilogue, BlockScheduler, workspaceStages>;

        QuantMatmulKernel quant_matmul_op;
        typename QuantMatmulKernel::Params params{problemShape,
                                                  reinterpret_cast<GM_ADDR>(aGM_),
                                                  layoutA,
                                                  reinterpret_cast<GM_ADDR>(bGM_),
                                                  layoutB,
                                                  reinterpret_cast<GM_ADDR>(scaleGM_),
                                                  layoutScale,
                                                  reinterpret_cast<GM_ADDR>(pertokenScaleGM_),
                                                  layoutPerTokenScale,
                                                  reinterpret_cast<GM_ADDR>(biasGM_),
                                                  layoutBias,
                                                  reinterpret_cast<GM_ADDR>(gm_peer_mem),
                                                  layoutD,
                                                  reinterpret_cast<GM_ADDR>(workspaceGM_),
                                                  reinterpret_cast<GM_ADDR>(cGM_),
                                                  commUtil,
                                                  MAX_BLOCK_COUNT,
                                                  TB};
        quant_matmul_op(params);
    } else {
        using L1TileShape = GemmShape<256, 128, 512>;
        using L0TileShape = GemmShape<256, 128, 128>;
        using EpilogueTileShape = MatrixShape<64, 128>;
        using TileRowBroadcastMul =
            Epilogue::Tile::TileRowBroadcastMul<ArchTag, RowBroadcastMulType, EpilogueTileShape>;

        using TileRowBroadcastAdd =
            Epilogue::Tile::TileRowBroadcastAdd<ArchTag, RowBroadcastAddType, EpilogueTileShape>;

        using TileBroadcastOneBlk =
            Epilogue::Tile::TileBroadcastOneBlk<ArchTag, BroadcastOneBlkType, EpilogueTileShape::ROW>;
        using TileOneBlkColumnBroadcastMul =
            Epilogue::Tile::TileOneBlkColumnBroadcastMul<ArchTag, OneBlkColumnBroadcastMulType, EpilogueTileShape>;

        using QuantBlockEpilogue =
            Epilogue::Block::BlockEpilogue<EpilogueDispatchPolicy, CType_, ScaleType, PerTokenScaleType, BiasType_,
                                           DType, TileRowBroadcastMul, TileRowBroadcastAdd, TileBroadcastOneBlk,
                                           TileOneBlkColumnBroadcastMul, TileCopyDequant, EpilogueTileScheduler>;

        // kernel level
        using BlockMmadOpt = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType_, BType_, CType_>;
        using QuantMatmulKernel =
            Gemm::Kernel::QuantMatmulAllToAllKernel<BlockMmadOpt, QuantBlockEpilogue, BlockScheduler, workspaceStages>;

        QuantMatmulKernel quant_matmul_op;
        typename QuantMatmulKernel::Params params{problemShape,
                                                  reinterpret_cast<GM_ADDR>(aGM_),
                                                  layoutA,
                                                  reinterpret_cast<GM_ADDR>(bGM_),
                                                  layoutB,
                                                  reinterpret_cast<GM_ADDR>(scaleGM_),
                                                  layoutScale,
                                                  reinterpret_cast<GM_ADDR>(pertokenScaleGM_),
                                                  layoutPerTokenScale,
                                                  reinterpret_cast<GM_ADDR>(biasGM_),
                                                  layoutBias,
                                                  reinterpret_cast<GM_ADDR>(gm_peer_mem),
                                                  layoutD,
                                                  reinterpret_cast<GM_ADDR>(workspaceGM_),
                                                  reinterpret_cast<GM_ADDR>(cGM_),
                                                  commUtil,
                                                  MAX_BLOCK_COUNT,
                                                  TB};
        quant_matmul_op(params);
    }
}

template <TemplateMMA2AClass>
__aicore__ inline void MatmulAlltoAll<TemplateMMA2AFunc>::Process()
{
    if constexpr (AscendC::IsSameType<AType, int8_t>::value) {
        QuantCatlassMatmul();
    } else {
        CatlassMatmul();
        if ASCEND_IS_AIV {
            commCount = DivCeil(commUtil.m_loop, commUtil.p_value);
            commUtil.ResetIpcFlags(2);
            PipeBarrier<PIPE_ALL>();

            for (int32_t commIdx = 0; commIdx < commCount; ++commIdx) {
                int32_t actual_p_value = commUtil.p_value;

                int64_t token_total = static_cast<int64_t>(commUtil.p_value) * commUtil.m0 * commUtil.n;
                if (commIdx == commCount - 1) {
                    token_total =
                        (commUtil.m - static_cast<int64_t>(commIdx) * commUtil.m0 * commUtil.p_value) * commUtil.n;
                }
                int64_t token_per_rank = token_total / rank_size;

                uint64_t flag_idx = commIdx % MAX_BLOCK_COUNT;
                WaitEvent(flag_idx);

                commUtil.SetAndWaitAivSync(flag_idx);
                commUtil.CrossRankSyncV1(FLAG_ZERO_IDX, commIdx + 1);
                commUtil.SetAndWaitAivSync(flag_idx);

                int64_t rank_offset = static_cast<int64_t>(commUtil.m) * commUtil.n / rank_size;
                if (commUtil.aiv_idx == 0 && commUtil.core_idx < rank_size) {
                    int64_t src_offset = static_cast<int64_t>(flag_idx) * commUtil.gm_a_pingpong_size +
                                         commUtil.gm_a_pingpong_size / rank_size * rank;
                    int64_t dst_offset = commUtil.core_idx * rank_offset + static_cast<int64_t>(commIdx) * commUtil.m0 *
                                                                               commUtil.p_value *
                                                                               (commUtil.n / rank_size);
                    commUtil.CopyGMToGM((__gm__ cType *)commUtil.buff[commUtil.core_idx] + src_offset,
                                        reinterpret_cast<__gm__ cType *>(cGM_) + dst_offset, token_per_rank);
                }

                commUtil.SetAndWaitAivSync(flag_idx);
                commUtil.CrossRankSyncV1(FLAG_ONE_IDX, commIdx + 1);
                commUtil.SetAndWaitAivSync(flag_idx);
                if (commIdx < commCount - 2) {
                    commUtil.SetAicSync(flag_idx);
                }
            }
            PipeBarrier<PIPE_ALL>();
            commUtil.ResetIpcFlags(1);
        }
    }
    SyncAll<false>();
}
} // namespace Mc2Kernel
#endif // MATMUL_ALLTO_ALL_H
