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
 * \file matmul_reduce_scatter_aiv_mode_smallM.h
 * \brief
 */

#ifndef MATMUL_REDUCE_SCATTER_AIV_MODE_SMALL_M_H
#define MATMUL_REDUCE_SCATTER_AIV_MODE_SMALL_M_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "../../common/op_kernel/moe_distribute_base.h"
#include "matmul_reduce_scatter_aiv_mode_util.h"
#include "../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/tla_catlass.hpp"
#include "../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/arch/tla_arch_arch.hpp"
#include "../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/layout/tla_layout_layout.hpp"
#include "../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/gemm/block/tla_gemm_block_mmad.hpp"
#include "../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/gemm/block/tla_gemm_block_swizzle.hpp"
#include "../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/gemm/tla_gemm_dispatch_policy.hpp"
#include "../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/gemm/tla_gemm_gemm_type.hpp"
#include "../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/tla_gemm_coord.hpp"
#include "matmul_smallM.hpp"

#include "matmul_reduce_scatter_aiv_mode.h"

using namespace Catlass;
using namespace AscendC;
using namespace matmulReduceScatterV2_aivmode_tiling;
using namespace matmulReduceScatterV2_util;
using namespace dequant;
using namespace padding;

namespace MatmulReduceScatterV2Impl {

using Index = Catlass::layout::VectorLayout::Index;
using ArchTag = Arch::AtlasA2;

template <TemplateMMReduceScatterV2Class>
class MatmulReduceScatterAivModeSmallM
    : public MatmulReduceScatterAivMode<TemplateMMReduceScatterV2Func,
                                        MatmulReduceScatterAivModeSmallM<TemplateMMReduceScatterV2Func>> {
public:
    using MatmulReduceScatterAivMode<TemplateMMReduceScatterV2Func,
                                     MatmulReduceScatterAivModeSmallM<TemplateMMReduceScatterV2Func>>::m;
    using MatmulReduceScatterAivMode<TemplateMMReduceScatterV2Func,
                                     MatmulReduceScatterAivModeSmallM<TemplateMMReduceScatterV2Func>>::k;
    using MatmulReduceScatterAivMode<TemplateMMReduceScatterV2Func,
                                     MatmulReduceScatterAivModeSmallM<TemplateMMReduceScatterV2Func>>::n;
    using MatmulReduceScatterAivMode<TemplateMMReduceScatterV2Func,
                                     MatmulReduceScatterAivModeSmallM<TemplateMMReduceScatterV2Func>>::k_align;
    using MatmulReduceScatterAivMode<TemplateMMReduceScatterV2Func,
                                     MatmulReduceScatterAivModeSmallM<TemplateMMReduceScatterV2Func>>::n_align;
    using MatmulReduceScatterAivMode<TemplateMMReduceScatterV2Func,
                                     MatmulReduceScatterAivModeSmallM<TemplateMMReduceScatterV2Func>>::nAlign16;
    using MatmulReduceScatterAivMode<TemplateMMReduceScatterV2Func,
                                     MatmulReduceScatterAivModeSmallM<TemplateMMReduceScatterV2Func>>::m0;
    using MatmulReduceScatterAivMode<TemplateMMReduceScatterV2Func,
                                     MatmulReduceScatterAivModeSmallM<TemplateMMReduceScatterV2Func>>::n0;
    using MatmulReduceScatterAivMode<TemplateMMReduceScatterV2Func,
                                     MatmulReduceScatterAivModeSmallM<TemplateMMReduceScatterV2Func>>::quantFlag;
    using MatmulReduceScatterAivMode<TemplateMMReduceScatterV2Func,
                                     MatmulReduceScatterAivModeSmallM<TemplateMMReduceScatterV2Func>>::needCastBias;

    using MatmulReduceScatterAivMode<TemplateMMReduceScatterV2Func,
                                     MatmulReduceScatterAivModeSmallM<TemplateMMReduceScatterV2Func>>::gm_a_src;
    using MatmulReduceScatterAivMode<TemplateMMReduceScatterV2Func,
                                     MatmulReduceScatterAivModeSmallM<TemplateMMReduceScatterV2Func>>::gm_b_src;
    using MatmulReduceScatterAivMode<TemplateMMReduceScatterV2Func,
                                     MatmulReduceScatterAivModeSmallM<TemplateMMReduceScatterV2Func>>::gm_accum;
    using MatmulReduceScatterAivMode<TemplateMMReduceScatterV2Func,
                                     MatmulReduceScatterAivModeSmallM<TemplateMMReduceScatterV2Func>>::biasGM_;
    using MatmulReduceScatterAivMode<TemplateMMReduceScatterV2Func,
                                     MatmulReduceScatterAivModeSmallM<TemplateMMReduceScatterV2Func>>::cGM_;
    using MatmulReduceScatterAivMode<TemplateMMReduceScatterV2Func,
                                     MatmulReduceScatterAivModeSmallM<TemplateMMReduceScatterV2Func>>::dequant_type;
    using MatmulReduceScatterAivMode<
        TemplateMMReduceScatterV2Func,
        MatmulReduceScatterAivModeSmallM<TemplateMMReduceScatterV2Func>>::isX2ScaleTypeInt64;
    using MatmulReduceScatterAivMode<
        TemplateMMReduceScatterV2Func,
        MatmulReduceScatterAivModeSmallM<TemplateMMReduceScatterV2Func>>::perChannelScaleGM_;
    using MatmulReduceScatterAivMode<TemplateMMReduceScatterV2Func,
                                     MatmulReduceScatterAivModeSmallM<TemplateMMReduceScatterV2Func>>::perTokenScaleGM_;
    using MatmulReduceScatterAivMode<TemplateMMReduceScatterV2Func,
                                     MatmulReduceScatterAivModeSmallM<TemplateMMReduceScatterV2Func>>::buff;
    using MatmulReduceScatterAivMode<TemplateMMReduceScatterV2Func,
                                     MatmulReduceScatterAivModeSmallM<TemplateMMReduceScatterV2Func>>::uBuf_;

    using MatmulReduceScatterAivMode<TemplateMMReduceScatterV2Func,
                                     MatmulReduceScatterAivModeSmallM<TemplateMMReduceScatterV2Func>>::p_value;
    using MatmulReduceScatterAivMode<TemplateMMReduceScatterV2Func,
                                     MatmulReduceScatterAivModeSmallM<TemplateMMReduceScatterV2Func>>::swizzl_count;
    using MatmulReduceScatterAivMode<TemplateMMReduceScatterV2Func,
                                     MatmulReduceScatterAivModeSmallM<TemplateMMReduceScatterV2Func>>::swizzl_direct;
    using MatmulReduceScatterAivMode<
        TemplateMMReduceScatterV2Func,
        MatmulReduceScatterAivModeSmallM<TemplateMMReduceScatterV2Func>>::max_ub_ping_pong_size;
    using MatmulReduceScatterAivMode<TemplateMMReduceScatterV2Func,
                                     MatmulReduceScatterAivModeSmallM<TemplateMMReduceScatterV2Func>>::rank;
    using MatmulReduceScatterAivMode<TemplateMMReduceScatterV2Func,
                                     MatmulReduceScatterAivModeSmallM<TemplateMMReduceScatterV2Func>>::rank_size;
    using MatmulReduceScatterAivMode<TemplateMMReduceScatterV2Func,
                                     MatmulReduceScatterAivModeSmallM<TemplateMMReduceScatterV2Func>>::aiv_idx;
    using MatmulReduceScatterAivMode<TemplateMMReduceScatterV2Func,
                                     MatmulReduceScatterAivModeSmallM<TemplateMMReduceScatterV2Func>>::core_idx;
    using MatmulReduceScatterAivMode<TemplateMMReduceScatterV2Func,
                                     MatmulReduceScatterAivModeSmallM<TemplateMMReduceScatterV2Func>>::block_id;
    using MatmulReduceScatterAivMode<TemplateMMReduceScatterV2Func,
                                     MatmulReduceScatterAivModeSmallM<TemplateMMReduceScatterV2Func>>::core_num;
    using MatmulReduceScatterAivMode<
        TemplateMMReduceScatterV2Func,
        MatmulReduceScatterAivModeSmallM<TemplateMMReduceScatterV2Func>>::loop_num_per_comm;
    using MatmulReduceScatterAivMode<TemplateMMReduceScatterV2Func,
                                     MatmulReduceScatterAivModeSmallM<TemplateMMReduceScatterV2Func>>::m_loop;
    using MatmulReduceScatterAivMode<TemplateMMReduceScatterV2Func,
                                     MatmulReduceScatterAivModeSmallM<TemplateMMReduceScatterV2Func>>::n_loop;
    using MatmulReduceScatterAivMode<TemplateMMReduceScatterV2Func,
                                     MatmulReduceScatterAivModeSmallM<TemplateMMReduceScatterV2Func>>::cal_count;

    using MatmulReduceScatterAivMode<
        TemplateMMReduceScatterV2Func,
        MatmulReduceScatterAivModeSmallM<TemplateMMReduceScatterV2Func>>::StartBeforeFirstStep;
    using MatmulReduceScatterAivMode<TemplateMMReduceScatterV2Func,
                                     MatmulReduceScatterAivModeSmallM<TemplateMMReduceScatterV2Func>>::EndFirstStep;
    using MatmulReduceScatterAivMode<TemplateMMReduceScatterV2Func,
                                     MatmulReduceScatterAivModeSmallM<TemplateMMReduceScatterV2Func>>::ResetIpcFlags;
    using MatmulReduceScatterAivMode<TemplateMMReduceScatterV2Func,
                                     MatmulReduceScatterAivModeSmallM<TemplateMMReduceScatterV2Func>>::CheckBuffFlagV2;
    using MatmulReduceScatterAivMode<TemplateMMReduceScatterV2Func,
                                     MatmulReduceScatterAivModeSmallM<TemplateMMReduceScatterV2Func>>::CheckBuffFlag;
    using MatmulReduceScatterAivMode<TemplateMMReduceScatterV2Func,
                                     MatmulReduceScatterAivModeSmallM<TemplateMMReduceScatterV2Func>>::SetBuffFlag;
    using MatmulReduceScatterAivMode<TemplateMMReduceScatterV2Func,
                                     MatmulReduceScatterAivModeSmallM<TemplateMMReduceScatterV2Func>>::CopyGmToUbuf;
    using MatmulReduceScatterAivMode<TemplateMMReduceScatterV2Func,
                                     MatmulReduceScatterAivModeSmallM<TemplateMMReduceScatterV2Func>>::CopyUbufToGm;
    using MatmulReduceScatterAivMode<
        TemplateMMReduceScatterV2Func,
        MatmulReduceScatterAivModeSmallM<TemplateMMReduceScatterV2Func>>::CopyUbufToGmUnknown;

    using MatmulReduceScatterAivMode<TemplateMMReduceScatterV2Func,
                                     MatmulReduceScatterAivModeSmallM<TemplateMMReduceScatterV2Func>>::Init;
    using MatmulReduceScatterAivMode<TemplateMMReduceScatterV2Func,
                                     MatmulReduceScatterAivModeSmallM<TemplateMMReduceScatterV2Func>>::Process;

public:
    int32_t m_per_rank;

    constexpr static int32_t L1TileShapeK = quantFlag ? TILE_SHAPE_512 : TILE_SHAPE_256;
    constexpr static int32_t L0TileShapeK = quantFlag ? TILE_SHAPE_128 : TILE_SHAPE_64;

    __aicore__ inline void InitOutputMatrixCToZero();
    __aicore__ inline void CatlassMatmul();
    __aicore__ inline void AIVComm();
    __aicore__ inline void NotifyRemoteDataCopy(const CommColumnSplitter &taskSplit, bool needDequant,
                                                bool needPerChannel, bool needPerToken);
    __aicore__ inline void CompleteRemoteDataCopy(const CommColumnSplitter &taskSplit, bool needDequant,
                                                  bool needPerChannel, bool needPerToken);
    __aicore__ inline void SendCopyCompleteSignal();
    __aicore__ inline void ComfirmRemoteCopyDone();
    __aicore__ inline void Dequant(Arch::Resource<ArchTag> &resource, int32_t calIdx, int32_t n_st, int32_t n_ed,
                                   int32_t n_nxt_ed, int32_t n_comm_block, bool needPerChannel, bool needPerToken);

    inline __aicore__ GemmCoord GetBlockIdCoord(int32_t loopOffset, int32_t mLoop, int32_t nLoop, int32_t swizzlDirect,
                                                int32_t swizzlCount);
    inline __aicore__ GemmCoord GetBlockLocCoord(GemmCoord blockIdxCoord);
    inline __aicore__ GemmCoord GetBlockSizeCoord(GemmCoord blockIdxCoord, GemmCoord blockLocCoord, int32_t mLoop,
                                                  int32_t mSize, int32_t nLoop, int32_t nSize, int32_t kSize);

    using MatmulReduceScatterAivMode<TemplateMMReduceScatterV2Func,
                                     MatmulReduceScatterAivModeSmallM<TemplateMMReduceScatterV2Func>>::Padding;
};

template <TemplateMMReduceScatterV2Class>
inline __aicore__ GemmCoord MatmulReduceScatterAivModeSmallM<TemplateMMReduceScatterV2Func>::GetBlockIdCoord(
    int32_t loopOffset, int32_t mLoop, int32_t nLoop, int32_t swizzlDirect, int32_t swizzlCount)
{
    uint32_t kIdx = 0;
    int64_t mIdx, nIdx;
    GetSwizzledBlockIdx(loopOffset, mLoop, nLoop, swizzlDirect, swizzlCount, mIdx, nIdx);
    return GemmCoord{static_cast<uint32_t>(mIdx), static_cast<uint32_t>(nIdx), kIdx}; // idx在uint32_t范围内
}

template <TemplateMMReduceScatterV2Class>
inline __aicore__ GemmCoord
MatmulReduceScatterAivModeSmallM<TemplateMMReduceScatterV2Func>::GetBlockLocCoord(GemmCoord blockIdxCoord)
{
    return GemmCoord{blockIdxCoord.m() * m0, blockIdxCoord.n() * n0, blockIdxCoord.k() * L1TileShapeK};
}

template <TemplateMMReduceScatterV2Class>
inline __aicore__ GemmCoord MatmulReduceScatterAivModeSmallM<TemplateMMReduceScatterV2Func>::GetBlockSizeCoord(
    GemmCoord blockIdxCoord, GemmCoord blockLocCoord, int32_t mLoop, int32_t mSize, int32_t nLoop, int32_t nSize,
    int32_t kSize)
{
    uint32_t mActual = (blockIdxCoord.m() == (mLoop - 1)) ? (mSize - blockLocCoord.m()) : m0;
    uint32_t nActual = (blockIdxCoord.n() == (nLoop - 1)) ? (nSize - blockLocCoord.n()) : n0;
    uint32_t kActual = kSize;
    return GemmCoord{mActual, nActual, kActual};
}

template <TemplateMMReduceScatterV2Class>
__aicore__ inline void MatmulReduceScatterAivModeSmallM<TemplateMMReduceScatterV2Func>::Dequant(
    Arch::Resource<ArchTag> &resource, int32_t calIdx, int32_t n_st, int32_t n_ed, int32_t n_nxt_ed,
    int32_t n_comm_block, bool needPerChannel, bool needPerToken)
{
    using CType = Gemm::GemmType<int32_t, layout::RowMajor>;
    using ScaleType = Gemm::GemmType<float, layout::VectorLayout>;
    using PerTokenScaleType = Gemm::GemmType<float, layout::VectorLayout>;
    using BiasGType = Gemm::GemmType<biasType, layout::VectorLayout>;
    using DType = Gemm::GemmType<cType, layout::RowMajor>;
    using RowBroadcastMulType = Gemm::GemmType<float, layout::RowMajor>;
    using RowBroadcastAddType = Gemm::GemmType<float, layout::RowMajor>;
    using EpilogueTileShape = MatrixShape<TILE_SHAPE_64, TILE_SHAPE_128>;
    using TileRowBroadcastMul = Epilogue::Tile::TileRowBroadcastMul<ArchTag, RowBroadcastMulType, EpilogueTileShape>;
    using TileRowBroadcastAdd = Epilogue::Tile::TileRowBroadcastAdd<ArchTag, RowBroadcastAddType, EpilogueTileShape>;
    using BroadcastOneBlkType = Gemm::GemmType<float, layout::RowMajor>;
    using TileBroadcastOneBlk =
        Epilogue::Tile::TileBroadcastOneBlk<ArchTag, BroadcastOneBlkType, EpilogueTileShape::ROW>;
    using OneBlkColumnBroadcastMulType = Gemm::GemmType<float, layout::RowMajor>;
    using TileOneBlkColumnBroadcastMul =
        Epilogue::Tile::TileOneBlkColumnBroadcastMul<ArchTag, OneBlkColumnBroadcastMulType, EpilogueTileShape>;
    using TileCopy = Epilogue::Tile::TileCopy<ArchTag, CType, ScaleType, PerTokenScaleType, DType>;
    using TileScheduler = Epilogue::Tile::EpilogueHorizontalTileSwizzle;
    using BlockEpilogue = Epilogue::Block::BlockEpilogue<ArchTag, CType, ScaleType, PerTokenScaleType, BiasGType, DType,
                                                         TileRowBroadcastMul, TileRowBroadcastAdd, TileBroadcastOneBlk,
                                                         TileOneBlkColumnBroadcastMul, TileCopy, TileScheduler>;

    using LayoutC = layout::RowMajor;

    BlockEpilogue blockEpilogue(resource);
    __gm__ float32_t *perChannelScale =
        needPerChannel ? reinterpret_cast<__gm__ float32_t *>(perChannelScaleGM_) : nullptr;
    __gm__ float32_t *perTokenScale = needPerToken ? reinterpret_cast<__gm__ float32_t *>(perTokenScaleGM_) : nullptr;
    __gm__ int32_t *workspace = needPerChannel ? reinterpret_cast<__gm__ int32_t *>(gm_accum) : nullptr;

    int32_t totalLoop = m_loop * n_loop;
    for (int32_t p = 0; p < p_value; p++) {
        int32_t loopIdx = calIdx * loop_num_per_comm + p * core_num + core_idx;
        if (loopIdx >= totalLoop) {
            break;
        }
        GemmCoord blockIdxCoord = GetBlockIdCoord(loopIdx, m_loop, n_loop, swizzl_direct, swizzl_count);
        GemmCoord blockLocCoord = GetBlockLocCoord(blockIdxCoord);
        GemmCoord blockSizeCoord = GetBlockSizeCoord(blockIdxCoord, blockLocCoord, m_loop, m, n_loop, n, 0);
        LayoutC layout_tmp;
        int64_t gmOffsetC;
        if (blockIdxCoord.n() < n_ed) {
            n_comm_block = n_ed - n_st;
            int32_t comm_loopIdx = loopIdx - m_loop * n_st;
            layout_tmp = {static_cast<uint32_t>(m), static_cast<uint32_t>(n_comm_block * n0)};
            GemmCoord comm_blockIdxCoord =
                GetBlockIdCoord(comm_loopIdx, m_loop, n_comm_block, swizzl_direct, swizzl_count);
            if (swizzl_direct && ((n_st / swizzl_count) & 1)) {
                comm_blockIdxCoord.m() = m_loop - comm_blockIdxCoord.m() - 1;
            }
            GemmCoord comm_blockLocCoord = GetBlockLocCoord(comm_blockIdxCoord);
            MatrixCoord offsetC_comm{comm_blockLocCoord.m(), comm_blockLocCoord.n()};
            gmOffsetC = static_cast<int64_t>(n0) * n_st * m + layout_tmp.GetOffset(offsetC_comm);
        } else {
            n_comm_block = n_nxt_ed - n_ed;
            int32_t comm_loopIdx = loopIdx - m_loop * n_ed;
            layout_tmp = {static_cast<uint32_t>(m), static_cast<uint32_t>(n_comm_block * n0)};
            GemmCoord comm_blockIdxCoord =
                GetBlockIdCoord(comm_loopIdx, m_loop, n_comm_block, swizzl_direct, swizzl_count);
            if (swizzl_direct && ((n_ed / swizzl_count) & 1) & 1) {
                comm_blockIdxCoord.m() = m_loop - comm_blockIdxCoord.m() - 1;
            }
            GemmCoord comm_blockLocCoord = GetBlockLocCoord(comm_blockIdxCoord);
            MatrixCoord offsetC_comm{comm_blockLocCoord.m(), comm_blockLocCoord.n()};
            gmOffsetC = static_cast<int64_t>(n0) * n_ed * m + layout_tmp.GetOffset(offsetC_comm);
        }

        __gm__ cType *gmPeerMem = reinterpret_cast<__gm__ cType *>(buff[rank]);

        layout::VectorLayout layoutPerChannelScale{blockSizeCoord.n()};
        layout::VectorLayout layoutPerTokenScale{blockSizeCoord.m()};
        layout::VectorLayout layoutBias{blockSizeCoord.n()};
        __gm__ biasType *gmBiasOffset =
            (biasGM_ != 0) ? (reinterpret_cast<__gm__ biasType *>(biasGM_) + blockLocCoord.n()) : nullptr;
        if (needPerChannel & needPerToken) {
            blockEpilogue(perChannelScale + blockLocCoord.n(), layoutPerChannelScale, perTokenScale + blockLocCoord.m(),
                          layoutPerTokenScale, gmBiasOffset, layoutBias, workspace + gmOffsetC, layout_tmp,
                          gmPeerMem + gmOffsetC, layout_tmp, blockSizeCoord);
        } else if (needPerChannel) {
            blockEpilogue(perChannelScale + blockLocCoord.n(), layoutPerChannelScale, gmBiasOffset, layoutBias,
                          workspace + gmOffsetC, layout_tmp, gmPeerMem + gmOffsetC, layout_tmp, blockSizeCoord);
        } else if (needPerToken) {
            blockEpilogue(perTokenScale + blockLocCoord.m(), layoutPerTokenScale, gmBiasOffset, layoutBias,
                          gmPeerMem + gmOffsetC, layout_tmp, gmPeerMem + gmOffsetC, layout_tmp, blockSizeCoord);
        }
    }
}

template <TemplateMMReduceScatterV2Class>
__aicore__ inline void MatmulReduceScatterAivModeSmallM<TemplateMMReduceScatterV2Func>::CatlassMatmul()
{
    if ASCEND_IS_AIC {
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
        if (m0 == TILE_SHAPE_128) {
            using L1TileShape = GemmShape<TILE_SHAPE_128, TILE_SHAPE_256, L1TileShapeK>; // m n k
            using L0TileShape = GemmShape<TILE_SHAPE_128, TILE_SHAPE_256, L0TileShapeK>;
            using BlockMmadOpt = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType_, BType_,
                                                        CType_, BiasType_, TileCopy>;
            using MatmulKernel = Gemm::Kernel::MatmulReduceScatterAivModeSmallM<void, void, BlockMmadOpt, void,
                                                                                BlockScheduler30, aicCalBias>;
            typename MatmulKernel::Params params{
                processSize,
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
        } else {
            using L1TileShape = GemmShape<TILE_SHAPE_256, TILE_SHAPE_128, L1TileShapeK>; // m n k
            using L0TileShape = GemmShape<TILE_SHAPE_256, TILE_SHAPE_128, L0TileShapeK>;
            using BlockMmadOpt = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType_, BType_,
                                                        CType_, BiasType_, TileCopy>;
            using MatmulKernel = Gemm::Kernel::MatmulReduceScatterAivModeSmallM<void, void, BlockMmadOpt, void,
                                                                                BlockScheduler30, aicCalBias>;
            typename MatmulKernel::Params rowMajorParams{
                processSize,
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
            MatmulKernel rowMajorMatmulOp;
            rowMajorMatmulOp(rowMajorParams);
        }
    }
}

template <TemplateMMReduceScatterV2Class>
__aicore__ inline void MatmulReduceScatterAivModeSmallM<TemplateMMReduceScatterV2Func>::InitOutputMatrixCToZero()
{
    uint64_t totalLength = static_cast<uint64_t>(m) * static_cast<uint64_t>(n) / rank_size;
    constexpr uint32_t BLOCK_SIZE = 32;
    constexpr uint32_t UB_BLOCK_NUM = 20;
    uint32_t alignNum = BLOCK_SIZE / sizeof(cType);
    const uint32_t aiv_num = core_num * 2;
    uint64_t totalLengthAligned = ((totalLength + alignNum - 1) / alignNum) * alignNum;
    // 计算整核数量
    uint32_t formerNum = (totalLengthAligned / alignNum) % aiv_num;
    // 计算整核的数据量
    uint64_t formerLength = ((totalLengthAligned / aiv_num + alignNum - 1) / alignNum) * alignNum;
    uint64_t formerTileNum = formerLength / alignNum / UB_BLOCK_NUM;
    uint32_t formerTileLength = -1;
    uint32_t formerLastTileLength = -1;
    if ((formerLength / alignNum) % UB_BLOCK_NUM == 0 || formerTileNum == 0) {
        if (formerTileNum == 0) {
            formerTileNum = 1;
        }
        if (formerLength < UB_BLOCK_NUM * alignNum) {
            formerTileLength = formerLength / alignNum * alignNum;
            formerLastTileLength = formerTileLength;
        } else {
            formerTileLength = UB_BLOCK_NUM * alignNum;
            formerLastTileLength = formerTileLength;
        }
    } else {
        formerTileNum = formerTileNum + 1;
        formerTileLength = UB_BLOCK_NUM * alignNum;
        formerLastTileLength = (formerLength - (formerTileNum - 1) * formerTileLength);
    }

    // 计算尾核数量
    uint32_t tailNum = aiv_num - formerNum;
    // 计算尾核的数据量
    uint64_t tailLength = (totalLengthAligned / aiv_num / alignNum) * alignNum;

    uint64_t tailTileNum = tailLength / alignNum / UB_BLOCK_NUM;
    uint32_t tailTileLength = -1;
    uint32_t tailLastTileLength = -1;
    if ((tailLength / alignNum) % UB_BLOCK_NUM == 0 || tailTileNum == 0) {
        if (tailTileNum == 0) {
            tailTileNum = 1;
        }
        if (tailLength < UB_BLOCK_NUM * alignNum) {
            tailTileLength = tailLength / alignNum * alignNum;
            tailLastTileLength = tailTileLength;
        } else {
            tailTileLength = UB_BLOCK_NUM * alignNum;
            tailLastTileLength = tailTileLength;
        }
    } else {
        tailTileNum = tailTileNum + 1;
        tailTileLength = UB_BLOCK_NUM * alignNum;
        tailLastTileLength = (tailLength - (tailTileNum - 1) * tailTileLength);
    }

    // init
    uint64_t tileLength = -1;
    uint64_t tileNum = -1;
    uint64_t lastTileLength = -1;
    AscendC::GlobalTensor<cType> cGm;
    if (block_id < formerNum) {
        tileNum = formerTileNum;
        tileLength = formerTileLength;
        lastTileLength = formerLastTileLength;

        cGm.SetGlobalBuffer((__gm__ cType *)cGM_ + formerLength * block_id, formerLength);
    } else {
        tileNum = tailTileNum;
        tileLength = tailTileLength;
        lastTileLength = tailLastTileLength;
        cGm.SetGlobalBuffer((__gm__ cType *)cGM_ + formerLength * formerNum + tailLength * (block_id - formerNum),
                            tailLength);
    }

    AscendC::LocalTensor<cType> cLocal(AscendC::TPosition::VECOUT, 0, tileLength);

    cType inputVal(0);
    AscendC::Duplicate<cType>(cLocal, inputVal, tileLength);

    for (int32_t i = 0; i < tileNum; i++) {
        if (i == (tileNum - 1)) {
            AscendC::DataCopy(cGm[i * tileLength], cLocal, lastTileLength);
        } else {
            AscendC::DataCopy(cGm[i * tileLength], cLocal, tileLength);
        }
    }
}

template <TemplateMMReduceScatterV2Class>
__aicore__ inline void MatmulReduceScatterAivModeSmallM<TemplateMMReduceScatterV2Func>::NotifyRemoteDataCopy(
    const CommColumnSplitter &taskSplit, bool needDequant, bool needPerChannel, bool needPerToken)
{
    Arch::Resource<ArchTag> resource;
    for (int32_t cal_idx = 0; cal_idx < cal_count; ++cal_idx) {
        uint64_t flag_idx = cal_idx % MAX_BLOCK_COUNT_SM;
        // 本次通信的起始列块索引
        int32_t n_st = taskSplit.GetCulumnStartPos(cal_idx);
        // 本次通信的结束列块索引
        int32_t n_ed = cal_idx == cal_count - 1 ? n_loop : taskSplit.GetCulumnEndPos(cal_idx);
        int32_t n_comm_block = n_ed - n_st;

        // 接受AIC计算完成的信号
        WaitEvent(flag_idx);

        if (needDequant) {
            int32_t n_nxt_ed = cal_idx == cal_count - 2 ? n_loop : taskSplit.GetCulumnEndPos(cal_idx + 1);
            Dequant(resource, cal_idx, n_st, n_ed, n_nxt_ed, n_comm_block, needPerChannel, needPerToken);
        }
        SyncAll<true>();
        if (core_idx < rank_size) {
            SetBuffFlag((__gm__ int32_t *)buff[core_idx] + FLAG_OFFSET + rank, cal_idx + 1);
        }
    }
}

template <TemplateMMReduceScatterV2Class>
__aicore__ inline void MatmulReduceScatterAivModeSmallM<TemplateMMReduceScatterV2Func>::ComfirmRemoteCopyDone()
{
    // 收到远端卡读写完成信号，并复位信号，然后退出kernel。
    if (core_idx < rank_size) {
        CheckBuffFlag((__gm__ int32_t *)buff[rank] + FLAG_OFFSET + core_idx, cal_count + 1);
        SetBuffFlag((__gm__ int32_t *)buff[rank] + FLAG_OFFSET + core_idx, -1);
    }
}

template <TemplateMMReduceScatterV2Class>
__aicore__ inline void MatmulReduceScatterAivModeSmallM<TemplateMMReduceScatterV2Func>::CompleteRemoteDataCopy(
    const CommColumnSplitter &taskSplit, bool needDequant, bool needPerChannel, bool needPerToken)
{
    Arch::Resource<ArchTag> resource;

    for (int32_t cal_idx = 0; cal_idx < cal_count; ++cal_idx) {
        // 本次通信的起始列块索引
        int32_t n_st = taskSplit.GetCulumnStartPos(cal_idx);
        // 本次通信的结束列块索引
        int32_t n_ed = cal_idx == cal_count - 1 ? n_loop : taskSplit.GetCulumnEndPos(cal_idx);
        int32_t n_comm_block = n_ed - n_st;

        if (needDequant) {
            uint64_t flag_idx = cal_idx % MAX_BLOCK_COUNT_SM;
            // 接受AIC计算完成的信号
            WaitEvent(flag_idx);
            int32_t n_nxt_ed = cal_idx == cal_count - 2 ? n_loop : taskSplit.GetCulumnEndPos(cal_idx + 1);
            Dequant(resource, cal_idx, n_st, n_ed, n_nxt_ed, n_comm_block, needPerChannel, needPerToken);
        }
        SyncAll<true>();
        StartBeforeFirstStep();

        int32_t actual_move_size_gm2ub = (n_ed - n_st) * n0; // 实际拷贝的部分行长度
        // 单次 mte 读的部分行数量
        int32_t n_repeat =
            (max_ub_ping_pong_size / actual_move_size_gm2ub) ? (max_ub_ping_pong_size / actual_move_size_gm2ub) : 1;
        int32_t loop_copy_count = (m_per_rank + n_repeat - 1) / n_repeat;
        // 最后一次 mte 读的部分行数量
        int32_t last_n_repeat = m_per_rank - n_repeat * (loop_copy_count - 1);

        // AIV 远程数据读，并完成 GM 累加：一个AIV单次搬运会顺序搬运n_repeat个部分行，提高带宽。
        if (core_idx < loop_copy_count) {
            auto ub_offset = USED_UB_SIZE / 2 / sizeof(cType);
            LocalTensor<cType> ubTensor = uBuf_.template AllocTensor<cType>();
            LocalTensor<cType> copyTensor0 = ubTensor;
            LocalTensor<cType> copyTensor1 = ubTensor[ub_offset];

            uint32_t i_loop_db = 0;
            // 当前AIV核心需要完成单次搬运个数
            uint32_t core_loop = loop_copy_count / core_num + (core_idx < (loop_copy_count % core_num) ? 1 : 0);
            bool is_first_copy = true;
            for (int32_t idx_core = 0; idx_core < core_loop; idx_core++) {
                int32_t loop_copy_idx = idx_core * core_num + core_idx;
                int32_t cur_n_repeat = loop_copy_idx == (loop_copy_count - 1) ? last_n_repeat : n_repeat;
                // 一个AIV从全部卡中读取n_repeat个部分行的数据，并拷贝到GM上完成reduce累加。
                for (int32_t rank_idx = 0; rank_idx < rank_size; rank_idx++) {
                    int32_t target_rank = (core_idx + rank_idx) % rank_size;
                    auto event_id = (i_loop_db & 1) ? EVENT_ID0 : EVENT_ID1;
                    auto ub_buff_st = (i_loop_db & 1) ? copyTensor0 : copyTensor1;
                    i_loop_db++;
                    // 第一次拷贝，确保远端数据准备就绪
                    if (is_first_copy) {
                        CheckBuffFlagV2((__gm__ int32_t *)buff[rank] + FLAG_OFFSET + target_rank, cal_idx);
                    }
                    WaitFlag<HardEvent::MTE3_MTE2>(event_id);
                    auto src_peermem = reinterpret_cast<__gm__ cType *>(buff[target_rank]) + m * n_st * n0 +
                                       rank * m_per_rank * actual_move_size_gm2ub +
                                       loop_copy_idx * n_repeat * actual_move_size_gm2ub;
                    CopyGmToUbuf(ub_buff_st, src_peermem, 1, cur_n_repeat * actual_move_size_gm2ub * sizeof(cType) / 32,
                                 0, 0);
                    SetFlag<HardEvent::MTE2_MTE3>(event_id);
                    WaitFlag<HardEvent::MTE2_MTE3>(event_id);
                    auto dst_gm = reinterpret_cast<__gm__ cType *>(cGM_) + loop_copy_idx * n_repeat * n + n_st * n0;
                    int32_t actual_move_size_ub2gm = n_ed * n0 >= n ? n - n_st * n0 : actual_move_size_gm2ub;
                    uint16_t dst_stride = ((n - actual_move_size_ub2gm) * sizeof(cType));
                    CopyUbufToGmUnknown(
                        nAlign16, dst_gm, ub_buff_st, cur_n_repeat, actual_move_size_ub2gm * sizeof(cType),
                        (actual_move_size_gm2ub - actual_move_size_ub2gm) * sizeof(cType) / 32, dst_stride);
                    SetFlag<HardEvent::MTE3_MTE2>(event_id);
                }
                is_first_copy = false;
            }
        }
        EndFirstStep();
    }
}

template <TemplateMMReduceScatterV2Class>
__aicore__ inline void MatmulReduceScatterAivModeSmallM<TemplateMMReduceScatterV2Func>::SendCopyCompleteSignal()
{
    // 通知远端卡整个读取过程完成，可以退出。
    if (core_idx < rank_size) {
        SetBuffFlag((__gm__ int32_t *)buff[core_idx] + FLAG_OFFSET + rank, cal_count + 1);
    }
}

template <TemplateMMReduceScatterV2Class>
__aicore__ inline void MatmulReduceScatterAivModeSmallM<TemplateMMReduceScatterV2Func>::AIVComm()
{
    if ASCEND_IS_AIV {
        Padding();
        ResetIpcFlags(rank_size);

        // 仅在x2ScaleType为INT64,输出为FP16时走fixPipe，其余场景均需要AIV做反量化
        bool needPerChannel = quantFlag && !(isX2ScaleTypeInt64 && std::is_same<cType, float16_t>::value);
        // 可简化为只校验dequant_type；host侧校验，只有输入为quant场景，dequantType才能传有效值。
        bool needPerToken = quantFlag && dequant_type == DequantType::PER_TOKEN;
        bool needDequant = needPerToken || needPerChannel;

        PipeBarrier<PIPE_ALL>();
        // 初始化输出矩阵C为0，为后续reduce累加准备。
        InitOutputMatrixCToZero();

        // 通过检查其他卡的初始话的flag位，确认其他卡准备接受本卡的flag远程写。
        if (core_idx < rank_size && 1 == aiv_idx) {
            CheckBuffFlag((__gm__ int32_t *)buff[core_idx] + FLAG_OFFSET + rank, 0);
        }

        SyncAll<true>();
        m_per_rank = m / rank_size;
        // 任务分解器：由于每次通信任务块不相同，后续通过该对象获取本次通信块的列索引范围。
        CommColumnSplitter taskSplit(loop_num_per_comm, m_loop, swizzl_direct ? swizzl_count : 1, n_loop);

        if (aiv_idx == 1) {
            NotifyRemoteDataCopy(taskSplit, needDequant, needPerChannel, needPerToken);
            SyncAll<true>();
            ComfirmRemoteCopyDone();
        } else {
            CompleteRemoteDataCopy(taskSplit, needDequant, needPerChannel, needPerToken);
            SyncAll<true>();
            SendCopyCompleteSignal();
        }
    }
}

} // namespace MatmulReduceScatterV2Impl
#endif // MATMUL_REDUCE_SCATTER_AIV_MODE_SMALL_M_H
