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
 * \file matmul_reduce_scatter_aiv_mode_dequant.h
 * \brief
 */
#ifndef MATMUL_REDUCE_SCATTER_AIV_MODE_DEQUANT_H
#define MATMUL_REDUCE_SCATTER_AIV_MODE_DEQUANT_H

#include "../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/tla_catlass.hpp"
#include "../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/arch/tla_arch_cross_core_sync.hpp"
#include "../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/arch/tla_arch_resource.hpp"
#include "../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/tla_coord.hpp"
#include "../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/layout/tla_layout_layout.hpp"
#include "../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/detail/tla_detail_callback.hpp"
#include "../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/tla_gemm_coord.hpp"
#include "../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/tla_matrix_coord.hpp"
#include "../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/epilogue/block/tla_block_epilogue.hpp"
#include "../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/epilogue/tile/tla_epilogue_tile_broadcast_mul.hpp"
#include "../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/epilogue/tile/tla_epilogue_tile_broadcast_one_blk.hpp"
#include "../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/epilogue/tile/tla_epilogue_tile_swizzle.hpp"
#include "../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/gemm/block/tla_gemm_block_mmad.hpp"
#include "../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/gemm/block/tla_gemm_block_swizzle.hpp"
#include "../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/gemm/tla_gemm_gemm_type.hpp"
#include "matmul_reduce_scatter_aiv_mode_block_epilogue_dequant.h"
#include "matmul_reduce_scatter_aiv_mode_util.h"
#include "mrs_v2_tile_broadcast_add.hpp"

using namespace matmulReduceScatterV2_util;
namespace dequant {
template <typename BiasType, typename OutputType>
class DequantRunner {
public:
    using ArchTag = Arch::AtlasA2;
    using ScaleType = Gemm::GemmType<float, layout::VectorLayout>;
    using PerTokenScaleType = Gemm::GemmType<float, layout::VectorLayout>;
    using BiasGType = Gemm::GemmType<BiasType, layout::VectorLayout>;
    using CType = Gemm::GemmType<int32_t, layout::RowMajor>;
    using DType = Gemm::GemmType<OutputType, layout::RowMajor>;
    using RowBroadcastMulType = Gemm::GemmType<float, layout::RowMajor>;
    using BroadcastOneBlkType = Gemm::GemmType<float, layout::RowMajor>;
    using OneBlkColumnBroadcastMulType = Gemm::GemmType<float, layout::RowMajor>;
    using RowBroadcastAddType = Gemm::GemmType<float, layout::RowMajor>;

    using EpilogueTileShape = MatrixShape<TILE_SHAPE_64, TILE_SHAPE_128>;
    using TileRowBroadcastMul = Epilogue::Tile::TileRowBroadcastMul<ArchTag, RowBroadcastMulType, EpilogueTileShape>;
    using TileRowBroadcastAdd = Epilogue::Tile::TileRowBroadcastAdd<ArchTag, RowBroadcastAddType, EpilogueTileShape>;
    using TileBroadcastOneBlk =
        Epilogue::Tile::TileBroadcastOneBlk<ArchTag, BroadcastOneBlkType, EpilogueTileShape::ROW>;
    using TileOneBlkColumnBroadcastMul =
        Epilogue::Tile::TileOneBlkColumnBroadcastMul<ArchTag, OneBlkColumnBroadcastMulType, EpilogueTileShape>;
    using TileCopy = Epilogue::Tile::TileCopy<ArchTag, CType, ScaleType, PerTokenScaleType, DType>;
    using TileScheduler = Epilogue::Tile::EpilogueHorizontalTileSwizzle;

    using BlockEpilogue = Epilogue::Block::BlockEpilogue<ArchTag, CType, ScaleType, PerTokenScaleType, BiasGType, DType,
                                                         TileRowBroadcastMul, TileRowBroadcastAdd, TileBroadcastOneBlk,
                                                         TileOneBlkColumnBroadcastMul, TileCopy, TileScheduler>;

    uint32_t DefaultSwizzleDirect = 0;
    uint32_t DefaultSwizzleCount = 3;

    __aicore__ explicit DequantRunner() = default;

    __aicore__ inline GemmCoord GetBlockIdCoord(int32_t loopOffset, int32_t mLoop, int32_t nLoop, int32_t swizzlDirect,
                                                int32_t swizzlCount)
    {
        uint32_t kIdx = 0;
        int64_t mIdx, nIdx;
        GetSwizzledBlockIdx(loopOffset, mLoop, nLoop, swizzlDirect, swizzlCount, mIdx, nIdx);
        return GemmCoord{static_cast<uint32_t>(mIdx), static_cast<uint32_t>(nIdx), kIdx}; // idx在uint32_t范围内
    }

    __aicore__ inline GemmCoord GetBlockLocCoord(GemmCoord blockIdxCoord)
    {
        return GemmCoord{blockIdxCoord.m() * m0, blockIdxCoord.n() * n0, 0};
    }

    __aicore__ inline GemmCoord GetBlockSizeCoord(GemmCoord blockIdxCoord, GemmCoord blockLocCoord, int32_t mLoop,
                                                  int32_t mSize, int32_t nLoop, int32_t nSize, int32_t kSize)
    {
        uint32_t mActual = (blockIdxCoord.m() == (mLoop - 1)) ? (mSize - blockLocCoord.m()) : m0;
        uint32_t nActual = (blockIdxCoord.n() == (nLoop - 1)) ? (nSize - blockLocCoord.n()) : n0;
        uint32_t kActual = kSize;
        return GemmCoord{mActual, nActual, kActual};
    }

    __aicore__ inline void RunMatmulReduceScatter(DEQUANT_ARGS_FUN())
    {
        m0 = tileM0;
        n0 = tileN0;
        uint32_t finalM = rowNum / rankSize;
        uint32_t finalN = colNum;
        uint32_t mLoop = DivCeil(finalM, m0);
        uint32_t nLoop = DivCeil(finalN, n0);
        uint32_t blockLoop = mLoop * nLoop;
        uint32_t totalLoop = blockLoop * rankSize;
        uint32_t blockSize = m0 * n0;
        uint32_t loopNumPerCom = pValue * coreNum;
        uint32_t loopSt = calIdx * loopNumPerCom;
        uint32_t peerMemBlockSize = blockSize * loopNumPerCom / rankSize;

        BlockEpilogue blockEpilogue(resource);
        layout::RowMajor layoutBlock{m0, n0};
        layout::RowMajor layoutOutput{finalM, finalN};
        __gm__ OutputType *gmPeerMem = reinterpret_cast<__gm__ OutputType *>(peerMem);
        __gm__ OutputType *gmOutput = reinterpret_cast<__gm__ OutputType *>(output);
        __gm__ BiasType *gmbias = reinterpret_cast<__gm__ BiasType *>(biasptr);
        for (int32_t p = 0; p < pValue; p++) {
            int32_t loopIdx = loopSt + p * coreNum + coreIdx;
            if (loopIdx >= totalLoop) {
                break;
            }
            int32_t dstRankIdx = loopIdx % rankSize;
            int32_t inRankIdx = loopIdx / rankSize;
            GemmCoord blockIdxCoord = GetBlockIdCoord(inRankIdx, mLoop, nLoop, swizzlDirect, swizzlCount);
            GemmCoord blockLocCoord = GetBlockLocCoord(blockIdxCoord);
            GemmCoord blockSizeCoord = GetBlockSizeCoord(blockIdxCoord, blockLocCoord, mLoop, finalM, nLoop, finalN, 0);

            layout::VectorLayout layoutPerChannelScale{blockSizeCoord.n()};
            layout::VectorLayout layoutBias{blockSizeCoord.n()};
            layout::VectorLayout layoutPerTokenScale{blockSizeCoord.m()};
            uint32_t dataBlockOffset = dstRankIdx * peerMemBlockSize + (loopIdx - loopSt) / rankSize * blockSize;
            layout::RowMajor layoutDst = layoutBlock;
            __gm__ OutputType *gmDst = gmPeerMem + dataBlockOffset;
            if (dstRankIdx == rankIdx) {
                MatrixCoord offsetC{blockLocCoord.m(), blockLocCoord.n()};
                layoutDst = layoutOutput;
                gmDst = gmOutput + layoutOutput.GetOffset(offsetC);
            }
            __gm__ BiasType *gmBiasOffset = (gmbias != nullptr) ? (gmbias + blockLocCoord.n()) : nullptr;
            if (needPerChannel && needPerToken) {
                blockEpilogue(perChannelScale + blockLocCoord.n(), layoutPerChannelScale,
                              perTokenScale + dstRankIdx * finalM + blockLocCoord.m(), layoutPerTokenScale,
                              gmBiasOffset, layoutBias, workspace + dataBlockOffset, layoutBlock, gmDst, layoutDst,
                              blockSizeCoord);
            } else if (needPerChannel) {
                blockEpilogue(perChannelScale + blockLocCoord.n(), layoutPerChannelScale, gmBiasOffset, layoutBias,
                              workspace + dataBlockOffset, layoutBlock, gmDst, layoutDst, blockSizeCoord);
            } else if (needPerToken) {
                blockEpilogue(perTokenScale + dstRankIdx * finalM + blockLocCoord.m(), layoutPerTokenScale,
                              gmBiasOffset, layoutBias, gmPeerMem + dataBlockOffset, layoutBlock, gmDst, layoutDst,
                              blockSizeCoord);
            }
        }
    }

private:
    uint32_t m0;
    uint32_t n0;
};
} // namespace dequant
#endif // MATMUL_REDUCE_SCATTER_AIV_MODE_DEQUANT_H
