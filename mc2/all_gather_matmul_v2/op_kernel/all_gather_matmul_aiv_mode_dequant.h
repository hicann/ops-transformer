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
 * \file all_gather_matmul_aiv_mode_dequant.h
 * \brief
 */

#ifndef CATLASS_GEMM_KERNEL_TEMPLATE_DEQUANT_HPP
#define CATLASS_GEMM_KERNEL_TEMPLATE_DEQUANT_HPP

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
#include "all_gather_matmul_aiv_mode_block_epilogue_dequant.h"

#define DEQUANT_ARGS_CALL() \
    rowNum, colNum, perChannelScale, perTokenScale, workspace, reinterpret_cast<GM_ADDR>(output), tileM0, tileN0, \
        pValue, swizzlDirect, swizzlCount, blockSt, blockSize, blockStInWorkspace, blockSizeInWorkspace, blockIdx, \
        coreNum, worldSize, resource, needPerChannel, needPerToken, isInt4Type

#define DEQUANT_ARGS_FUN() \
    uint32_t rowNum, uint32_t colNum, __gm__ float32_t *perChannelScale, __gm__ float32_t *perTokenScale, \
        __gm__ int32_t *workspace, GM_ADDR output, uint32_t tileM0, uint32_t tileN0, uint32_t pValue, \
        uint32_t swizzlDirect, uint32_t swizzlCount, int64_t blockSt, int64_t blockSize, int64_t blockStInWorkspace, \
        int64_t blockSizeInWorkspace, uint32_t blockIdx, uint32_t coreNum, uint32_t worldSize, \
        Arch::Resource<Arch::AtlasA2> resource, bool needPerChannel = false, bool needPerToken = false, \
                                                     bool isInt4Type = false

template <typename OutputType>
class DequantRunner {
public:
    using ArchTag = Arch::AtlasA2;
    using ScaleType = Gemm::GemmType<float, layout::VectorLayout>;
    using PerTokenScaleType = Gemm::GemmType<float, layout::VectorLayout>;
    using CType = Gemm::GemmType<int32_t, layout::RowMajor>;
    using DType = Gemm::GemmType<OutputType, layout::RowMajor>;
    using RowBroadcastMulType = Gemm::GemmType<float, layout::RowMajor>;
    using BroadcastOneBlkType = Gemm::GemmType<float, layout::RowMajor>;
    using OneBlkColumnBroadcastMulType = Gemm::GemmType<float, layout::RowMajor>;

    using EpilogueTileShape = MatrixShape<64, 128>;
    using TileRowBroadcastMul = Epilogue::Tile::TileRowBroadcastMul<ArchTag, RowBroadcastMulType, EpilogueTileShape>;
    using TileBroadcastOneBlk =
        Epilogue::Tile::TileBroadcastOneBlk<ArchTag, BroadcastOneBlkType, EpilogueTileShape::ROW>;
    using TileOneBlkColumnBroadcastMul =
        Epilogue::Tile::TileOneBlkColumnBroadcastMul<ArchTag, OneBlkColumnBroadcastMulType, EpilogueTileShape>;
    using TileCopy = Epilogue::Tile::TileCopy<ArchTag, CType, ScaleType, PerTokenScaleType, DType>;
    using TileScheduler = Epilogue::Tile::EpilogueHorizontalTileSwizzle;

    using BlockEpilogue =
        Epilogue::Block::BlockEpilogue<ArchTag, CType, ScaleType, PerTokenScaleType, DType, TileRowBroadcastMul,
                                       TileBroadcastOneBlk, TileOneBlkColumnBroadcastMul, TileCopy, TileScheduler>;

    uint32_t DefaultSwizzleDirect = 0;
    uint32_t DefaultSwizzleCount = 3;

    __aicore__ explicit DequantRunner() = default;

    FORCE_INLINE_AICORE GemmCoord GetBlockIdCoord(int32_t loopOffset, int32_t mLoop, int32_t nLoop,
                                                  int32_t swizzlDirect, int32_t swizzlCount)
    {
        uint32_t kIdx = 0;
        int64_t mIdx, nIdx;
        GetBlockIdx(loopOffset, mLoop, nLoop, swizzlDirect, swizzlCount, mIdx, nIdx);
        return GemmCoord{static_cast<uint32_t>(mIdx), static_cast<uint32_t>(nIdx), kIdx}; // idx在uint32_t范围内
    }

    FORCE_INLINE_AICORE GemmCoord GetBlockLocCoord(GemmCoord blockIdxCoord)
    {
        return GemmCoord{blockIdxCoord.m() * m0, blockIdxCoord.n() * n0, 0};
    }

    FORCE_INLINE_AICORE GemmCoord GetBlockSizeCoord(GemmCoord blockIdxCoord, GemmCoord blockLocCoord, int32_t mLoop,
                                                    int32_t mSize, int32_t nLoop, int32_t nSize, int32_t kSize)
    {
        uint32_t mActual = (blockIdxCoord.m() == (mLoop - 1)) ? (mSize - blockLocCoord.m()) : m0;
        uint32_t nActual = (blockIdxCoord.n() == (nLoop - 1)) ? (nSize - blockLocCoord.n()) : n0;
        uint32_t kActual = kSize;
        return GemmCoord{mActual, nActual, kActual};
    }

    FORCE_INLINE_AICORE void Run(DEQUANT_ARGS_FUN())
    {
        m0 = tileM0;
        n0 = tileN0;
        uint32_t m = rowNum;
        uint32_t n = colNum;
        uint32_t mLoop = DivCeil(m, m0);
        uint32_t nLoop = DivCeil(n, n0);
        uint32_t blockLoop = mLoop * nLoop;
        uint32_t totalLoop = blockLoop * worldSize;

        if (needPerChannel && !needPerToken) {
            DefaultSwizzleDirect = 0;
        } else if (!needPerChannel && needPerToken) {
            DefaultSwizzleDirect = 1;
        }

        BlockEpilogue blockEpilogue(resource);
        layout::RowMajor layoutC{m, n};
        __gm__ OutputType *gmD = reinterpret_cast<__gm__ OutputType *>(output);
        for (uint32_t loopIdx = blockIdx; loopIdx < totalLoop; loopIdx += coreNum) {
            uint32_t dataBlockIdx = loopIdx / blockLoop;
            uint32_t blockLoopIdx = loopIdx % blockLoop;
            GemmCoord blockIdxCoord =
                GetBlockIdCoord(blockLoopIdx, mLoop, nLoop, DefaultSwizzleDirect, DefaultSwizzleCount);
            GemmCoord blockLocCoord = GetBlockLocCoord(blockIdxCoord);
            GemmCoord blockSizeCoord = GetBlockSizeCoord(blockIdxCoord, blockLocCoord, mLoop, m, nLoop, n, 0);

            MatrixCoord offsetC{blockLocCoord.m(), blockLocCoord.n()};
            int64_t outputDataOffset = dataBlockIdx * blockSize + blockSt + layoutC.GetOffset(offsetC);
            int64_t workspaceDataOffset =
                dataBlockIdx * blockSizeInWorkspace + blockStInWorkspace + layoutC.GetOffset(offsetC);
            int64_t perTokenScaleOffset = (dataBlockIdx * blockSize + blockSt) / n + blockLocCoord.m();
            layout::VectorLayout layoutPerChannelScale{blockSizeCoord.n()};
            layout::VectorLayout layoutPerTokenScale{blockSizeCoord.m()};
            if (needPerChannel && needPerToken) {
                blockEpilogue(perChannelScale + blockLocCoord.n(), layoutPerChannelScale,
                              perTokenScale + perTokenScaleOffset, layoutPerTokenScale, workspace + workspaceDataOffset,
                              layoutC, gmD + outputDataOffset, layoutC, blockSizeCoord, isInt4Type);
            } else if (needPerChannel) {
                blockEpilogue(perChannelScale + blockLocCoord.n(), layoutPerChannelScale,
                              workspace + workspaceDataOffset, layoutC, gmD + outputDataOffset, layoutC, blockSizeCoord,
                              isInt4Type);
            } else if (needPerToken) {
                blockEpilogue(perTokenScale + perTokenScaleOffset, layoutPerTokenScale, gmD + outputDataOffset, layoutC,
                              blockSizeCoord, isInt4Type);
            }
        }
    }

private:
    uint32_t m0;
    uint32_t n0;
};

#endif // CATLASS_GEMM_KERNEL_TEMPLATE_DEQUANT_HPP
