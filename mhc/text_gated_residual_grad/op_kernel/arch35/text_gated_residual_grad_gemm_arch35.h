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
 * \file text_gated_residual_grad_gemm_arch35.h
 * \brief Ascend 950 CATLASS GEMM backend for TextGatedResidualGrad.
 */
#ifndef TEXT_GATED_RESIDUAL_GRAD_GEMM_ARCH35_H
#define TEXT_GATED_RESIDUAL_GRAD_GEMM_ARCH35_H

#include "../text_gated_residual_grad_constants.h"

#if !defined(CATLASS_ARCH) || CATLASS_ARCH != HC_GRAD_ARCH_A5
#error "text_gated_residual_grad_gemm_arch35.h requires CATLASS_ARCH=3510"
#endif

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include <type_traits>

#include "kernel_operator.h"
#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/kernel/basic_matmul_tla.hpp"
#include "catlass/layout/layout.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"
#include "text_gated_residual_grad_pipeline_arch35.h"

namespace TextGatedResidualGrad {

template <typename D_T>
class HcGradGemm {
public:
    // The A5 Cube backend has no runtime state; keep the same initialization interface as arch22.
    __aicore__ inline void Init(uint32_t /*aicNum*/) {}

    // C1a: dact = dup @ wUp, split-K across all AIC; FP32 partials land in workspace.
    static __aicore__ inline void RunC1aSplitK(GM_ADDR dup, GM_ADDR wUp, GM_ADDR partial, uint32_t rows, uint32_t r,
                                               uint32_t d, uint32_t splitkFactor)
    {
        using BlockMmad = BlockMmadT<C1aTile, C1aTile, RowMajor, RowMajor, float>;
        using BlockScheduler =
            Catlass::Gemm::Block::SplitkGemmIdentityBlockSwizzle<HC_GRAD_SWIZZLE_OFFSET, HC_GRAD_SWIZZLE_M_FIRST>;
        Catlass::GemmCoord problemShape{rows, r, d};
        BlockScheduler scheduler(problemShape,
                                 Catlass::GemmCoord(HC_GRAD_CUBE_TILE_M_A5, HC_GRAD_C1A_TILE_N, HC_GRAD_CUBE_TILE_K),
                                 splitkFactor);
        const uint32_t coreLoops = scheduler.GetCoreLoops();

        Catlass::Arch::Resource<ArchTag> resource;
        BlockMmad blockMmad(resource);

        AscendC::GlobalTensor<D_T> gmA;
        gmA.SetGlobalBuffer(reinterpret_cast<__gm__ D_T*>(dup));
        AscendC::GlobalTensor<D_T> gmB;
        gmB.SetGlobalBuffer(reinterpret_cast<__gm__ D_T*>(wUp));
        AscendC::GlobalTensor<float> gmW;
        gmW.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(partial));

        auto layoutA = MakeLayout<RowMajor>(rows, d, d);
        auto layoutB = MakeLayout<RowMajor>(d, r, r);
        auto layoutC = MakeLayout<RowMajor>(rows, r, r);
        auto tensorA = tla::MakeTensor(gmA, layoutA, Catlass::Arch::PositionGM{});
        auto tensorB = tla::MakeTensor(gmB, layoutB, Catlass::Arch::PositionGM{});
        const size_t lenC = static_cast<size_t>(rows) * r;

        for (uint32_t loopIdx = AscendC::GetBlockIdx(); loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
            const Catlass::GemmCoord blockCoord = scheduler.GetBlockCoord(loopIdx);
            const uint32_t slice = scheduler.GetSplitkSliceIdx(loopIdx);
            const Catlass::GemmCoord actual = scheduler.GetActualBlockShape(blockCoord, slice);
            auto tensorBlockA = tla::GetTile(
                tensorA, tla::MakeCoord(blockCoord.m() * HC_GRAD_CUBE_TILE_M_A5, blockCoord.k() * HC_GRAD_CUBE_TILE_K),
                tla::MakeShape(actual.m(), actual.k()));
            auto tensorBlockB = tla::GetTile(
                tensorB, tla::MakeCoord(blockCoord.k() * HC_GRAD_CUBE_TILE_K, blockCoord.n() * HC_GRAD_C1A_TILE_N),
                tla::MakeShape(actual.k(), actual.n()));
            auto tensorC = tla::MakeTensor(gmW[lenC * slice], layoutC, Catlass::Arch::PositionGM{});
            auto tensorBlockC = tla::GetTile(
                tensorC, tla::MakeCoord(blockCoord.m() * HC_GRAD_CUBE_TILE_M_A5, blockCoord.n() * HC_GRAD_C1A_TILE_N),
                tla::MakeShape(actual.m(), actual.n()));
            blockMmad(tensorBlockA, tensorBlockB, tensorBlockC, actual);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
    }

    // C1b: dwUpPartial = dup^T @ act
    static __aicore__ inline void RunC1b(GM_ADDR dup, GM_ADDR act, GM_ADDR dwUpPartial, uint32_t rows, uint32_t r,
                                         uint32_t d)
    {
        RunGemm<C1Tile, C1Tile, ColumnMajor, RowMajor, float>(d, r, rows, dup, d, act, r, dwUpPartial, r);
    }

    // Compute dwDown/dwGate = packedT[0:R+gateRows] @ xn in one M=R+gateRows GEMM.
    // Separate M=R and M=gateRows GEMMs would reread all of xn [rows,D] from GM to L1
    // for the small gate GEMM. Combining them avoids this repeated transfer.
    // The contiguous FP32 result [R+gateRows,D] holds R dwDown rows followed by gateRows dwGate rows;
    // REDUCE_WD separates them into their respective FP32 accumulators.
    static __aicore__ inline void RunG56Merged(GM_ADDR packedT, GM_ADDR xn, GM_ADDR dwOut, uint32_t rows, uint32_t r,
                                               uint32_t d, uint32_t gateRows, bool hasGate)
    {
        const uint32_t m = hasGate ? (r + gateRows) : r;
        RunGemm<C1Tile, C1Tile, RowMajor, RowMajor, float>(m, d, rows, packedT, rows, xn, d, dwOut, d);
    }

    // Single-task fast paths: accumulate straight into the FP32 outputs.
    static __aicore__ inline void RunC1bDirect(GM_ADDR dup, GM_ADDR act, GM_ADDR dwUp, uint32_t rows, uint32_t r,
                                               uint32_t d)
    {
        RunGemm<C1Tile, C1Tile, ColumnMajor, RowMajor, float>(d, r, rows, dup, d, act, r, dwUp, r);
    }

    // For one task, compute separately: dwDown goes directly to its FP32 accumulator and dwGate to an aligned partial.
    // This saves an r*D*4-byte REDUCE_WD copy for dwDown, at the cost of rereading xn for the gate GEMM.
    static __aicore__ inline void RunG56Direct(GM_ADDR packedT, GM_ADDR xn, GM_ADDR dwDown, GM_ADDR dwGate,
                                               uint32_t rows, uint32_t r, uint32_t d, uint32_t gateRows, bool hasGate)
    {
        RunGemm<C1Tile, C1Tile, RowMajor, RowMajor, float>(r, d, rows, packedT, rows, xn, d, dwDown, d);
        if (hasGate) {
            RunGemm<C1Tile, C1Tile, RowMajor, RowMajor, float>(gateRows, d, rows, packedT + r * rows * sizeof(D_T),
                                                               rows, xn, d, dwGate, d);
        }
    }

    // dxnorm = ddown @ w_down + gpre @ w_gate. Concatenate wDown/wGate along K into wConcat,
    // then compute both branches in one K=R+C GEMM that writes dx2.
    // One L0C accumulation and FIXP avoid writing and reading a separate FP32 [N,D] intermediate.
    static __aicore__ inline void RunDx(GM_ADDR packed, GM_ADDR wDown, GM_ADDR wGate, GM_ADDR wConcat, GM_ADDR dxOut,
                                        uint32_t rows, uint32_t d, uint32_t r, uint32_t c, uint32_t packedStride,
                                        bool hasGate)
    {
        if (hasGate) {
            RunGemm<Dx2Tile, Dx2Tile, RowMajor, RowMajor, float>(rows, d, r + c, packed, packedStride, wConcat, d,
                                                                 dxOut, d);
        } else {
            RunGemm<Dx2Tile, Dx2Tile, RowMajor, RowMajor, float>(rows, d, r, packed, packedStride, wDown, d, dxOut, d);
        }
    }

    // Read packed directly as ColumnMajor to compute packed^T @ xn without materializing packedT.
    static __aicore__ inline void RunG56MergedPacked(GM_ADDR packed, GM_ADDR xn, GM_ADDR dwOut, uint32_t rows,
                                                     uint32_t r, uint32_t d, uint32_t packedStride, uint32_t gateRows,
                                                     bool hasGate)
    {
        const uint32_t m = hasGate ? (r + gateRows) : r;
        RunGemm<C1Tile, C1Tile, ColumnMajor, RowMajor, float>(m, d, rows, packed, packedStride, xn, d, dwOut, d);
    }

    static __aicore__ inline void RunG56DirectPacked(GM_ADDR packed, GM_ADDR xn, GM_ADDR dwDown, GM_ADDR dwGate,
                                                     uint32_t rows, uint32_t r, uint32_t d, uint32_t packedStride,
                                                     uint32_t gateRows, bool hasGate)
    {
        RunGemm<C1Tile, C1Tile, ColumnMajor, RowMajor, float>(r, d, rows, packed, packedStride, xn, d, dwDown, d);
        if (hasGate) {
            RunGemm<C1Tile, C1Tile, ColumnMajor, RowMajor, float>(gateRows, d, rows, packed + r * sizeof(D_T),
                                                                  packedStride, xn, d, dwGate, d);
        }
    }

private:
    using ArchTag = Catlass::Arch::Ascend950;
    // Ping-pong arguments: ArchTag, UnitFlag=false, HF32=false, L0C_STAGES=1,
    // L1_RESIDENT=false, L1A_STAGES=4, L1B_STAGES=4, L0A_STAGES=2, L0B_STAGES=2.
    // The main 256x256 tile uses 256 KiB of single-stage FP32 L0C; C1a's 256x128 tile uses 128 KiB.
    // Disable UnitFlag and use CATLASS's explicit MMAD/Fixpipe synchronization.
    // CATLASS v1.6+ provides the mutex implementation; older headers use event synchronization.
    // Both policies keep the same tile geometry, buffer stages and FP32 accumulation.
#if __has_include("catlass/gemm/block/block_mmad_pingpong_mutex_tla.hpp")
    using DispatchPolicy =
        Catlass::Gemm::MmadPingpongMutex<ArchTag, false, false, HC_GRAD_L0C_STAGES, false, HC_GRAD_L1_STAGES,
                                         HC_GRAD_L1_STAGES, HC_GRAD_L0_STAGES, HC_GRAD_L0_STAGES>;
#else
    using DispatchPolicy =
        Catlass::Gemm::MmadPingpong<ArchTag, false, false, HC_GRAD_L0C_STAGES, false, HC_GRAD_L1_STAGES,
                                    HC_GRAD_L1_STAGES, HC_GRAD_L0_STAGES, HC_GRAD_L0_STAGES>;
#endif
    using RowMajor = Catlass::layout::RowMajor;
    using ColumnMajor = Catlass::layout::ColumnMajor;

    // Main tile: N=256 uses 256 KiB of single-stage L0C and halves rereads of A across N tiles compared with N=128.
    using C1Tile =
        tla::Shape<tla::Int<HC_GRAD_CUBE_TILE_M_A5>, tla::Int<HC_GRAD_CUBE_TILE_N_A5>, tla::Int<HC_GRAD_CUBE_TILE_K>>;
    // C1a's explicit split-K tile loop uses N=128; N=256 produced incorrect results for its 64-wide tail.
    using C1aTile =
        tla::Shape<tla::Int<HC_GRAD_CUBE_TILE_M_A5>, tla::Int<HC_GRAD_C1A_TILE_N>, tla::Int<HC_GRAD_CUBE_TILE_K>>;
    using Dx2Tile =
        tla::Shape<tla::Int<HC_GRAD_CUBE_TILE_M_A5>, tla::Int<HC_GRAD_CUBE_TILE_N_A5>, tla::Int<HC_GRAD_CUBE_TILE_K>>;

    template <class L1TileShape, class L0TileShape, class LayoutTagA, class LayoutTagB, class C_T>
    using BlockMmadT = Catlass::Gemm::Block::BlockMmadTla<
        DispatchPolicy, L1TileShape, L0TileShape, D_T, D_T, C_T, void,
        Catlass::Gemm::Tile::PackedTileCopyTla<ArchTag, D_T, LayoutTagA, D_T, LayoutTagB, C_T, RowMajor>>;

    template <class BlockMmad, uint32_t Direction>
    using MatmulKernelT = Catlass::Gemm::Kernel::BasicMatmulTla<
        BlockMmad, void, Catlass::Gemm::Block::GemmIdentityBlockSwizzle<HC_GRAD_SWIZZLE_OFFSET, Direction>>;

    template <class LayoutTag>
    static __aicore__ inline auto MakeLayout(uint32_t rows, uint32_t cols, int64_t leadingDim)
    {
        LayoutTag tag{rows, cols, leadingDim};
        return tla::MakeLayoutFromTag(tag);
    }

    template <class MatmulKernel>
    static __aicore__ inline void Invoke(Catlass::GemmCoord const& problemShape, GM_ADDR ptrA,
                                         typename MatmulKernel::LayoutA const& layoutA, GM_ADDR ptrB,
                                         typename MatmulKernel::LayoutB const& layoutB, GM_ADDR ptrC,
                                         typename MatmulKernel::LayoutC const& layoutC)
    {
        typename MatmulKernel::Params params{problemShape, ptrA, layoutA, ptrB, layoutB, ptrC, layoutC};
        MatmulKernel matmul;
        matmul.template operator()<AscendC::AIC>(params);
    }

    template <class L1TileShape, class L0TileShape, class LayoutTagA, class LayoutTagB, class C_T>
    static __aicore__ inline void RunGemm(uint32_t m, uint32_t n, uint32_t k, GM_ADDR ptrA, int64_t lda, GM_ADDR ptrB,
                                          int64_t ldb, GM_ADDR ptrC, int64_t ldc)
    {
        using BlockMmad = BlockMmadT<L1TileShape, L0TileShape, LayoutTagA, LayoutTagB, C_T>;
        static_assert(BlockMmad::L1_TILE_M == BlockMmad::L0_TILE_M && BlockMmad::L1_TILE_N == BlockMmad::L0_TILE_N,
                      "L1/L0 tile M and N must match");
        static_assert(BlockMmad::L0_TILE_K <= BlockMmad::L1_TILE_K, "L0 tile K exceeds L1 tile K");
        static_assert(
            BlockMmad::L1A_TILE_SIZE * BlockMmad::L1A_STAGES + BlockMmad::L1B_TILE_SIZE * BlockMmad::L1B_STAGES <=
                ArchTag::L1_SIZE,
            "GEMM tile exceeds L1");
        static_assert(BlockMmad::L0A_TILE_SIZE * BlockMmad::L0A_STAGES <= ArchTag::L0A_SIZE, "GEMM tile exceeds L0A");
        static_assert(BlockMmad::L0B_TILE_SIZE * BlockMmad::L0B_STAGES <= ArchTag::L0B_SIZE, "GEMM tile exceeds L0B");
        static_assert(BlockMmad::L0C_TILE_SIZE * BlockMmad::L0C_STAGES <= ArchTag::L0C_SIZE, "GEMM tile exceeds L0C");
        static_assert(std::is_same_v<typename BlockMmad::ElementAccumulator, float>,
                      "HcGradGemm requires FP32 accumulation");

        Catlass::GemmCoord problemShape{m, n, k};
        auto layoutA = MakeLayout<LayoutTagA>(m, k, lda);
        auto layoutB = MakeLayout<LayoutTagB>(k, n, ldb);
        auto layoutC = MakeLayout<RowMajor>(m, n, ldc);
        if (m >= n) {
            using MatmulKernel = MatmulKernelT<BlockMmad, HC_GRAD_SWIZZLE_M_FIRST>;
            Invoke<MatmulKernel>(problemShape, ptrA, layoutA, ptrB, layoutB, ptrC, layoutC);
        } else {
            using MatmulKernel = MatmulKernelT<BlockMmad, HC_GRAD_SWIZZLE_N_FIRST>;
            Invoke<MatmulKernel>(problemShape, ptrA, layoutA, ptrB, layoutB, ptrC, layoutC);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
    }
};

} // namespace TextGatedResidualGrad

#endif // TEXT_GATED_RESIDUAL_GRAD_GEMM_ARCH35_H
