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
 * \file text_gated_residual_grad_gemm_arch22.h
 * \brief Atlas A2/A3 CATLASS GEMM backend for TextGatedResidualGrad.
 */
#ifndef TEXT_GATED_RESIDUAL_GRAD_GEMM_ARCH22_H
#define TEXT_GATED_RESIDUAL_GRAD_GEMM_ARCH22_H

#include "../text_gated_residual_grad_constants.h"

#if !defined(CATLASS_ARCH) || CATLASS_ARCH != HC_GRAD_ARCH_A2
#error "text_gated_residual_grad_gemm_arch22.h requires CATLASS_ARCH=2201"
#endif

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include <type_traits>

#include "kernel_operator.h"
#include "catlass/catlass.hpp"
#include "catlass/arch/local_tensor_buffer.hpp"

#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A2
// Limit AIC VECCALC UB allocation only for this operator's A2/A3 kernel.
// CATLASS's full-UB default blocks L0C-to-GM Fixpipe; use a local specialization instead of patching shared code.
namespace Catlass::Arch {
template <>
struct LocalTensorBuffer<AtlasA2, AscendC::TPosition::VECCALC> : LocalTensorBufferBase {
public:
    static constexpr AscendC::TPosition Position = AscendC::TPosition::VECCALC;

    CATLASS_DEVICE LocalTensorBuffer()
    {
        AscendC::TBuf<AscendC::TPosition::VECCALC> tbufVECCALC;
        uint32_t ubBytes = AtlasA2::UB_SIZE;
        if ASCEND_IS_AIC {
            ubBytes = TextGatedResidualGrad::HC_GRAD_A2_CUBE_UB_BYTES;
        }
        GetTPipePtr()->InitBuffer(tbufVECCALC, ubBytes);
        tensor = tbufVECCALC.Get<uint8_t>();
    }
};
} // namespace Catlass::Arch
#endif

#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/layout/layout.hpp"
#include "../arch35/text_gated_residual_grad_pipeline_arch35.h"

namespace TextGatedResidualGrad {

// A2/A3 (arch22) uses the same algorithm as arch35, with these platform differences:
//   1) Cube uses AtlasA2 BlockMmad (Gemm::Block::BlockMmad + MmadAtlasA2Pingpong),
//      as other A2 operators do; A5 MmadPingpongMutex supports Ascend950 only.
//   2) C1/DX use 128x128x64 tiles for A2's 128 KiB L0C, consuming 64 KiB per tile.
//      BlockMmad uses one L0C stage and double buffering for L1/L0A/L0B.
//   3) MmadAtlasA2Pingpong<false> disables unit flags and uses explicit M_FIX/FIX_M synchronization.
//   4) Block strides use the tiling AIC count; GetBlockNum() is not the AIC count in a mixed kernel.
// As on A5, C1b and G56 read Vector results directly as ColumnMajor without materializing transposes.
template <typename D_T>
class HcGradGemmArch22 {
    static constexpr uint32_t C1TileM = HC_GRAD_CUBE_TILE_M_A2;
    static constexpr uint32_t C1TileN = HC_GRAD_C1A_TILE_N;
    static constexpr uint32_t C1TileK = HC_GRAD_CUBE_TILE_K;

public:
    __aicore__ inline void Init(uint32_t aicNum)
    {
        aicNum_ = aicNum;
    }

    // C1a: dact = dup @ wUp. As on A5, distribute all (m,n,slice) blocks across AICs in one loop,
    // treating split-K slices as virtual cores. Each slice writes its own FP32 partial,
    // which Vector REDUCE_DACT merges; slices do not run as separate serial GEMMs.
    __aicore__ inline void RunC1aSplitK(GM_ADDR dup, GM_ADDR wUp, GM_ADDR partial, uint32_t rows, uint32_t r,
                                        uint32_t d, uint32_t splitkFactor)
    {
        using BlockScheduler =
            Catlass::Gemm::Block::SplitkGemmIdentityBlockSwizzle<HC_GRAD_SWIZZLE_OFFSET, HC_GRAD_SWIZZLE_M_FIRST>;
        Catlass::GemmCoord problemShape{rows, r, d};
        BlockScheduler scheduler(problemShape, Catlass::GemmCoord(C1TileM, C1TileN, C1TileK), splitkFactor);
        const uint32_t coreLoops = scheduler.GetCoreLoops();

        Catlass::Arch::Resource<ArchTag> resource;
        C1BlockMmad blockMmad(resource);

        AscendC::GlobalTensor<D_T> gmA;
        gmA.SetGlobalBuffer(reinterpret_cast<__gm__ D_T*>(dup));
        AscendC::GlobalTensor<D_T> gmB;
        gmB.SetGlobalBuffer(reinterpret_cast<__gm__ D_T*>(wUp));
        AscendC::GlobalTensor<float> gmW;
        gmW.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(partial));

        RowMajor layoutA{rows, d, d};
        RowMajor layoutB{d, r, r};
        RowMajor layoutC{rows, r, r};
        const size_t lenC = static_cast<size_t>(rows) * r;
        for (uint32_t loopIdx = AscendC::GetBlockIdx(); loopIdx < coreLoops; loopIdx += aicNum_) {
            const Catlass::GemmCoord blockCoord = scheduler.GetBlockCoord(loopIdx);
            const uint32_t slice = scheduler.GetSplitkSliceIdx(loopIdx);
            const Catlass::GemmCoord actual = scheduler.GetActualBlockShape(blockCoord, slice);
            const MatrixCoord offsetA{blockCoord.m() * C1TileM, blockCoord.k() * C1TileK};
            const MatrixCoord offsetB{blockCoord.k() * C1TileK, blockCoord.n() * C1TileN};
            const MatrixCoord offsetC{blockCoord.m() * C1TileM, blockCoord.n() * C1TileN};
            blockMmad(gmA[layoutA.GetOffset(offsetA)], layoutA, gmB[layoutB.GetOffset(offsetB)], layoutB,
                      gmW[lenC * slice + layoutC.GetOffset(offsetC)], layoutC, actual);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
    }

    // C1b: dwUpPartial = dup^T @ act, reading dup as ColumnMajor [D,rows], as on A5.
    __aicore__ inline void RunC1b(GM_ADDR dup, GM_ADDR act, GM_ADDR dwUpPartial, uint32_t rows, uint32_t r, uint32_t d)
    {
        RunC1GemmColumnMajor(d, r, rows, dup, d, act, r, dwUpPartial, r);
    }

    // For multiple tasks, use one M=R+gateRows GEMM to avoid rereading xn in a separate small gate GEMM.
    __aicore__ inline void RunG56Merged(GM_ADDR packedT, GM_ADDR xn, GM_ADDR dwOut, uint32_t rows, uint32_t r,
                                        uint32_t d, uint32_t gateRows, bool hasGate)
    {
        const uint32_t m = hasGate ? (r + gateRows) : r;
        RunC1Gemm(m, d, rows, packedT, rows, xn, d, dwOut, d);
    }

    __aicore__ inline void RunG56MergedPacked(GM_ADDR packed, GM_ADDR xn, GM_ADDR dwOut, uint32_t rows, uint32_t r,
                                              uint32_t d, uint32_t packedStride, uint32_t gateRows, bool hasGate)
    {
        const uint32_t m = hasGate ? (r + gateRows) : r;
        RunC1GemmColumnMajor(m, d, rows, packed, packedStride, xn, d, dwOut, d);
    }

    __aicore__ inline void RunG56DirectPacked(GM_ADDR packed, GM_ADDR xn, GM_ADDR dwDown, GM_ADDR dwGate, uint32_t rows,
                                              uint32_t r, uint32_t d, uint32_t packedStride, uint32_t gateRows,
                                              bool hasGate)
    {
        RunC1GemmColumnMajor(r, d, rows, packed, packedStride, xn, d, dwDown, d);
        if (hasGate) {
            RunC1GemmColumnMajor(gateRows, d, rows, packed + r * sizeof(D_T), packedStride, xn, d, dwGate, d);
        }
    }

    // Single-task fast paths: accumulate straight into the FP32 outputs.
    __aicore__ inline void RunC1bDirect(GM_ADDR dup, GM_ADDR act, GM_ADDR dwUp, uint32_t rows, uint32_t r, uint32_t d)
    {
        RunC1GemmColumnMajor(d, r, rows, dup, d, act, r, dwUp, r);
    }

    __aicore__ inline void RunG56Direct(GM_ADDR packedT, GM_ADDR xn, GM_ADDR dwDown, GM_ADDR dwGate, uint32_t rows,
                                        uint32_t r, uint32_t d, uint32_t gateRows, bool hasGate)
    {
        RunC1Gemm(r, d, rows, packedT, rows, xn, d, dwDown, d);
        if (hasGate) {
            // As on A5, write gateRows aligned to 16 into dwGatePartial, which has room for all padded rows.
            // Vector REDUCE_WD copies only the first c*d elements into the dwGate FP32 accumulator.
            RunC1Gemm(gateRows, d, rows, packedT + r * rows * sizeof(D_T), rows, xn, d, dwGate, d);
        }
    }

    // dxnorm = ddown @ w_down + dlogits @ w_gate: concatenate weights along K into wConcat for one GEMM.
    __aicore__ inline void RunDx(GM_ADDR packed, GM_ADDR wDown, GM_ADDR wGate, GM_ADDR wConcat, GM_ADDR dxOut,
                                 uint32_t rows, uint32_t d, uint32_t r, uint32_t c, uint32_t packedStride, bool hasGate)
    {
        if (hasGate) {
            RunDxGemm(rows, d, r + c, packed, packedStride, wConcat, d, dxOut, d);
        } else {
            RunDxGemm(rows, d, r, packed, packedStride, wDown, d, dxOut, d);
        }
    }

private:
    using ArchTag = Catlass::Arch::AtlasA2;
    using DispatchPolicy = Catlass::Gemm::MmadAtlasA2Pingpong<false>;
    using RowMajor = Catlass::layout::RowMajor;
    using ColumnMajor = Catlass::layout::ColumnMajor;
    using MatrixCoord = Catlass::MatrixCoord;

    static_assert(C1TileM * C1TileN * sizeof(float) <= Catlass::Arch::AtlasA2::L0C_SIZE,
                  "A2/A3 L0C tile exceeds 128KB");

    using AType = Catlass::Gemm::GemmType<D_T, RowMajor>;
    using BType = Catlass::Gemm::GemmType<D_T, RowMajor>;
    using CType = Catlass::Gemm::GemmType<float, RowMajor>;

    using C1BlockMmad =
        Catlass::Gemm::Block::BlockMmad<DispatchPolicy, Catlass::GemmShape<C1TileM, C1TileN, C1TileK>,
                                        Catlass::GemmShape<C1TileM, C1TileN, C1TileK>, AType, BType, CType>;

    using ATypeColumnMajor = Catlass::Gemm::GemmType<D_T, Catlass::layout::ColumnMajor>;
    using C1BlockMmadColumnMajor =
        Catlass::Gemm::Block::BlockMmad<DispatchPolicy, Catlass::GemmShape<C1TileM, C1TileN, C1TileK>,
                                        Catlass::GemmShape<C1TileM, C1TileN, C1TileK>, ATypeColumnMajor, BType, CType>;

    template <class BlockMmad, uint32_t Direction, class LayoutAT = RowMajor>
    __aicore__ inline void RunBlockLoop(uint32_t m, uint32_t n, uint32_t k, GM_ADDR ptrA, int64_t lda, GM_ADDR ptrB,
                                        int64_t ldb, GM_ADDR ptrC, int64_t ldc)
    {
        using BlockScheduler = Catlass::Gemm::Block::GemmIdentityBlockSwizzle<HC_GRAD_SWIZZLE_OFFSET, Direction>;
        constexpr uint32_t tileM = BlockMmad::L1TileShape::M;
        constexpr uint32_t tileN = BlockMmad::L1TileShape::N;
        constexpr uint32_t tileK = BlockMmad::L1TileShape::K;
        Catlass::GemmCoord problemShape{m, n, k};
        BlockScheduler scheduler(problemShape, MatrixCoord(tileM, tileN));
        const uint32_t coreLoops = scheduler.GetCoreLoops();

        Catlass::Arch::Resource<ArchTag> resource;
        BlockMmad blockMmad(resource);

        AscendC::GlobalTensor<D_T> gmA;
        gmA.SetGlobalBuffer(reinterpret_cast<__gm__ D_T*>(ptrA));
        AscendC::GlobalTensor<D_T> gmB;
        gmB.SetGlobalBuffer(reinterpret_cast<__gm__ D_T*>(ptrB));
        AscendC::GlobalTensor<float> gmC;
        gmC.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(ptrC));

        LayoutAT layoutA{m, k, lda};
        RowMajor layoutB{k, n, ldb};
        RowMajor layoutC{m, n, ldc};
        for (uint32_t loopIdx = AscendC::GetBlockIdx(); loopIdx < coreLoops; loopIdx += aicNum_) {
            const Catlass::GemmCoord blockCoord = scheduler.GetBlockCoord(loopIdx);
            const Catlass::GemmCoord actual = scheduler.GetActualBlockShape(blockCoord);
            const MatrixCoord offsetA{blockCoord.m() * tileM, blockCoord.k() * tileK};
            const MatrixCoord offsetB{blockCoord.k() * tileK, blockCoord.n() * tileN};
            const MatrixCoord offsetC{blockCoord.m() * tileM, blockCoord.n() * tileN};
            blockMmad(gmA[layoutA.GetOffset(offsetA)], layoutA, gmB[layoutB.GetOffset(offsetB)], layoutB,
                      gmC[layoutC.GetOffset(offsetC)], layoutC, actual);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
    }

    __aicore__ inline void RunC1Gemm(uint32_t m, uint32_t n, uint32_t k, GM_ADDR ptrA, int64_t lda, GM_ADDR ptrB,
                                     int64_t ldb, GM_ADDR ptrC, int64_t ldc)
    {
        if (m >= n) {
            RunBlockLoop<C1BlockMmad, HC_GRAD_SWIZZLE_M_FIRST>(m, n, k, ptrA, lda, ptrB, ldb, ptrC, ldc);
        } else {
            RunBlockLoop<C1BlockMmad, HC_GRAD_SWIZZLE_N_FIRST>(m, n, k, ptrA, lda, ptrB, ldb, ptrC, ldc);
        }
    }

    __aicore__ inline void RunC1GemmColumnMajor(uint32_t m, uint32_t n, uint32_t k, GM_ADDR ptrA, int64_t lda,
                                                GM_ADDR ptrB, int64_t ldb, GM_ADDR ptrC, int64_t ldc)
    {
        if (m >= n) {
            RunBlockLoop<C1BlockMmadColumnMajor, HC_GRAD_SWIZZLE_M_FIRST, ColumnMajor>(m, n, k, ptrA, lda, ptrB, ldb,
                                                                                       ptrC, ldc);
        } else {
            RunBlockLoop<C1BlockMmadColumnMajor, HC_GRAD_SWIZZLE_N_FIRST, ColumnMajor>(m, n, k, ptrA, lda, ptrB, ldb,
                                                                                       ptrC, ldc);
        }
    }

    __aicore__ inline void RunDxGemm(uint32_t m, uint32_t n, uint32_t k, GM_ADDR ptrA, int64_t lda, GM_ADDR ptrB,
                                     int64_t ldb, GM_ADDR ptrC, int64_t ldc)
    {
        RunC1Gemm(m, n, k, ptrA, lda, ptrB, ldb, ptrC, ldc);
    }

    uint32_t aicNum_{1};
};

} // namespace TextGatedResidualGrad

#endif // TEXT_GATED_RESIDUAL_GRAD_GEMM_ARCH22_H
