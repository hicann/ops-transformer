/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef TEXT_GATED_RESIDUAL_GRAD_PIPELINE_ARCH22_H
#define TEXT_GATED_RESIDUAL_GRAD_PIPELINE_ARCH22_H

#include "../text_gated_residual_grad_constants.h"
#include "text_gated_residual_grad_nn_adapter.h"
#include "catlass/catlass.hpp"
#include "catlass/arch/local_tensor_buffer.hpp"

namespace Catlass::Arch {
// AIC Fixpipe needs the remaining UB. Keep this restriction local to this op.
template <>
struct LocalTensorBuffer<AtlasA2, AscendC::TPosition::VECCALC> : LocalTensorBufferBase {
    static constexpr AscendC::TPosition Position = AscendC::TPosition::VECCALC;
    CATLASS_DEVICE LocalTensorBuffer()
    {
        AscendC::TBuf<AscendC::TPosition::VECCALC> buffer;
        GetTPipePtr()->InitBuffer(buffer, TextGatedResidualGrad::HC_GRAD_A2_CUBE_UB_BYTES);
        tensor = buffer.Get<uint8_t>();
    }
};
} // namespace Catlass::Arch

#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "text_gated_residual_grad_dw_panel_arch22.h"
#include "text_gated_residual_grad_c1a_pipeline_arch22.h"
#include "text_gated_residual_grad_dx_pipeline_arch22.h"

namespace TextGatedResidualGrad {
// Use the buffer lifetimes of text_gated_residual's MatMul: prefetch A/B,
// release each L1 slot after its last L1->L0 load, and wait on the particular
// L0C slot before reusing it. CATLASS keeps the pending panel across calls,
// so the final panel of one output tile overlaps the first load of the next.
template <typename T, uint32_t TileM = 128, uint32_t TileN = 128, bool TransposeA = true, bool Dx = false>
__aicore__ inline void RunHcOperandPipeline(GM_ADDR a, GM_ADDR b, GM_ADDR c, uint32_t m, uint32_t n, uint32_t k,
                                            uint32_t lda, uint32_t ldb, uint32_t ldc, const MatmulTilingData& tiling)
{
    using Arch = Catlass::Arch::AtlasA2;
    using RowMajor = Catlass::layout::RowMajor;
    using LayoutA = std::conditional_t<TransposeA, Catlass::layout::ColumnMajor, RowMajor>;
    using AType = Catlass::Gemm::GemmType<T, LayoutA>;
    using BType = Catlass::Gemm::GemmType<T, RowMajor>;
    using CType = Catlass::Gemm::GemmType<float, RowMajor>;
    constexpr uint32_t tileM = TileM;
    constexpr uint32_t tileN = TileN;
    constexpr uint32_t baseK = 64;
    constexpr uint32_t panelK = 256;
    constexpr uint32_t l0cStages = 2;
    // Use two L1 slots with one panel of lookahead. Disable K shuffle and
    // keep the NN baseK so each output follows the same FP32 summation order.
    using Policy = Catlass::Gemm::MmadAtlasA2PreloadAsync<1, 2, 2, 2, l0cStages, false, false>;
    using SubtileBlock = Catlass::Gemm::Block::BlockMmad<Policy, Catlass::GemmShape<128, 128, panelK>,
                                                         Catlass::GemmShape<128, 128, baseK>, AType, BType, CType>;
    using DwBlock = std::conditional_t<TileM == 128 && TileN == 128, SubtileBlock, HcGradDwPanel<T, TileM, TileN>>;
    using Block =
        std::conditional_t<Dx, HcGradDxPipeline<T>, std::conditional_t<TransposeA, DwBlock, HcGradC1aPipeline<T>>>;
    using NnA = matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T, TransposeA>;
    using NnB = matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T>;
    using NnC = matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;
    HcGradNnStrideBlock schedule;
    schedule.template Init<NnA, NnB, NnC, NnC>(&tiling);
    schedule.lda = lda;
    schedule.ldb = ldb;
    schedule.ldc = ldc;

    AscendC::GlobalTensor<T> ga, gb;
    AscendC::GlobalTensor<float> gc;
    ga.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(a));
    gb.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(b));
    gc.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(c));
    LayoutA layoutA{m, k, lda};
    RowMajor layoutB{k, n, ldb};
    RowMajor layoutC{m, n, ldc};
    Catlass::Arch::Resource<Arch> resource;
    {
        Block block(resource);
        // These updates intentionally match MatmulBaseKernel::Process.
        bool reverse = true;
        for (uint64_t mTile = 0; mTile < schedule.params_.mTileCntL2; ++mTile) {
            reverse = !reverse;
            for (uint64_t i = 0; i < schedule.params_.nTileCntL2; ++i) {
                const uint64_t nTile = reverse ? schedule.params_.nTileCntL2 - i - 1 : i;
                schedule.UpdateBlockCnt(mTile, nTile);
                schedule.InitBlockIndex(0);
                for (uint64_t round = 0; round < schedule.params_.realRound; ++round) {
                    if (schedule.params_.rowOrder == 0) {
                        schedule.UpdateBasicIndex(round);
                    }
                    if (schedule.params_.index < schedule.params_.totalTileCnt) {
                        schedule.UpdateBlockParams(mTile, nTile);
                        schedule.template CalcGMOffset<NnA, NnB, NnC, NnC>(mTile, nTile);
                        const uint32_t rows = static_cast<uint32_t>(schedule.params_.singleCoreM);
                        const uint32_t cols = static_cast<uint32_t>(schedule.params_.singleCoreN);
                        // Keep each group on the original NN tile owner.
                        for (uint32_t row = 0; row < rows; row += tileM) {
                            const uint32_t rowsThisTile = rows - row < tileM ? rows - row : tileM;
                            for (uint32_t col = 0; col < cols; col += tileN) {
                                const uint32_t colsThisTile = cols - col < tileN ? cols - col : tileN;
                                const uint64_t outputOffset =
                                    schedule.offset_.offsetC + static_cast<uint64_t>(row) * ldc + col;
                                const Catlass::GemmCoord actual{rowsThisTile, colsThisTile, k};
                                block(ga[schedule.offset_.offsetA + layoutA.GetOffset({row, 0})], layoutA,
                                      gb[schedule.offset_.offsetB + col], layoutB, gc[outputOffset], layoutC, actual);
                            }
                        }
                    }
                    schedule.UpdateBlockIndex();
                }
            }
        }
    } // The BlockMmad destructor drains the pending panel and slot events.
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
}

template <typename T>
__aicore__ inline bool TryHcOperandPipeline(GM_ADDR a, GM_ADDR b, GM_ADDR c, uint32_t m, uint32_t n, uint32_t k,
                                            uint32_t lda, uint32_t ldb, uint32_t ldc, const MatmulTilingData& tiling)
{
    const auto& t = tiling.matmulTiling;
    // A 64-wide MMAD step and 16-element alignment match the NN reduction
    // granularity and CATLASS' operand and output layout requirements.
    if (k < 256 || m % 16 || n % 16 || k % 16 || lda % 16 || ldb % 16 || ldc % 16 || t.baseK != 64 || t.baseM % 16 ||
        t.baseN % 16 || t.baseM > 256 || t.baseN > 256 || t.singleCoreM != t.baseM || t.singleCoreN != t.baseN ||
        t.singleCoreK != k) {
        return false;
    }
    // Reuse the shared operand when the NN tile contains two 128x128 subtiles.
    if (k % HcGradDwPanel<T, 256, 128>::PANEL_K == 0) {
        if (t.baseM == 256 && t.baseN == 128) {
            RunHcOperandPipeline<T, 256, 128>(a, b, c, m, n, k, lda, ldb, ldc, tiling);
            return true;
        }
        if (t.baseM == 128 && t.baseN == 256) {
            RunHcOperandPipeline<T, 128, 256>(a, b, c, m, n, k, lda, ldb, ldc, tiling);
            return true;
        }
    }
    RunHcOperandPipeline<T>(a, b, c, m, n, k, lda, ldb, ldc, tiling);
    return true;
}
} // namespace TextGatedResidualGrad
#endif
