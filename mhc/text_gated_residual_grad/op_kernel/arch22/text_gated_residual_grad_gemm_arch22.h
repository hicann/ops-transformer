/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// A2/A3 Cube backend: ops-nn ordinary MatMulV3 kernel with fusion strides.
#ifndef TEXT_GATED_RESIDUAL_GRAD_GEMM_ARCH22_H
#define TEXT_GATED_RESIDUAL_GRAD_GEMM_ARCH22_H
#include <type_traits>
#include "../arch35/text_gated_residual_grad_tiling_data_arch35.h"
#include "text_gated_residual_grad_pipeline_arch22.h"
#include "nn_base/tgrg_mat_mul_nd2nz.h"
#include "nn_base/tgrg_batch_mat_mul_v3_common_kernel.h"
#if defined(__CCE_ENABLE_ASAN__)
#include "../arch35/text_gated_residual_grad_pipeline_arch35.h"
#endif
namespace TextGatedResidualGrad {
template <typename D_T>
class HcGradGemmArch22 {
public:
    __aicore__ inline void Init(uint32_t, const TextGatedResidualGradTilingDataArch35* data, GM_ADDR workspace)
    {
        data_ = data;
        scratch_ = workspace + data_->nnScratchOffset;
    }

    __aicore__ inline uint32_t ConversionPlan(uint32_t rows) const
    {
        const uint32_t fullRows = data_->scheduleMode == HC_GRAD_FULL_BATCH_SCHEDULE ?
                                      static_cast<uint32_t>(data_->fullBatchRows) :
                                      (data_->n < data_->taskRows ? data_->n : data_->taskRows);
        return (rows == fullRows || rows == data_->n) ? 0U : 1U;
    }

    __aicore__ inline bool NeedsNnConversion(uint32_t stage, uint32_t plan) const
    {
        return (data_->nnMixMask[plan] & (1U << stage)) != 0U;
    }

    __aicore__ inline void ConvertNnInputs(uint32_t stage, uint32_t task, GM_ADDR a, GM_ADDR b,
                                           AscendC::TBuf<AscendC::TPosition::VECCALC>& ub)
    {
        const auto& nn = data_->nnBase[task][stage];
        if (nn.matmulRunInfo.nd2nzA != 0) {
            MatrixAtoNZV2<D_T>(scratch_, a, nn.matmulTiling, nn.matmulRunInfo.transA, ub, nn.baseAN, nn.baseAD);
        }
        if (nn.matmulRunInfo.nd2nzB != 0) {
            AscendC::PipeBarrier<PIPE_ALL>();
            MatrixBtoNZV2<D_T>(ConvertedB(nn), b, nn.matmulTiling, false, ub, nn.baseBN, nn.baseBD);
        }
    }
    __aicore__ inline void RunC1aSlabs(GM_ADDR dup, GM_ADDR partial, uint32_t tilesPerSlab, uint16_t readyFlag)
    {
        const auto& nn = data_->nnBase[0][0];
        using InputA = matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, D_T>;
        using InputB = matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::NZ, D_T>;
        using Output = matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;
        using Bias = matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, D_T>;
        AscendC::TPipe pipe;
        MatmulBaseKernel<InputA, InputB, Output, Bias> kernel;
        kernel.Init(dup, scratch_, partial, nullptr, nullptr, nullptr, &nn, &pipe);
        kernel.Process(0U, 0U, tilesPerSlab, readyFlag);
        kernel.End();
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
        pipe.Destroy();
    }

    __aicore__ inline void RunC1aSplitK(GM_ADDR dup, GM_ADDR wUp, GM_ADDR partial, uint32_t rows, uint32_t r,
                                        uint32_t d, uint32_t splitkFactor)
    {
        // Host A2/A3 always uses splitkFactor=1, full-K FP32 accumulation.
#if defined(__CCE_ENABLE_ASAN__)
        if (rows % 16U == 0U && TryTrackedFullBatch<HC_GRAD_NN_STAGE_C1A, false, false>(rows, dup, wUp, partial)) {
            return;
        }
#endif
        RunNn<false>(0, rows, rows, r, d, dup, d, wUp, r, partial, r);
    }
    // C1b: dwUpPartial = dup^T @ act, reading dup as ColumnMajor [D,rows], as on A5.
    __aicore__ inline void RunC1b(GM_ADDR dup, GM_ADDR act, GM_ADDR dwUpPartial, uint32_t rows, uint32_t r, uint32_t d,
                                  bool accumulate = false)
    {
        RunNn<true>(1, rows, d, r, rows, dup, d, act, r, dwUpPartial, r, accumulate);
    }

    // For multiple tasks, use one M=R+gateRows GEMM to avoid rereading xn in a separate small gate GEMM.
    __aicore__ inline void RunG56Merged(GM_ADDR packedT, GM_ADDR xn, GM_ADDR dwOut, uint32_t rows, uint32_t r,
                                        uint32_t d, uint32_t gateRows, bool hasGate)
    {
        const uint32_t m = hasGate ? (r + gateRows) : r;
        RunNn<false>(HC_GRAD_NN_STAGE_G56, rows, m, d, rows, packedT, rows, xn, d, dwOut, d);
    }

    __aicore__ inline void RunG56MergedPacked(GM_ADDR packed, GM_ADDR xn, GM_ADDR dwOut, uint32_t rows, uint32_t r,
                                              uint32_t d, uint32_t packedStride, uint32_t gateRows, bool hasGate,
                                              bool accumulate = false)
    {
#if defined(__CCE_ENABLE_ASAN__)
        if (TryTrackedFullBatch<HC_GRAD_NN_STAGE_G56, true, true>(rows, packed, xn, dwOut)) {
            return;
        }
#endif
        const uint32_t m = hasGate ? (r + gateRows) : r;
        RunNn<true>(HC_GRAD_NN_STAGE_G56, rows, m, d, rows, packed, packedStride, xn, d, dwOut, d, accumulate);
    }

    __aicore__ inline void RunG56DirectPacked(GM_ADDR packed, GM_ADDR xn, GM_ADDR dwDown, GM_ADDR dwGate, uint32_t rows,
                                              uint32_t r, uint32_t d, uint32_t packedStride, uint32_t gateRows,
                                              bool hasGate)
    {
        RunNn<true>(HC_GRAD_NN_STAGE_G56, rows, r, d, rows, packed, packedStride, xn, d, dwDown, d);
        if (hasGate) {
            RunNn<true>(HC_GRAD_NN_STAGE_GATE, rows, gateRows, d, rows, packed + r * sizeof(D_T), packedStride, xn, d,
                        dwGate, d);
        }
    }

    // Single-task fast paths: accumulate straight into the FP32 outputs.
    __aicore__ inline void RunC1bDirect(GM_ADDR dup, GM_ADDR act, GM_ADDR dwUp, uint32_t rows, uint32_t r, uint32_t d)
    {
#if defined(__CCE_ENABLE_ASAN__)
        if (TryTrackedFullBatch<HC_GRAD_NN_STAGE_C1B, true, false>(rows, dup, act, dwUp)) {
            return;
        }
#endif
        RunNn<true>(1, rows, d, r, rows, dup, d, act, r, dwUp, r);
    }

    __aicore__ inline void RunG56Direct(GM_ADDR packedT, GM_ADDR xn, GM_ADDR dwDown, GM_ADDR dwGate, uint32_t rows,
                                        uint32_t r, uint32_t d, uint32_t gateRows, bool hasGate)
    {
        RunNn<false>(HC_GRAD_NN_STAGE_G56, rows, r, d, rows, packedT, rows, xn, d, dwDown, d);
        if (hasGate) {
            // As on A5, write gateRows aligned to 16 into dwGatePartial, which has room for all padded rows.
            // Vector REDUCE_WD copies only the first c*d elements into the dwGate FP32 accumulator.
            RunNn<false>(HC_GRAD_NN_STAGE_GATE, rows, gateRows, d, rows, packedT + r * rows * sizeof(D_T), rows, xn, d,
                         dwGate, d);
        }
    }

    // dxnorm = ddown @ w_down + dlogits @ w_gate: concatenate weights along K into wConcat for one GEMM.
    __aicore__ inline void RunDx(GM_ADDR packed, GM_ADDR wDown, GM_ADDR wGate, GM_ADDR wConcat, GM_ADDR dxOut,
                                 uint32_t rows, uint32_t d, uint32_t r, uint32_t c, uint32_t packedStride, bool hasGate)
    {
        if (data_->lowpDx != 0) {
            if (hasGate) {
                RunNn<false, D_T>(HC_GRAD_NN_STAGE_DX, rows, rows, d, r + c, packed, packedStride, wConcat, d, dxOut,
                                  d);
            } else {
                RunNn<false, D_T>(HC_GRAD_NN_STAGE_DX, rows, rows, d, r, packed, packedStride, wDown, d, dxOut, d);
            }
            return;
        }
        if (hasGate) {
            RunNn<false>(HC_GRAD_NN_STAGE_DX, rows, rows, d, r + c, packed, packedStride, wConcat, d, dxOut, d);
        } else {
            RunNn<false>(HC_GRAD_NN_STAGE_DX, rows, rows, d, r, packed, packedStride, wDown, d, dxOut, d);
        }
    }

private:
    __aicore__ inline GM_ADDR ConvertedB(const MatmulTilingData& nn) const
    {
        if (nn.matmulRunInfo.nd2nzA == 0) {
            return scratch_;
        }
        // BF16 and FP16 both use a 16-element C0. Match NN's A-then-B workspace layout.
        const uint64_t m = nn.matmulTiling.M;
        const uint64_t k = nn.matmulTiling.Ka;
        return scratch_ + ((m + 15U) / 16U * 16U) * ((k + 15U) / 16U * 16U) * sizeof(D_T);
    }

    template <bool TransA, CubeFormat FormatA, CubeFormat FormatB, typename OUT_T = float>
#if defined(__CCE_ENABLE_ASAN__)
    __aicore__ __attribute__((always_inline)) inline void RunNativeBody(GM_ADDR a, GM_ADDR b, GM_ADDR c,
                                                                        const MatmulTilingData& nn,
                                                                        bool accumulate = false)
#else
    __aicore__ __attribute__((noinline)) void RunNative(GM_ADDR a, GM_ADDR b, GM_ADDR c, const MatmulTilingData& nn,
                                                        bool accumulate = false)
#endif
    {
        using InputA = matmul::MatmulType<AscendC::TPosition::GM, FormatA, D_T, TransA>;
        using InputB = matmul::MatmulType<AscendC::TPosition::GM, FormatB, D_T>;
        using Output = matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, OUT_T>;
        using Bias = matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, D_T>;
        AscendC::TPipe pipe;
        MatmulBaseKernel<InputA, InputB, Output, Bias> kernel;
        kernel.Init(a, b, c, nullptr, nullptr, nullptr, &nn, &pipe);
        kernel.Process(0U, accumulate);
        kernel.End();
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
        pipe.Destroy();
    }

#if defined(__CCE_ENABLE_ASAN__)
    template <bool TransA, CubeFormat FormatA, CubeFormat FormatB, typename OUT_T = float>
    __aicore__ __attribute__((noinline)) void RunNative(GM_ADDR a, GM_ADDR b, GM_ADDR c, const MatmulTilingData& nn,
                                                        bool accumulate = false)
    {
        RunNativeBody<TransA, FormatA, FormatB, OUT_T>(a, b, c, nn, accumulate);
    }

    // Let CCEC track the Fixpipe writes consumed by the full-batch Vector stages.
    // Preserve the native tiling and A/B substitutions used by RunNn.
    template <uint32_t Stage, bool TransA, bool ConvertA>
    __aicore__ __attribute__((always_inline)) inline bool TryTrackedFullBatch(uint32_t rows, GM_ADDR a, GM_ADDR b,
                                                                              GM_ADDR c)
    {
        constexpr uint32_t mask = 1U << Stage;
        if (data_->scheduleMode != HC_GRAD_FULL_BATCH_SCHEDULE || rows != data_->n ||
            (data_->nnBaseMask[0] & data_->nnMixMask[0] & mask) == 0U) {
            return false;
        }
        const auto& nn = data_->nnBase[0][Stage];
        if ((nn.matmulRunInfo.nd2nzA != 0U) != ConvertA || (nn.matmulRunInfo.nd2nzB != 0U) == ConvertA) {
            return false;
        }
        if constexpr (ConvertA) {
            RunNativeBody<TransA, CubeFormat::NZ, CubeFormat::ND>(scratch_, b, c, nn);
        } else {
            RunNativeBody<TransA, CubeFormat::ND, CubeFormat::NZ>(a, scratch_, c, nn);
        }
        return true;
    }
#endif

    __aicore__ __attribute__((noinline)) void RunNativeWeightBatch(GM_ADDR a, GM_ADDR b, GM_ADDR c,
                                                                   const BatchMatmulTilingData& nn, bool accumulate)
    {
        using InputA = matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, D_T, true>;
        using InputB = matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, D_T>;
        using Output = matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;
        using Bias = matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, D_T>;
        AscendC::TPipe pipe;
        BatchMatMulCommonKernel<InputA, InputB, Output, Bias> kernel;
        kernel.Init(a, b, c, nullptr, nullptr, nullptr, &nn, &pipe);
        kernel.Process(accumulate);
        kernel.End();
        AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
        pipe.Destroy();
    }

    template <bool TRANS_A, typename OUT_T = float>
    __aicore__ __attribute__((noinline)) void RunNn(uint32_t stage, uint32_t rows, uint32_t m, uint32_t n, uint32_t k,
                                                    GM_ADDR a, uint32_t lda, GM_ADDR b, uint32_t ldb, GM_ADDR c,
                                                    uint32_t ldc, bool accumulate = false)
    {
        // ND-to-NZ only writes valid M rows, while LoadData reads complete 16-row tiles.
        // Initialize the Cube scratch for an unaligned C1a M tail before the NN kernel
        // owns L1. This also covers both ping-pong operand regions without altering K.
        if (stage == 0 && (m % 16) != 0) {
            constexpr uint32_t l1Bytes = 512 * 1024; // A2/A3 Cube L1 capacity.
            AscendC::TPipe paddingPipe;
            AscendC::TBuf<AscendC::TPosition::A1> paddingBuf;
            paddingPipe.InitBuffer(paddingBuf, l1Bytes);
            auto padding = paddingBuf.Get<uint16_t>();
            AscendC::InitConstValueParams<uint16_t> params{1, l1Bytes / 32, 0, 0};
            AscendC::InitConstValue(padding, params);
            AscendC::PipeBarrier<PIPE_ALL>();
            paddingPipe.Destroy();
        }
        const uint32_t fullRows = data_->scheduleMode == HC_GRAD_FULL_BATCH_SCHEDULE ?
                                      static_cast<uint32_t>(data_->fullBatchRows) :
                                      (data_->n < data_->taskRows ? data_->n : data_->taskRows);
        const uint32_t task = (rows == fullRows || rows == data_->n) ? 0 : 1;
        const auto& nn = data_->nnBase[task][stage];
        if ((data_->nnBaseMask[task] & (1U << stage)) != 0) {
            if ((data_->nnMixMask[task] & (1U << stage)) != 0) {
                const bool convertA = nn.matmulRunInfo.nd2nzA != 0;
                const bool convertB = nn.matmulRunInfo.nd2nzB != 0;
                if (convertA && convertB) {
                    RunNative<TRANS_A, CubeFormat::NZ, CubeFormat::NZ, OUT_T>(scratch_, ConvertedB(nn), c, nn,
                                                                              accumulate);
                } else if (convertA) {
                    RunNative<TRANS_A, CubeFormat::NZ, CubeFormat::ND, OUT_T>(scratch_, b, c, nn, accumulate);
                } else {
                    RunNative<TRANS_A, CubeFormat::ND, CubeFormat::NZ, OUT_T>(a, scratch_, c, nn, accumulate);
                }
                return;
            }
            if (lda == (TRANS_A ? m : k) && ldb == n && ldc == n) {
                if constexpr (TRANS_A) {
                    if (task == 0 && (data_->nnWeightBatchMask & (1U << stage)) != 0) {
                        RunNativeWeightBatch(a, b, c, data_->nnWeightBatch[stage - 1], accumulate);
                        return;
                    }
                }
                RunNative<TRANS_A, CubeFormat::ND, CubeFormat::ND, OUT_T>(a, b, c, nn, accumulate);
                return;
            }
            AscendC::TPipe pipe;
            HcGradNnKernel<D_T, TRANS_A, OUT_T> kernel;
            kernel.Init(a, b, c, nullptr, nullptr, nullptr, &nn, &pipe);
            kernel.SetStrides(m, k, lda, ldb, ldc);
            kernel.Process(0U, accumulate);
            kernel.End();
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
            pipe.Destroy();
            return;
        }
        MatmulTilingData tiling{};
        // Materialize the NN parameter representation, including alignment padding.
        // Some optimized aggregate-copy instructions are not tracked by initcheck.
        volatile unsigned char* parameterBytes = reinterpret_cast<volatile unsigned char*>(&tiling);
        const auto* sourceBytes = reinterpret_cast<const unsigned char*>(&nn);
        for (uint32_t i = 0; i < sizeof(tiling); ++i) {
            parameterBytes[i] = sourceBytes[i];
        }
        if constexpr (!std::is_same_v<OUT_T, float>) {
            AscendC::TPipe pipe;
            HcGradNnKernel<D_T, TRANS_A, OUT_T> kernel;
            kernel.Init(a, b, c, nullptr, nullptr, nullptr, &tiling, &pipe);
            kernel.SetStrides(m, k, lda, ldb, ldc);
            kernel.Process();
            kernel.End();
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
            pipe.Destroy();
            return;
        }
        if constexpr (TRANS_A) {
            if (!accumulate && TryHcOperandPipeline<D_T>(a, b, c, m, n, k, lda, ldb, ldc, tiling)) {
                return;
            }
        } else if (stage == 0 && k >= 1024 && k % 256 == 0 && m % 16 == 0 && n % 16 == 0 && lda % 16 == 0 &&
                   ldb % 16 == 0 && ldc % 16 == 0 && tiling.matmulTiling.singleCoreK == k &&
                   tiling.matmulTiling.baseK == 64 && tiling.matmulTiling.singleCoreM == tiling.matmulTiling.baseM &&
                   tiling.matmulTiling.singleCoreN == tiling.matmulTiling.baseN && tiling.matmulTiling.baseM >= 64 &&
                   tiling.matmulTiling.baseM <= 128 && tiling.matmulTiling.baseM % 16 == 0 &&
                   tiling.matmulTiling.baseN <= 256 && tiling.matmulTiling.baseN % 16 == 0) {
            RunHcOperandPipeline<D_T, 128, 256, false>(a, b, c, m, n, k, lda, ldb, ldc, tiling);
            return;
        }
        if constexpr (!TRANS_A) {
            const auto& tile = tiling.matmulTiling;
            if (stage == 3 && (k == 320 || k == 324) && m % 16 == 0 && n % 16 == 0 && lda % 16 == 0 && ldb % 16 == 0 &&
                ldc % 16 == 0 && tile.singleCoreK == k && tile.baseK == 64 && tile.baseM == 128 && tile.baseN == 256 &&
                tile.singleCoreM == tile.baseM && tile.singleCoreN == tile.baseN) {
                // Split the NN tile locally along N to keep two independent L0C tiles.
                RunHcOperandPipeline<D_T, 128, 256, false, true>(a, b, c, m, n, k, lda, ldb, ldc, tiling);
                return;
            }
        }
        AscendC::TPipe pipe;
        HcGradNnKernel<D_T, TRANS_A, OUT_T> kernel;
        kernel.Init(a, b, c, nullptr, nullptr, nullptr, &tiling, &pipe);
        kernel.SetStrides(m, k, lda, ldb, ldc);
        kernel.Process(0U, accumulate);
        kernel.End();
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
        pipe.Destroy();
    }
    const TextGatedResidualGradTilingDataArch35* data_{nullptr};
    GM_ADDR scratch_{nullptr};
};
} // namespace TextGatedResidualGrad
#endif
