/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MHC_PRE_SINKHORN_BACKWARD_VECTOR_ARCH35_H
#define MHC_PRE_SINKHORN_BACKWARD_VECTOR_ARCH35_H

#include "kernel_operator.h"
#include "mhc_pre_sinkhorn_backward_data_arch35.h"

namespace MhcPreSinkhornBackwardNs {
using namespace AscendC;
using namespace AscendC::MicroAPI;

constexpr uint32_t ROWS_PER_AIV = 16;
constexpr uint32_t K_TILE_VALUE = 256;
constexpr uint32_t FLOAT_C0 = 8;
constexpr uint32_t ND_GAP = 8;
constexpr uint32_t VECTOR_LENGTH = 64;
// Reserve UB beyond the row-scan buffer budget (per-row gradHin dot x) so aligned 64-lane reads stay within bounds.
constexpr uint32_t SMALL_BUFFER_SLACK = VECTOR_LENGTH;
constexpr uint32_t PARAM_ALPHA = 0;
constexpr uint32_t ROW_INPUT_Q = 0;
constexpr uint32_t ROW_INPUT_INV_RMS = 32;
constexpr uint32_t ROW_INPUT_GRAD_POST = 40;
constexpr uint32_t ROW_INPUT_SK = 48;
constexpr uint32_t ROW_INPUT_ELEMENTS = 64;
constexpr uint32_t ITER_INPUT_SUM = 0;
constexpr uint32_t ITER_INPUT_COL_SUM = 8;
constexpr uint32_t ITER_INPUT_NORM = 16;
constexpr uint32_t PREPARE_BATCH = 4;
constexpr uint32_t TILE_SCALAR_INV_RMS = ROWS_PER_AIV * FLOAT_C0;
constexpr uint32_t TILE_SCALAR_ELEMENTS = TILE_SCALAR_INV_RMS + ROWS_PER_AIV;
constexpr uint32_t X_STAGE_ND_ELEMENTS = ROWS_PER_AIV * (K_TILE_VALUE + ND_GAP);
constexpr uint32_t X_STAGE_NZ_OFFSET = X_STAGE_ND_ELEMENTS;
constexpr uint32_t X_STAGE_ELEMENTS = X_STAGE_ND_ELEMENTS + ROWS_PER_AIV * K_TILE_VALUE;
constexpr uint32_t STAGE_POOL_BUFFER_COUNT = 16;
// Shared buffer pool size for Prepare, including row-scan double buffers and iteration caches.
constexpr uint32_t STAGE_POOL_BYTES = 160 * 1024;
// Reserve fixed Prepare pool overhead beyond row-scan double buffers for alignment and small-buffer slack.
constexpr uint32_t SCAN_BUDGET_MARGIN_BYTES = 1024;
// Minimum row-scan chunk: C is a multiple of 128, so 128-aligned chunks keep the tail a multiple of 128.
constexpr uint32_t SCAN_CHUNK_ALIGN = 128;

using BackwardStagePool = TBufPool<QuePosition::VECCALC, STAGE_POOL_BUFFER_COUNT>;

constexpr CastTrait MHC_BWD_B16_TO_FP32 = {
    RegLayout::ZERO,
    SatMode::UNKNOWN,
    MaskMergeMode::ZEROING,
    RoundMode::UNKNOWN,
};

constexpr CastTrait MHC_BWD_FP32_TO_B16 = {
    RegLayout::ZERO,
    SatMode::NO_SAT,
    MaskMergeMode::ZEROING,
    RoundMode::CAST_RINT,
};

__aicore__ inline uint32_t CeilDivU32(uint32_t value, uint32_t divisor)
{
    return (value + divisor - 1) / divisor;
}

__aicore__ inline uint32_t AlignU32(uint32_t value, uint32_t align)
{
    return CeilDivU32(value, align) * align;
}

constexpr uint32_t ConstCeilDivU32(uint32_t value, uint32_t divisor)
{
    return (value + divisor - 1) / divisor;
}

constexpr uint32_t ConstAlignU32(uint32_t value, uint32_t align)
{
    return ConstCeilDivU32(value, align) * align;
}

__aicore__ constexpr bool IsPow2U32(uint32_t value)
{
    return value != 0 && (value & (value - 1)) == 0;
}

__aicore__ constexpr uint32_t Log2U32(uint32_t value)
{
    uint32_t log = 0;
    while ((1U << log) < value) {
        ++log;
    }
    return log;
}

constexpr uint32_t RoundUpPow2U32(uint32_t value)
{
    uint32_t pow2 = 1;
    while (pow2 < value) {
        pow2 <<= 1;
    }
    return pow2;
}

// Layout derived from runtime N (1 to 8).
struct MhcBwdShape {
    uint32_t n = 0;
    uint32_t hValue = 0; // hcMix = 2N + N^2
    // MM2 transposed LoadA requires kStep to be an even number of 8-column fractals.
    uint32_t hAlign = 0;
    // MM1 reduction K: zero-pad to a multiple of 8 with a minimum of 16, the minimum Mmad K granularity.
    uint32_t kMM1 = 0;
    // Load at least 16 phi rows to keep LoadB mStep nonzero.
    uint32_t kMM1Load = 0;
    // MM2 output rows: pad odd hcMix with one zero row for equal dualDstCtl=1 splits, with a minimum of 16.
    uint32_t mm2M = 0;
    uint32_t mm2RowsPerAiv = 0;

    // Per-row state layout: [g_h_pre(N)][grad_post(N)][sk(N^2)][gz(hcMix)][inv_rms(1)], with segments aligned to 8.
    uint32_t stateGhPre = 0;
    uint32_t stateGradPost = 0;
    uint32_t stateSk = 0;
    uint32_t stateGz = 0;
    uint32_t stateInvRms = 0;
    uint32_t smallStateElements = 0;

    uint32_t paramBias = 0;
    uint32_t paramElements = 0;

    // Per-row input staging layout, with all segments aligned to 8.
    uint32_t rowInputInvRms = 0;
    uint32_t rowInputGradPost = 0;
    uint32_t rowInputSk = 0;
    uint32_t rowInputElementsAligned = 0;

    uint32_t iterInputElements = 0;
    uint32_t skMatStride = 0;
    uint32_t batchIterColSum = 0;
    uint32_t batchIterNorm = 0;
    uint32_t batchIterElements = 0;
    uint32_t scalarReduceElements = 0;
    uint32_t scalarReduceAligned = 0;
    bool skContiguousBatch = false;
    uint32_t batchSkStride = 0;
    uint32_t batchSkElements = 0;

    __aicore__ inline void Init(uint32_t nValue)
    {
        n = nValue;
        hValue = n * n + 2 * n;
        hAlign = AlignU32(hValue, 16);
        kMM1 = hValue < 16 ? 16 : AlignU32(hValue, 8);
        kMM1Load = kMM1 < 16 ? 16 : kMM1;
        mm2M = hValue < 16 ? 16 : hValue + (hValue & 1);
        mm2RowsPerAiv = mm2M / 2;

        stateGhPre = 0;
        stateGradPost = FLOAT_C0;
        stateSk = stateGradPost + FLOAT_C0;
        stateGz = stateSk + AlignU32(n * n, 8);
        stateInvRms = stateGz + AlignU32(hValue, 8);
        smallStateElements = stateInvRms + FLOAT_C0;
        paramBias = FLOAT_C0;
        paramElements = FLOAT_C0 + hValue;

        rowInputInvRms = hAlign;
        rowInputGradPost = hAlign + FLOAT_C0;
        rowInputSk = rowInputGradPost + FLOAT_C0;
        rowInputElementsAligned = rowInputSk + AlignU32(n * n, 8);

        iterInputElements = AlignU32(ITER_INPUT_NORM + n * n, 8);
        skMatStride = AlignU32(n * n, FLOAT_C0);
        batchIterColSum = AlignU32(PREPARE_BATCH * n, 8);
        batchIterNorm = 2 * batchIterColSum;
        batchIterElements = AlignU32(batchIterNorm + PREPARE_BATCH * skMatStride, 8);
        scalarReduceElements = hValue + 3; // [alpha(3)][bias(hcMix)]
        scalarReduceAligned = AlignU32(scalarReduceElements, 8);
        skContiguousBatch = (n * n % FLOAT_C0 == 0) && (PREPARE_BATCH * n * n <= VECTOR_LENGTH);
        batchSkStride = skContiguousBatch ? n * n : skMatStride;
        batchSkElements = stateSk + PREPARE_BATCH * batchSkStride;
    }
};

template <typename T>
__aicore__ inline void CopyGmToUb(const GlobalTensor<T>& src, const LocalTensor<T>& dst, uint32_t count)
{
    DataCopyPadExtParams<T> pad = {false, 0, 0, 0};
    DataCopyExtParams copy = {1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
    DataCopyPad(dst, src, copy, pad);
}

template <typename T>
__aicore__ inline void CopyUbToGm(const GlobalTensor<T>& dst, const LocalTensor<T>& src, uint32_t count)
{
    DataCopyExtParams copy = {1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
    DataCopyPad(dst, src, copy);
}

// For small N, wait for MTE2 completion to avoid device exceptions from consecutive small GM reads.
__aicore__ inline void SyncMte2Reads()
{
    TEventID evtId = GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>();
    SetFlag<HardEvent::MTE2_V>(evtId);
    WaitFlag<HardEvent::MTE2_V>(evtId);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(evtId);
}

// Wait for MTE3 completion to avoid device exceptions from consecutive small GM writes.
__aicore__ inline void SyncMte3Writes()
{
    TEventID evtId = GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>();
    SetFlag<HardEvent::MTE3_MTE2>(evtId);
    WaitFlag<HardEvent::MTE3_MTE2>(evtId);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(evtId);
}

// Copy count elements using register-width transfers; src and dst must be aligned to 8 elements.
__aicore__ inline void VFCopySmall(const LocalTensor<float>& dst, const LocalTensor<float>& src, uint32_t count)
{
    __ubuf__ float* dstAddr = (__ubuf__ float*)dst.GetPhyAddr();
    __ubuf__ float* srcAddr = (__ubuf__ float*)src.GetPhyAddr();
    uint16_t loopCount = static_cast<uint16_t>(CeilDivU32(count, VECTOR_LENGTH));
    __VEC_SCOPE__
    {
        RegTensor<float> data;
        uint32_t sreg = count;
        for (uint16_t i = 0; i < loopCount; ++i) {
            uint32_t offset = static_cast<uint32_t>(i) * VECTOR_LENGTH;
            MaskReg mask = UpdateMask<float>(sreg);
            LoadAlign(data, srcAddr + offset);
            StoreAlign(dstAddr + offset, data, mask);
        }
    }
}

__aicore__ inline void VFAccumulate(const LocalTensor<float>& dst, const LocalTensor<float>& src, uint32_t count)
{
    __ubuf__ float* dstAddr = (__ubuf__ float*)dst.GetPhyAddr();
    __ubuf__ float* srcAddr = (__ubuf__ float*)src.GetPhyAddr();
    uint16_t loopCount = static_cast<uint16_t>(CeilDivU32(count, VECTOR_LENGTH));
    __VEC_SCOPE__
    {
        RegTensor<float> dstReg;
        RegTensor<float> srcReg;
        uint32_t sreg = count;
        for (uint16_t i = 0; i < loopCount; ++i) {
            uint32_t offset = static_cast<uint32_t>(i) * VECTOR_LENGTH;
            MaskReg mask = UpdateMask<float>(sreg);
            LoadAlign(dstReg, dstAddr + offset);
            LoadAlign(srcReg, srcAddr + offset);
            Add(dstReg, dstReg, srcReg, mask);
            StoreAlign(dstAddr + offset, dstReg, mask);
        }
    }
}

__aicore__ inline void VFSumRows(const LocalTensor<float>& dst, const LocalTensor<float>& src, uint16_t rows,
                                 uint32_t count, uint32_t rowStride = K_TILE_VALUE)
{
    __ubuf__ float* dstAddr = (__ubuf__ float*)dst.GetPhyAddr();
    __ubuf__ float* srcAddr = (__ubuf__ float*)src.GetPhyAddr();
    uint16_t repeatTimes = static_cast<uint16_t>(CeilDivU32(count, VECTOR_LENGTH));
    __VEC_SCOPE__
    {
        RegTensor<float> sum;
        RegTensor<float> partial;
        uint32_t sreg = count;
        for (uint16_t repeat = 0; repeat < repeatTimes; ++repeat) {
            uint32_t offset = repeat * VECTOR_LENGTH;
            MaskReg mask = UpdateMask<float>(sreg);
            Duplicate(sum, 0.0f);
            for (uint16_t row = 0; row < rows; ++row) {
                LoadAlign(partial, srcAddr + row * rowStride + offset);
                Add(sum, sum, partial, mask);
            }
            StoreAlign(dstAddr + offset, sum, mask);
        }
    }
}

__aicore__ inline void VFAdd(const LocalTensor<float>& dst, const LocalTensor<float>& lhs,
                             const LocalTensor<float>& rhs, uint32_t count)
{
    __ubuf__ float* dstAddr = (__ubuf__ float*)dst.GetPhyAddr();
    __ubuf__ float* lhsAddr = (__ubuf__ float*)lhs.GetPhyAddr();
    __ubuf__ float* rhsAddr = (__ubuf__ float*)rhs.GetPhyAddr();
    uint16_t loopCount = static_cast<uint16_t>(CeilDivU32(count, VECTOR_LENGTH));
    __VEC_SCOPE__
    {
        RegTensor<float> lhsReg;
        RegTensor<float> rhsReg;
        RegTensor<float> dstReg;
        uint32_t sreg = count;
        for (uint16_t i = 0; i < loopCount; ++i) {
            uint32_t offset = static_cast<uint32_t>(i) * VECTOR_LENGTH;
            MaskReg mask = UpdateMask<float>(sreg);
            Load(lhsReg, lhsAddr + offset);
            Load(rhsReg, rhsAddr + offset);
            Add(dstReg, lhsReg, rhsReg, mask);
            StoreAlign(dstAddr + offset, dstReg, mask);
        }
    }
}

// x is ND with one extra 32-byte gap per row. y is NZ with M aligned to eight FP32 values.
// validCols counts valid columns; PAD_GQ copies only valid column fractals, and the caller clears padding in the tail
// fractal.
template <bool PAD_ROWS = false, bool PAD_GQ = false>
__aicore__ inline void TransNdToNz(const LocalTensor<float>& y, const LocalTensor<float>& x, uint16_t rows,
                                   uint16_t cols, uint16_t srcRowStride, uint16_t validRows = ROWS_PER_AIV,
                                   uint16_t validCols = 0)
{
    __ubuf__ float* dst = (__ubuf__ float*)y.GetPhyAddr();
    __ubuf__ float* src = (__ubuf__ float*)x.GetPhyAddr();
    uint16_t rowsAlign = AlignU32(rows, FLOAT_C0);
    uint16_t colsAlign = AlignU32(cols, FLOAT_C0);
    uint16_t colBlocks = (PAD_GQ ? CeilDivU32(validCols, FLOAT_C0) : colsAlign / FLOAT_C0);
    uint16_t padBlocks = (PAD_GQ ? colsAlign / FLOAT_C0 : colBlocks) - colBlocks;
    uint16_t rowBlocks = rowsAlign / FLOAT_C0;
    uint32_t srcBlockStride = srcRowStride / FLOAT_C0;
    __VEC_SCOPE__
    {
        RegTensor<float> data;
        RegTensor<float> zero;
        MaskReg mask = CreateMask<float>();
        MaskReg rowMask;
        uint32_t validElements;
        if constexpr (PAD_ROWS || PAD_GQ) {
            Duplicate(zero, 0.0f);
        }
        for (uint16_t colBlock = 0; colBlock < colBlocks; ++colBlock) {
            if constexpr (PAD_ROWS) {
                validElements = validRows * FLOAT_C0;
            }
            for (uint16_t rowBlock = 0; rowBlock < rowBlocks; ++rowBlock) {
                if constexpr (PAD_ROWS) {
                    rowMask = UpdateMask<float>(validElements);
                    LoadAlign<float, DataCopyMode::DATA_BLOCK_COPY>(
                        data, src + rowBlock * FLOAT_C0 * srcRowStride + colBlock * FLOAT_C0, srcBlockStride, rowMask);
                    Select(data, data, zero, rowMask);
                } else {
                    LoadAlign<float, DataCopyMode::DATA_BLOCK_COPY>(
                        data, src + rowBlock * FLOAT_C0 * srcRowStride + colBlock * FLOAT_C0, srcBlockStride, mask);
                }
                StoreAlign(dst + colBlock * rowsAlign * FLOAT_C0 + rowBlock * VECTOR_LENGTH, data, mask);
            }
        }
        if constexpr (PAD_GQ) {
            // MM2 transposed loads read the full H_ALIGN width; clear padded fractals beyond the valid column fractals
            // here.
            for (uint16_t padBlock = 0; padBlock < padBlocks; ++padBlock) {
                uint16_t colBlock = colBlocks + padBlock;
                for (uint16_t rowBlock = 0; rowBlock < rowBlocks; ++rowBlock) {
                    StoreAlign(dst + colBlock * rowsAlign * FLOAT_C0 + rowBlock * VECTOR_LENGTH, zero, mask);
                }
            }
        }
    }
}

// Row-scan reduction: dst[n] += sum_i lhs[i] * rhs[n*count + i]. Accumulate chunks with RMW; the caller clears dst
// first.
template <typename T>
__aicore__ inline void VFReducePreGrad(const LocalTensor<float>& dst, const LocalTensor<T>& lhs,
                                       const LocalTensor<T>& rhs, uint32_t count, uint32_t nValue)
{
    __ubuf__ float* dstAddr = (__ubuf__ float*)dst.GetPhyAddr();
    __ubuf__ T* lhsAddr = (__ubuf__ T*)lhs.GetPhyAddr();
    __ubuf__ T* rhsAddr = (__ubuf__ T*)rhs.GetPhyAddr();
    uint16_t loopCount = static_cast<uint16_t>(CeilDivU32(count, VECTOR_LENGTH));
    __VEC_SCOPE__
    {
        RegTensor<T> lhsB16;
        RegTensor<T> rhsB16;
        RegTensor<float> lhsFp32;
        RegTensor<float> rhsFp32;
        RegTensor<float> product;
        RegTensor<float> partial;
        RegTensor<float> sum;
        RegTensor<float> partOld;
        RegTensor<float> partNew;
        MaskReg fullMask = CreateMask<float>();
        MaskReg mask1 = CreateMask<float, MaskPattern::VL1>();
        for (uint16_t n = 0; n < static_cast<uint16_t>(nValue); ++n) {
            Duplicate(sum, 0.0f);
            uint32_t sreg = count;
            for (uint16_t i = 0; i < loopCount; ++i) {
                uint32_t offset = static_cast<uint32_t>(i) * VECTOR_LENGTH;
                MaskReg mask = UpdateMask<float>(sreg);
                LoadAlign<T, LoadDist::DIST_UNPACK_B16>(lhsB16, lhsAddr + offset);
                LoadAlign<T, LoadDist::DIST_UNPACK_B16>(rhsB16, rhsAddr + n * count + offset);
                Cast<float, T, MHC_BWD_B16_TO_FP32>(lhsFp32, lhsB16, mask);
                Cast<float, T, MHC_BWD_B16_TO_FP32>(rhsFp32, rhsB16, mask);
                Mul(product, lhsFp32, rhsFp32, mask);
                Add(sum, sum, product, mask);
            }
            Reduce<Reg::ReduceType::SUM>(partial, sum, fullMask);
            Load(partOld, dstAddr + n);
            Add(partNew, partOld, partial, mask1);
            Store(dstAddr + n, partNew, 1);
        }
    }
}

// With GLOBAL=true, registers hold the entire batch of B*N^2 elements; otherwise process one matrix at a time.
template <uint32_t BATCH, bool GLOBAL>
__aicore__ inline void VFSinkhornBackwardPass(__ubuf__ float* skAddr, __ubuf__ float* sumAddr, __ubuf__ float* normAddr,
                                              uint32_t iterElements, uint32_t colSumOffset, uint16_t iterations,
                                              uint32_t validElems, uint32_t matrixRowBase, uint32_t matrixNormOffset,
                                              uint32_t n, bool pow2Index, uint32_t logN, float hcEps)
{
    uint32_t NN = n * n;
    __VEC_SCOPE__
    {
        uint32_t sreg = validElems;
        MaskReg mask = UpdateMask<float>(sreg);
        MaskReg fullMask = CreateMask<uint32_t>();
        RegTensor<uint32_t> order;
        RegTensor<uint32_t> bits;
        RegTensor<uint32_t> divisor;
        RegTensor<uint32_t> colIndex;
        RegTensor<uint32_t> sumColIndex;
        RegTensor<uint32_t> matrixBase;
        RegTensor<uint32_t> rowBase;
        RegTensor<uint32_t> rowIndex;
        RegTensor<uint32_t> index;
        Arange((RegTensor<int32_t>&)order, static_cast<int32_t>(0));
        if constexpr (GLOBAL) {
            // Precompute pow2Index/logN on the scalar side to avoid unsupported ctpop instructions in vector code.
            if (pow2Index) {
                Duplicate(bits, n - 1);
                And(colIndex, order, bits, fullMask);
                Duplicate(bits, (BATCH * n - 1) * n);
                And(rowBase, order, bits, fullMask);
                ShiftRights(rowIndex, rowBase, static_cast<int16_t>(logN), fullMask);
                if constexpr (BATCH != 1) {
                    Duplicate(bits, (BATCH - 1) * NN);
                    And(matrixBase, order, bits, fullMask);
                    ShiftRights(sumColIndex, matrixBase, static_cast<int16_t>(logN), fullMask);
                    Add(sumColIndex, sumColIndex, colIndex, fullMask);
                    Add(colIndex, colIndex, matrixBase, fullMask);
                }
            } else {
                Duplicate(divisor, n);
                Div(rowIndex, order, divisor, fullMask);
                Mul(rowBase, rowIndex, divisor, fullMask);
                Sub(colIndex, order, rowBase, fullMask);
                if constexpr (BATCH != 1) {
                    RegTensor<uint32_t> matIdx;
                    RegTensor<uint32_t> divisorNN;
                    Duplicate(divisorNN, NN);
                    Div(matIdx, order, divisorNN, fullMask);
                    Mul(matrixBase, matIdx, divisorNN, fullMask);
                    Mul(sumColIndex, matIdx, divisor, fullMask);
                    Add(sumColIndex, sumColIndex, colIndex, fullMask);
                    Add(colIndex, colIndex, matrixBase, fullMask);
                }
            }
        } else {
            // Single-matrix pass: lane l maps to matrix element l, with matrix-component offsets added as immediates.
            // Lanes outside the mask also execute Gather, so clamp indices to the register width.
            Duplicate(divisor, n);
            Div(rowIndex, order, divisor, fullMask);
            Mul(rowBase, rowIndex, divisor, fullMask);
            Sub(colIndex, order, rowBase, fullMask);
            Adds(rowIndex, rowIndex, matrixRowBase, fullMask);
            Adds(sumColIndex, colIndex, matrixRowBase, fullMask);
            Duplicate(bits, static_cast<uint32_t>(VECTOR_LENGTH - 1));
            And(rowIndex, rowIndex, bits, fullMask);
            And(sumColIndex, sumColIndex, bits, fullMask);
        }

        RegTensor<float> grad;
        RegTensor<float> norm;
        RegTensor<float> denom;
        RegTensor<float> denomSquared;
        RegTensor<float> product;
        RegTensor<float> correction;
        RegTensor<float> sum;
        RegTensor<float> part;
        LoadAlign(grad, skAddr);
        for (uint16_t step = 0; step < iterations; ++step) {
            uint32_t iterOffset = (iterations - 1 - step) * iterElements;
            LoadAlign(norm, normAddr + iterOffset + matrixNormOffset);
            LoadAlign(sum, sumAddr + iterOffset + colSumOffset);
            if constexpr (BATCH == 1) {
                Gather(denom, sum, colIndex);
            } else {
                Gather(denom, sum, sumColIndex);
            }
            Mul(product, grad, norm, mask);
            Duplicate(correction, 0.0f);
            for (uint16_t k = 0; k < static_cast<uint16_t>(n); ++k) {
                Adds(index, colIndex, static_cast<uint32_t>(k) * n, fullMask);
                Gather(part, product, index);
                Add(correction, correction, part, mask);
            }
            Mul(denomSquared, denom, denom, mask);
            Div(correction, correction, denomSquared, mask);
            Div(grad, grad, denom, mask);
            Sub(grad, grad, correction, mask);

            // In the final row backward pass, subtract the forward hcEps from normOut[0] and sumOut[0].
            if (iterOffset == 0U) {
                Adds(norm, norm, -hcEps, mask);
            }
            Mul(product, grad, norm, mask);
            Duplicate(correction, 0.0f);
            for (uint16_t k = 0; k < static_cast<uint16_t>(n); ++k) {
                Adds(index, rowBase, static_cast<uint32_t>(k), fullMask);
                Gather(part, product, index);
                Add(correction, correction, part, mask);
            }
            LoadAlign(sum, sumAddr + iterOffset);
            Gather(denom, sum, rowIndex);
            if (iterOffset == 0U) {
                Adds(denom, denom, -hcEps, mask);
            }
            Sub(grad, grad, correction, mask);
            Div(grad, grad, denom, mask);
        }
        StoreAlign(skAddr, grad, mask);
    }
}

// Run batched backward when sk is packed contiguously for the batch; otherwise process one matrix at a time.
template <uint32_t BATCH = 1>
__aicore__ inline void VFSinkhornBackward(const MhcBwdShape& shape, const LocalTensor<float>& sk,
                                          const LocalTensor<float>& iterInput, uint16_t iterations, float hcEps,
                                          uint32_t validRows = 1)
{
    uint32_t n = shape.n;
    uint32_t NN = n * n;
    bool pow2Index = IsPow2U32(n) && (BATCH == 1 || IsPow2U32(BATCH * n));
    uint32_t logN = Log2U32(n);
    uint32_t iterElements = BATCH == 1 ? shape.iterInputElements : shape.batchIterElements;
    uint32_t colSumOffset = BATCH == 1 ? ITER_INPUT_COL_SUM : shape.batchIterColSum;
    uint32_t normOffset = BATCH == 1 ? ITER_INPUT_NORM : shape.batchIterNorm;
    __ubuf__ float* skAddr = (__ubuf__ float*)sk[shape.stateSk].GetPhyAddr();
    __ubuf__ float* sumAddr = (__ubuf__ float*)iterInput[ITER_INPUT_SUM].GetPhyAddr();
    __ubuf__ float* normAddr = (__ubuf__ float*)iterInput[normOffset].GetPhyAddr();
    if constexpr (BATCH == 1) {
        VFSinkhornBackwardPass<BATCH, true>(skAddr, sumAddr, normAddr, iterElements, colSumOffset, iterations, NN, 0, 0,
                                            n, pow2Index, logN, hcEps);
    } else if (shape.skContiguousBatch) {
        VFSinkhornBackwardPass<BATCH, true>(skAddr, sumAddr, normAddr, iterElements, colSumOffset, iterations,
                                            validRows * NN, 0, 0, n, pow2Index, logN, hcEps);
    } else {
        uint32_t skStride = shape.smallStateElements;
        for (uint32_t matrix = 0; matrix < BATCH; ++matrix) {
            if (matrix >= validRows) {
                continue;
            }
            VFSinkhornBackwardPass<BATCH, false>(skAddr + matrix * skStride, sumAddr, normAddr, iterElements,
                                                 colSumOffset, iterations, NN, matrix * n, matrix * shape.skMatStride,
                                                 n, pow2Index, logN, hcEps);
        }
    }
}

template <bool BATCHED = false>
__aicore__ inline void VFPrepareGqAndPartials(const MhcBwdShape& shape, const LocalTensor<float>& q,
                                              const LocalTensor<float>& params, const LocalTensor<float>& state,
                                              const LocalTensor<float>& sk, const LocalTensor<float>& gq,
                                              const LocalTensor<float>& gr, const LocalTensor<float>& rowPart,
                                              uint32_t localRow, float hcEps, float negativeInvK, uint16_t rows = 1)
{
    uint32_t n = shape.n;
    uint32_t hValue = shape.hValue;
    uint32_t hAlign = shape.hAlign;
    uint32_t paramBias = shape.paramBias;
    uint32_t stateGz = shape.stateGz;
    uint32_t stateInvRms = shape.stateInvRms;
    uint32_t stateGhPre = shape.stateGhPre;
    uint32_t stateGradPost = shape.stateGradPost;
    uint32_t smallState = shape.smallStateElements;
    // The sk read stride is N*N when packed in batchSk; otherwise sk resides in each state row's sk slot.
    uint32_t skReadStride = (BATCHED && shape.skContiguousBatch) ? n * n : smallState;
    __ubuf__ float* qBase = (__ubuf__ float*)q.GetPhyAddr();
    __ubuf__ float* paramAddr = (__ubuf__ float*)params.GetPhyAddr();
    __ubuf__ float* stateBase = (__ubuf__ float*)state.GetPhyAddr();
    __ubuf__ float* skBase = (__ubuf__ float*)sk.GetPhyAddr();
    __ubuf__ float* gqBase = (__ubuf__ float*)gq.GetPhyAddr() + localRow * hAlign;
    __ubuf__ float* grBase = (__ubuf__ float*)gr.GetPhyAddr() + localRow;
    __ubuf__ float* partAddr = (__ubuf__ float*)rowPart.GetPhyAddr();
    // h_pre = sigmoid(z) + hcEps; recompute sigmoid(z) from z for the gate derivative, independent of hcEps.
    (void)hcEps;
    __VEC_SCOPE__
    {
        uint32_t sregN = n;
        MaskReg maskN = UpdateMask<float>(sregN);
        MaskReg mask1 = CreateMask<float, MaskPattern::VL1>();
        uint32_t smallGroupSreg = n;
        uint32_t residualGroupSreg = n * n;
        MaskReg smallGroupMask = UpdateMask<float>(smallGroupSreg);
        MaskReg residualGroupMask = UpdateMask<float>(residualGroupSreg);
        RegTensor<float> qReg;
        RegTensor<float> uReg;
        RegTensor<float> biasReg;
        RegTensor<float> alphaReg;
        RegTensor<float> invReg;
        RegTensor<float> zReg;
        RegTensor<float> negReg;
        RegTensor<float> expReg;
        RegTensor<float> sigmoidReg;
        RegTensor<float> oneReg;
        RegTensor<float> gateReg;
        RegTensor<float> upstreamReg;
        RegTensor<float> gzReg;
        RegTensor<float> maxReg;
        RegTensor<float> maxBroadcast;
        RegTensor<float> weighted;
        RegTensor<float> weightedSum;
        RegTensor<float> weightedBroadcast;
        RegTensor<float> zeroReg;
        RegTensor<float> correction;
        RegTensor<int32_t> order;
        RegTensor<int32_t> sentinel;
        RegTensor<int32_t> candidate;
        RegTensor<int32_t> firstIndex;
        RegTensor<int32_t> firstBroadcast;
        MaskReg equalMax;
        MaskReg firstMask;
        RegTensor<float> grSum;
        RegTensor<float> gradNorm;
        RegTensor<float> product;
        RegTensor<float> partial;
        Duplicate(oneReg, 1.0f);
        Duplicate(zeroReg, 0.0f);
        Arange(order, static_cast<int32_t>(0));
        Duplicate(sentinel, static_cast<int32_t>(n));
        // Keep cross-row partials in registers; only GZ scratch needs a local barrier.
        RegTensor<float> alphaPre;
        RegTensor<float> alphaPost;
        RegTensor<float> alphaRes;
        RegTensor<float> biasPre;
        RegTensor<float> biasPost;
        RegTensor<float> biasRes;
        // Compute partials only for this call; the kernel accumulates across calls.
        Duplicate(alphaPre, 0.0f);
        Duplicate(alphaPost, 0.0f);
        Duplicate(alphaRes, 0.0f);
        Duplicate(biasPre, 0.0f);
        Duplicate(biasPost, 0.0f);
        Duplicate(biasRes, 0.0f);
        for (uint16_t batchRow = 0; batchRow < rows; ++batchRow) {
            LoadAlign<float, LoadDist::DIST_BRC_B32>(invReg, stateBase + batchRow * smallState + stateInvRms);

            LoadAlign(qReg, qBase + batchRow * hAlign);
            LoadAlign(biasReg, paramAddr + paramBias);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(alphaReg, paramAddr + PARAM_ALPHA);
            Mul(uReg, qReg, invReg, maskN);
            Mul(zReg, uReg, alphaReg, maskN);
            Add(zReg, zReg, biasReg, maskN);
            Neg(negReg, zReg, maskN);
            Exp(expReg, negReg, maskN);
            Adds(expReg, expReg, 1.0f, maskN);
            Div(sigmoidReg, oneReg, expReg, maskN);
            Exp(gateReg, zReg, maskN);
            Adds(gateReg, gateReg, 1.0f, maskN);
            Div(gateReg, oneReg, gateReg, maskN);
            Mul(gzReg, sigmoidReg, gateReg, maskN);
            LoadAlign(upstreamReg, stateBase + batchRow * smallState + stateGhPre);
            Mul(gzReg, gzReg, upstreamReg, maskN);
            StoreAlign(stateBase + batchRow * smallState + stateGz, gzReg, maskN);

            Load(qReg, qBase + batchRow * hAlign + n);
            Load(biasReg, paramAddr + paramBias + n);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(alphaReg, paramAddr + PARAM_ALPHA + 1);
            Mul(uReg, qReg, invReg, maskN);
            Mul(zReg, uReg, alphaReg, maskN);
            Add(zReg, zReg, biasReg, maskN);
            Neg(negReg, zReg, maskN);
            Exp(expReg, negReg, maskN);
            Adds(expReg, expReg, 1.0f, maskN);
            Div(sigmoidReg, oneReg, expReg, maskN);
            Exp(gateReg, zReg, maskN);
            Adds(gateReg, gateReg, 1.0f, maskN);
            Div(gateReg, oneReg, gateReg, maskN);
            Mul(gzReg, sigmoidReg, gateReg, maskN);
            LoadAlign(upstreamReg, stateBase + batchRow * smallState + stateGradPost);
            Mul(gzReg, gzReg, upstreamReg, maskN);
            Muls(gzReg, gzReg, 2.0f, maskN);
            Store(stateBase + batchRow * smallState + stateGz + n, gzReg, n);

            LoadAlign<float, LoadDist::DIST_BRC_B32>(alphaReg, paramAddr + PARAM_ALPHA + 2);
            for (uint16_t row = 0; row < static_cast<uint16_t>(n); ++row) {
                Load(qReg, qBase + batchRow * hAlign + 2 * n + row * n);
                Load(biasReg, paramAddr + paramBias + 2 * n + row * n);
                Mul(uReg, qReg, invReg, maskN);
                Mul(zReg, uReg, alphaReg, maskN);
                Add(zReg, zReg, biasReg, maskN);
                Reduce<Reg::ReduceType::MAX>(maxReg, zReg, maskN);
                Duplicate(maxBroadcast, maxReg, maskN);
                Compare<float, CMPMODE::EQ>(equalMax, zReg, maxBroadcast, maskN);
                Select(candidate, order, sentinel, equalMax);
                Reduce<Reg::ReduceType::MIN>(firstIndex, candidate, maskN);
                Duplicate(firstBroadcast, firstIndex, maskN);
                Compare<int32_t, CMPMODE::EQ>(firstMask, order, firstBroadcast, maskN);
                Sub(expReg, zReg, maxBroadcast, maskN);
                Exp(expReg, expReg, maskN);
                Load(upstreamReg, skBase + batchRow * skReadStride + row * n);
                Mul(weighted, expReg, upstreamReg, maskN);
                Reduce<Reg::ReduceType::SUM>(weightedSum, weighted, maskN);
                Duplicate(weightedBroadcast, weightedSum, maskN);
                Select(correction, weightedBroadcast, zeroReg, firstMask);
                Sub(gzReg, weighted, correction, maskN);
                Store(stateBase + batchRow * smallState + stateGz + 2 * n + row * n, gzReg, n);
            }
            // The group reductions below reload the GZ scratch written above.
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

            Duplicate(grSum, 0.0f);
            for (uint16_t group = 0; group < 2; ++group) {
                Load(gzReg, stateBase + batchRow * smallState + stateGz + group * n);
                Load(qReg, qBase + batchRow * hAlign + group * n);
                LoadAlign<float, LoadDist::DIST_BRC_B32>(alphaReg, paramAddr + PARAM_ALPHA + group);
                Mul(gradNorm, gzReg, alphaReg, smallGroupMask);
                Mul(product, gradNorm, invReg, smallGroupMask);
                Store(gqBase + batchRow * hAlign + group * n, product, n);
                Mul(product, qReg, gradNorm, smallGroupMask);
                Reduce<Reg::ReduceType::SUM>(partial, product, smallGroupMask);
                Add(grSum, grSum, partial, mask1);

                Mul(uReg, qReg, invReg, smallGroupMask);
                Mul(product, uReg, gzReg, smallGroupMask);
                Reduce<Reg::ReduceType::SUM>(partial, product, smallGroupMask);
                if (group == 0) {
                    Add(alphaPre, alphaPre, partial, mask1);
                    Add(biasPre, biasPre, gzReg, smallGroupMask);
                } else {
                    Add(alphaPost, alphaPost, partial, mask1);
                    Add(biasPost, biasPost, gzReg, smallGroupMask);
                }
            }

            // Use unaligned loads when 2N is not a multiple of 8.
            if (2 * n % 8 == 0) {
                LoadAlign(gzReg, stateBase + batchRow * smallState + stateGz + 2 * n);
                LoadAlign(qReg, qBase + batchRow * hAlign + 2 * n);
            } else {
                Load(gzReg, stateBase + batchRow * smallState + stateGz + 2 * n);
                Load(qReg, qBase + batchRow * hAlign + 2 * n);
            }
            LoadAlign<float, LoadDist::DIST_BRC_B32>(alphaReg, paramAddr + PARAM_ALPHA + 2);
            Mul(gradNorm, gzReg, alphaReg, residualGroupMask);
            Mul(product, gradNorm, invReg, residualGroupMask);
            // Use unaligned stores when 2N is not a multiple of 8.
            if (2 * n % 8 == 0) {
                StoreAlign(gqBase + batchRow * hAlign + 2 * n, product, residualGroupMask);
            } else {
                Store(gqBase + batchRow * hAlign + 2 * n, product, n * n);
            }
            Mul(product, qReg, gradNorm, residualGroupMask);
            Reduce<Reg::ReduceType::SUM>(partial, product, residualGroupMask);
            Add(grSum, grSum, partial, mask1);

            Mul(uReg, qReg, invReg, residualGroupMask);
            Mul(product, uReg, gzReg, residualGroupMask);
            Reduce<Reg::ReduceType::SUM>(partial, product, residualGroupMask);
            Add(alphaRes, alphaRes, partial, mask1);
            Add(biasRes, biasRes, gzReg, residualGroupMask);
            // The RMS coefficient is invariant across all K tiles of this row.
            Mul(product, invReg, invReg, mask1);
            Mul(product, product, invReg, mask1);
            Mul(product, product, grSum, mask1);
            Muls(product, product, negativeInvK, mask1);
            Store(grBase + batchRow, product, 1);
        }
        Store(partAddr, alphaPre, 1);
        Store(partAddr + 1, alphaPost, 1);
        Store(partAddr + 2, alphaRes, 1);
        Store(partAddr + 3, biasPre, n);
        Store(partAddr + 3 + n, biasPost, n);
        Store(partAddr + 3 + 2 * n, biasRes, n * n);
    }
}

template <typename T, typename OUT_T = T>
__aicore__ inline void VFComputeGradX(const LocalTensor<OUT_T>& gxOut, const LocalTensor<float>& xNd,
                                      const LocalTensor<T>& xIn, const LocalTensor<T>& gradHin,
                                      const LocalTensor<float>& mm1, const LocalTensor<float>& scalars,
                                      const LocalTensor<float>& gr, uint16_t validRows, uint32_t count,
                                      uint32_t hPreStride = FLOAT_C0, uint32_t colOffset = 0)
{
    __ubuf__ OUT_T* gxAddr = (__ubuf__ OUT_T*)gxOut.GetPhyAddr();
    __ubuf__ float* xNdAddr = (__ubuf__ float*)xNd.GetPhyAddr();
    __ubuf__ T* xAddr = (__ubuf__ T*)xIn.GetPhyAddr();
    __ubuf__ T* ghAddr = (__ubuf__ T*)gradHin.GetPhyAddr();
    __ubuf__ float* mmAddr = (__ubuf__ float*)mm1.GetPhyAddr();
    __ubuf__ float* scalarAddr = (__ubuf__ float*)scalars.GetPhyAddr();
    __ubuf__ float* grAddr = (__ubuf__ float*)gr.GetPhyAddr();
    uint16_t loopCount = static_cast<uint16_t>(CeilDivU32(count, VECTOR_LENGTH));
    __VEC_SCOPE__
    {
        RegTensor<T> xB16;
        RegTensor<T> ghB16;
        RegTensor<T> gxB16;
        RegTensor<float> xFp32;
        RegTensor<float> ghFp32;
        RegTensor<float> mmReg;
        RegTensor<float> hPreReg;
        RegTensor<float> coeffReg;
        RegTensor<float> result;
        MaskReg fullMask = CreateMask<float>();
        for (uint16_t row = 0; row < validRows; ++row) {
            LoadAlign<float, LoadDist::DIST_BRC_B32>(hPreReg, scalarAddr + row * hPreStride);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(coeffReg, grAddr + row);
            uint32_t sreg = count;
            for (uint16_t i = 0; i < loopCount; ++i) {
                uint32_t col = static_cast<uint32_t>(i) * VECTOR_LENGTH;
                // Pack source segments (x/gh) contiguously with row stride count; write destinations (xNd/gx/mm1) at
                // tile-global column offsets to support K_TILE segments across channels, each reading its channel's
                // grad_hin and h_pre.
                uint32_t srcOffset = row * count + col;
                uint32_t dstOffset = row * K_TILE_VALUE + colOffset + col;
                MaskReg mask = UpdateMask<float>(sreg);
                LoadAlign<T, LoadDist::DIST_UNPACK_B16>(xB16, xAddr + srcOffset);
                LoadAlign<T, LoadDist::DIST_UNPACK_B16>(ghB16, ghAddr + srcOffset);
                Cast<float, T, MHC_BWD_B16_TO_FP32>(xFp32, xB16, mask);
                Cast<float, T, MHC_BWD_B16_TO_FP32>(ghFp32, ghB16, mask);
                StoreAlign(xNdAddr + row * (K_TILE_VALUE + ND_GAP) + colOffset + col, xFp32, mask);
                Mul(ghFp32, ghFp32, hPreReg, mask);
                Mul(xFp32, xFp32, coeffReg, mask);
                Add(result, ghFp32, xFp32, mask);
                if constexpr (IsSameType<OUT_T, float>::value) {
                    StoreAlign(gxAddr + dstOffset, result, mask);
                } else {
                    LoadAlign(mmReg, mmAddr + dstOffset);
                    Add(result, result, mmReg, mask);
                    Cast<T, float, MHC_BWD_FP32_TO_B16>(gxB16, result, mask);
                    StoreAlign<T, StoreDist::DIST_PACK_B32>(gxAddr + dstOffset, gxB16, mask);
                }
            }
        }
    }
}

template <typename T>
__aicore__ inline void VFFinishGradX(const LocalTensor<T>& dst, const LocalTensor<float>& base,
                                     const LocalTensor<float>& mm1, uint32_t count)
{
    __ubuf__ T* dstAddr = (__ubuf__ T*)dst.GetPhyAddr();
    __ubuf__ float* baseAddr = (__ubuf__ float*)base.GetPhyAddr();
    __ubuf__ float* mmAddr = (__ubuf__ float*)mm1.GetPhyAddr();
    uint16_t loops = static_cast<uint16_t>(CeilDivU32(count, VECTOR_LENGTH));
    __VEC_SCOPE__
    {
        RegTensor<float> value;
        RegTensor<float> mm;
        RegTensor<T> result;
        uint32_t sreg = count;
        for (uint16_t i = 0; i < loops; ++i) {
            uint32_t offset = static_cast<uint32_t>(i) * VECTOR_LENGTH;
            MaskReg mask = UpdateMask<float>(sreg);
            LoadAlign(value, baseAddr + offset);
            LoadAlign(mm, mmAddr + offset);
            Add(value, value, mm, mask);
            Cast<T, float, MHC_BWD_FP32_TO_B16>(result, value, mask);
            StoreAlign<T, StoreDist::DIST_PACK_B32>(dstAddr + offset, result, mask);
        }
    }
}

template <typename T>
class MhcPreSinkhornBackwardVector {
    MhcBwdShape shape_;

public:
    __aicore__ inline void InitShape(uint32_t n)
    {
        shape_.Init(n);
    }

    __aicore__ inline void Init(GM_ADDR gradHin, GM_ADDR gradHPost, GM_ADDR gradHRes, GM_ADDR x, GM_ADDR phi,
                                GM_ADDR alpha, GM_ADDR bias, GM_ADDR hPre, GM_ADDR hcBeforeNorm, GM_ADDR invRms,
                                GM_ADDR sumOut, GM_ADDR normOut, GM_ADDR gradX, GM_ADDR gradPhi, GM_ADDR gradAlpha,
                                GM_ADDR gradBias, const MhcPreSinkhornBackwardArch35TilingData* tiling)
    {
        gradHinGm_.SetGlobalBuffer((__gm__ T*)gradHin);
        gradHPostGm_.SetGlobalBuffer((__gm__ float*)gradHPost);
        gradHResGm_.SetGlobalBuffer((__gm__ float*)gradHRes);
        xGm_.SetGlobalBuffer((__gm__ T*)x);
        // Disable L2 caching for single-pass x/grad_x accesses to keep phi cached.
        xGm_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        phiGm_.SetGlobalBuffer((__gm__ float*)phi);
        alphaGm_.SetGlobalBuffer((__gm__ float*)alpha);
        biasGm_.SetGlobalBuffer((__gm__ float*)bias);
        hPreGm_.SetGlobalBuffer((__gm__ float*)hPre);
        qGm_.SetGlobalBuffer((__gm__ float*)hcBeforeNorm);
        invRmsGm_.SetGlobalBuffer((__gm__ float*)invRms);
        sumGm_.SetGlobalBuffer((__gm__ float*)sumOut);
        normGm_.SetGlobalBuffer((__gm__ float*)normOut);
        gradXGm_.SetGlobalBuffer((__gm__ T*)gradX);
        gradXGm_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        gradPhiGm_.SetGlobalBuffer((__gm__ float*)gradPhi);
        gradAlphaGm_.SetGlobalBuffer((__gm__ float*)gradAlpha);
        gradBiasGm_.SetGlobalBuffer((__gm__ float*)gradBias);
        tiling_ = tiling;
    }

    template <uint32_t BUFFER_COUNT>
    __aicore__ inline void InitPermanentBuffers(TBufPool<QuePosition::VECCALC, BUFFER_COUNT>& pool,
                                                uint32_t cacheRows = 0)
    {
        pool.InitBuffer(paramsQue_, 1, shape_.paramElements * sizeof(float));
        if (cacheRows != 0) {
            pool.InitBuffer(hPreCacheBuf_, AlignU32(cacheRows * shape_.n, FLOAT_C0) * sizeof(float));
            hPreCache_ = hPreCacheBuf_.template Get<float>();
        }
    }

    __aicore__ inline void LoadHPre(const LocalTensor<float>& dst, int64_t rowBase, uint32_t rows)
    {
        CopyGmToUb(hPreGm_[rowBase * shape_.n], dst, rows * shape_.n);
    }

    __aicore__ inline void CacheHPre(const LocalTensor<float>& src, uint32_t cacheRow, uint32_t rows)
    {
        DataCopy(hPreCache_[cacheRow * shape_.n], src, AlignU32(rows * shape_.n, FLOAT_C0));
    }

    __aicore__ inline void InitPrepareStage(BackwardStagePool& pool, uint32_t batchRows = 1)
    {
        uint32_t inputBuffers = batchRows == 1 ? 1 : 2;
        uint32_t iterElements = batchRows == 1 ? shape_.iterInputElements : shape_.batchIterElements;
        uint32_t fixedBytes = tiling_->iterTimes * iterElements * sizeof(float) +
                              inputBuffers * shape_.rowInputElementsAligned * sizeof(float) +
                              (batchRows * shape_.hAlign + SMALL_BUFFER_SLACK) * sizeof(float) +
                              (batchRows * shape_.smallStateElements + SMALL_BUFFER_SLACK) * sizeof(float) +
                              (shape_.batchSkElements + SMALL_BUFFER_SLACK) * sizeof(float) + SCAN_BUDGET_MARGIN_BYTES;
        // Choose the row-scan chunk size from the remaining stage pool capacity.
        uint32_t budget = STAGE_POOL_BYTES > fixedBytes ? STAGE_POOL_BYTES - fixedBytes : SCAN_CHUNK_ALIGN;
        uint32_t chunkCap = budget / (inputBuffers * (shape_.n + 1) * sizeof(T));
        chunkCap = chunkCap / SCAN_CHUNK_ALIGN * SCAN_CHUNK_ALIGN;
        scanChunk_ = chunkCap < SCAN_CHUNK_ALIGN ? SCAN_CHUNK_ALIGN : chunkCap;
        if (static_cast<uint32_t>(tiling_->c) < scanChunk_) {
            scanChunk_ = static_cast<uint32_t>(tiling_->c);
        }
        scanChunks_ = CeilDivU32(static_cast<uint32_t>(tiling_->c), scanChunk_);
        pool.InitBuffer(scanXQue_, inputBuffers, shape_.n * scanChunk_ * sizeof(T));
        pool.InitBuffer(scanGhQue_, inputBuffers, scanChunk_ * sizeof(T));
        pool.InitBuffer(rowInputQue_, inputBuffers, shape_.rowInputElementsAligned * sizeof(float));
        pool.InitBuffer(iterInputQue_, 1,
                        tiling_->iterTimes * iterElements * sizeof(float) + SMALL_BUFFER_SLACK * sizeof(float));
        pool.InitBuffer(qBuf_, (batchRows * shape_.hAlign + SMALL_BUFFER_SLACK) * sizeof(float));
        pool.InitBuffer(stateBuf_, (batchRows * shape_.smallStateElements + SMALL_BUFFER_SLACK) * sizeof(float));
        pool.InitBuffer(batchSkBuf_, (shape_.batchSkElements + SMALL_BUFFER_SLACK) * sizeof(float));
        q_ = qBuf_.Get<float>();
        state_ = stateBuf_.Get<float>();
        batchSk_ = batchSkBuf_.Get<float>();
    }

    __aicore__ inline void InitGradXStage(BackwardStagePool& pool, bool deferMm1 = false)
    {
        pool.InitBuffer(tileInputQue_, 2, ROWS_PER_AIV * K_TILE_VALUE * sizeof(T));
        pool.InitBuffer(tileScalarQue_, 1, TILE_SCALAR_ELEMENTS * sizeof(float));
        // Double-buffer grad_x to overlap vector output with MTE3 writeback.
        pool.InitBuffer(gxOutQue_, 2, ROWS_PER_AIV * K_TILE_VALUE * sizeof(T));
        pool.InitBuffer(xStageQue_, deferMm1 ? 2 : 1,
                        (X_STAGE_ELEMENTS + (deferMm1 ? ROWS_PER_AIV * K_TILE_VALUE : 0)) * sizeof(float));
    }

    __aicore__ inline void LoadConstants()
    {
        LocalTensor<float> params = paramsQue_.AllocTensor<float>();
        CopyGmToUb(alphaGm_, params[PARAM_ALPHA], 3);
        CopyGmToUb(biasGm_, params[shape_.paramBias], shape_.hValue);
        paramsQue_.EnQue(params);
        params_ = paramsQue_.DeQue<float>();
    }

    __aicore__ inline void ReleaseConstants()
    {
        paramsQue_.FreeTensor(params_);
    }

    __aicore__ inline void PrepareRow(int64_t row, uint32_t localRow, const LocalTensor<float>& gqNd,
                                      const LocalTensor<float>& gr, const LocalTensor<float>& rowPart)
    {
        Duplicate(state_[shape_.stateGhPre], 0.0f, shape_.n);
        PrefetchRowInput(row);
        for (uint32_t chunk = 0; chunk < scanChunks_; ++chunk) {
            LocalTensor<T> scanGh = ScanChunkIn(row, chunk);
            LocalTensor<T> scanX = ScanXChunkIn(row, chunk);
            VFReducePreGrad(state_[shape_.stateGhPre], scanGh, scanX, ScanChunkLen(chunk),
                            static_cast<uint32_t>(tiling_->n));
            scanGhQue_.FreeTensor(scanGh);
            scanXQue_.FreeTensor(scanX);
        }
        ConsumeRowInput(row, q_, state_);
        LocalTensor<float> iterInput = iterInputQue_.AllocTensor<float>();
        LoadIterInput(iterInput, row, 1);
        iterInputQue_.EnQue(iterInput);
        iterInput = iterInputQue_.DeQue<float>();
        VFSinkhornBackward(shape_, state_, iterInput, tiling_->iterTimes, tiling_->hcEps);
        iterInputQue_.FreeTensor(iterInput);
        VFPrepareGqAndPartials(shape_, q_, params_, state_, state_[shape_.stateSk], gqNd, gr, rowPart, localRow,
                               tiling_->hcEps, -1.0f / static_cast<float>(tiling_->k));
    }

    __aicore__ inline void PrepareRows(int64_t rowBase, uint32_t rows, const LocalTensor<float>& gqNd,
                                       const LocalTensor<float>& gr, const LocalTensor<float>& rowPart)
    {
        for (uint32_t row = 0; row < rows; ++row) {
            Duplicate(state_[row * shape_.smallStateElements + shape_.stateGhPre], 0.0f, shape_.n);
        }
        // Double-buffer the row scan to overlap the next chunk's transfer with the current chunk's reduction.
        uint32_t totalChunks = rows * scanChunks_;
        PrefetchScanChunk(rowBase, 0);
        PrefetchRowInput(rowBase);
        for (uint32_t idx = 0; idx < totalChunks; ++idx) {
            uint32_t row = idx / scanChunks_;
            uint32_t chunk = idx % scanChunks_;
            if (idx + 1 < totalChunks) {
                uint32_t nextRow = (idx + 1) / scanChunks_;
                PrefetchScanChunk(rowBase + nextRow, (idx + 1) % scanChunks_);
                if ((idx + 1) % scanChunks_ == 0) {
                    PrefetchRowInput(rowBase + nextRow);
                }
            }
            LocalTensor<T> scanGh = scanGhQue_.DeQue<T>();
            LocalTensor<T> scanX = scanXQue_.DeQue<T>();
            VFReducePreGrad(state_[row * shape_.smallStateElements + shape_.stateGhPre], scanGh, scanX,
                            ScanChunkLen(chunk), static_cast<uint32_t>(tiling_->n));
            scanGhQue_.FreeTensor(scanGh);
            scanXQue_.FreeTensor(scanX);
            if (chunk == 0) {
                ConsumeRowInput(rowBase + row, q_[row * shape_.hAlign], state_[row * shape_.smallStateElements]);
            }
        }
        CopySkToBatch(rows);
        LocalTensor<float> iterInput = iterInputQue_.AllocTensor<float>();
        LoadBatchIterInput(iterInput, rowBase, rows);
        iterInputQue_.EnQue(iterInput);
        iterInput = iterInputQue_.DeQue<float>();
        if (shape_.skContiguousBatch) {
            VFSinkhornBackward<PREPARE_BATCH>(shape_, batchSk_, iterInput, tiling_->iterTimes, tiling_->hcEps, rows);
        } else {
            VFSinkhornBackward<PREPARE_BATCH>(shape_, state_, iterInput, tiling_->iterTimes, tiling_->hcEps, rows);
        }
        iterInputQue_.FreeTensor(iterInput);
        if (shape_.skContiguousBatch) {
            VFPrepareGqAndPartials<true>(shape_, q_, params_, state_, batchSk_[shape_.stateSk], gqNd, gr, rowPart, 0,
                                         tiling_->hcEps, -1.0f / static_cast<float>(tiling_->k),
                                         static_cast<uint16_t>(rows));
        } else {
            VFPrepareGqAndPartials<true>(shape_, q_, params_, state_, state_[shape_.stateSk], gqNd, gr, rowPart, 0,
                                         tiling_->hcEps, -1.0f / static_cast<float>(tiling_->k),
                                         static_cast<uint16_t>(rows));
        }
    }

    template <bool DEFER_MM1 = false>
    __aicore__ inline void ComputeGradXAndStageX(int64_t rowBase, uint32_t validRows, uint32_t kBegin, uint32_t kValid,
                                                 const LocalTensor<float>& mm1, const LocalTensor<float>& gr,
                                                 uint32_t cacheRow = 0)
    {
        LocalTensor<float> xStage = xStageQue_.AllocTensor<float>();
        LocalTensor<float> xNd = xStage;
        LocalTensor<float> xNz = xStage[X_STAGE_NZ_OFFSET];
        if (validRows == 0) {
            AscendC::Duplicate(xNz, 0.0f, ROWS_PER_AIV * K_TILE_VALUE);
            xStageQue_.EnQue(xStage);
            return;
        }
        if (validRows != 0) {
            LocalTensor<T> gxOut;
            if constexpr (!DEFER_MM1) {
                gxOut = gxOutQue_.AllocTensor<T>();
            }
            // Read cross-channel K_TILE data in segments at channel boundaries and write each result into the same
            // buffer at kOff.
            uint32_t kOff = 0;
            while (kOff < kValid) {
                uint32_t nIdx = (kBegin + kOff) / static_cast<uint32_t>(tiling_->c);
                uint32_t cBegin = (kBegin + kOff) - nIdx * static_cast<uint32_t>(tiling_->c);
                uint32_t segLen = static_cast<uint32_t>(tiling_->c) - cBegin;
                if (segLen > kValid - kOff) {
                    segLen = kValid - kOff;
                }
                // Copy source segments contiguously with row stride segLen; use tile-global column indices and row
                // stride K_TILE in the destination.
                DataCopyPadExtParams<T> inputPad = {false, 0, 0, 0};
                DataCopyExtParams inputCopy = {static_cast<uint16_t>(validRows),
                                               static_cast<uint32_t>(segLen * sizeof(T)),
                                               static_cast<uint32_t>((tiling_->k - segLen) * sizeof(T)), 0, 0};
                LocalTensor<T> xIn = tileInputQue_.AllocTensor<T>();
                DataCopyPad(xIn, xGm_[rowBase * tiling_->k + kBegin + kOff], inputCopy, inputPad);
                tileInputQue_.EnQue(xIn);
                LocalTensor<T> ghIn = tileInputQue_.AllocTensor<T>();
                inputCopy.srcStride = static_cast<uint32_t>((tiling_->c - segLen) * sizeof(T));
                DataCopyPad(ghIn, gradHinGm_[rowBase * tiling_->c + cBegin], inputCopy, inputPad);
                tileInputQue_.EnQue(ghIn);
                LocalTensor<float> scalars;
                if constexpr (DEFER_MM1) {
                    scalars = hPreCache_[cacheRow * shape_.n + nIdx];
                } else {
                    scalars = tileScalarQue_.AllocTensor<float>();
                    DataCopyExtParams scalarCopy = {static_cast<uint16_t>(validRows), sizeof(float),
                                                    static_cast<int64_t>((shape_.n - 1) * sizeof(float)), 0, 0};
                    DataCopyPadExtParams<float> scalarPad = {false, 0, 0, 0};
                    DataCopyPad(scalars, hPreGm_[rowBase * shape_.n + nIdx], scalarCopy, scalarPad);
                    tileScalarQue_.EnQue(scalars);
                }

                xIn = tileInputQue_.DeQue<T>();
                ghIn = tileInputQue_.DeQue<T>();
                if constexpr (!DEFER_MM1) {
                    scalars = tileScalarQue_.DeQue<float>();
                }
                if constexpr (DEFER_MM1) {
                    VFComputeGradX<T, float>(xStage[X_STAGE_ELEMENTS], xNd, xIn, ghIn, mm1, scalars, gr, validRows,
                                             segLen, shape_.n, kOff);
                } else {
                    VFComputeGradX(gxOut, xNd, xIn, ghIn, mm1, scalars, gr, validRows, segLen, FLOAT_C0, kOff);
                }
                tileInputQue_.FreeTensor(xIn);
                tileInputQue_.FreeTensor(ghIn);
                if constexpr (!DEFER_MM1) {
                    tileScalarQue_.FreeTensor(scalars);
                }
                kOff += segLen;
            }
            if constexpr (!DEFER_MM1) {
                StoreGradX(gxOut, rowBase, validRows, kBegin, kValid);
            }
        }
        if (validRows == ROWS_PER_AIV) {
            TransNdToNz(xNz, xNd, ROWS_PER_AIV, K_TILE_VALUE, K_TILE_VALUE + ND_GAP);
        } else {
            TransNdToNz<true>(xNz, xNd, ROWS_PER_AIV, K_TILE_VALUE, K_TILE_VALUE + ND_GAP, validRows);
        }
        xStageQue_.EnQue(xStage);
    }

    __aicore__ inline LocalTensor<float> DeQueXStage(uint32_t slot = 0)
    {
        activeXStage_[slot] = xStageQue_.DeQue<float>();
        return activeXStage_[slot];
    }

    __aicore__ inline void FinishGradX(int64_t rowBase, uint32_t validRows, uint32_t kBegin, uint32_t kValid,
                                       const LocalTensor<float>& mm1, uint32_t slot = 0)
    {
        if (validRows != 0) {
            LocalTensor<T> output = gxOutQue_.AllocTensor<T>();
            VFFinishGradX(output, activeXStage_[slot][X_STAGE_ELEMENTS], mm1, validRows * K_TILE_VALUE);
            StoreGradX(output, rowBase, validRows, kBegin, kValid);
        }
        xStageQue_.FreeTensor(activeXStage_[slot]);
    }

    __aicore__ inline void FreeXStage(LocalTensor<float>& xStage)
    {
        xStageQue_.FreeTensor(xStage);
    }

    __aicore__ inline GlobalTensor<float>& GradPhiGm()
    {
        return gradPhiGm_;
    }
    __aicore__ inline GlobalTensor<float>& GradAlphaGm()
    {
        return gradAlphaGm_;
    }
    __aicore__ inline GlobalTensor<float>& GradBiasGm()
    {
        return gradBiasGm_;
    }
    __aicore__ inline GlobalTensor<float>& PhiGm()
    {
        return phiGm_;
    }
    __aicore__ inline GlobalTensor<T>& XGm()
    {
        return xGm_;
    }

private:
    __aicore__ inline uint32_t ScanChunkLen(uint32_t chunk) const
    {
        uint32_t remain = static_cast<uint32_t>(tiling_->c) - chunk * scanChunk_;
        return remain < scanChunk_ ? remain : scanChunk_;
    }

    __aicore__ inline void PrefetchScanChunk(int64_t row, uint32_t chunk)
    {
        uint32_t len = ScanChunkLen(chunk);
        uint32_t lenBytes = len * sizeof(T);
        uint32_t gapBytes = static_cast<uint32_t>(tiling_->c - len) * sizeof(T);
        DataCopyPadExtParams<T> pad = {false, 0, 0, 0};
        LocalTensor<T> scanGh = scanGhQue_.AllocTensor<T>();
        DataCopyExtParams ghCopy = {1, lenBytes, 0, 0, 0};
        DataCopyPad(scanGh, gradHinGm_[row * tiling_->c + chunk * scanChunk_], ghCopy, pad);
        scanGhQue_.EnQue(scanGh);
        LocalTensor<T> scanX = scanXQue_.AllocTensor<T>();
        // x rows use [shape_.n][C] layout; copy the current C chunk as N fractal blocks with gap C minus the chunk
        // length.
        DataCopyExtParams xCopy = {static_cast<uint16_t>(shape_.n), lenBytes, gapBytes, 0, 0};
        DataCopyPad(scanX, xGm_[row * tiling_->k + chunk * scanChunk_], xCopy, pad);
        scanXQue_.EnQue(scanX);
    }

    __aicore__ inline LocalTensor<T> ScanChunkIn(int64_t row, uint32_t chunk)
    {
        PrefetchScanChunk(row, chunk);
        return scanGhQue_.DeQue<T>();
    }

    __aicore__ inline LocalTensor<T> ScanXChunkIn(int64_t row, uint32_t chunk)
    {
        return scanXQue_.DeQue<T>();
    }

    __aicore__ inline void PrefetchRowInput(int64_t row)
    {
        LocalTensor<float> input = rowInputQue_.AllocTensor<float>();
        if (shape_.hValue < 16) {
            // For small N (hcMix<16), wait for MTE2 completion between four small GM reads to separate consecutive
            // small descriptors.
            CopyGmToUb(qGm_[row * shape_.hValue], input[ROW_INPUT_Q], shape_.hValue);
            SyncMte2Reads();
            CopyGmToUb(invRmsGm_[row], input[shape_.rowInputInvRms], 1);
            SyncMte2Reads();
            CopyGmToUb(gradHPostGm_[row * shape_.n], input[shape_.rowInputGradPost], shape_.n);
            SyncMte2Reads();
            CopyGmToUb(gradHResGm_[row * shape_.n * shape_.n], input[shape_.rowInputSk], shape_.n * shape_.n);
        } else {
            CopyGmToUb(qGm_[row * shape_.hValue], input[ROW_INPUT_Q], shape_.hValue);
            CopyGmToUb(invRmsGm_[row], input[shape_.rowInputInvRms], 1);
            CopyGmToUb(gradHPostGm_[row * shape_.n], input[shape_.rowInputGradPost], shape_.n);
            CopyGmToUb(gradHResGm_[row * shape_.n * shape_.n], input[shape_.rowInputSk], shape_.n * shape_.n);
        }
        rowInputQue_.EnQue(input);
    }

    __aicore__ inline void ConsumeRowInput(int64_t row, const LocalTensor<float>& q, const LocalTensor<float>& state)
    {
        LocalTensor<float> input = rowInputQue_.DeQue<float>();
        VFCopySmall(q, input[ROW_INPUT_Q], shape_.hValue);
        VFCopySmall(state[shape_.stateInvRms], input[shape_.rowInputInvRms], FLOAT_C0);
        VFCopySmall(state[shape_.stateGradPost], input[shape_.rowInputGradPost], FLOAT_C0);
        VFCopySmall(state[shape_.stateSk], input[shape_.rowInputSk], shape_.n * shape_.n);
        rowInputQue_.FreeTensor(input);
    }

    __aicore__ inline void CopySkToBatch(uint32_t rows)
    {
        // Gather sk into batchSk for the packed batch layout.
        if (shape_.skContiguousBatch) {
            for (uint32_t row = 0; row < rows; ++row) {
                DataCopy(batchSk_[shape_.stateSk + row * shape_.n * shape_.n],
                         state_[row * shape_.smallStateElements + shape_.stateSk], shape_.n * shape_.n);
            }
        }
    }

    __aicore__ inline void LoadIterInput(const LocalTensor<float>& iterInput, int64_t row, uint32_t rows)
    {
        DataCopyPadExtParams<float> iterPad = {false, 0, 0, 0};
        DataCopyExtParams iterCopy = {static_cast<uint16_t>(tiling_->iterTimes),
                                      static_cast<uint32_t>(shape_.n * sizeof(float)),
                                      static_cast<uint32_t>((2 * tiling_->bs * shape_.n - shape_.n) * sizeof(float)),
                                      (shape_.iterInputElements - FLOAT_C0) / FLOAT_C0, 0};
        DataCopyPad(iterInput[ITER_INPUT_SUM], sumGm_[row * shape_.n], iterCopy, iterPad);
        if (shape_.hValue < 16) {
            SyncMte2Reads();
        }
        DataCopyPad(iterInput[ITER_INPUT_COL_SUM], sumGm_[(tiling_->bs + row) * shape_.n], iterCopy, iterPad);
        if (shape_.hValue < 16) {
            SyncMte2Reads();
        }
        iterCopy.blockLen = shape_.n * shape_.n * sizeof(float);
        iterCopy.srcStride = (2 * tiling_->bs - 1) * shape_.n * shape_.n * sizeof(float);
        iterCopy.dstStride = (shape_.iterInputElements - shape_.n * shape_.n) / FLOAT_C0;
        DataCopyPad(iterInput[ITER_INPUT_NORM], normGm_[row * shape_.n * shape_.n], iterCopy, iterPad);
    }

    __aicore__ inline void LoadBatchIterInput(const LocalTensor<float>& iterInput, int64_t rowBase, uint32_t rows)
    {
        DataCopyPadExtParams<float> pad = {false, 0, 0, 0};
        DataCopyExtParams copy = {static_cast<uint16_t>(tiling_->iterTimes),
                                  static_cast<uint32_t>(rows * shape_.n * sizeof(float)),
                                  static_cast<uint32_t>((2 * tiling_->bs - rows) * shape_.n * sizeof(float)),
                                  (shape_.batchIterElements - AlignU32(rows * shape_.n, FLOAT_C0)) / FLOAT_C0, 0};
        DataCopyPad(iterInput, sumGm_[rowBase * shape_.n], copy, pad);
        if (shape_.hValue < 16) {
            SyncMte2Reads();
        }
        DataCopyPad(iterInput[shape_.batchIterColSum], sumGm_[(tiling_->bs + rowBase) * shape_.n], copy, pad);
        // Align each norm matrix to 8 for LoadAlign reads.
        for (uint32_t matrix = 0; matrix < rows; ++matrix) {
            if (shape_.hValue < 16) {
                SyncMte2Reads();
            }
            DataCopyExtParams normCopy = {
                static_cast<uint16_t>(tiling_->iterTimes), static_cast<uint32_t>(shape_.n * shape_.n * sizeof(float)),
                static_cast<uint32_t>((2 * tiling_->bs - 1) * shape_.n * shape_.n * sizeof(float)),
                (shape_.batchIterElements - shape_.skMatStride) / FLOAT_C0, 0};
            DataCopyPad(iterInput[shape_.batchIterNorm + matrix * shape_.skMatStride],
                        normGm_[(rowBase + matrix) * shape_.n * shape_.n], normCopy, pad);
        }
    }

    __aicore__ inline void StoreGradX(LocalTensor<T>& output, int64_t rowBase, uint32_t validRows, uint32_t kBegin,
                                      uint32_t kValid)
    {
        gxOutQue_.EnQue(output);
        output = gxOutQue_.DeQue<T>();
        // The source row stride is K_TILE, with cross-channel segments at global column offsets; specify UB row gaps in
        // 32-byte blocks. Write only kValid valid columns to GM, whose row stride is k.
        DataCopyExtParams copy = {static_cast<uint16_t>(validRows), static_cast<uint32_t>(kValid * sizeof(T)),
                                  static_cast<uint32_t>((K_TILE_VALUE - kValid) * sizeof(T) / 32),
                                  static_cast<uint32_t>((tiling_->k - kValid) * sizeof(T)), 0};
        DataCopyPad(gradXGm_[rowBase * tiling_->k + kBegin], output, copy);
        gxOutQue_.FreeTensor(output);
    }

    LocalTensor<float> activeXStage_[2];
    TBuf<TPosition::VECCALC> hPreCacheBuf_;
    LocalTensor<float> hPreCache_;
    const MhcPreSinkhornBackwardArch35TilingData* tiling_ = nullptr;
    uint32_t scanChunk_ = SCAN_CHUNK_ALIGN;
    uint32_t scanChunks_ = 1;
    GlobalTensor<T> gradHinGm_;
    GlobalTensor<float> gradHPostGm_;
    GlobalTensor<float> gradHResGm_;
    GlobalTensor<T> xGm_;
    GlobalTensor<float> phiGm_;
    GlobalTensor<float> alphaGm_;
    GlobalTensor<float> biasGm_;
    GlobalTensor<float> hPreGm_;
    GlobalTensor<float> qGm_;
    GlobalTensor<float> invRmsGm_;
    GlobalTensor<float> sumGm_;
    GlobalTensor<float> normGm_;
    GlobalTensor<T> gradXGm_;
    GlobalTensor<float> gradPhiGm_;
    GlobalTensor<float> gradAlphaGm_;
    GlobalTensor<float> gradBiasGm_;
    TQue<QuePosition::VECIN, 2> scanXQue_;
    TQue<QuePosition::VECIN, 2> scanGhQue_;
    TQue<QuePosition::VECIN, 1> paramsQue_;
    TQue<QuePosition::VECIN, 2> rowInputQue_;
    TQue<QuePosition::VECIN, 1> iterInputQue_;
    TQue<QuePosition::VECIN, 2> tileInputQue_;
    TQue<QuePosition::VECIN, 1> tileScalarQue_;
    TQue<QuePosition::VECOUT, 2> gxOutQue_;
    TQue<QuePosition::VECOUT, 2> xStageQue_;
    TBuf<TPosition::VECCALC> qBuf_;
    TBuf<TPosition::VECCALC> stateBuf_;
    TBuf<TPosition::VECCALC> batchSkBuf_;
    LocalTensor<float> q_;
    LocalTensor<float> params_;
    LocalTensor<float> state_;
    LocalTensor<float> batchSk_;
};
} // namespace MhcPreSinkhornBackwardNs

#endif // MHC_PRE_SINKHORN_BACKWARD_VECTOR_ARCH35_H
