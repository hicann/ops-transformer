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
 * \file mhc_pre_sinkhorn_backward_sinkhorn_simd_arch35.h
 * \brief Sinkhorn 反向的寄存器布局与向量计算
 */

#ifndef MHC_PRE_SINKHORN_BACKWARD_SINKHORN_SIMD_ARCH35_H
#define MHC_PRE_SINKHORN_BACKWARD_SINKHORN_SIMD_ARCH35_H

#include "kernel_operator.h"
#include "mhc_pre_sinkhorn_backward_data_arch35.h"

namespace MhcPreSinkhornBackwardSinkhornNs {
using namespace AscendC;
namespace Reg = AscendC::MicroAPI;

// 列反向共用公式：correction = sum_rows(grad * norm) / denominator^2；grad = grad / denominator - correction。
__simd_callee__ inline __attribute__((always_inline)) void ScaleColumnCorrection(
    Reg::RegTensor<float>& correction, Reg::RegTensor<float>& denominator, Reg::RegTensor<float>& denominatorSquared,
    Reg::MaskReg& mask)
{
    Reg::Mul(denominatorSquared, denominator, denominator, mask);
    Reg::Div(correction, correction, denominatorSquared, mask);
}

__simd_callee__ inline __attribute__((always_inline)) void ApplyColumnCorrection(Reg::RegTensor<float>& grad,
                                                                                 Reg::RegTensor<float>& correction,
                                                                                 Reg::RegTensor<float>& denominator,
                                                                                 Reg::MaskReg& mask)
{
    Reg::Div(grad, grad, denominator, mask);
    Reg::Sub(grad, grad, correction, mask);
}

// 按 stride 收集元素，保持从第 0 项开始累加的顺序。
__simd_callee__ inline __attribute__((always_inline)) void SumGatheredValues(Reg::RegTensor<float>& sum,
                                                                             Reg::RegTensor<float>& values,
                                                                             Reg::RegTensor<uint32_t>& baseIndex,
                                                                             uint32_t count, uint32_t stride,
                                                                             Reg::MaskReg& fullMask, Reg::MaskReg& mask)
{
    Reg::RegTensor<uint32_t> index;
    Reg::RegTensor<float> partial;
    Reg::Duplicate(sum, 0.0f);
    for (uint16_t k = 0; k < static_cast<uint16_t>(count); ++k) {
        Reg::Adds(index, baseIndex, static_cast<uint32_t>(k) * stride, fullMask);
        Reg::Gather(partial, values, index);
        Reg::Add(sum, sum, partial, mask);
    }
}

// 行修正项为 sum_cols(grad * norm)；宽度 4 保留逐项累加，宽度 8 使用八通道块归约。
template <uint32_t REDUCE_WIDTH>
__simd_callee__ inline __attribute__((always_inline)) void CalcRowCorrection(
    Reg::RegTensor<float>& grad, Reg::RegTensor<float>& norm, Reg::RegTensor<uint32_t>& reduceIndex,
    Reg::RegTensor<float>& product, Reg::RegTensor<float>& correction, bool firstIteration, float eps,
    Reg::MaskReg& fullMask, Reg::MaskReg& mask)
{
    if (firstIteration) {
        Reg::Adds(norm, norm, -eps, mask);
    }
    Reg::Mul(product, grad, norm, mask);
    if constexpr (REDUCE_WIDTH == 4U) {
        // 保留四项显式展开，避免 VF 编译器生成运行时循环。
        Reg::RegTensor<uint32_t> index;
        Reg::RegTensor<float> part;
        Reg::Duplicate(correction, 0.0f);
        Reg::Adds(index, reduceIndex, 0U, fullMask);
        Reg::Gather(part, product, index);
        Reg::Add(correction, correction, part, mask);
        Reg::Adds(index, reduceIndex, 1U, fullMask);
        Reg::Gather(part, product, index);
        Reg::Add(correction, correction, part, mask);
        Reg::Adds(index, reduceIndex, 2U, fullMask);
        Reg::Gather(part, product, index);
        Reg::Add(correction, correction, part, mask);
        Reg::Adds(index, reduceIndex, 3U, fullMask);
        Reg::Gather(part, product, index);
        Reg::Add(correction, correction, part, mask);
    } else {
        static_assert(REDUCE_WIDTH == 8U, "Unsupported row reduction width");
        Reg::ReduceSumWithDataBlock(correction, product, mask);
        Reg::Gather(correction, correction, reduceIndex);
    }
}

// 行反向：grad = (grad - correction) / denominator；前向第 0 轮的行和需扣除 epsilon。
__simd_callee__ inline __attribute__((always_inline)) void ApplyRowCorrection(Reg::RegTensor<float>& grad,
                                                                              Reg::RegTensor<float>& correction,
                                                                              Reg::RegTensor<float>& denominator,
                                                                              bool firstIteration, float eps,
                                                                              Reg::MaskReg& mask)
{
    if (firstIteration) {
        Reg::Adds(denominator, denominator, -eps, mask);
    }
    Reg::Sub(grad, grad, correction, mask);
    Reg::Div(grad, grad, denominator, mask);
}

__simd_callee__ inline __attribute__((always_inline)) void CalcCompactColumnBackward(
    Reg::RegTensor<float>& grad, Reg::RegTensor<float>& norm, __ubuf__ float* sums, Reg::RegTensor<uint32_t>& sumIndex,
    Reg::RegTensor<uint32_t>& columnBase, uint32_t n, Reg::MaskReg& fullMask, Reg::MaskReg& mask)
{
    Reg::RegTensor<float> sum, denominator, product, correction, denominatorSquared;
    Reg::LoadAlign(sum, sums);
    Reg::Gather(denominator, sum, sumIndex);
    Reg::Mul(product, grad, norm, mask);
    SumGatheredValues(correction, product, columnBase, n, n, fullMask, mask);
    ScaleColumnCorrection(correction, denominator, denominatorSquared, mask);
    ApplyColumnCorrection(grad, correction, denominator, mask);
}

__simd_callee__ inline __attribute__((always_inline)) void CalcCompactRowBackward(
    Reg::RegTensor<float>& grad, Reg::RegTensor<float>& norm, __ubuf__ float* sums, Reg::RegTensor<uint32_t>& sumIndex,
    Reg::RegTensor<uint32_t>& rowBase, uint32_t n, bool firstIteration, float eps, Reg::MaskReg& fullMask,
    Reg::MaskReg& mask)
{
    Reg::RegTensor<float> sum, denominator, product, correction;
    if (firstIteration) {
        Reg::Adds(norm, norm, -eps, mask);
    }
    Reg::Mul(product, grad, norm, mask);
    SumGatheredValues(correction, product, rowBase, n, 1U, fullMask, mask);
    Reg::LoadAlign(sum, sums);
    Reg::Gather(denominator, sum, sumIndex);
    ApplyRowCorrection(grad, correction, denominator, firstIteration, eps, mask);
}

// sumBase/sumOffset 指定当前矩阵行和的位置。
template <uint32_t REDUCE_WIDTH>
__simd_callee__ inline __attribute__((always_inline)) void CalcPackedRowBackward(
    Reg::RegTensor<float>& grad, Reg::RegTensor<float>& norm, Reg::RegTensor<float>& sums,
    Reg::RegTensor<uint32_t>& reduceIndex, Reg::RegTensor<uint32_t>& sumBase, uint32_t sumOffset, bool firstIteration,
    float eps, Reg::MaskReg& fullMask, Reg::MaskReg& mask)
{
    Reg::RegTensor<float> product, correction, denominator;
    Reg::RegTensor<uint32_t> sumIndex;
    CalcRowCorrection<REDUCE_WIDTH>(grad, norm, reduceIndex, product, correction, firstIteration, eps, fullMask, mask);
    Reg::Adds(sumIndex, sumBase, sumOffset, fullMask);
    Reg::Gather(denominator, sums, sumIndex);
    ApplyRowCorrection(grad, correction, denominator, firstIteration, eps, mask);
}

// 8×8 矩阵布局通过转置和块归约计算列修正项。
__simd_callee__ inline __attribute__((always_inline)) void CalcMatrixColumnBackward(
    Reg::RegTensor<float>& grad, Reg::RegTensor<float>& norm, Reg::RegTensor<float>& sums,
    Reg::RegTensor<uint32_t>& sumIndex, Reg::RegTensor<uint32_t>& columnIndex, Reg::RegTensor<uint32_t>& transposeIndex,
    Reg::MaskReg& mask)
{
    Reg::RegTensor<float> denominator, product, transposedProduct, correction, denominatorSquared;
    Reg::Gather(denominator, sums, sumIndex);
    Reg::Mul(product, grad, norm, mask);
    Reg::Gather(transposedProduct, product, transposeIndex);
    Reg::ReduceSumWithDataBlock(correction, transposedProduct, mask);
    Reg::Gather(correction, correction, columnIndex);
    ScaleColumnCorrection(correction, denominator, denominatorSquared, mask);
    ApplyColumnCorrection(grad, correction, denominator, mask);
}

// 逐行累加列修正项，加载的 norm 留给后续行反向复用。
__simd_callee__ inline __attribute__((always_inline)) void AccumulatePackedColumnRow(
    Reg::RegTensor<float>& correction, Reg::RegTensor<float>& product, Reg::RegTensor<float>& gradRow,
    Reg::RegTensor<float>& normRow, __ubuf__ float* normAddress, Reg::RegTensor<uint32_t>& index, Reg::MaskReg& mask)
{
    Reg::DataCopyGather(normRow, normAddress, index, mask);
    Reg::Mul(product, gradRow, normRow, mask);
    Reg::Add(correction, correction, product, mask);
}

template <uint32_t N>
__simd_callee__ inline __attribute__((always_inline)) void LoadMatrixRegister(Reg::RegTensor<float>& value,
                                                                              __ubuf__ float* address,
                                                                              Reg::RegTensor<uint32_t>& compactIndex,
                                                                              Reg::MaskReg& mask)
{
    if constexpr (N == 8U) {
        Reg::LoadAlign(value, address);
    } else {
        Reg::DataCopyGather(value, address, compactIndex, mask);
    }
}

template <uint32_t N>
__simd_callee__ inline __attribute__((always_inline)) void StoreMatrixRegister(__ubuf__ float* address,
                                                                               Reg::RegTensor<float>& value,
                                                                               Reg::RegTensor<uint32_t>& compactIndex,
                                                                               Reg::MaskReg& mask)
{
    if constexpr (N == 8U) {
        Reg::StoreAlign(address, value, mask);
    } else {
        Reg::DataCopyScatter(address, value, compactIndex, mask);
    }
}

// 每个通道处理一个矩阵元素，Gather 下标限制在当前矩阵内。
// 列、行修正项均按原顺序逐项累加 N 个元素。
template <uint32_t SpecializedN = 0U>
__aicore__ inline void VFDoBackwardCompactTile(const SinkhornTilePlan& plan, __ubuf__ float* gradOutputAddr,
                                               __ubuf__ float* tileAddr, uint32_t tileMatrixCount, uint16_t iterations,
                                               float eps)
{
    const uint32_t nSize = SpecializedN != 0U ? SpecializedN : plan.nSize;
    const uint32_t matrixElements = SpecializedN != 0U ? SpecializedN * SpecializedN : plan.matrixElements;
    const uint32_t matricesPerVector = SpecializedN != 0U ? MatricesPerVector(SpecializedN) : plan.matricesPerVector;
    const uint32_t sumElementsPerIteration = SpecializedN != 0U ?
                                                 AlignFloatBlock(MatricesPerTile(SpecializedN) * SpecializedN) :
                                                 plan.sumElementsPerIteration;
    const uint32_t normElementsPerIteration =
        SpecializedN != 0U ? AlignFloatBlock(MatricesPerTile(SpecializedN) * matrixElements) : 0U;
    const uint32_t iterationStride =
        SpecializedN != 0U ? 2U * sumElementsPerIteration + normElementsPerIteration : plan.iterationStride;
    const uint32_t iterationDataOffset =
        SpecializedN != 0U ? normElementsPerIteration + VECTOR_ELEMENTS : plan.iterationDataOffset;
    const uint32_t logN = SpecializedN != 0U ? (SpecializedN == 2U ? 1U :
                                                SpecializedN == 4U ? 2U :
                                                SpecializedN == 8U ? 3U :
                                                                     0U) :
                                               plan.logN;
    const bool powerOfTwo = SpecializedN != 0U ? (SpecializedN & (SpecializedN - 1U)) == 0U : plan.powerOfTwo;
    const bool alignedVectors = SpecializedN != 0U ?
                                    (MatricesPerVector(SpecializedN) * matrixElements) % FLOAT_BLOCK_ELEMENTS == 0U &&
                                        (MatricesPerTile(SpecializedN) * matrixElements) % FLOAT_BLOCK_ELEMENTS == 0U :
                                    plan.alignedVectors;
    __ubuf__ float* iterationData = tileAddr + iterationDataOffset;
    __VEC_SCOPE__
    {
        Reg::MaskReg fullMask = Reg::CreateMask<uint32_t>();
        Reg::RegTensor<uint32_t> laneIndex, zeroIndex, divisor, rowIndex, rowBase, columnIndex, matrixBase;
        Reg::RegTensor<uint32_t> colSumIndex, colGatherBase, bits;
        Reg::Arange((Reg::RegTensor<int32_t>&)laneIndex, static_cast<int32_t>(0));
        Reg::Duplicate(zeroIndex, 0U);
        if (nSize == 1U) {
            Reg::Adds(rowIndex, laneIndex, 0U, fullMask);
            Reg::Adds(rowBase, laneIndex, 0U, fullMask);
            Reg::Adds(matrixBase, laneIndex, 0U, fullMask);
            Reg::Duplicate(columnIndex, 0U);
            Reg::Adds(colSumIndex, laneIndex, 0U, fullMask);
        } else if (powerOfTwo) {
            Reg::Duplicate(bits, nSize - 1U);
            Reg::And(columnIndex, laneIndex, bits, fullMask);
            Reg::Duplicate(bits, ~(nSize - 1U));
            Reg::And(rowBase, laneIndex, bits, fullMask);
            Reg::ShiftRights(rowIndex, laneIndex, static_cast<int16_t>(logN), fullMask);
            Reg::Duplicate(bits, ~(matrixElements - 1U));
            Reg::And(matrixBase, laneIndex, bits, fullMask);
            Reg::ShiftRights(colSumIndex, matrixBase, static_cast<int16_t>(logN), fullMask);
            Reg::Add(colSumIndex, colSumIndex, columnIndex, fullMask);
        } else {
            Reg::Duplicate(divisor, nSize);
            Reg::Div(rowIndex, laneIndex, divisor, fullMask);
            Reg::Mul(rowBase, rowIndex, divisor, fullMask);
            Reg::Sub(columnIndex, laneIndex, rowBase, fullMask);
            Reg::RegTensor<uint32_t> matrixIndex, nnDivisor;
            Reg::Duplicate(nnDivisor, matrixElements);
            Reg::Div(matrixIndex, laneIndex, nnDivisor, fullMask);
            Reg::Mul(matrixBase, matrixIndex, nnDivisor, fullMask);
            Reg::Mul(colSumIndex, matrixIndex, divisor, fullMask);
            Reg::Add(colSumIndex, colSumIndex, columnIndex, fullMask);
        }
        Reg::Add(colGatherBase, matrixBase, columnIndex, fullMask);

        Reg::RegTensor<float> gradReg, rowNormReg;
        // N<=5 时一个寄存器组即可容纳整个分块；N>5 时可能需要第二组。
        const uint16_t batches = SpecializedN != 0U && SpecializedN <= 5U ?
                                     1U :
                                     static_cast<uint16_t>(tileMatrixCount > matricesPerVector ? 2U : 1U);
        for (uint16_t batch = 0; batch < batches; ++batch) {
            const uint32_t matrix = SpecializedN != 0U && SpecializedN <= 5U ? 0U : batch * matricesPerVector;
            uint32_t currentRows = tileMatrixCount - matrix;
            if constexpr (SpecializedN == 0U || SpecializedN > 5U) {
                if (currentRows > matricesPerVector) {
                    currentRows = matricesPerVector;
                }
            }
            uint32_t validElements = currentRows * matrixElements;
            uint32_t count = validElements;
            Reg::MaskReg activeMask = Reg::UpdateMask<float>(count);
            // N>5 时按单矩阵分两组处理；复用下标寄存器，降低迭代期间的寄存器占用。
            if (matrix != 0U) {
                Reg::Adds(rowIndex, rowIndex, nSize, fullMask);
                Reg::Adds(colSumIndex, colSumIndex, nSize, fullMask);
            }
            if constexpr (SpecializedN != 4U) {
                Reg::Select(rowIndex, rowIndex, zeroIndex, activeMask);
                Reg::Select(colSumIndex, colSumIndex, zeroIndex, activeMask);
                Reg::Select(rowBase, rowBase, zeroIndex, activeMask);
                Reg::Select(colGatherBase, colGatherBase, zeroIndex, activeMask);
            }

            const uint32_t matrixOffset = static_cast<uint32_t>(matrix) * matrixElements;
            if (alignedVectors) {
                Reg::LoadAlign(gradReg, tileAddr + matrixOffset);
            } else {
                Reg::Load(gradReg, tileAddr + matrixOffset);
            }
            for (uint16_t step = 0; step < iterations; ++step) {
                const uint32_t iterationOffset = (iterations - 1U - step) * iterationStride;
                if (alignedVectors) {
                    Reg::LoadAlign(rowNormReg,
                                   iterationData + iterationOffset + 2U * sumElementsPerIteration + matrixOffset);
                } else {
                    Reg::Load(rowNormReg,
                              iterationData + iterationOffset + 2U * sumElementsPerIteration + matrixOffset);
                }
                CalcCompactColumnBackward(gradReg, rowNormReg,
                                          iterationData + iterationOffset + sumElementsPerIteration, colSumIndex,
                                          colGatherBase, nSize, fullMask, activeMask);
                CalcCompactRowBackward(gradReg, rowNormReg, iterationData + iterationOffset, rowIndex, rowBase, nSize,
                                       iterationOffset == 0U, eps, fullMask, activeMask);
            }
            if (alignedVectors) {
                Reg::StoreAlign(gradOutputAddr + matrixOffset, gradReg, activeMask);
            } else {
                Reg::Store(gradOutputAddr + matrixOffset, gradReg, validElements);
            }
        }
    }
}

// N=4/6 按行打包多个矩阵；列修正项逐行累加，行修正项分别使用 4 项累加或 8 通道块归约。
template <uint32_t N>
__aicore__ inline void VFDoBackwardRowPackedTile(__ubuf__ float* gradOutputAddr, __ubuf__ float* tileAddr,
                                                 uint32_t tileMatrixCount, uint16_t iterations, float eps)
{
    static_assert(N == 4U || N == 6U, "Row-packed Sinkhorn supports N=4/6");
    constexpr uint32_t rowWidth = N == 4U ? 4U : 8U;
    constexpr uint32_t matricesPerTile = VECTOR_ELEMENTS / rowWidth;
    constexpr uint32_t sumElementsPerIteration = AlignFloatBlock(matricesPerTile * N);
    constexpr uint32_t normElementsPerIteration = AlignFloatBlock(matricesPerTile * N * N);
    constexpr uint32_t iterationDataOffset = normElementsPerIteration + VECTOR_ELEMENTS;
    constexpr uint32_t iterationStride = 2U * sumElementsPerIteration + normElementsPerIteration;
    __ubuf__ float* iterationData = tileAddr + iterationDataOffset;
    __VEC_SCOPE__
    {
        Reg::MaskReg fullMask = Reg::CreateMask<uint32_t>();
        Reg::RegTensor<uint32_t> laneIndex, matrix, columnIndex, scale, index, sumBase, sumIndex, rowBase;
        Reg::Arange((Reg::RegTensor<int32_t>&)laneIndex, 0);
        Reg::ShiftRights(matrix, laneIndex, static_cast<int16_t>(N == 4U ? 2 : 3), fullMask);
        Reg::Duplicate(scale, rowWidth - 1U);
        Reg::And(columnIndex, laneIndex, scale, fullMask);
        Reg::Duplicate(scale, N * N);
        Reg::Mul(index, matrix, scale, fullMask);
        Reg::Add(index, index, columnIndex, fullMask);
        if constexpr (N == 6U) {
            Reg::Duplicate(scale, N);
            Reg::Mul(sumBase, matrix, scale, fullMask);
            Reg::Add(sumIndex, sumBase, columnIndex, fullMask);
        }
        Reg::Duplicate(scale, ~(rowWidth - 1U));
        Reg::And(rowBase, laneIndex, scale, fullMask);
        uint32_t lanes = tileMatrixCount * rowWidth;
        Reg::MaskReg activeMask = Reg::UpdateMask<float>(lanes);
        if constexpr (N == 6U) {
            Reg::MaskReg columns;
            Reg::CompareScalar<uint32_t, CMPMODE::LT>(columns, columnIndex, N, fullMask);
            Reg::And(activeMask, activeMask, columns, fullMask);
        }
        Reg::RegTensor<float> gradRow0Reg, gradRow1Reg, gradRow2Reg, gradRow3Reg, gradRow4Reg, gradRow5Reg;
        Reg::RegTensor<float> rowNormRow0Reg, rowNormRow1Reg, rowNormRow2Reg, rowNormRow3Reg, rowNormRow4Reg,
            rowNormRow5Reg;
        Reg::RegTensor<float> sumReg, denominatorReg, denominatorSquaredReg, productReg, correctionReg;
        Reg::DataCopyGather(gradRow0Reg, tileAddr + 0U * N, index, activeMask);
        Reg::DataCopyGather(gradRow1Reg, tileAddr + 1U * N, index, activeMask);
        Reg::DataCopyGather(gradRow2Reg, tileAddr + 2U * N, index, activeMask);
        Reg::DataCopyGather(gradRow3Reg, tileAddr + 3U * N, index, activeMask);
        if constexpr (N == 6U) {
            Reg::DataCopyGather(gradRow4Reg, tileAddr + 4U * N, index, activeMask);
            Reg::DataCopyGather(gradRow5Reg, tileAddr + 5U * N, index, activeMask);
        }
        for (uint16_t step = 0; step < iterations; ++step) {
            const uint32_t iterationOffset = (iterations - 1U - step) * iterationStride;
            // 列反向。
            if constexpr (N == 4U) {
                Reg::LoadAlign(denominatorReg, iterationData + iterationOffset + sumElementsPerIteration);
            } else {
                Reg::LoadAlign(sumReg, iterationData + iterationOffset + sumElementsPerIteration);
                Reg::Gather(denominatorReg, sumReg, sumIndex);
            }
            Reg::Duplicate(correctionReg, 0.0f);
            AccumulatePackedColumnRow(correctionReg, productReg, gradRow0Reg, rowNormRow0Reg,
                                      iterationData + iterationOffset + 2U * sumElementsPerIteration + 0U * N, index,
                                      activeMask);
            AccumulatePackedColumnRow(correctionReg, productReg, gradRow1Reg, rowNormRow1Reg,
                                      iterationData + iterationOffset + 2U * sumElementsPerIteration + 1U * N, index,
                                      activeMask);
            AccumulatePackedColumnRow(correctionReg, productReg, gradRow2Reg, rowNormRow2Reg,
                                      iterationData + iterationOffset + 2U * sumElementsPerIteration + 2U * N, index,
                                      activeMask);
            AccumulatePackedColumnRow(correctionReg, productReg, gradRow3Reg, rowNormRow3Reg,
                                      iterationData + iterationOffset + 2U * sumElementsPerIteration + 3U * N, index,
                                      activeMask);
            if constexpr (N == 6U) {
                AccumulatePackedColumnRow(correctionReg, productReg, gradRow4Reg, rowNormRow4Reg,
                                          iterationData + iterationOffset + 2U * sumElementsPerIteration + 4U * N,
                                          index, activeMask);
                AccumulatePackedColumnRow(correctionReg, productReg, gradRow5Reg, rowNormRow5Reg,
                                          iterationData + iterationOffset + 2U * sumElementsPerIteration + 5U * N,
                                          index, activeMask);
            }
            ScaleColumnCorrection(correctionReg, denominatorReg, denominatorSquaredReg, activeMask);
            ApplyColumnCorrection(gradRow0Reg, correctionReg, denominatorReg, activeMask);
            ApplyColumnCorrection(gradRow1Reg, correctionReg, denominatorReg, activeMask);
            ApplyColumnCorrection(gradRow2Reg, correctionReg, denominatorReg, activeMask);
            ApplyColumnCorrection(gradRow3Reg, correctionReg, denominatorReg, activeMask);
            if constexpr (N == 6U) {
                ApplyColumnCorrection(gradRow4Reg, correctionReg, denominatorReg, activeMask);
                ApplyColumnCorrection(gradRow5Reg, correctionReg, denominatorReg, activeMask);
            }
            // 行反向。
            Reg::LoadAlign(sumReg, iterationData + iterationOffset);
            CalcPackedRowBackward<rowWidth>(gradRow0Reg, rowNormRow0Reg, sumReg, N == 4U ? rowBase : matrix,
                                            N == 4U ? rowBase : sumBase, 0U, iterationOffset == 0U, eps, fullMask,
                                            activeMask);
            CalcPackedRowBackward<rowWidth>(gradRow1Reg, rowNormRow1Reg, sumReg, N == 4U ? rowBase : matrix,
                                            N == 4U ? rowBase : sumBase, 1U, iterationOffset == 0U, eps, fullMask,
                                            activeMask);
            CalcPackedRowBackward<rowWidth>(gradRow2Reg, rowNormRow2Reg, sumReg, N == 4U ? rowBase : matrix,
                                            N == 4U ? rowBase : sumBase, 2U, iterationOffset == 0U, eps, fullMask,
                                            activeMask);
            CalcPackedRowBackward<rowWidth>(gradRow3Reg, rowNormRow3Reg, sumReg, N == 4U ? rowBase : matrix,
                                            N == 4U ? rowBase : sumBase, 3U, iterationOffset == 0U, eps, fullMask,
                                            activeMask);
            if constexpr (N == 6U) {
                CalcPackedRowBackward<rowWidth>(gradRow4Reg, rowNormRow4Reg, sumReg, matrix, sumBase, 4U,
                                                iterationOffset == 0U, eps, fullMask, activeMask);
                CalcPackedRowBackward<rowWidth>(gradRow5Reg, rowNormRow5Reg, sumReg, matrix, sumBase, 5U,
                                                iterationOffset == 0U, eps, fullMask, activeMask);
            }
        }
        Reg::DataCopyScatter(gradOutputAddr + 0U * N, gradRow0Reg, index, activeMask);
        Reg::DataCopyScatter(gradOutputAddr + 1U * N, gradRow1Reg, index, activeMask);
        Reg::DataCopyScatter(gradOutputAddr + 2U * N, gradRow2Reg, index, activeMask);
        Reg::DataCopyScatter(gradOutputAddr + 3U * N, gradRow3Reg, index, activeMask);
        if constexpr (N == 6U) {
            Reg::DataCopyScatter(gradOutputAddr + 4U * N, gradRow4Reg, index, activeMask);
            Reg::DataCopyScatter(gradOutputAddr + 5U * N, gradRow5Reg, index, activeMask);
        }
        Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
    }
}

// 每个 DMA 分块加载多个矩阵，每个矩阵补齐为 8x8 并在迭代期间保留在寄存器中。
// 通过转置，行、列均可使用 8 通道块归约。
template <uint32_t N>
__aicore__ inline void VFDoBackwardMatrixTile(__ubuf__ float* gradOutputAddr, __ubuf__ float* tileAddr,
                                              uint32_t tileMatrixCount, uint16_t iterations, float eps)
{
    static_assert(N == 6U || N == 8U, "Packed Sinkhorn supports N=6/8");
    constexpr uint32_t sumElementsPerIteration = AlignFloatBlock(MatricesPerTile(N) * N);
    constexpr uint32_t normElementsPerIteration = AlignFloatBlock(MatricesPerTile(N) * N * N);
    constexpr uint32_t iterationDataOffset = normElementsPerIteration + VECTOR_ELEMENTS;
    constexpr uint32_t iterationStride = 2U * sumElementsPerIteration + normElementsPerIteration;
    __ubuf__ float* iterationData = tileAddr + iterationDataOffset;
    __VEC_SCOPE__
    {
        Reg::MaskReg fullMask = Reg::CreateMask<uint32_t>();
        Reg::RegTensor<uint32_t> laneIndex, rowIndex, columnIndex, scale, compactIndex, transposeIndex, rowSumIndex,
            colSumIndex;
        Reg::Arange((Reg::RegTensor<int32_t>&)laneIndex, 0);
        Reg::ShiftRights(rowIndex, laneIndex, static_cast<int16_t>(3), fullMask);
        Reg::Duplicate(scale, 7U);
        Reg::And(columnIndex, laneIndex, scale, fullMask);
        Reg::ShiftLefts(transposeIndex, columnIndex, static_cast<int16_t>(3), fullMask);
        Reg::Add(transposeIndex, transposeIndex, rowIndex, fullMask);
        Reg::Duplicate(scale, N);
        Reg::Mul(compactIndex, rowIndex, scale, fullMask);
        Reg::Add(compactIndex, compactIndex, columnIndex, fullMask);
        Reg::MaskReg activeMask = fullMask;
        if constexpr (N == 6U) {
            Reg::MaskReg rowMask, colMask;
            Reg::CompareScalar<uint32_t, CMPMODE::LT>(rowMask, rowIndex, N, fullMask);
            Reg::CompareScalar<uint32_t, CMPMODE::LT>(colMask, columnIndex, N, fullMask);
            Reg::And(activeMask, rowMask, colMask, fullMask);
        }
        Reg::RegTensor<float> gradReg, rowNormReg, sumReg, denominatorReg, productReg, correctionReg;
        for (uint16_t matrix = 0; matrix < static_cast<uint16_t>(tileMatrixCount); ++matrix) {
            const uint32_t offset = static_cast<uint32_t>(matrix) * N * N;
            Reg::Adds(rowSumIndex, rowIndex, static_cast<uint32_t>(matrix) * N, fullMask);
            Reg::Adds(colSumIndex, columnIndex, static_cast<uint32_t>(matrix) * N, fullMask);
            LoadMatrixRegister<N>(gradReg, tileAddr + offset, compactIndex, activeMask);
            for (uint16_t step = 0; step < iterations; ++step) {
                const uint32_t iterationOffset = (iterations - 1U - step) * iterationStride;
                LoadMatrixRegister<N>(rowNormReg,
                                      iterationData + iterationOffset + 2U * sumElementsPerIteration + offset,
                                      compactIndex, activeMask);
                // 列反向。
                Reg::LoadAlign(sumReg, iterationData + iterationOffset + sumElementsPerIteration);
                CalcMatrixColumnBackward(gradReg, rowNormReg, sumReg, colSumIndex, columnIndex, transposeIndex,
                                         activeMask);
                CalcRowCorrection<8U>(gradReg, rowNormReg, rowIndex, productReg, correctionReg, iterationOffset == 0U,
                                      eps, fullMask, activeMask);
                // 行反向。
                Reg::LoadAlign(sumReg, iterationData + iterationOffset);
                Reg::Gather(denominatorReg, sumReg, rowSumIndex);
                ApplyRowCorrection(gradReg, correctionReg, denominatorReg, iterationOffset == 0U, eps, activeMask);
            }
            StoreMatrixRegister<N>(gradOutputAddr + offset, gradReg, compactIndex, activeMask);
        }
        Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
    }
}

// N=8 主块：4 个寄存器分别保存完整矩阵，展开每轮计算以便编译器交错调度。
__aicore__ inline void VFDoBackwardInterleavedTile(__ubuf__ float* gradOutputAddr, __ubuf__ float* tileAddr,
                                                   uint32_t tileMatrixCount, uint16_t iterations, float eps)
{
    constexpr uint32_t N = 8U;
    constexpr uint32_t sumElementsPerIteration = AlignFloatBlock(MatricesPerTile(N) * N);
    constexpr uint32_t normElementsPerIteration = AlignFloatBlock(MatricesPerTile(N) * N * N);
    constexpr uint32_t iterationDataOffset = normElementsPerIteration + VECTOR_ELEMENTS;
    constexpr uint32_t iterationStride = 2U * sumElementsPerIteration + normElementsPerIteration;
    __ubuf__ float* iterationData = tileAddr + iterationDataOffset;
    __VEC_SCOPE__
    {
        Reg::MaskReg fullMask = Reg::CreateMask<uint32_t>();
        Reg::RegTensor<uint32_t> laneIndex, rowIndex, columnIndex, scale, compactIndex, transposeIndex, rowSumIndex,
            colSumIndex;
        Reg::Arange((Reg::RegTensor<int32_t>&)laneIndex, 0);
        Reg::ShiftRights(rowIndex, laneIndex, static_cast<int16_t>(3), fullMask);
        Reg::Duplicate(scale, 7U);
        Reg::And(columnIndex, laneIndex, scale, fullMask);
        Reg::ShiftLefts(transposeIndex, columnIndex, static_cast<int16_t>(3), fullMask);
        Reg::Add(transposeIndex, transposeIndex, rowIndex, fullMask);
        Reg::Duplicate(scale, N);
        Reg::Mul(compactIndex, rowIndex, scale, fullMask);
        Reg::Add(compactIndex, compactIndex, columnIndex, fullMask);
        Reg::MaskReg activeMask = fullMask;
        Reg::RegTensor<float> gradMatrix0Reg, gradMatrix1Reg, gradMatrix2Reg, gradMatrix3Reg, rowNormMatrix0Reg,
            rowNormMatrix1Reg, rowNormMatrix2Reg, rowNormMatrix3Reg;
        Reg::RegTensor<float> sumReg;
        for (uint16_t base = 0; base < static_cast<uint16_t>(tileMatrixCount); base += 4U) {
            uint32_t activeLanes0 = base + 0U < tileMatrixCount ? 64U : 0U;
            Reg::MaskReg matrixMask0 = Reg::UpdateMask<float>(activeLanes0);
            Reg::And(matrixMask0, matrixMask0, activeMask, fullMask);
            LoadMatrixRegister<N>(gradMatrix0Reg, tileAddr + (base + 0U) * N * N, compactIndex, matrixMask0);
            uint32_t activeLanes1 = base + 1U < tileMatrixCount ? 64U : 0U;
            Reg::MaskReg matrixMask1 = Reg::UpdateMask<float>(activeLanes1);
            Reg::And(matrixMask1, matrixMask1, activeMask, fullMask);
            LoadMatrixRegister<N>(gradMatrix1Reg, tileAddr + (base + 1U) * N * N, compactIndex, matrixMask1);
            uint32_t activeLanes2 = base + 2U < tileMatrixCount ? 64U : 0U;
            Reg::MaskReg matrixMask2 = Reg::UpdateMask<float>(activeLanes2);
            Reg::And(matrixMask2, matrixMask2, activeMask, fullMask);
            LoadMatrixRegister<N>(gradMatrix2Reg, tileAddr + (base + 2U) * N * N, compactIndex, matrixMask2);
            uint32_t activeLanes3 = base + 3U < tileMatrixCount ? 64U : 0U;
            Reg::MaskReg matrixMask3 = Reg::UpdateMask<float>(activeLanes3);
            Reg::And(matrixMask3, matrixMask3, activeMask, fullMask);
            LoadMatrixRegister<N>(gradMatrix3Reg, tileAddr + (base + 3U) * N * N, compactIndex, matrixMask3);
            for (uint16_t step = 0; step < iterations; ++step) {
                const uint32_t iterationOffset = (iterations - 1U - step) * iterationStride;
                LoadMatrixRegister<N>(
                    rowNormMatrix0Reg,
                    iterationData + iterationOffset + 2U * sumElementsPerIteration + (base + 0U) * N * N, compactIndex,
                    matrixMask0);
                LoadMatrixRegister<N>(
                    rowNormMatrix1Reg,
                    iterationData + iterationOffset + 2U * sumElementsPerIteration + (base + 1U) * N * N, compactIndex,
                    matrixMask1);
                LoadMatrixRegister<N>(
                    rowNormMatrix2Reg,
                    iterationData + iterationOffset + 2U * sumElementsPerIteration + (base + 2U) * N * N, compactIndex,
                    matrixMask2);
                LoadMatrixRegister<N>(
                    rowNormMatrix3Reg,
                    iterationData + iterationOffset + 2U * sumElementsPerIteration + (base + 3U) * N * N, compactIndex,
                    matrixMask3);
                // 列反向。
                Reg::LoadAlign(sumReg, iterationData + iterationOffset + sumElementsPerIteration);
                Reg::Adds(colSumIndex, columnIndex, static_cast<uint32_t>(base + 0U) * N, fullMask);
                CalcMatrixColumnBackward(gradMatrix0Reg, rowNormMatrix0Reg, sumReg, colSumIndex, columnIndex,
                                         transposeIndex, matrixMask0);
                Reg::Adds(colSumIndex, columnIndex, static_cast<uint32_t>(base + 1U) * N, fullMask);
                CalcMatrixColumnBackward(gradMatrix1Reg, rowNormMatrix1Reg, sumReg, colSumIndex, columnIndex,
                                         transposeIndex, matrixMask1);
                Reg::Adds(colSumIndex, columnIndex, static_cast<uint32_t>(base + 2U) * N, fullMask);
                CalcMatrixColumnBackward(gradMatrix2Reg, rowNormMatrix2Reg, sumReg, colSumIndex, columnIndex,
                                         transposeIndex, matrixMask2);
                Reg::Adds(colSumIndex, columnIndex, static_cast<uint32_t>(base + 3U) * N, fullMask);
                CalcMatrixColumnBackward(gradMatrix3Reg, rowNormMatrix3Reg, sumReg, colSumIndex, columnIndex,
                                         transposeIndex, matrixMask3);
                // 行反向。
                Reg::LoadAlign(sumReg, iterationData + iterationOffset);
                CalcPackedRowBackward<8U>(gradMatrix0Reg, rowNormMatrix0Reg, sumReg, rowIndex, rowIndex,
                                          static_cast<uint32_t>(base + 0U) * N, iterationOffset == 0U, eps, fullMask,
                                          matrixMask0);
                CalcPackedRowBackward<8U>(gradMatrix1Reg, rowNormMatrix1Reg, sumReg, rowIndex, rowIndex,
                                          static_cast<uint32_t>(base + 1U) * N, iterationOffset == 0U, eps, fullMask,
                                          matrixMask1);
                CalcPackedRowBackward<8U>(gradMatrix2Reg, rowNormMatrix2Reg, sumReg, rowIndex, rowIndex,
                                          static_cast<uint32_t>(base + 2U) * N, iterationOffset == 0U, eps, fullMask,
                                          matrixMask2);
                CalcPackedRowBackward<8U>(gradMatrix3Reg, rowNormMatrix3Reg, sumReg, rowIndex, rowIndex,
                                          static_cast<uint32_t>(base + 3U) * N, iterationOffset == 0U, eps, fullMask,
                                          matrixMask3);
            }
            StoreMatrixRegister<N>(gradOutputAddr + (base + 0U) * N * N, gradMatrix0Reg, compactIndex, matrixMask0);
            StoreMatrixRegister<N>(gradOutputAddr + (base + 1U) * N * N, gradMatrix1Reg, compactIndex, matrixMask1);
            StoreMatrixRegister<N>(gradOutputAddr + (base + 2U) * N * N, gradMatrix2Reg, compactIndex, matrixMask2);
            StoreMatrixRegister<N>(gradOutputAddr + (base + 3U) * N * N, gradMatrix3Reg, compactIndex, matrixMask3);
        }
        Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
    }
}

} // namespace MhcPreSinkhornBackwardSinkhornNs

#endif // MHC_PRE_SINKHORN_BACKWARD_SINKHORN_SIMD_ARCH35_H
