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
 * \file mhc_pre_sinkhorn_backward_sinkhorn_arch35.h
 * \brief V0 阶段的 Sinkhorn 反向调度与状态加载
 */

#ifndef MHC_PRE_SINKHORN_BACKWARD_SINKHORN_ARCH35_H
#define MHC_PRE_SINKHORN_BACKWARD_SINKHORN_ARCH35_H

#include "mhc_pre_sinkhorn_backward_sinkhorn_simd_arch35.h"

namespace MhcPreSinkhornBackwardSinkhornNs {

// 每个 tile 使用一个队列槽保存梯度及各轮 [行和 | 列和 | 行 norm]，列/行反向共用行 norm。
// 完成全部迭代后，将梯度交给后续 V0 计算。
class SinkhornBackward {
public:
    using InputQueue = TQue<QuePosition::VECIN, 0>;

    __aicore__ inline void Init(GM_ADDR gradient, GM_ADDR sums, GM_ADDR norms, const SinkhornTilePlan& plan,
                                uint64_t totalMatrixCount, uint32_t iterations, float eps);

    // 按 Host 计划分派布局特化。
    __aicore__ inline void Process(uint64_t matrixStart, uint32_t matrixCount, LocalTensor<float>& gradOutput,
                                   InputQueue& queue);

private:
    // 各布局共用分块、同步和 DMA 流程。
    template <uint32_t SpecializedN = 0U, bool N4RowPacked = false>
    __aicore__ inline void ProcessTiles(uint64_t matrixStart, uint32_t matrixCount, LocalTensor<float>& gradOutput,
                                        InputQueue& queue);

    template <uint32_t SpecializedN = 0U, bool N4RowPacked = false>
    __aicore__ inline void CopyInTile(LocalTensor<float>& stateBuf, uint64_t matrixStart, uint32_t tileMatrixCount);

    SinkhornTilePlan plan_;
    GlobalTensor<float> gradYGm_;
    GlobalTensor<float> sumGm_;
    GlobalTensor<float> normGm_;
    uint64_t totalMatrixCount_;
    uint16_t iterations_;
    float eps_;
};

__aicore__ inline void SinkhornBackward::Init(GM_ADDR gradient, GM_ADDR sums, GM_ADDR norms,
                                              const SinkhornTilePlan& plan, uint64_t totalMatrixCount,
                                              uint32_t iterations, float eps)
{
    gradYGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(gradient));
    sumGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(sums));
    normGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(norms));
    plan_ = plan;
    totalMatrixCount_ = totalMatrixCount;
    iterations_ = static_cast<uint16_t>(iterations);
    eps_ = eps;
}

// 各特化共用分块和 DMA 流程，主要区别是每块矩阵数及寄存器布局：
// <>：通用 N 的紧凑布局，每个通道对应一个矩阵元素；<4U>：同布局，固定 N=4，每块 4 个矩阵。
// <4U, true>：每块 16 个矩阵，4 个行寄存器并行计算。
// <6U>：每块 8 个矩阵，6 个行寄存器，每行补齐到 8 列。
// <8U>：每块 4 个矩阵，4 个完整矩阵寄存器交错计算。
// N=6/8 的尾块不足 4 个矩阵时，改用逐矩阵 VF。
__aicore__ inline void SinkhornBackward::Process(uint64_t matrixStart, uint32_t matrixCount,
                                                 LocalTensor<float>& gradOutput, InputQueue& queue)
{
    if (plan_.path == SinkhornPath::N4_COMPACT) {
        ProcessTiles<4U>(matrixStart, matrixCount, gradOutput, queue);
    } else if (plan_.path == SinkhornPath::N4_ROW_PACKED) {
        ProcessTiles<4U, true>(matrixStart, matrixCount, gradOutput, queue);
    } else if (plan_.path == SinkhornPath::N6_ROW_PACKED) {
        ProcessTiles<6U>(matrixStart, matrixCount, gradOutput, queue);
    } else if (plan_.path == SinkhornPath::N8_INTERLEAVED) {
        ProcessTiles<8U>(matrixStart, matrixCount, gradOutput, queue);
    } else {
        ProcessTiles<>(matrixStart, matrixCount, gradOutput, queue);
    }
    PipeBarrier<PIPE_V>();
}

template <uint32_t SpecializedN, bool N4RowPacked>
__aicore__ inline void SinkhornBackward::ProcessTiles(uint64_t matrixStart, uint32_t matrixCount,
                                                      LocalTensor<float>& gradOutput, InputQueue& queue)
{
    static_assert(!N4RowPacked || SpecializedN == 4U, "The 16-matrix row layout requires N=4");
    const uint32_t matricesPerTile = N4RowPacked        ? 16U :
                                     SpecializedN != 0U ? MatricesPerTile(SpecializedN) :
                                                          plan_.matricesPerTile;
    const uint32_t matrixElements = SpecializedN != 0U ? SpecializedN * SpecializedN : plan_.matrixElements;
    for (uint32_t tileMatrixOffset = 0; tileMatrixOffset < matrixCount; tileMatrixOffset += matricesPerTile) {
        const uint32_t remain = matrixCount - tileMatrixOffset;
        const uint32_t tileMatrixCount = remain < matricesPerTile ? remain : matricesPerTile;
        LocalTensor<float> stateBuf;
        queue.AllocTensor<float>(stateBuf);
        CopyInTile<SpecializedN, N4RowPacked>(stateBuf, matrixStart + tileMatrixOffset, tileMatrixCount);
        queue.EnQue(stateBuf);
        queue.DeQue<float>(stateBuf);
        if constexpr (N4RowPacked) {
            VFDoBackwardRowPackedTile<4U>((__ubuf__ float*)gradOutput.GetPhyAddr() + tileMatrixOffset * matrixElements,
                                          (__ubuf__ float*)stateBuf.GetPhyAddr(), tileMatrixCount, iterations_, eps_);
        } else if constexpr (SpecializedN == 6U || SpecializedN == 8U) {
            __ubuf__ float* output = (__ubuf__ float*)gradOutput.GetPhyAddr() + tileMatrixOffset * matrixElements;
            __ubuf__ float* input = (__ubuf__ float*)stateBuf.GetPhyAddr();
            // 小尾块使用逐矩阵 VF。
            if (tileMatrixCount < 4U) {
                VFDoBackwardMatrixTile<SpecializedN>(output, input, tileMatrixCount, iterations_, eps_);
            } else if constexpr (SpecializedN == 6U) {
                VFDoBackwardRowPackedTile<6U>(output, input, tileMatrixCount, iterations_, eps_);
            } else {
                VFDoBackwardInterleavedTile(output, input, tileMatrixCount, iterations_, eps_);
            }
        } else {
            VFDoBackwardCompactTile<SpecializedN>(
                plan_, (__ubuf__ float*)gradOutput.GetPhyAddr() + tileMatrixOffset * matrixElements,
                (__ubuf__ float*)stateBuf.GetPhyAddr(), tileMatrixCount, iterations_, eps_);
        }
        // Vector 读取完成后才允许 MTE2 复用队列槽。
        queue.FreeTensor(stateBuf);
    }
}

template <uint32_t SpecializedN, bool N4RowPacked>
__aicore__ inline void SinkhornBackward::CopyInTile(LocalTensor<float>& stateBuf, uint64_t matrixStart,
                                                    uint32_t tileMatrixCount)
{
    const uint32_t nSize = SpecializedN != 0U ? SpecializedN : plan_.nSize;
    const uint32_t matrixElements = SpecializedN != 0U ? SpecializedN * SpecializedN : plan_.matrixElements;
    static_assert(!N4RowPacked || SpecializedN == 4U, "The 16-matrix row layout requires N=4");
    const uint32_t matricesPerTile = N4RowPacked        ? 16U :
                                     SpecializedN != 0U ? MatricesPerTile(SpecializedN) :
                                                          plan_.matricesPerTile;
    const uint32_t sumElementsPerIteration =
        SpecializedN != 0U ? AlignFloatBlock(matricesPerTile * nSize) : plan_.sumElementsPerIteration;
    const uint32_t normElementsPerIteration =
        SpecializedN != 0U ? AlignFloatBlock(matricesPerTile * matrixElements) : 0U;
    const uint32_t iterationDataOffset =
        SpecializedN != 0U ? normElementsPerIteration + VECTOR_ELEMENTS : plan_.iterationDataOffset;
    const uint32_t iterationStride =
        SpecializedN != 0U ? 2U * sumElementsPerIteration + normElementsPerIteration : plan_.iterationStride;
    if constexpr (N4RowPacked) {
        static_assert(ScratchElements(4U, MAX_ITERATIONS, 16U) * sizeof(float) <= MaxScratchBytes(),
                      "N=4 tile exceeds the existing Sinkhorn input buffer");
    }
    DataCopyPadExtParams<float> pad = {false, 0, 0, 0};
    DataCopyExtParams copy = {1, static_cast<uint32_t>(tileMatrixCount * matrixElements * sizeof(float)), 0, 0, 0};
    DataCopyPad(stateBuf, gradYGm_[matrixStart * matrixElements], copy, pad);

    copy.blockCount = iterations_;
    copy.blockLen = tileMatrixCount * nSize * sizeof(float);
    copy.srcStride = (2U * totalMatrixCount_ - tileMatrixCount) * nSize * sizeof(float);
    copy.dstStride = (iterationStride - AlignFloatBlock(tileMatrixCount * nSize)) / FLOAT_BLOCK_ELEMENTS;
    DataCopyPad(stateBuf[iterationDataOffset], sumGm_[matrixStart * nSize], copy, pad);
    DataCopyPad(stateBuf[iterationDataOffset + sumElementsPerIteration],
                sumGm_[(totalMatrixCount_ + matrixStart) * nSize], copy, pad);

    // 一次跨步 DMA 加载各轮行 norm，列、行反向复用该切片。
    copy.blockLen = tileMatrixCount * matrixElements * sizeof(float);
    copy.srcStride = (2U * totalMatrixCount_ - tileMatrixCount) * matrixElements * sizeof(float);
    copy.dstStride = (iterationStride - AlignFloatBlock(tileMatrixCount * matrixElements)) / FLOAT_BLOCK_ELEMENTS;
    DataCopyPad(stateBuf[iterationDataOffset + 2U * sumElementsPerIteration], normGm_[matrixStart * matrixElements],
                copy, pad);
}
} // namespace MhcPreSinkhornBackwardSinkhornNs

#endif // MHC_PRE_SINKHORN_BACKWARD_SINKHORN_ARCH35_H
