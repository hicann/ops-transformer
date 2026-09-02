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
 * \file block_sparse_attention_grad.h
 * \brief Block Sparse Attention Grad Interface
 */

#ifndef BLOCK_SPARSE_ATTENTION_GRAD_INTERFACE_H
#define BLOCK_SPARSE_ATTENTION_GRAD_INTERFACE_H

#include "kernel_operator.h"
#include "block_sparse_attention_grad_kernel.h"

using namespace NpuArch;

namespace BSA {
template <typename InputDtype = half,
          uint32_t InputLayout = 0> // 0=TND, 1=BNSD
__aicore__ inline void BlockSparseAttentionGradInfer(GM_ADDR dout, GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR out,
                                                     GM_ADDR softmaxLse, GM_ADDR blockSparseMask, GM_ADDR blockShape,
                                                     GM_ADDR attentionMask, GM_ADDR actualQseqlen,
                                                     GM_ADDR actualKvseqlen, GM_ADDR dq, GM_ADDR dk, GM_ADDR dv,
                                                     GM_ADDR workspace, GM_ADDR tiling)
{
    using ArchTag = Arch::AtlasA2;

    constexpr uint32_t Q_TILE_CEIL = 128;

    // Cube1 ：(S = Q * K^T) 和 dP = dOut * V^T
    using ElementA1 = InputDtype;
    using LayoutA1 = layout::RowMajor;
    using ElementB1 = InputDtype;
    using LayoutB1 = layout::ColumnMajor;
    using ElementC1 = float;
    using LayoutC1 = layout::RowMajor;
    using A1Type = Gemm::GemmType<ElementA1, LayoutA1>;
    using B1Type = Gemm::GemmType<ElementB1, LayoutB1>;
    using C1Type = Gemm::GemmType<ElementC1, LayoutC1>;
    using DispatchPolicyCube1 = Gemm::MmadAtlasA2SBSAG2;
    using L1TileShape1 = GemmShape<128, 128, 128>;
    using L0TileShape1 = GemmShape<128, 128, 128>;
    using BlockMmadCube1 =
        Gemm::Block::BlockMmad<DispatchPolicyCube1, L1TileShape1, L0TileShape1, A1Type, B1Type, C1Type>;

    // Cube2 ：dQ = dS * K
    using ElementA2 = InputDtype;
    using LayoutA2 = layout::RowMajor;
    using ElementB2 = InputDtype;
    using LayoutB2 = layout::RowMajor;
    using ElementC2 = float;
    using LayoutC2 = layout::RowMajor;
    using A2Type = Gemm::GemmType<ElementA2, LayoutA2>;
    using B2Type = Gemm::GemmType<ElementB2, LayoutB2>;
    using C2Type = Gemm::GemmType<ElementC2, LayoutC2>;
    using DispatchPolicyCube2 = Gemm::MmadAtlasA2SBSAG2;
    using L1TileShape2 = GemmShape<128, 128, 128>;
    using L0TileShape2 = GemmShape<128, 128, 128>;
    using BlockMmadCube2 =
        Gemm::Block::BlockMmad<DispatchPolicyCube2, L1TileShape2, L0TileShape2, A2Type, B2Type, C2Type>;

    // Cube3 ：dK = dS^T * Q 和 dV = P^T * dOut
    using ElementA3 = InputDtype;
    using LayoutA3 = layout::ColumnMajor;
    using ElementB3 = InputDtype;
    using LayoutB3 = layout::RowMajor;
    using ElementC3 = float;
    using LayoutC3 = layout::RowMajor;
    using A3Type = Gemm::GemmType<ElementA3, LayoutA3>;
    using B3Type = Gemm::GemmType<ElementB3, LayoutB3>;
    using C3Type = Gemm::GemmType<ElementC3, LayoutC3>;
    using DispatchPolicyCube3 = Gemm::MmadAtlasA2SBSAG2;
    using L1TileShape3 = GemmShape<128, 128, 128>;
    using L0TileShape3 = GemmShape<128, 128, 128>;
    using BlockMmadCube3 =
        Gemm::Block::BlockMmad<DispatchPolicyCube3, L1TileShape3, L1TileShape3, A3Type, B3Type, C3Type>;

    // Epilogue
    using ElementOutput = InputDtype;
    using LayoutOutput = layout::RowMajor;
    using OutputType = Gemm::GemmType<ElementOutput, LayoutOutput>;

    using ElementUpdate = float;
    using LayoutUpdate = layout::RowMajor;
    using UpdateType = Gemm::GemmType<ElementUpdate, LayoutUpdate>;

    using ElementInput = float;
    using LayoutInput = layout::RowMajor;
    using InputType = Gemm::GemmType<ElementInput, LayoutInput>;

    // VEC_Pre ：dQ/dK/dV的workspace清零
    using EpilogueAtlasA2FAGPre = Epilogue::EpilogueAtlasA2FAGPre;
    using EpilogueFAGPre = Epilogue::Block::BlockEpilogue<EpilogueAtlasA2FAGPre, OutputType, UpdateType, InputType>;

    // VEC_Sfmg ：dP = SoftmaxGrad(dOut, out)
    using EpilogueAtlasA2FAGSfmg = Epilogue::EpilogueAtlasA2FAGPre;
    using EpilogueFAGSfmg = Epilogue::Block::SoftmaxGrad<InputDtype, float, InputLayout>;

    // VEC_Op：P = simple_softmax(S)，再计算dS = P * Sub(dP, Sfmg)  【cube1 输出S = Q*K^T 及 dP = dOut * V^T】
    using EpilogueAtlasA2FAGOp = Epilogue::EpilogueAtlasA2FAGPre;
    using EpilogueFAGOp = Epilogue::Block::SimpltSoftmax<InputDtype, float, InputLayout>;

    // VEC_Post：dQ*scale和dK*scale，并搬运输出dQ/dK/dV
    using EpilogueAtlasA2FAGPost = Epilogue::EpilogueAtlasA2FAGPre;
    using EpilogueFAGPost = Epilogue::Block::BlockPost<InputLayout, InputDtype, UpdateType, InputDtype>;

    // Kernel instantiation
    using BSAGKernel = BlockSparseAttentionGradKernel<BlockMmadCube1, BlockMmadCube2, BlockMmadCube3, EpilogueFAGPre,
                                                      EpilogueFAGSfmg, EpilogueFAGOp, EpilogueFAGPost, InputLayout>;
    typename BSAGKernel::Params params{
        dout,           q,  k,  v,  out,       softmaxLse, blockSparseMask, blockShape, attentionMask, actualQseqlen,
        actualKvseqlen, dq, dk, dv, workspace, tiling};
    BSAGKernel kernel;
    kernel(params);
}
} // namespace BSA

#endif // BLOCK_SPARSE_ATTENTION_GRAD_INTERFACE_H
