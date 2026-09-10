/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file generic_block_sparse_attention_kernel_interface.cpp
 * \brief W8A8 Generic Sparse Attention Kernel Interface (arch22 only)
 */
#include "kernel_operator.h"
#if (__CCE_AICORE__ == 220)
#include "arch22/generic_block_sparse_attention_kernel_arch22.h"
#endif

using namespace NpuArch;

#if (__CCE_AICORE__ == 220)

using namespace GsaKernelArch22;

// 伪量化接口：KV 为 int8，在线伪量化（MSD 分段），输出类型与 Q 一致。
// LSE / FD / halfSM 在 INT8 tiling key 下已禁用，这里固定 LseMode::NONE。
// softmaxLse 形参保留以对齐 GsaKernelParamsArch22 布局，入口固定传 nullptr。
template <class InDtype, class SMDtype>
__global__ __aicore__ void GsaInferIntfAntiquantArch22(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR sparseBlockIdx,
                                                       GM_ADDR sparseBlockCount, GM_ADDR metaData, GM_ADDR cuSeqLengths,
                                                       GM_ADDR cuSeqLengthsKv, GM_ADDR sequsedQ, GM_ADDR sequsedKv,
                                                       GM_ADDR blockTable, GM_ADDR qDequantScale, GM_ADDR kDequantScale,
                                                       GM_ADDR vDequantScale, GM_ADDR o, GM_ADDR softmaxLse,
                                                       GM_ADDR workspace, GM_ADDR tiling)
{
    using ArchTag = Arch::AtlasA2;
    using ElementQ = int8_t;     // 预处理后 Query 类型 (int8)，原始 Q 类型由 ElementO 表示
    using ElementK = int8_t;     // KV 存储类型 (int8)
    using ElementV = int8_t;     // KV 存储类型 (int8)
    using ElementS = int32_t;    // BMM1 输出: int8×int8→int32
    using ElementP = int8_t;     // Softmax 输出: int8 (供 BMM2 使用)
    using ElementO = InDtype;    // 输出类型与原始 Q 一致 (half/bfloat16_t)
    using ElementOTmp = int32_t; // BMM2 输出: int8×int8→int32

    using LayoutQ = layout::RowMajor;
    using LayoutK = layout::ColumnMajor;
    using LayoutS = layout::RowMajor;
    using LayoutP = layout::RowMajor;
    using LayoutV = layout::RowMajor;
    using LayoutO = layout::RowMajor;
    using LayoutOTmp = layout::RowMajor;

    // QK matmul: int8 × int8 → int32.  The L1 K tile is widened to 256 so
    // D=256 is handled by the existing two-pass L0 K loop; D=128 simply
    // consumes the first half and keeps the same tiling key.
    using L1TileShapeQK = GemmShape<128, 128, 256>;
    using L0TileShapeQK = GemmShape<128, 128, 128>;
    // The pseudo-quantized KV cache is PA_NZ.  The policy bit selects the
    // direct [D/C0, blockSize, C0] loader in BlockMmadQK.
    using DispatchPolicyQK = Gemm::MmadAtlasA2SFAIQK<true, false>;
    using QType = Gemm::GemmType<ElementQ, LayoutQ>;
    using KType = Gemm::GemmType<ElementK, LayoutK>;
    using SType = Gemm::GemmType<ElementS, LayoutS>;
    using BlockMmadQK = Gemm::Block::BlockMmad<DispatchPolicyQK, L1TileShapeQK, L0TileShapeQK, QType, KType, SType>;

    // Online softmax: int32 输入 → int8 输出
    using PType = Gemm::GemmType<ElementP, LayoutP>;
    using DispatchPolicyOnlineSoftmax = Epilogue::EpilogueAtlasA2OnlineSoftmax<Epilogue::LseMode::NONE, SMDtype>;
    using MaskType = Gemm::GemmType<int8_t, layout::RowMajor>;
    using EpilogueOnlineSoftmax = Epilogue::Block::BlockEpilogue<DispatchPolicyOnlineSoftmax, PType, SType, MaskType>;

    // PV matmul: int8 × int8 → int32
    using L1TileShapePV = GemmShape<128, 128, 256>;
    // D=256 produces a 128x256 int32 tile.  A 64-row L0 M tile keeps each
    // ping-pong C buffer within the 64 KiB A2 capacity.
    using L0TileShapePV = GemmShape<64, 128, 128>;
    // Match QK: load the PA_NZ V cache directly without an ND staging copy.
    using DispatchPolicyPV = Gemm::MmadAtlasA2SFAIPV<true, false>;
    using VType = Gemm::GemmType<ElementV, LayoutV>;
    using OTmpType = Gemm::GemmType<ElementOTmp, LayoutOTmp>;
    using BlockMmadPV = Gemm::Block::BlockMmad<DispatchPolicyPV, L1TileShapePV, L0TileShapePV, PType, VType, OTmpType>;

    // Rescale O: int32 → float → half/bf16
    using DispatchPolicyRescaleO = Epilogue::EpilogueAtlasA2RescaleO<Epilogue::LseMode::NONE, SMDtype>;
    using OType = Gemm::GemmType<ElementO, LayoutO>;
    using OTmpUpdateType = Gemm::GemmType<ElementOTmp, LayoutOTmp>;
    using LseType = Gemm::GemmType<float, layout::RowMajor>;
    using EpilogueRescaleO =
        Epilogue::Block::BlockEpilogue<DispatchPolicyRescaleO, OType, OTmpType, OTmpUpdateType, LseType>;

    using GsaKernel = GsaRegularKernelArch22<BlockMmadQK, EpilogueOnlineSoftmax, BlockMmadPV, EpilogueRescaleO>;

    GsaKernelParamsArch22 params{q,
                                 k,
                                 v,
                                 sparseBlockIdx,
                                 sparseBlockCount,
                                 metaData,
                                 cuSeqLengths,
                                 cuSeqLengthsKv,
                                 sequsedQ,
                                 sequsedKv,
                                 blockTable,
                                 qDequantScale,
                                 kDequantScale,
                                 vDequantScale,
                                 o,
                                 softmaxLse,
                                 workspace,
                                 tiling};
    GsaKernel gsaKernel;
    gsaKernel(params);
}

#endif
