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
 * \file rain_fusion_attention_interface.cpp
 * \brief Rain Fusion Attention Interface
 */
#include "kernel_operator.h"
#include "rain_fusion_attention_kernel.h"

using namespace NpuArch;

namespace RainFusion {
// Rain Fusion Attention Infer Interface
// This template provides the entry point for rain fusion attention kernels
// where attention is computed only on selected KV blocks based on selectIdx
template <typename InputDtype = half, typename SoftmaxDtype = float,
          Epilogue::LseMode lseMode = Epilogue::LseMode::NONE,
          uint32_t QueryLayout = 0,   // 0=TND, 1=BNSD
          uint32_t KvCacheLayout = 0> // 0=TND, 1=BNSD
__global__ __aicore__ void RainFusionAttentionInfer(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR mask, GM_ADDR blockTables, GM_ADDR o, GM_ADDR actualQseqlen,
    GM_ADDR actualKvseqlen, GM_ADDR blockShape,
    GM_ADDR selectIdx,    // Sparse block selection indices [T, headNum, maxKvBlockNum]
    GM_ADDR selectNumIdx, // Number of selected blocks per Q block [T, headNum]
    GM_ADDR workspace,    // S matrix (QK^T result)
    GM_ADDR lse, GM_ADDR tiling)
{
    using ArchTag = Arch::AtlasA2;
    using ElementQ = InputDtype;
    using LayoutQ = layout::RowMajor;
    using ElementK = InputDtype;
    using LayoutK = layout::ColumnMajor;
    using ElementV = InputDtype;
    using LayoutV = layout::RowMajor;
    using ElementS = SoftmaxDtype;
    using LayoutS = layout::RowMajor;
    using ElementP = InputDtype;
    using LayoutP = layout::RowMajor;
    using ElementO = InputDtype;
    using LayoutO = layout::RowMajor;
    using ElementLse = float;
    using LayoutLse = layout::RowMajor;
    using ElementMask = int8_t;
    using LayoutMask = layout::RowMajor;
    using ElementOTmp = SoftmaxDtype;
    using LayoutOTmp = layout::RowMajor;
    using ElementUpdate = SoftmaxDtype;
    using LayoutUpdate = layout::RowMajor;

    // Use sparse-specific dispatch policies
    using L1TileShapeQK = GemmShape<Q_TILE_CEIL, 128, 128>;
    using L0TileShapeQK = GemmShape<128, 128, 128>;
    using DispatchPolicyQK = Gemm::MmadAtlasA2SFAIQK<false, false>;
    using QType = Gemm::GemmType<ElementQ, LayoutQ>;
    using KType = Gemm::GemmType<ElementK, LayoutK>;
    using SType = Gemm::GemmType<ElementS, LayoutS>;
    using BlockMmadQK = Gemm::Block::BlockMmad<DispatchPolicyQK, L1TileShapeQK, L0TileShapeQK, QType, KType, SType>;

    using L1TileShapePV = GemmShape<128, 128, 256>;
    using L0TileShapePV = GemmShape<128, 128, 128>;
    using DispatchPolicyPV = Gemm::MmadAtlasA2SFAIPV<false, false>;
    using PType = Gemm::GemmType<ElementP, LayoutP>;
    using VType = Gemm::GemmType<ElementV, LayoutV>;
    using OTmpType = Gemm::GemmType<ElementOTmp, LayoutOTmp>;
    using BlockMmadPV = Gemm::Block::BlockMmad<DispatchPolicyPV, L1TileShapePV, L0TileShapePV, PType, VType, OTmpType>;

    // Epilogue policies for sparse attention
    using DispatchPolicyOnlineSoftmax = Epilogue::EpilogueAtlasA2OnlineSoftmax<lseMode, SoftmaxDtype>;
    using MaskType = Gemm::GemmType<ElementMask, LayoutMask>;
    using EpilogueOnlineSoftmax = Epilogue::Block::BlockEpilogue<DispatchPolicyOnlineSoftmax, PType, SType, MaskType>;
    using DispatchPolicyRescaleO = Epilogue::EpilogueAtlasA2RescaleO<lseMode, SoftmaxDtype>;
    using OType = Gemm::GemmType<ElementO, LayoutO>;
    using OUpdateType = Gemm::GemmType<ElementUpdate, LayoutUpdate>;
    using LseType = Gemm::GemmType<ElementLse, LayoutLse>;
    using EpilogueRescaleO =
        Epilogue::Block::BlockEpilogue<DispatchPolicyRescaleO, OType, OTmpType, OUpdateType, LseType>;

    // Kernel instantiation
    using RainFusionAttentionKernelType =
        RainFusionAttentionKernel<BlockMmadQK, BlockMmadPV, EpilogueOnlineSoftmax, EpilogueRescaleO,
                                  false,          // PAGED_CACHE_FLAG
                                  QueryLayout,    // QUERY_LAYOUT
                                  KvCacheLayout>; // KV_CACHE_LAYOUT
    RainFusionAttentionKernelParams params{
        q, k, v, mask, blockTables, actualQseqlen, actualKvseqlen, selectIdx, selectNumIdx, o, lse, workspace, tiling};

    // Call rain fusion attention kernel
    RainFusionAttentionKernelType rainFusionAttenInfer;
    rainFusionAttenInfer(params);
}
} // namespace RainFusion
