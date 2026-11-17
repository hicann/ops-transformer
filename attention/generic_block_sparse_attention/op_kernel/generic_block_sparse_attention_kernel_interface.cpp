/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file generic_block_sparse_attention_kernel_interface.cpp
 * \brief Generic Sparse Attention Kernel Interface
 */
#include "kernel_operator.h"
#if (__CCE_AICORE__ == 220)
#include "arch22/generic_block_sparse_attention_kernel_arch22.h"
#include "arch22/generic_block_sparse_attention_kernel_arch22_union.h"
#endif
#if (__CCE_AICORE__ == 310)
#include "arch35/generic_block_sparse_attention_kernel_arch35.h"
#include "arch35/generic_block_sparse_attention_kernel_arch35_union.h"
#if !defined(TILING_KEY_VAR) || (TILING_KEY_VAR >= 9250002900000020ULL && TILING_KEY_VAR <= 9250002900000023ULL) || \
    (TILING_KEY_VAR >= 9250002900000032ULL && TILING_KEY_VAR <= 9250002900000035ULL) || \
    (TILING_KEY_VAR >= 9250002900000040ULL && TILING_KEY_VAR <= 9250002900000047ULL) || \
    (TILING_KEY_VAR >= 9250002900000050ULL && TILING_KEY_VAR <= 9250002900000055ULL)
#include "arch35/generic_block_sparse_attention_kernel_arch35_mxfp8_full_quant.h"
#endif
#if defined(ORIG_DTYPE_QUERY) && defined(ORIG_DTYPE_KEY) && (ORIG_DTYPE_QUERY == DT_FLOAT8_E4M3FN) && \
    (ORIG_DTYPE_KEY == DT_FLOAT4_E2M1)
#include "arch35/generic_block_sparse_attention_kernel_arch35_a8c4.h"
#endif
#if defined(ORIG_DTYPE_QUERY) && defined(ORIG_DTYPE_KEY) && (ORIG_DTYPE_QUERY == DT_FLOAT4_E2M1) && \
    (ORIG_DTYPE_KEY == DT_FLOAT4_E2M1)
#include "arch35/generic_block_sparse_attention_kernel_arch35_mxfp4_full_quant.h"
#endif // MXFP4-only headers
#endif

using namespace NpuArch;

#if (__CCE_AICORE__ == 220)

using namespace GbsaKernelArch22;

template <class InDtype, class SMDtype, class REDtype, Epilogue::LseMode lseMode, bool isPaged, bool hasResidualBlock,
          bool useKvTile512>
__aicore__ inline void GbsaInferIntfRegularArch22Impl(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR sparseBlockIdx,
                                                      GM_ADDR sparseBlockCount, GM_ADDR metaData, GM_ADDR cuSeqLengths,
                                                      GM_ADDR cuSeqLengthsKv, GM_ADDR sequsedQ, GM_ADDR sequsedKv,
                                                      GM_ADDR blockTable, GM_ADDR o, GM_ADDR softmaxLse,
                                                      GM_ADDR workspace, GM_ADDR tiling)
{
    using ArchTag = Arch::AtlasA2;
    using ElementQ = InDtype;
    using ElementK = InDtype;
    using ElementV = InDtype;
    using ElementS = SMDtype;
    using ElementP = InDtype;
    using ElementO = InDtype;
    using ElementOTmp = REDtype;

    using LayoutQ = layout::RowMajor;
    using LayoutK = layout::ColumnMajor;
    using LayoutS = layout::RowMajor;
    using LayoutP = layout::RowMajor;
    using LayoutV = layout::RowMajor;
    using LayoutO = layout::RowMajor;
    using LayoutOTmp = layout::RowMajor;

    // QK matmul
    // Like A5 for D=256, ping-pong K in 128-token chunks so a full P and V tile fit in L1.
    using L1TileShapeQK = std::conditional_t<useKvTile512, GemmShape<16, 128, 128>, GemmShape<128, 128, 128>>;
    using L0TileShapeQK = L1TileShapeQK;
    using DispatchPolicyQK = Gemm::MmadAtlasA2SFAIQK<isPaged, false, hasResidualBlock>;
    using QType = Gemm::GemmType<ElementQ, LayoutQ>;
    using KType = Gemm::GemmType<ElementK, LayoutK>;
    using SType = Gemm::GemmType<ElementS, LayoutS>;
    using BlockMmadQK = Gemm::Block::BlockMmad<DispatchPolicyQK, L1TileShapeQK, L0TileShapeQK, QType, KType, SType>;

    // Online softmax
    using PType = Gemm::GemmType<ElementP, LayoutP>;
    using DispatchPolicyOnlineSoftmax = Epilogue::EpilogueAtlasA2OnlineSoftmax<lseMode, SMDtype>;
    using MaskType = Gemm::GemmType<int8_t, layout::RowMajor>;
    using EpilogueOnlineSoftmax = Epilogue::Block::BlockEpilogue<DispatchPolicyOnlineSoftmax, PType, SType, MaskType>;

    // PV matmul
    using L1TileShapePV =
        std::conditional_t<useKvTile512, GemmShape<16, 128, GbsaSparsePairCopy::KV_COPY_MAX_TILE_TOKENS>,
                           GemmShape<128, 128, 256>>;
    using L0TileShapePV = std::conditional_t<useKvTile512, GemmShape<16, 128, 128>, GemmShape<128, 128, 128>>;
    using DispatchPolicyPV = Gemm::MmadAtlasA2SFAIPV<isPaged, false, hasResidualBlock>;
    using VType = Gemm::GemmType<ElementV, LayoutV>;
    using OTmpType = Gemm::GemmType<ElementOTmp, LayoutOTmp>;
    using BlockMmadPV = Gemm::Block::BlockMmad<DispatchPolicyPV, L1TileShapePV, L0TileShapePV, PType, VType, OTmpType>;

    // Rescale O (REDtype=float → high-prec rescale; Softmax online gm/dm/gl are float)
    using DispatchPolicyRescaleO = Epilogue::EpilogueAtlasA2RescaleO<lseMode, REDtype>;
    using OType = Gemm::GemmType<ElementO, LayoutO>;
    using OTmpUpdateType = Gemm::GemmType<ElementOTmp, LayoutOTmp>;
    using LseType = Gemm::GemmType<float, layout::RowMajor>;
    using EpilogueRescaleO =
        Epilogue::Block::BlockEpilogue<DispatchPolicyRescaleO, OType, OTmpType, OTmpUpdateType, LseType>;

    using GbsaKernel = GbsaRegularKernelArch22<BlockMmadQK, EpilogueOnlineSoftmax, BlockMmadPV, EpilogueRescaleO,
                                               isPaged, hasResidualBlock>;

    GbsaKernelParamsArch22 params{
        q,         k,          v, sparseBlockIdx, sparseBlockCount, metaData, cuSeqLengths, cuSeqLengthsKv, sequsedQ,
        sequsedKv, blockTable, o, softmaxLse,     workspace,        tiling};
    GbsaKernel gbsaKernel;
    gbsaKernel(params);
}

template <class InDtype, class SMDtype, class REDtype = SMDtype, Epilogue::LseMode lseMode = Epilogue::LseMode::NONE,
          bool isPaged = true, bool hasResidualBlock = false>
__global__ __aicore__ void GbsaInferIntfRegularArch22(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR sparseBlockIdx,
                                                      GM_ADDR sparseBlockCount, GM_ADDR metaData, GM_ADDR cuSeqLengths,
                                                      GM_ADDR cuSeqLengthsKv, GM_ADDR sequsedQ, GM_ADDR sequsedKv,
                                                      GM_ADDR blockTable, GM_ADDR o, GM_ADDR softmaxLse,
                                                      GM_ADDR workspace, GM_ADDR tiling)
{
    // Static kernels embed tiling data and pass nullptr for the GM tiling argument.
    GET_TILING_DATA_WITH_STRUCT(GenericBlockSparseAttn::GenericBlockSparseAttentionTilingData, tilingDataLocal, tiling);
    const auto* data = &tilingDataLocal;
    const bool smallM = data->embeddingSize > 0U && data->embeddingSize <= 256U && data->groupSize > 0U &&
                        data->groupSize <= 16U && data->blockShapeX == 1U;
    if (smallM && data->baseTileInfo.kvBaseTile == GbsaSparsePairCopy::KV_COPY_MAX_TILE_TOKENS) {
        GbsaInferIntfRegularArch22Impl<InDtype, SMDtype, REDtype, lseMode, isPaged, hasResidualBlock, true>(
            q, k, v, sparseBlockIdx, sparseBlockCount, metaData, cuSeqLengths, cuSeqLengthsKv, sequsedQ, sequsedKv,
            blockTable, o, softmaxLse, workspace, tiling);
    } else {
        GbsaInferIntfRegularArch22Impl<InDtype, SMDtype, REDtype, lseMode, isPaged, hasResidualBlock, false>(
            q, k, v, sparseBlockIdx, sparseBlockCount, metaData, cuSeqLengths, cuSeqLengthsKv, sequsedQ, sequsedKv,
            blockTable, o, softmaxLse, workspace, tiling);
    }
}

// maskType=3 only — Union ND / exactMask on Arch22 (FP32 softmax).
template <class InDtype, class SMDtype, class REDtype, Epilogue::LseMode lseMode, bool isPaged, uint32_t qLayout,
          uint32_t kvLayout>
__global__ __aicore__ void GbsaInferIntfUnionArch22(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR sparseBlockIdx,
                                                    GM_ADDR sparseBlockCount, GM_ADDR metaData, GM_ADDR cuSeqLengths,
                                                    GM_ADDR cuSeqLengthsKv, GM_ADDR sequsedQ, GM_ADDR sequsedKv,
                                                    GM_ADDR blockTable, GM_ADDR o, GM_ADDR softmaxLse,
                                                    GM_ADDR workspace, GM_ADDR tiling, GM_ADDR attenMask)
{
    using ArchTag = Arch::AtlasA2;
    using ElementQ = InDtype;
    using ElementK = InDtype;
    using ElementV = InDtype;
    using ElementS = SMDtype;
    using ElementP = InDtype;
    using ElementO = InDtype;
    using ElementOTmp = REDtype;

    using LayoutQ = layout::RowMajor;
    using LayoutK = layout::ColumnMajor;
    using LayoutS = layout::RowMajor;
    using LayoutP = layout::RowMajor;
    using LayoutV = layout::RowMajor;
    using LayoutO = layout::RowMajor;
    using LayoutOTmp = layout::RowMajor;

    using L1TileShapeQK = GemmShape<128, 128, 128>;
    using L0TileShapeQK = GemmShape<128, 128, 128>;
    using DispatchPolicyQK = Gemm::MmadAtlasA2SFAIQK<isPaged, false, false, true>;
    using QType = Gemm::GemmType<ElementQ, LayoutQ>;
    using KType = Gemm::GemmType<ElementK, LayoutK>;
    using SType = Gemm::GemmType<ElementS, LayoutS>;
    using BlockMmadQK = Gemm::Block::BlockMmad<DispatchPolicyQK, L1TileShapeQK, L0TileShapeQK, QType, KType, SType>;

    using PType = Gemm::GemmType<ElementP, LayoutP>;
    using DispatchPolicyOnlineSoftmax = Epilogue::EpilogueAtlasA2OnlineSoftmax<lseMode, SMDtype>;
    using MaskType = Gemm::GemmType<int8_t, layout::RowMajor>;
    using EpilogueOnlineSoftmax = Epilogue::Block::BlockEpilogue<DispatchPolicyOnlineSoftmax, PType, SType, MaskType>;

    using L1TileShapePV = GemmShape<128, 128, 256>;
    using L0TileShapePV = GemmShape<128, 128, 128>;
    using DispatchPolicyPV = Gemm::MmadAtlasA2SFAIPV<isPaged, false, false, true>;
    using VType = Gemm::GemmType<ElementV, LayoutV>;
    using OTmpType = Gemm::GemmType<ElementOTmp, LayoutOTmp>;
    using BlockMmadPV = Gemm::Block::BlockMmad<DispatchPolicyPV, L1TileShapePV, L0TileShapePV, PType, VType, OTmpType>;

    using DispatchPolicyRescaleO = Epilogue::EpilogueAtlasA2RescaleO<lseMode, REDtype>;
    using OType = Gemm::GemmType<ElementO, LayoutO>;
    using OTmpUpdateType = Gemm::GemmType<ElementOTmp, LayoutOTmp>;
    using LseType = Gemm::GemmType<float, layout::RowMajor>;
    using EpilogueRescaleO =
        Epilogue::Block::BlockEpilogue<DispatchPolicyRescaleO, OType, OTmpType, OTmpUpdateType, LseType>;

    using GbsaKernel = GbsaUnionKernelArch22<BlockMmadQK, EpilogueOnlineSoftmax, BlockMmadPV, EpilogueRescaleO, isPaged,
                                             qLayout, kvLayout>;

    GbsaKernelParamsArch22 params{
        q,         k,          v, sparseBlockIdx, sparseBlockCount, metaData, cuSeqLengths, cuSeqLengthsKv, sequsedQ,
        sequsedKv, blockTable, o, softmaxLse,     workspace,        tiling,   attenMask};
    GbsaKernel gbsaKernel;
    gbsaKernel(params);
}

#endif
#if (__CCE_AICORE__ == 310)

using namespace GbsaKernelArch35;

#if defined(ORIG_DTYPE_QUERY) && defined(ORIG_DTYPE_KEY) && (ORIG_DTYPE_QUERY == DT_FLOAT4_E2M1) && \
    (ORIG_DTYPE_KEY == DT_FLOAT4_E2M1)
template <uint32_t D, class OutputT, Format QFormat, Format KvFormat, bool IsPaged, bool IsCausal, bool HasResidual,
          uint32_t FixedQueryTile = 0U>
__aicore__ __attribute__((noinline)) static void GbsaInferMxfp4Head(const GbsaKernelArch35::Mxfp4::KernelParams& params)
{
    GET_TILING_DATA_WITH_STRUCT(GenericBlockSparseAttn::GenericBlockSparseAttentionTilingData, data, params.tiling);
    const uint32_t Q = FixedQueryTile != 0U ? FixedQueryTile : data.baseTileInfo.qBaseTile;
    const uint32_t KV = data.baseTileInfo.kvBaseTile;
    static_assert(D == 64U || D == 128U || D == 192U || D == 256U, "Unsupported MXFP4 head size");
    // Validate the host-selected tile pair before constructing any device layout.
    const bool validTile =
        ((Q == 16U || Q == 32U) && KV == 512U) || (Q == 64U && KV == 256U) || (Q == 128U && KV == 128U);
    if (data.groupSize == 0U || data.groupSize > 128U || !validTile) {
        return;
    }
    using ArchTag = Arch::AtlasA5;
    using ElementQ = fp4x2_e2m1_t;
    using ElementK = fp4x2_e2m1_t;
    using ElementP = fp4x2_e2m1_t;
    using ElementV = fp4x2_e2m1_t;
    using ElementS = float;
    using ElementOTmp = float;
    using ElementO = OutputT;
    using LayoutP = layout::zN;

    // MXFP4 storage and compute extents use the host-selected runtime tiles.
    // Keep the BlockMmad interface, with dynamic dimension carrier types.
    using L1TileShapeQK = tla::Shape<uint32_t, uint32_t, uint32_t>;
    using L0TileShapeQK = tla::Shape<uint32_t, uint32_t, uint32_t>;
    using L1TileShapePV = tla::Shape<uint32_t, uint32_t, uint32_t>;
    using L0TileShapePV = tla::Shape<uint32_t, uint32_t, uint32_t>;
    using DispatchPolicyQK = Gemm::MmadAtlasA5GbsaQKMxfp4<IsPaged>;
    using DispatchPolicyPV = Gemm::MmadAtlasA5GbsaPVMxfp4<IsPaged>;
    using TileCopyQK = Gemm::Tile::PackedTileCopyMxfp4<IsPaged, false>;
    using TileCopyPV = Gemm::Tile::PackedTileCopyMxfp4<IsPaged, true>;
    using TileMmadQK = Gemm::Tile::TileMmadTla<ArchTag, ElementQ, typename TileCopyQK::LayoutTagL1A>;
    using TileMmadPV = Gemm::Tile::TileMmadTla<ArchTag, ElementP, LayoutP>;
    using BlockMmadQK = Gemm::Block::BlockMmadTla<DispatchPolicyQK, L1TileShapeQK, L0TileShapeQK, ElementQ, ElementK,
                                                  ElementS, void, TileCopyQK, TileMmadQK>;
    using BlockMmadPV = Gemm::Block::BlockMmadTla<DispatchPolicyPV, L1TileShapePV, L0TileShapePV, ElementP, ElementV,
                                                  ElementOTmp, void, TileCopyPV, TileMmadPV>;
    using DispatchPolicySoftmax = Epilogue::EpilogueAtlasA5Mxfp4Softmax;
    using EpilogueOnlineSoftmax = Epilogue::Block::BlockEpilogue<DispatchPolicySoftmax>;
    using DispatchPolicyPScale = Epilogue::EpilogueAtlasA5Mxfp4PScale;
    using EpilogueComputePScale = Epilogue::Block::BlockEpilogue<DispatchPolicyPScale>;
    using DispatchPolicyRescaleO = Epilogue::EpilogueAtlasA5Mxfp4RescaleO;
    using EpilogueRescaleO = Epilogue::Block::BlockEpilogue<DispatchPolicyRescaleO, ElementO>;
    using DispatchPolicyPrepareKvScale = Epilogue::EpilogueAtlasA5Mxfp4KvScale<IsPaged>;
    using EpiloguePrepareKvScale = Epilogue::Block::BlockEpilogue<DispatchPolicyPrepareKvScale>;
    using DispatchPolicyPrepareOffsets = Epilogue::EpilogueAtlasA5PrepareKvOffsetsMxfp4<IsPaged, HasResidual>;
    using EpiloguePrepareOffsets = Epilogue::Block::BlockEpilogue<DispatchPolicyPrepareOffsets>;
    using Kernel = GbsaMxfp4FullQuantKernelArch35<BlockMmadQK, BlockMmadPV, EpilogueOnlineSoftmax,
                                                  EpilogueComputePScale, EpilogueRescaleO, EpiloguePrepareOffsets,
                                                  EpiloguePrepareKvScale, QFormat, KvFormat, IsCausal, HasResidual>;
    Kernel kernel(D, Q, KV);
    // Keep the straight-through body only for the minimum query-tile family.
    kernel(params, FixedQueryTile == 16U);
}

template <uint32_t D, class OutputT, Format QFormat, Format KvFormat, bool IsPaged, bool IsCausal, bool HasResidual>
__aicore__ inline void GbsaInferMxfp4Dimension(
    const Mxfp4::KernelParams& params, const GenericBlockSparseAttn::GenericBlockSparseAttentionTilingData& data)
{
    if (data.groupSize <= 16U) {
        // Every minimum query tile keeps its address arithmetic constant.
        GbsaInferMxfp4Head<D, OutputT, QFormat, KvFormat, IsPaged, IsCausal, HasResidual, 16U>(params);
    } else {
        GbsaInferMxfp4Head<D, OutputT, QFormat, KvFormat, IsPaged, IsCausal, HasResidual>(params);
    }
}

template <class OutputT, Format QFormat, Format KvFormat, bool IsPaged, bool IsCausal, bool HasResidual>
__aicore__ inline void GbsaInferInterfaceMxfp4(const Mxfp4::KernelParams& params)
{
    GET_TILING_DATA_WITH_STRUCT(GenericBlockSparseAttn::GenericBlockSparseAttentionTilingData, data, params.tiling);
    if (data.embeddingSize == 64U) {
        GbsaInferMxfp4Dimension<64, OutputT, QFormat, KvFormat, IsPaged, IsCausal, HasResidual>(params, data);
    } else if (data.embeddingSize == 128U) {
        GbsaInferMxfp4Dimension<128, OutputT, QFormat, KvFormat, IsPaged, IsCausal, HasResidual>(params, data);
    } else if (data.embeddingSize == 192U) {
        GbsaInferMxfp4Dimension<192, OutputT, QFormat, KvFormat, IsPaged, IsCausal, HasResidual>(params, data);
    } else if (data.embeddingSize == 256U) {
        GbsaInferMxfp4Dimension<256, OutputT, QFormat, KvFormat, IsPaged, IsCausal, HasResidual>(params, data);
    }
}

#endif // MXFP4-only entry helpers

template <class InDtype, class SMDtype, class REDtype, Format qFormat, Format kvFormat, bool isPaged,
          bool isCausal = true, bool hasResidualBlock = false, Epilogue::LseMode lseMode = Epilogue::LseMode::NONE,
          Epilogue::LseFormat lseFormat = Epilogue::LseFormat::TN1, class OutDtype = InDtype>
__aicore__ inline void GbsaInferIntfRegularImpl(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR sparseBlockIdx,
                                                GM_ADDR sparseBlockCount, GM_ADDR metaData, GM_ADDR cuSeqLengths,
                                                GM_ADDR cuSeqLengthsKv, GM_ADDR sequsedQ, GM_ADDR sequsedKv,
                                                GM_ADDR blockTable, GM_ADDR o, GM_ADDR softmaxLse, GM_ADDR workspace,
                                                GM_ADDR tiling)
{
    using ArchTag = Arch::AtlasA5;
    using ElementQ = InDtype;
    using ElementK = InDtype;
    using ElementV = InDtype;
    using ElementS = SMDtype;
    using ElementP = InDtype;
    using ElementO = OutDtype;
    using ElementOTmp = REDtype;

    using LayoutQ = layout::RowMajor;
    using LayoutK = layout::ColumnMajor;
    using LayoutS = layout::RowMajor;
    using LayoutPDummy = layout::zN;
    using LayoutV = layout::RowMajor;
    using LayoutO = layout::RowMajor;
    using LayoutOTmp = layout::RowMajor;

    // Sparse KV offset preparation
    using DispatchPolicyPrepareKvOffsets =
        Epilogue::EpilogueAtlasA5PrepareKvOffsets<isPaged, hasResidualBlock, sizeof(ElementK)>;
    using EpiloguePrepareKvOffsets = Epilogue::Block::BlockEpilogue<DispatchPolicyPrepareKvOffsets>;

    // QK matmul
    using L1TileShapeQK = Shape<Int<128>, Int<128>, Int<128>>;
    using L0TileShapeQK = Shape<Int<128>, Int<128>, Int<128>>;
    using DispatchPolicyQK = Gemm::MmadAtlasA5BsaQK<isPaged, hasResidualBlock, true>;
    static constexpr auto MM1_COPY_MODE = AscendC::IsSameType<ElementS, float>::value ?
                                              Gemm::Tile::CopyL0CToUBMode::SPLIT_M :
                                              Gemm::Tile::CopyL0CToUBMode::NO_SPLIT;
    using TileCopyQK = Gemm::Tile::PackedTileCopyTlaToUB<ArchTag, ElementQ, LayoutQ, ElementK, LayoutK, ElementS,
                                                         LayoutS, void, MM1_COPY_MODE>;
    using BlockMmadQK = Gemm::Block::BlockMmadTla<DispatchPolicyQK, L1TileShapeQK, L0TileShapeQK, ElementQ, ElementK,
                                                  ElementS, void, TileCopyQK>;

    // Online softmax
    using PType = Gemm::GemmType<ElementP, LayoutPDummy>;
    using SType = Gemm::GemmType<ElementS, LayoutS>;
    using DispatchPolicyOnlineSoftmax = Epilogue::EpilogueOnlineSoftmaxBsa;
    using EpilogueOnlineSoftmax = Epilogue::Block::BlockEpilogue<DispatchPolicyOnlineSoftmax, PType, SType>;

    // PV matmul
    using L1TileShapePV = Shape<Int<128>, Int<128>, Int<128>>;
    using L0TileShapePV = Shape<Int<128>, Int<128>, Int<128>>;
    using DispatchPolicyPV = Gemm::MmadAtlasA5BsaPV<isPaged, hasResidualBlock, true>;
    using TileCopyPV =
        Gemm::Tile::PackedTileCopyTlaToUB<ArchTag, ElementP, LayoutPDummy, ElementV, LayoutV, ElementOTmp, LayoutOTmp,
                                          void, Gemm::Tile::CopyL0CToUBMode::SPLIT_M>;
    using BlockMmadPV = Gemm::Block::BlockMmadTla<DispatchPolicyPV, L1TileShapePV, L0TileShapePV, ElementP, ElementV,
                                                  ElementOTmp, void, TileCopyPV>;

    // Rescale O
    using DispatchPolicyRescaleO = Epilogue::EpilogueAtlasA5BsaRescaleO<lseMode, lseFormat>;
    using TileCopyRescaleO = Epilogue::Tile::TileCopyRescaleO<ArchTag, ElementO, LayoutO, LayoutOTmp>;
    using EpilogueRescaleO = Epilogue::Block::BlockEpilogue<DispatchPolicyRescaleO, ElementO, ElementOTmp, ElementS,
                                                            ElementK, TileCopyRescaleO, Arch::PositionL0C>;

    using GbsaKernel =
        GbsaRegularKernelArch35<EpiloguePrepareKvOffsets, BlockMmadQK, EpilogueOnlineSoftmax, BlockMmadPV,
                                EpilogueRescaleO, qFormat, kvFormat, isPaged, hasResidualBlock, isCausal>;

    GbsaKernelParamsArch35 params{
        q,         k,          v, sparseBlockIdx, sparseBlockCount, metaData, cuSeqLengths, cuSeqLengthsKv, sequsedQ,
        sequsedKv, blockTable, o, softmaxLse,     workspace,        tiling};
    GbsaKernel gbsaKernel;
    gbsaKernel(params);
}

template <class InDtype, class SMDtype, class REDtype, Format qFormat, Format kvFormat, bool isPaged,
          bool isCausal = true, bool hasResidualBlock = false, Epilogue::LseMode lseMode = Epilogue::LseMode::NONE,
          Epilogue::LseFormat lseFormat = Epilogue::LseFormat::TN1, class OutDtype = InDtype>
__global__ __aicore__ void GbsaInferIntfRegular(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR sparseBlockIdx,
                                                GM_ADDR sparseBlockCount, GM_ADDR metaData, GM_ADDR cuSeqLengths,
                                                GM_ADDR cuSeqLengthsKv, GM_ADDR sequsedQ, GM_ADDR sequsedKv,
                                                GM_ADDR blockTable, GM_ADDR o, GM_ADDR softmaxLse, GM_ADDR workspace,
                                                GM_ADDR tiling)
{
    GbsaInferIntfRegularImpl<InDtype, SMDtype, REDtype, qFormat, kvFormat, isPaged, isCausal, hasResidualBlock, lseMode,
                             lseFormat, OutDtype>(q, k, v, sparseBlockIdx, sparseBlockCount, metaData, cuSeqLengths,
                                                  cuSeqLengthsKv, sequsedQ, sequsedKv, blockTable, o, softmaxLse,
                                                  workspace, tiling);
}

// maskType=3 only — Union* templates + GbsaUnionKernelArch35 (DN / exactMask).
template <class InDtype, class SMDtype, class REDtype, Format qFormat, Format kvFormat, bool isPaged,
          Epilogue::LseMode lseMode = Epilogue::LseMode::NONE, Epilogue::LseFormat lseFormat = Epilogue::LseFormat::TN1>
__global__ __aicore__ void GbsaInferIntfUnion(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR sparseBlockIdx,
                                              GM_ADDR sparseBlockCount, GM_ADDR metaData, GM_ADDR cuSeqLengths,
                                              GM_ADDR cuSeqLengthsKv, GM_ADDR sequsedQ, GM_ADDR sequsedKv,
                                              GM_ADDR blockTable, GM_ADDR o, GM_ADDR softmaxLse, GM_ADDR workspace,
                                              GM_ADDR tiling, GM_ADDR attenMask)
{
    using ArchTag = Arch::AtlasA5;
    using ElementQ = InDtype;
    using ElementK = InDtype;
    using ElementV = InDtype;
    using ElementS = SMDtype;
    using ElementP = InDtype;
    using ElementO = InDtype;
    using ElementOTmp = REDtype;

    using LayoutQ = layout::RowMajor;
    using LayoutK = layout::ColumnMajor;
    using LayoutS = layout::RowMajor;
    using LayoutPDummy = layout::nZ;
    using LayoutV = layout::RowMajor;
    using LayoutO = layout::RowMajor;
    using LayoutOTmp = layout::RowMajor;

    using DispatchPolicyPrepareKvOffsets = Epilogue::EpilogueAtlasA5PrepareKvOffsets<isPaged, false, sizeof(ElementK)>;
    using EpiloguePrepareKvOffsets = Epilogue::Block::BlockEpilogue<DispatchPolicyPrepareKvOffsets>;

    using L1TileShapeQK = Shape<Int<128>, Int<128>, Int<128>>;
    using L0TileShapeQK = Shape<Int<128>, Int<128>, Int<128>>;
    using DispatchPolicyQK = Gemm::MmadAtlasA5UnionQK<isPaged, false, true>;
    using TileCopyQK = Gemm::Tile::PackedTileCopyTlaToUB<ArchTag, ElementQ, LayoutQ, ElementK, LayoutK, ElementS,
                                                         LayoutS, void, Gemm::Tile::CopyL0CToUBMode::SPLIT_N>;
    using BlockMmadQK = Gemm::Block::BlockMmadTla<DispatchPolicyQK, L1TileShapeQK, L0TileShapeQK, ElementQ, ElementK,
                                                  ElementS, void, TileCopyQK>;

    using PType = Gemm::GemmType<ElementP, LayoutPDummy>;
    using SType = Gemm::GemmType<ElementS, LayoutS>;
    using DispatchPolicyOnlineSoftmax = Epilogue::EpilogueOnlineSoftmaxUnion;
    using EpilogueOnlineSoftmax = Epilogue::Block::BlockEpilogue<DispatchPolicyOnlineSoftmax, PType, SType>;

    using L1TileShapePV = Shape<Int<128>, Int<128>, Int<128>>;
    using L0TileShapePV = Shape<Int<128>, Int<128>, Int<128>>;
    using DispatchPolicyPV = Gemm::MmadAtlasA5BsaPV<isPaged, false, true>;
    using TileCopyPV =
        Gemm::Tile::PackedTileCopyTlaToUB<ArchTag, ElementP, LayoutPDummy, ElementV, LayoutV, ElementOTmp, LayoutOTmp,
                                          void, Gemm::Tile::CopyL0CToUBMode::SPLIT_M32>;
    using BlockMmadPV = Gemm::Block::BlockMmadTla<DispatchPolicyPV, L1TileShapePV, L0TileShapePV, ElementP, ElementV,
                                                  ElementOTmp, void, TileCopyPV>;

    using DispatchPolicyRescaleO = Epilogue::EpilogueAtlasA5UnionRescaleO<lseMode, lseFormat>;
    using TileCopyRescaleO = Epilogue::Tile::TileCopyRescaleO<ArchTag, ElementO, LayoutO, LayoutOTmp>;
    using EpilogueRescaleO = Epilogue::Block::BlockEpilogue<DispatchPolicyRescaleO, ElementO, ElementOTmp, ElementS,
                                                            ElementK, TileCopyRescaleO, Arch::PositionL0C>;

    using GbsaKernel = GbsaUnionKernelArch35<EpiloguePrepareKvOffsets, BlockMmadQK, EpilogueOnlineSoftmax, BlockMmadPV,
                                             EpilogueRescaleO, qFormat, kvFormat, isPaged, false>;

    GbsaKernelParamsArch35 params{
        q,         k,          v, sparseBlockIdx, sparseBlockCount, metaData, cuSeqLengths, cuSeqLengthsKv, sequsedQ,
        sequsedKv, blockTable, o, softmaxLse,     workspace,        tiling,   attenMask};
    GbsaKernel gbsaKernel;
    gbsaKernel(params);
}

// maskType=3 + quantType=5: same Union DN / exactMask path as IntfUnion, but FP8
// QKV/P with SoftmaxUnion ×448 and Rescale ÷448. O is SMDtype (fp16/bf16).
template <class InDtype, class SMDtype, class REDtype, Format qFormat, Format kvFormat, bool isPaged,
          Epilogue::LseMode lseMode = Epilogue::LseMode::NONE, Epilogue::LseFormat lseFormat = Epilogue::LseFormat::TN1>
__global__ __aicore__ void GbsaInferIntfUnionFp8(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR sparseBlockIdx,
                                                 GM_ADDR sparseBlockCount, GM_ADDR metaData, GM_ADDR cuSeqLengths,
                                                 GM_ADDR cuSeqLengthsKv, GM_ADDR sequsedQ, GM_ADDR sequsedKv,
                                                 GM_ADDR blockTable, GM_ADDR o, GM_ADDR softmaxLse, GM_ADDR workspace,
                                                 GM_ADDR tiling, GM_ADDR attenMask)
{
    using ArchTag = Arch::AtlasA5;
    using ElementQ = InDtype;
    using ElementK = InDtype;
    using ElementV = InDtype;
    using ElementS = float;
    using ElementP = InDtype;
    using ElementO = SMDtype;
    using ElementOTmp = REDtype;

    using LayoutQ = layout::RowMajor;
    using LayoutK = layout::ColumnMajor;
    using LayoutS = layout::RowMajor;
    // SoftmaxUnion FP8 Step1 SoftmaxUnion-half baseline: nZ P (SoftmaxUnion CopyP + MergeHalfNz).
    using LayoutPDummy = layout::nZ;
    using LayoutV = layout::RowMajor;
    using LayoutO = layout::RowMajor;
    using LayoutOTmp = layout::RowMajor;

    using DispatchPolicyPrepareKvOffsets = Epilogue::EpilogueAtlasA5PrepareKvOffsets<isPaged, false, sizeof(ElementK)>;
    using EpiloguePrepareKvOffsets = Epilogue::Block::BlockEpilogue<DispatchPolicyPrepareKvOffsets>;

    using L1TileShapeQK = Shape<Int<128>, Int<128>, Int<128>>;
    using L0TileShapeQK = Shape<Int<128>, Int<128>, Int<128>>;
    using DispatchPolicyQK = Gemm::MmadAtlasA5UnionQK<isPaged, false, true>;
    using TileCopyQK = Gemm::Tile::PackedTileCopyTlaToUB<ArchTag, ElementQ, LayoutQ, ElementK, LayoutK, ElementS,
                                                         LayoutS, void, Gemm::Tile::CopyL0CToUBMode::SPLIT_N>;
    using BlockMmadQK = Gemm::Block::BlockMmadTla<DispatchPolicyQK, L1TileShapeQK, L0TileShapeQK, ElementQ, ElementK,
                                                  ElementS, void, TileCopyQK>;

    using PType = Gemm::GemmType<ElementP, LayoutPDummy>;
    using SType = Gemm::GemmType<ElementS, LayoutS>;
    using DispatchPolicyOnlineSoftmax = Epilogue::EpilogueOnlineSoftmaxUnion;
    using EpilogueOnlineSoftmax = Epilogue::Block::BlockEpilogue<DispatchPolicyOnlineSoftmax, PType, SType>;

    using L1TileShapePV = Shape<Int<128>, Int<128>, Int<128>>;
    using L0TileShapePV = Shape<Int<128>, Int<128>, Int<128>>;
    using DispatchPolicyPV = Gemm::MmadAtlasA5BsaPV<isPaged, false, true>;
    using TileCopyPV =
        Gemm::Tile::PackedTileCopyTlaToUB<ArchTag, ElementP, LayoutPDummy, ElementV, LayoutV, ElementOTmp, LayoutOTmp,
                                          void, Gemm::Tile::CopyL0CToUBMode::SPLIT_M32>;
    using BlockMmadPV = Gemm::Block::BlockMmadTla<DispatchPolicyPV, L1TileShapePV, L0TileShapePV, ElementP, ElementV,
                                                  ElementOTmp, void, TileCopyPV>;

    using DispatchPolicyRescaleO = Epilogue::EpilogueAtlasA5UnionRescaleO<lseMode, lseFormat>;
    using TileCopyRescaleO = Epilogue::Tile::TileCopyRescaleO<ArchTag, ElementO, LayoutO, LayoutOTmp>;
    using EpilogueRescaleO = Epilogue::Block::BlockEpilogue<DispatchPolicyRescaleO, ElementO, ElementOTmp, ElementS,
                                                            ElementK, TileCopyRescaleO, Arch::PositionL0C>;

    using GbsaKernel = GbsaUnionKernelArch35<EpiloguePrepareKvOffsets, BlockMmadQK, EpilogueOnlineSoftmax, BlockMmadPV,
                                             EpilogueRescaleO, qFormat, kvFormat, isPaged, true>;

    GbsaKernelParamsArch35 params{
        q,         k,          v, sparseBlockIdx, sparseBlockCount, metaData, cuSeqLengths, cuSeqLengthsKv, sequsedQ,
        sequsedKv, blockTable, o, softmaxLse,     workspace,        tiling,   attenMask};
    GbsaKernel gbsaKernel;
    gbsaKernel(params);
}

template <class InDtype, class SMDtype, class REDtype, Format qFormat, Format kvFormat, bool isPaged,
          Epilogue::LseMode lseMode = Epilogue::LseMode::NONE, Epilogue::LseFormat lseFormat = Epilogue::LseFormat::TN1,
          class OutDtype = SMDtype>
__global__ __aicore__ void GbsaInferInterfaceFullQuant(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR sparseBlockIdx,
                                                       GM_ADDR sparseBlockCount, GM_ADDR metaData, GM_ADDR cuSeqLengths,
                                                       GM_ADDR cuSeqLengthsKv, GM_ADDR sequsedQ, GM_ADDR sequsedKv,
                                                       GM_ADDR blockTable, GM_ADDR qDequantScale, GM_ADDR kDequantScale,
                                                       GM_ADDR vDequantScale, GM_ADDR o, GM_ADDR softmaxLse,
                                                       GM_ADDR workspace, GM_ADDR tiling)
{
    // Per-tensor dequantization scales are not used by this FP8 path.
    // FP8 full-quant is causal and does not use an implicit residual block.
    GbsaInferIntfRegularImpl<InDtype, SMDtype, REDtype, qFormat, kvFormat, isPaged, true, false, lseMode, lseFormat,
                             OutDtype>(q, k, v, sparseBlockIdx, sparseBlockCount, metaData, cuSeqLengths,
                                       cuSeqLengthsKv, sequsedQ, sequsedKv, blockTable, o, softmaxLse, workspace,
                                       tiling);
}
#if !defined(TILING_KEY_VAR) || (TILING_KEY_VAR >= 9250002900000020ULL && TILING_KEY_VAR <= 9250002900000023ULL) || \
    (TILING_KEY_VAR >= 9250002900000032ULL && TILING_KEY_VAR <= 9250002900000035ULL) || \
    (TILING_KEY_VAR >= 9250002900000040ULL && TILING_KEY_VAR <= 9250002900000047ULL) || \
    (TILING_KEY_VAR >= 9250002900000050ULL && TILING_KEY_VAR <= 9250002900000055ULL)
namespace GbsaMxScaleMode {
constexpr auto PIPELINE = 4U;           // Grouped K-scale streaming.
constexpr auto COALESCED_PIPELINE = 5U; // Retained prefill dispatch alias of grouped streaming.
} // namespace GbsaMxScaleMode

template <class OutDtype, uint32_t ScaleMode = GbsaMxScaleMode::PIPELINE, Format QFormat = Format::BSND,
          Format KvFormat = QFormat, bool IsPaged = false>
__global__ __aicore__ void GbsaInferInterfaceMxfp8Packed(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR sparseBlockIdx,
                                                         GM_ADDR sparseBlockCount, GM_ADDR metaData,
                                                         GM_ADDR cuSeqLengths, GM_ADDR cuSeqLengthsKv, GM_ADDR sequsedQ,
                                                         GM_ADDR sequsedKv, GM_ADDR blockTable, GM_ADDR qDequantScale,
                                                         GM_ADDR kDequantScale, GM_ADDR vDequantScale, GM_ADDR o,
                                                         GM_ADDR softmaxLse, GM_ADDR workspace, GM_ADDR tiling,
                                                         GM_ADDR pQuantScale = nullptr)
{
    static_assert(ScaleMode == GbsaMxScaleMode::PIPELINE || ScaleMode == GbsaMxScaleMode::COALESCED_PIPELINE,
                  "Unsupported MXFP8 scale mode");
    using ArchTag = Arch::AtlasA5;
    using ElementQ = fp8_e4m3fn_t;
    using ElementK = fp8_e4m3fn_t;
    using ElementV = fp8_e4m3fn_t;
    // Keep QK scores in FP32 through Fixpipe and online softmax. OutDtype
    // controls only the final attention output, including legacy precision=1.
    using ElementS = float;
    using ElementP = fp8_e4m3fn_t;
    using ElementO = OutDtype;
    using ElementOTmp = float;

    using LayoutQ = layout::RowMajor;
    using LayoutK = layout::ColumnMajor;
    using LayoutS = layout::RowMajor;
    using LayoutPDummy = layout::zN;
    using LayoutV = layout::RowMajor;
    using LayoutO = layout::RowMajor;
    using LayoutOTmp = layout::RowMajor;

    using L1TileShapeQK = Shape<Int<128>, Int<128>, Int<128>>;
    // Packed tiles contain one, two or four N128 groups, selected by the head ratio.
    using L0TileShapeQK = Shape<Int<128>, Int<128>, Int<256>>;
    // The packed matrix loader gathers dense sparse blocks in logical index order.
    using DispatchPolicyQK = Gemm::MmadAtlasA5GbsaQKMxfp8;
    static constexpr auto MM1_COPY_MODE = AscendC::IsSameType<ElementS, float>::value ?
                                              Gemm::Tile::CopyL0CToUBMode::SPLIT_M :
                                              Gemm::Tile::CopyL0CToUBMode::NO_SPLIT;
    using TileCopyQKBase = Gemm::Tile::PackedTileCopyTlaToUB<ArchTag, ElementQ, LayoutQ, ElementK, LayoutK, ElementS,
                                                             LayoutS, void, MM1_COPY_MODE>;
    using TileCopyQK = Gemm::Tile::GbsaMxPackedQKTileCopy<TileCopyQKBase>;
    using TileMmadQK = Gemm::Tile::GbsaMxPackedQKSingleKMmad<L0TileShapeQK>;
    using BlockMmadQK = Gemm::Block::BlockMmadTla<DispatchPolicyQK, L1TileShapeQK, L0TileShapeQK, ElementQ, ElementK,
                                                  ElementS, void, TileCopyQK, TileMmadQK>;

    using DispatchPolicyOnlineSoftmax = Epilogue::EpilogueOnlineSoftmaxGbsaMxfp8;
    using PType = Gemm::GemmType<ElementP, LayoutPDummy>;
    using SType = Gemm::GemmType<ElementS, LayoutS>;
    using EpilogueOnlineSoftmax = Epilogue::Block::BlockEpilogue<DispatchPolicyOnlineSoftmax, PType, SType>;

    using L1TileShapePV = Shape<Int<128>, Int<128>, Int<128>>;
    using L0TileShapePV = Shape<Int<128>, Int<128>, Int<256>>;
    // The packed matrix loader gathers dense sparse blocks in logical index order.
    using DispatchPolicyPV = Gemm::MmadAtlasA5GbsaPVMxfp8;
    using TileCopyPVBase =
        Gemm::Tile::PackedTileCopyTlaToUB<ArchTag, ElementP, LayoutPDummy, ElementV, LayoutV, ElementOTmp, LayoutOTmp,
                                          void, Gemm::Tile::CopyL0CToUBMode::SPLIT_M>;
    using TileCopyPV = Gemm::Tile::GbsaMxPackedPVTileCopy<TileCopyPVBase>;
    using TileMmadPV = Gemm::Tile::TileMmadTla<ArchTag, ElementP, typename TileCopyPV::LayoutTagL1A>;
    using BlockMmadPV = Gemm::Block::BlockMmadTla<DispatchPolicyPV, L1TileShapePV, L0TileShapePV, ElementP, ElementV,
                                                  ElementOTmp, void, TileCopyPV, TileMmadPV>;

    using DispatchPolicyRescaleO =
        Epilogue::EpilogueAtlasA5GbsaRescaleOMxfp8<Epilogue::LseMode::NONE, Epilogue::LseFormat::TN1>;
    using TileCopyRescaleOBase = Epilogue::Tile::TileCopyRescaleO<ArchTag, ElementO, LayoutO, LayoutOTmp>;
    using TileCopyRescaleO = Epilogue::Tile::TileCopyRescaleOMxV<TileCopyRescaleOBase, true, true>;
    using EpilogueRescaleO = Epilogue::Block::BlockEpilogue<DispatchPolicyRescaleO, ElementO, ElementOTmp, ElementS,
                                                            ElementK, TileCopyRescaleO, Arch::PositionL0C>;

    using GbsaKernel = GbsaMxfp8PackedKernelArch35<BlockMmadQK, EpilogueOnlineSoftmax, BlockMmadPV, EpilogueRescaleO,
                                                   GbsaMxQkVariant::D192, QFormat, KvFormat, IsPaged>;

    GbsaFullQuantKernelParamsArch35 params{q,
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
    params.pQuantScale = pQuantScale;
    GET_TILING_DATA_WITH_STRUCT(GenericBlockSparseAttn::GenericBlockSparseAttentionTilingData, packedTiling, tiling);
    if constexpr (ScaleMode == GbsaMxScaleMode::COALESCED_PIPELINE) {
        // Host selects this policy only for D256. Do not emit unused D192
        // implementations into the coalesced entry and its device image.
        using WideL0Shape = Shape<Int<128>, Int<128>, Int<256>>;
        using WideQK = Gemm::Block::BlockMmadTla<DispatchPolicyQK, L1TileShapeQK, WideL0Shape, ElementQ, ElementK,
                                                 ElementS, void, TileCopyQK, Gemm::Tile::GbsaMxMmad>;
        using WideKernel = GbsaMxfp8PackedKernelArch35<WideQK, EpilogueOnlineSoftmax, BlockMmadPV, EpilogueRescaleO,
                                                       GbsaMxQkVariant::D256, QFormat, KvFormat, IsPaged>;
        WideKernel kernel;
        kernel(params);
    } else if (packedTiling.embeddingSize == 256U) {
        // N128/K256 MMADs form the logical KV tile selected by the host.
        using WideL0Shape = Shape<Int<128>, Int<128>, Int<256>>;
        using WideQK = Gemm::Block::BlockMmadTla<DispatchPolicyQK, L1TileShapeQK, WideL0Shape, ElementQ, ElementK,
                                                 ElementS, void, TileCopyQK, Gemm::Tile::GbsaMxMmad>;
        using WideKernel = GbsaMxfp8PackedKernelArch35<WideQK, EpilogueOnlineSoftmax, BlockMmadPV, EpilogueRescaleO,
                                                       GbsaMxQkVariant::D256, QFormat, KvFormat, IsPaged>;
        WideKernel kernel;
        kernel(params);
    } else if (!IsPaged && packedTiling.embeddingSize == 192U && packedTiling.blockShapeY == 8U &&
               packedTiling.kvHeads == 4U && packedTiling.groupSize == 16U &&
               AscendC::IsSameType<OutDtype, bfloat16_t>::value &&
               packedTiling.qSeqCapacity == packedTiling.kvSeqCapacity && packedTiling.qSeqCapacity >= 1024U &&
               packedTiling.qSeqCapacity <= 8192U) {
        // Retain the dense CSA dispatch alias with an explicit actual D192.
        // Both aliases use grouped KScale preparation on the two AIVs.
        using DenseCsaKernel =
            GbsaMxfp8PackedKernelArch35<BlockMmadQK, EpilogueOnlineSoftmax, BlockMmadPV, EpilogueRescaleO,
                                        GbsaMxQkVariant::D192_PREFILL, QFormat, KvFormat, IsPaged>;
        DenseCsaKernel kernel;
        kernel(params);
    } else {
        GbsaKernel gbsaKernel;
        gbsaKernel(params);
    }
}
#endif // Packed MXFP8 dispatch

#if defined(ORIG_DTYPE_QUERY) && defined(ORIG_DTYPE_KEY) && (ORIG_DTYPE_QUERY == DT_FLOAT8_E4M3FN) && \
    (ORIG_DTYPE_KEY == DT_FLOAT4_E2M1)
template <class BaseCopy>
struct A8C4PreloadedKvTileCopy : BaseCopy {
    template <class TensorB>
    struct CopyGmToL1B {
        template <class Dst, class Src>
        __aicore__ inline void operator()(const Dst&, const Src&)
        {}
    };
};

template <class OutDtype, Format qFormat, Format kvFormat, bool isPaged, bool isCausal, Epilogue::LseMode lseMode,
          Epilogue::LseFormat lseFormat>
__global__ __aicore__ void GbsaInferInterfaceA8C4(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR sparseIdx,
                                                  GM_ADDR sparseCount, GM_ADDR metadata, GM_ADDR cuQ, GM_ADDR cuKv,
                                                  GM_ADDR usedQ, GM_ADDR usedKv, GM_ADDR pages, GM_ADDR kScale,
                                                  GM_ADDR vScale, GM_ADDR out, GM_ADDR lse, GM_ADDR workspace,
                                                  GM_ADDR tiling)
{
    using ArchTag = Arch::AtlasA5;
    using Fp8 = fp8_e4m3fn_t;
    using QKCopy = A8C4PreloadedKvTileCopy<
        Gemm::Tile::PackedTileCopyTlaToUB<ArchTag, Fp8, layout::RowMajor, Fp8, layout::ColumnMajor, float,
                                          layout::RowMajor, void, Gemm::Tile::CopyL0CToUBMode::SPLIT_M>>;
    using PVBaseCopy = A8C4PreloadedKvTileCopy<
        Gemm::Tile::PackedTileCopyTlaToUB<ArchTag, Fp8, layout::zN, Fp8, layout::RowMajor, float, layout::RowMajor,
                                          void, Gemm::Tile::CopyL0CToUBMode::SPLIT_M>>;
    using Tile = Shape<Int<128>, Int<128>, Int<128>>;
    using QK = Gemm::Block::BlockMmadTla<Gemm::MmadAtlasA5BsaQK<isPaged>, Tile, Tile, Fp8, Fp8, float, void, QKCopy>;
    using PV =
        Gemm::Block::BlockMmadTla<Gemm::MmadAtlasA5BsaPV<isPaged>, Tile, Tile, Fp8, Fp8, float, void, PVBaseCopy>;
    using RescaleCopy = Epilogue::Tile::TileCopyRescaleO<ArchTag, OutDtype, layout::RowMajor, layout::RowMajor>;
    // FP4 storage tag disables the legacy A8C8 P*448 output compensation.
    using Rescale = Epilogue::Block::BlockEpilogue<Epilogue::EpilogueAtlasA5BsaRescaleO<lseMode, lseFormat>, OutDtype,
                                                   float, float, fp4x2_e2m1_t, RescaleCopy, Arch::PositionL0C>;
    GbsaFullQuantKernelParamsArch35 params{q,      k,      v,     sparseIdx, sparseCount, metadata,
                                           cuQ,    cuKv,   usedQ, usedKv,    pages,       nullptr,
                                           kScale, vScale, out,   lse,       workspace,   tiling};
    GET_TILING_DATA_WITH_STRUCT(GenericBlockSparseAttn::GenericBlockSparseAttentionTilingData, tilingData, tiling);
    using Kernel = GbsaA8C4KernelArch35<QK, Epilogue::Block::A8C4OnlineSoftmax, PV, Rescale, qFormat, kvFormat, isPaged,
                                        false, isCausal>;
    Kernel kernel;
    kernel(params, &tilingData);
}
#endif
#endif // Ascend 950
