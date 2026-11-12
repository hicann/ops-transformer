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
#endif
#if (__CCE_AICORE__ == 310)
#include "arch35/generic_block_sparse_attention_kernel_arch35.h"
#if !defined(TILING_KEY_VAR) || (TILING_KEY_VAR >= 9250002900000020ULL && TILING_KEY_VAR <= 9250002900000023ULL) || \
    (TILING_KEY_VAR >= 9250002900000032ULL && TILING_KEY_VAR <= 9250002900000035ULL) || \
    (TILING_KEY_VAR >= 9250002900000040ULL && TILING_KEY_VAR <= 9250002900000047ULL) || \
    (TILING_KEY_VAR >= 9250002900000050ULL && TILING_KEY_VAR <= 9250002900000055ULL)
#include "arch35/generic_block_sparse_attention_kernel_arch35_mxfp8_full_quant.h"
#endif
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

#endif
#if (__CCE_AICORE__ == 310)

using namespace GbsaKernelArch35;

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

template <class InDtype, class SMDtype, class REDtype, Format qFormat,
          Epilogue::LseMode lseMode = Epilogue::LseMode::NONE, Epilogue::LseFormat lseFormat = Epilogue::LseFormat::TN1>
__global__ __aicore__ void GbsaInferInterfaceFullQuant(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR sparseBlockIdx,
                                                       GM_ADDR sparseBlockCount, GM_ADDR metaData, GM_ADDR cuSeqLengths,
                                                       GM_ADDR cuSeqLengthsKv, GM_ADDR sequsedQ, GM_ADDR sequsedKv,
                                                       GM_ADDR blockTable, GM_ADDR qDequantScale, GM_ADDR kDequantScale,
                                                       GM_ADDR vDequantScale, GM_ADDR o, GM_ADDR softmaxLse,
                                                       GM_ADDR workspace, GM_ADDR tiling, GM_ADDR pQuantScale = nullptr)
{
    GbsaInferIntfRegularImpl<InDtype, SMDtype, REDtype, qFormat, Format::BSND, true, true, false, lseMode, lseFormat,
                             SMDtype>(q, k, v, sparseBlockIdx, sparseBlockCount, metaData, cuSeqLengths, cuSeqLengthsKv,
                                      sequsedQ, sequsedKv, blockTable, o, softmaxLse, workspace, tiling);
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
#endif // Ascend 950
