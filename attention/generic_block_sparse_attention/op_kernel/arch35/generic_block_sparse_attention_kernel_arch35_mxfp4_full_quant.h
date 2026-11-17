/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GENERIC_BLOCK_SPARSE_ATTENTION_KERNEL_ARCH35_MXFP4_FULL_QUANT_H
#define GENERIC_BLOCK_SPARSE_ATTENTION_KERNEL_ARCH35_MXFP4_FULL_QUANT_H

#include "kernel_utils.hpp"
#include "../attn_infra/gemm/block/gbsa_block_mmad_qk_arch35_mxfp4.hpp"
#include "../attn_infra/gemm/block/gbsa_block_mmad_pv_arch35_mxfp4.hpp"
#include "../attn_infra/epilogue/block/gbsa_block_epilogue_softmax_arch35_high_prec_mxfp4.hpp"
#include "../attn_infra/epilogue/block/gbsa_block_epilogue_compute_pscale_arch35_mxfp4.hpp"
#include "../attn_infra/epilogue/block/gbsa_block_epilogue_rescale_o_arch35_mxfp4.hpp"
#include "../attn_infra/epilogue/block/gbsa_block_epilogue_prepare_kv_scale_arch35_mxfp4.hpp"
#include "../attn_infra/epilogue/block/gbsa_block_epilogue_prepare_kv_offsets_arch35_mxfp4.hpp"
#include "gbsa_mxfp4_kernel_utils.hpp"
#include "gbsa_layout_utils.hpp"
#include "gbsa_mxfp4_fd_combine_arch35.h"
#include "../attn_infra/gemm/block/gbsa_block_mmad_arch35_mxfp4_utils.hpp"
#include "../attn_infra/epilogue/block/gbsa_block_epilogue_arch35_mxfp4_utils.hpp"
#include "../tla/gbsa_tla_tensor.hpp"

namespace GbsaKernelArch35 {

template <class BlockMmadQK, class BlockMmadPV, class EpilogueOnlineSoftmax, class EpilogueComputePScale,
          class EpilogueRescaleO, class EpiloguePrepareOffsets, class EpiloguePrepareKvScale, Format QFormat,
          Format KvFormat, bool IsCausal, bool HasResidualBlock>
class GbsaMxfp4FullQuantKernelArch35 {
public:
    using ArchTag = typename BlockMmadQK::ArchTag;
    using Resource = NpuArch::Arch::Resource<ArchTag>;
    // GM offsets and zN copies use packed bytes; L0/MMAD operands use logical FP4 elements.
    using ElementStorage = typename BlockMmadQK::ElementStorage;
    using ElementQ = typename BlockMmadQK::ElementA;
    using ElementK = typename BlockMmadQK::ElementB;
    using ElementS = typename BlockMmadQK::ElementC;
    using ElementP = typename BlockMmadPV::ElementA;
    using ElementV = typename BlockMmadPV::ElementB;
    using ElementOTmp = typename BlockMmadPV::ElementC;
    using ElementO = typename EpilogueRescaleO::ElementO;
    using ElementLse = typename EpilogueRescaleO::ElementLse;
    uint32_t HeadDim, VecRows, QueryTile, KV;
    static constexpr bool IsPaged = BlockMmadQK::DispatchPolicy::PAGED_CACHE_FLAG;
    using Ub = Mxfp4::UBufTileHelper;
    using Descriptor = Mxfp4::DescriptorTileHelper<IsPaged>;
    using L1 = typename BlockMmadQK::Layout;

    Ub ub;
    L1 l1;
    Descriptor descriptor;
    __aicore__ GbsaMxfp4FullQuantKernelArch35(uint32_t d, uint32_t q, uint32_t kv)
        : HeadDim(d),
          VecRows(((q / 2U + 15U) / 16U) * 16U),
          QueryTile(q),
          KV(kv)
    {
        CalcOnChipBufTileInfo();
        CalcUBufTileInfo();
        CalcDescriptorTileInfo();
    }

    __aicore__ inline void operator()(const Mxfp4::KernelParams& params, bool enableDirectTask = true)
    {
        GET_TILING_DATA_WITH_STRUCT(GenericBlockSparseAttn::GenericBlockSparseAttentionTilingData, tiling,
                                    params.tiling);
        writeLse = tiling.returnSoftmaxLse != 0U;
        const auto* meta = reinterpret_cast<__gm__ const GbsaMetadata::Metadata*>(params.metaData);
        if constexpr (!HasResidualBlock) {
            // This metadata range covers all actual queries. With no implicit
            // residual block, an empty range makes every output (including
            // seqused storage padding) zero, independently of either layout.
            if (meta->maxSparseCount == 0 && meta->packedCountRangeTag == GbsaMetadata::PACKED_COUNT_RANGE_TAG) {
#ifdef __DAV_VEC__
                StoreEmptyTensor(params, tiling);
#endif
                return;
            }
        }
        const bool flashDecode = (meta->fdScheduleFlags & GbsaMetadata::FD_SCHEDULE_ENABLED) != 0U;
        // A residual adds at most one block to this conservative token bound.
        const uint32_t maximum = meta->packedCountRangeTag == GbsaMetadata::PACKED_COUNT_RANGE_TAG ?
                                     static_cast<uint32_t>(meta->maxSparseCount) :
                                     tiling.topK;
        const uint64_t bound =
            (static_cast<uint64_t>(maximum) + static_cast<uint32_t>(HasResidualBlock)) * tiling.blockShapeY;
        const bool singleTile = !flashDecode && bound <= KV;
        const bool directTask =
            enableDirectTask && singleTile && static_cast<uint32_t>(meta->saTotalTaskNum) <= AscendC::GetBlockNum();
        if (directTask) {
            Run<true>(params, tiling, meta, false);
            return;
        }
        Run<false>(params, tiling, meta, flashDecode);
        if (flashDecode) {
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::SyncAll<false>();
#ifdef __DAV_VEC__
            GbsaMxfp4FdCombineArch35<ElementO, QFormat> combine(resource, HeadDim);
            combine(params, tiling, meta);
#endif
        }
    }

private:
    __aicore__ static constexpr bool PreferScaleLookahead(uint32_t dimension, uint32_t blockRows)
    {
        const uint32_t pairs = dimension / 64U;
        return (pairs & (pairs - 1U)) != 0U || static_cast<uint64_t>(blockRows) * (dimension / 32U) >= 256U;
    }

    __aicore__ inline void CalcOnChipBufTileInfo()
    {
        l1.D = HeadDim;
        l1.Q = QueryTile;
        l1.QUERY_TILE = l1.Q;
        l1.R = VecRows;
        l1.QUERY_ROWS = 2 * l1.R;
        l1.HEAD_CAPACITY = l1.D <= 128 ? 128 : 256;
        l1.KV_ROWS = KV;
        l1.K_ROWS = KV;
        l1.Q_SLOT_BYTES = l1.Q * l1.HEAD_CAPACITY / 2;
        l1.K_OFFSET = l1.Q_OFFSET + l1.Q_SLOTS * l1.Q_SLOT_BYTES;
        l1.K_SLOT_BYTES = l1.K_ROWS * l1.HEAD_CAPACITY / 2;
        l1.P_OFFSET = l1.K_OFFSET + l1.K_SLOTS * l1.K_SLOT_BYTES;
        l1.P_SLOT_BYTES = l1.QUERY_ROWS * l1.KV_ROWS / 2;
        l1.V_OFFSET = l1.P_OFFSET + l1.P_SLOTS * l1.P_SLOT_BYTES;
        l1.V_SLOT_BYTES = l1.KV_ROWS * l1.HEAD_CAPACITY / 2;
        l1.QSCALE_OFFSET = l1.V_OFFSET + l1.V_SLOTS * l1.V_SLOT_BYTES;
        l1.QSCALE_SLOT_BYTES = ((l1.Q * l1.HEAD_CAPACITY / 32 + 255) / 256) * 256;
        l1.KSCALE_OFFSET = l1.QSCALE_OFFSET + l1.Q_SLOTS * l1.QSCALE_SLOT_BYTES;
        l1.KSCALE_SLOT_BYTES = ((l1.K_ROWS * l1.HEAD_CAPACITY / 32 + 255) / 256) * 256;
        l1.PSCALE_OFFSET = l1.KSCALE_OFFSET + l1.K_SLOTS * l1.KSCALE_SLOT_BYTES;
        l1.PSCALE_SLOT_BYTES = ((l1.QUERY_ROWS * l1.KV_ROWS / 32 + 255) / 256) * 256;
        l1.VSCALE_OFFSET = l1.PSCALE_OFFSET + l1.P_SLOTS * l1.PSCALE_SLOT_BYTES;
        l1.VSCALE_SLOT_BYTES = ((l1.KV_ROWS * l1.HEAD_CAPACITY / 32 + 255) / 256) * 256;
        l1.TOTAL_BYTES = l1.VSCALE_OFFSET + l1.VSCALE_SLOT_BYTES;
        l1.PV_COLUMNS = l1.D < 128 ? l1.D : 128;
    }

    __aicore__ inline void CalcUBufTileInfo()
    {
        ub.D = HeadDim;
        ub.Q = QueryTile;
        ub.R = VecRows;
        ub.KV = KV;
        ub.S_SLOT_BYTES = ub.R * ub.KV * sizeof(float);
        ub.PV_OFFSET = ub.S_OFFSET + 2 * ub.S_SLOT_BYTES;
        ub.PV_SLOTS = ub.D > 128 && ub.Q == 128 ? 1 : 2;
        ub.PV_SLOT_BYTES = ub.R * ub.D * sizeof(float);
        ub.O_OFFSET = ub.PV_OFFSET + ub.PV_SLOTS * ub.PV_SLOT_BYTES;
        ub.P_OFFSET = ub.O_OFFSET + ub.R * ub.D * sizeof(float);
        ub.P_SLOT_BYTES = ub.R * ub.KV / 2;
        ub.PSCALE_SLOT_BYTES = ub.R * ub.KV / 32;
        ub.LOCAL_MAX_OFFSET = ub.P_OFFSET + 2 * ub.P_SLOT_BYTES;
        ub.LOCAL_MAX_SLOT_BYTES = ub.R * (ub.KV / 32) * sizeof(float);
        ub.GROUP_SUM_OFFSET = ub.LOCAL_MAX_OFFSET + ub.LOCAL_MAX_SLOT_BYTES;
        ub.TILE_G_OFFSET = ub.GROUP_SUM_OFFSET + ub.LOCAL_MAX_SLOT_BYTES;
        ub.OUT_G_OFFSET = ub.TILE_G_OFFSET + ub.STATS_SLOTS * ub.VECTOR_BYTES;
        ub.TILE_DM_OFFSET = ub.OUT_G_OFFSET + ub.VECTOR_BYTES;
        ub.LOCAL_ROW_SUM_OFFSET = ub.TILE_DM_OFFSET + ub.STATS_SLOTS * ub.VECTOR_BYTES;
        ub.OUT_L_OFFSET = ub.LOCAL_ROW_SUM_OFFSET + ub.LOCAL_ROW_SUM_BYTES;
        ub.TILE_MAX_OFFSET = ub.OUT_L_OFFSET + ub.VECTOR_BYTES;
        ub.DESC_OFFSET = ub.TILE_MAX_OFFSET + ub.VECTOR_BYTES;
        ub.KSCALE_GROUPS = (ub.KV + 255) / 256;
        ub.KSCALE_INPUT_SLOTS = 3 * ub.KSCALE_GROUPS;
        ub.KSCALE_OUTPUT_BYTES = (ub.D == 192U || ub.D == 256U) && ub.Q <= 32U ? 256U * (ub.D / 32U) : 1024U;
        ub.KSCALE_OUTPUT_OFFSET = ub.DESC_OFFSET + ub.KSCALE_INPUT_SLOTS * ub.KSCALE_INPUT_BYTES;
        ub.LOCAL_DESC_OFFSET = ub.KSCALE_OUTPUT_OFFSET + ub.KSCALE_OUTPUT_BYTES;
        ub.LSE_PACKED_OFFSET = ub.LOCAL_DESC_OFFSET + (ub.KV > 256 ? 2 : 1) * 4160;
        ub.DESC_BYTES = ub.LSE_PACKED_OFFSET - ub.DESC_OFFSET;
        ub.OUTPUT_SCRATCH_OFFSET = ub.P_OFFSET;
        ub.OUTPUT_SCRATCH_BYTES = 2 * ub.P_SLOT_BYTES;
        ub.LSE_OFFSET = ub.LSE_PACKED_OFFSET + ub.R * 32;
        ub.PSCALE_OFFSET = ub.LSE_OFFSET + ub.VECTOR_BYTES;
        ub.TOTAL_BYTES = ub.PSCALE_OFFSET + ub.PSCALE_SLOT_BYTES;
    }

    __aicore__ inline void CalcDescriptorTileInfo()
    {
        descriptor.D = HeadDim;
        descriptor.Q = QueryTile;
        descriptor.KV = KV;
        descriptor.ROWS =
            (descriptor.D == 256U || (IsPaged && descriptor.D == 128U) || (!IsPaged && descriptor.D == 192U)) &&
                    descriptor.Q <= 32U && descriptor.KV == 512U ?
                256U :
                128U;
        descriptor.GROUPS = (descriptor.KV + descriptor.ROWS - 1U) / descriptor.ROWS;
        descriptor.GROUPS_PER_BANK = (descriptor.GROUPS + 1U) / 2U;
        descriptor.INPUT_BYTES = descriptor.ROWS * 32U;
        descriptor.INPUT_SLOTS = 3U * descriptor.GROUPS_PER_BANK;
        descriptor.LOCAL_DESC_SLOTS = descriptor.GROUPS_PER_BANK;
        descriptor.LOCAL_DESC_SLOT_BYTES = (8U + descriptor.ROWS * 4U) * sizeof(uint64_t);
    }

#ifdef __DAV_VEC__
    __aicore__ inline void StoreEmptyTensor(const Mxfp4::KernelParams& params,
                                            const GenericBlockSparseAttn::GenericBlockSparseAttentionTilingData& tiling)
    {
        constexpr uint32_t chunkBytes = 16384U;
        constexpr uint32_t outputElements = chunkBytes / sizeof(ElementO);
        constexpr uint32_t lseElements = chunkBytes / sizeof(float);
        const uint32_t core = AscendC::GetBlockIdx();
        const uint32_t cores = AscendC::GetBlockNum() * AscendC::GetSubBlockNum();
        const uint64_t rows = tiling.totalQTokens * tiling.numHeads;
        const uint64_t elements = rows * HeadDim;
        auto zeros = resource.ubBuf.template GetBufferByByte<ElementO>(0U);
        auto emptyLse = resource.ubBuf.template GetBufferByByte<float>(chunkBytes);
        AscendC::Duplicate(zeros, static_cast<ElementO>(0), outputElements);
        if (writeLse) {
            AscendC::Duplicate(emptyLse, -3.402823466e+38F, lseElements);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::GlobalTensor<ElementO> output;
        output.SetGlobalBuffer(reinterpret_cast<__gm__ ElementO*>(params.o));
        for (uint64_t offset = static_cast<uint64_t>(core) * outputElements; offset < elements;
             offset += static_cast<uint64_t>(cores) * outputElements) {
            const uint32_t count = static_cast<uint32_t>(min(static_cast<uint64_t>(outputElements), elements - offset));
            AscendC::DataCopyExtParams copy = {1U, count * static_cast<uint32_t>(sizeof(ElementO)), 0, 0, 0};
            AscendC::DataCopyPad(output[offset], zeros, copy);
        }
        if (writeLse) {
            AscendC::GlobalTensor<float> lse;
            lse.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(params.softmaxLse));
            for (uint64_t offset = static_cast<uint64_t>(core) * lseElements; offset < rows;
                 offset += static_cast<uint64_t>(cores) * lseElements) {
                const uint32_t count = static_cast<uint32_t>(min(static_cast<uint64_t>(lseElements), rows - offset));
                AscendC::DataCopyExtParams copy = {1U, count * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
                AscendC::DataCopyPad(lse[offset], emptyLse, copy);
            }
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }
#endif

    struct FdPipelineTile {
#ifdef __DAV_CUBE__
        uint64_t kvByteOffset;
        uint64_t vScaleOffset;
        uint32_t kvTokens;
        uint32_t vScalePairs;
#endif
#ifdef __DAV_VEC__
        uint64_t outputOffset;
        uint64_t lseOffset;
        int32_t partialId;
#endif
        uint32_t tileCount;
        uint32_t tileId;
    };

    template <bool DirectTask>
    __aicore__ inline void Run(const Mxfp4::KernelParams& params,
                               const GenericBlockSparseAttn::GenericBlockSparseAttentionTilingData& tiling,
                               __gm__ const GbsaMetadata::Metadata* meta, bool flashDecode)
    {
        const uint32_t taskCount = meta->saTotalTaskNum;
        allQStored = taskCount == static_cast<uint64_t>(tiling.totalQTokens) * tiling.kvHeads;
        InitCrossCoreDstBuf();
        uint32_t core = AscendC::GetBlockIdx();
        const uint32_t coreNum = AscendC::GetBlockNum();
        uint32_t bank = 0U;
#ifdef __DAV_VEC__
        bank = AscendC::GetSubBlockIdx();
        core /= AscendC::GetSubBlockNum();

#endif
        uint32_t taskStart = core, taskEnd = taskCount, taskStep = coreNum;
        uint32_t firstBlock = 0U, lastBlock = 0U;
        if (!DirectTask && flashDecode) {
            taskStart = taskEnd = 0U;
            taskStep = 1U;
            if (core < static_cast<uint32_t>(meta->fdActiveCoreNum)) {
                const auto& schedule = meta->decodeSchedules[core];
                taskStart = schedule.baseTaskStart;
                taskEnd = schedule.baseTaskEnd;
                firstBlock = schedule.firstBlockStart;
                lastBlock = schedule.lastBlockEnd;
            }
        }
#ifdef __DAV_VEC__
        const uint32_t scaleLookahead =
            PreferScaleLookahead(HeadDim, tiling.blockShapeY) && taskStart + taskStep < taskEnd ? 1U : 0U;
#endif
        reuseQuerySeq = allQStored && taskEnd > taskStart && taskEnd - taskStart > taskStep;
        const uint32_t rows0 = (tiling.groupSize + 1U) / 2U;
        const uint32_t validRows = bank == 0U ? rows0 : tiling.groupSize - rows0;
        const uint32_t rowOffset = bank * rows0;
        const uint32_t groupWords = CalcKvOffsetGroupWords<IsPaged>(descriptor.ROWS, tiling.blockShapeY);
        const uint32_t tokenStrideBytes = GbsaLayoutHelper<KvFormat>::TokenStride(tiling.kvHeads, HeadDim / 2U);
        const uint32_t qHeadStride = QFormat == Format::BNSD ? tiling.qSeqCapacity : 1U;
        const uint32_t lseHeadStride = QFormat == Format::TND ? 1U : tiling.qSeqCapacity;
        AscendC::GlobalTensor<ElementStorage> gQ, gK, gV;
        AscendC::GlobalTensor<uint8_t> gQScale, gKScale, gVScale;
        gQ.SetGlobalBuffer(reinterpret_cast<__gm__ ElementStorage*>(params.q));
        gK.SetGlobalBuffer(reinterpret_cast<__gm__ ElementStorage*>(params.k));
        gV.SetGlobalBuffer(reinterpret_cast<__gm__ ElementStorage*>(params.v));
        gQScale.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(params.qDequantScale));
        gKScale.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(params.kDequantScale));
        AscendC::GlobalTensor<ElementLse> gLse;
        gVScale.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(params.vDequantScale));
        gLse.SetGlobalBuffer(reinterpret_cast<__gm__ ElementLse*>(params.softmaxLse));
        AscendC::GlobalTensor<ElementO> gO;
        gO.SetGlobalBuffer(reinterpret_cast<__gm__ ElementO*>(params.o));
        AscendC::GlobalTensor<int64_t> cuQ, cuKv;
        cuQ.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(params.cuSeqLengths));
        cuKv.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(params.cuSeqLengthsKv));
        AscendC::GlobalTensor<int32_t> gSequsedQ, gSequsedKv, gSparseIdx, gSparseCount, gBlockTable;
        gSequsedQ.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(params.sequsedQ));
        gSequsedKv.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(params.sequsedKv));
        gSparseIdx.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(params.sparseBlockIdx));
        gSparseCount.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(params.sparseBlockCount));
        gBlockTable.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(params.blockTable));
        AscendC::GlobalTensor<float> gPartialN, gPartialStats;
        if (flashDecode) {
            gPartialN.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(params.workspace + tiling.fdPartialOOffset));
            gPartialStats.SetGlobalBuffer(
                reinterpret_cast<__gm__ float*>(params.workspace + tiling.fdPartialLseOffset));
        }
        AscendC::GlobalTensor<uint64_t> gKvOffsets;
        gKvOffsets.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(params.workspace + tiling.kvOffsetInfoOffset) +
                                   static_cast<uint64_t>(core) * Mxfp4::DESCRIPTOR_WORKSPACE_SLOTS *
                                       tiling.kvOffsetTileWords);
        auto ubSLayout = tla::MakeLayout<float, NpuArch::layout::RowMajor>(VecRows, KV);
        auto ubPVLayout = tla::MakeLayout<float, NpuArch::layout::RowMajor>(VecRows, HeadDim);
        auto gQLayout = tla::MakeLayout(tla::MakeShape(tiling.groupSize, HeadDim / 2),
                                        tla::MakeStride(qHeadStride * HeadDim / 2, tla::Int<1>{}));
        auto gKVLayout = MakeKvLayout(tiling);
#ifdef __DAV_CUBE__
        BlockMmadQK blockMmadQK(resource, l1, descriptor);
        BlockMmadPV blockMmadPV(resource, l1, descriptor);
        for (uint32_t slot = 0; slot < Mxfp4::DESCRIPTOR_STAGES; ++slot) {
            Mxfp4::CubePublishBoth<PIPE_MTE1>(static_cast<uint32_t>(Mxfp4::Flag::DESCRIPTOR_BASE) + slot);
        }
        for (uint32_t slot = 0; slot < Mxfp4::P_STAGES; ++slot) {
            Mxfp4::CubePublishBoth<PIPE_MTE1>(static_cast<uint32_t>(Mxfp4::Flag::P_BASE) + slot);
        }
#endif
#ifdef __DAV_VEC__
        EpilogueOnlineSoftmax epilogueOnlineSoftmax(resource, ub, tiling.log2Cx);
        EpilogueComputePScale epilogueComputePScale(resource, ub, tiling.log2CxCeil);
        EpilogueRescaleO epilogueRescaleO(resource, ub, tiling.log2CxCeil);
        EpiloguePrepareKvScale prepareKvScale(resource, l1, ub, descriptor);
        NpuArch::Epilogue::Block::UBufTileHelper offsetsLayout{};
        offsetsLayout.maskStartOffset = ub.LOCAL_DESC_OFFSET;
        // EndGroup consumes local records before publication. A single group
        // per bank can reuse one buffer: MTE3_S event 5 protects its DMA source.
        // QK and delayed PV retain the separately protected GM descriptor slots.
        offsetsLayout.kvOffsetUbBufNum = descriptor.LOCAL_DESC_SLOTS;
        EpiloguePrepareOffsets prepareOffsets(resource, offsetsLayout, KV, descriptor.ROWS, tiling.blockShapeY,
                                              tiling.blockSize, tokenStrideBytes, tiling.kStride0 / 2U, HeadDim / 2U);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID5);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID6);
        InitializeVectorMailboxes();
        // sequsedQ leaves storage padding outside the metadata task space.
        if (params.sequsedQ != nullptr && !allQStored) {
            uint32_t paddingTask = 0U;
            for (uint32_t batch = 0U; batch < tiling.batch; ++batch) {
                BatchSeqInfo qSeq{};
                ReadBatchSeqInfo<QFormat>(cuQ, gSequsedQ, true, batch, qSeq, tiling.qSeqCapacity);
                const uint32_t length = qSeq.storageLen;
                const uint32_t actual = qSeq.actualLen;
                for (uint32_t token = actual; token < length; ++token) {
                    for (uint32_t head = 0U; head < tiling.kvHeads; ++head, ++paddingTask) {
                        if (paddingTask % coreNum != core) {
                            continue;
                        }
                        const uint32_t qHead = head * tiling.groupSize + rowOffset;
                        const uint64_t outputOffset =
                            GetTensorOffset<QFormat>(qSeq, token, qHead, tiling.numHeads, HeadDim);
                        const uint64_t lseOffset = GetLseOffset<QFormat>(qSeq, token, qHead, tiling.numHeads);
                        auto taskLse = gLse;
                        if (writeLse) {
                            taskLse = gLse[lseOffset];
                        }
                        epilogueRescaleO.StoreEmpty(gO[outputOffset], taskLse, validRows, writeLse, qHeadStride,
                                                    lseHeadStride);
                    }
                }
            }
        }
#endif
#ifdef __DAV_VEC__
        epilogueOnlineSoftmax.InitTask(validRows, tiling.scaleValue);
#endif
        using Tile = FdPipelineTile;
        Tile pending[DirectTask ? 1U : Mxfp4::P_STAGES];
        uint32_t sequence = 0U;
        for (uint32_t taskId = taskStart; taskId < taskEnd; taskId += taskStep) {
            Mxfp4::TaskInfo task{};
            uint32_t visible = 0U;
            if (!DecodeTask(taskId, params, tiling, cuQ, cuKv, gSequsedQ, gSequsedKv, gSparseIdx, gSparseCount,
                            gBlockTable, task, visible)) {
                if constexpr (DirectTask) {
                    break;
                }
                continue;
            }
            int32_t partialId = -1;
            if (!DirectTask && flashDecode) {
                uint32_t id = 0U, count = 0U;
                if (GsaFd::FindPartialTask(meta, taskId, core, id, count)) {
                    partialId = static_cast<int32_t>(id);
                }
                const uint32_t blocks =
                    task.explicitBlocks + static_cast<uint32_t>(HasResidualBlock && visible % tiling.blockShapeY != 0U);
                const uint32_t begin = taskId == taskStart ? min(firstBlock, blocks) : 0U;
                const uint32_t end = taskId + 1U == taskEnd ? min(lastBlock, blocks) : blocks;
                auto indices = gSparseIdx[task.sparseOffset];
                task.kvTokens = 0U;
                CalcSparseKvSize<HasResidualBlock>(indices, task.explicitBlocks, begin, end, tiling.blockShapeY,
                                                   visible, task.kvTokens);
                task.sparseOffset += min(begin, task.explicitBlocks);
                task.explicitBlocks = min(end, task.explicitBlocks) - min(begin, task.explicitBlocks);
                task.tileCount = (task.kvTokens + KV - 1U) / KV;
            }
            const uint32_t taskBegin = sequence;
            if (task.tileCount != 0U) {
#ifdef __DAV_CUBE__
                auto query = tla::MakeTensor(gQ[task.qByteOffset], gQLayout, NpuArch::Arch::PositionGM{});
                blockMmadQK.LoadQuery(query, gQScale[task.qByteOffset / 16U], tiling.groupSize);
                blockMmadPV.InitializeValueScale(
                    gVScale, task.vScaleOffset, task.vScalePairs,
                    KvFormat == Format::BNSD ? HeadDim * 2U : tiling.kvHeads * HeadDim * 2U);
#endif
#ifdef __DAV_VEC__
                for (uint32_t tile = 0; tile < (DirectTask ? 1U : min(task.tileCount, Mxfp4::DESCRIPTOR_PRE_LAUNCH));
                     ++tile) {
                    PrepareTile(prepareOffsets, prepareKvScale, gKScale, gKvOffsets, gSparseIdx, gBlockTable, tiling,
                                task, tile, taskBegin);
                }
#endif
            }
            // Empty tasks occupy one queue position so their output is cleared
            // only after all preceding PV consumers have finished.
            for (uint32_t tileId = 0; tileId < (DirectTask ? 1U : max(task.tileCount, 1U)); ++tileId, ++sequence) {
                auto& item = pending[DirectTask ? 0U : sequence % Mxfp4::P_STAGES];
                // Keep only the fields needed by the delayed PV consumer.
                // Query addresses, sparse-list cursors and batch decoding
                // state belong to the producer and need no queue copy.
#ifdef __DAV_CUBE__
                item.kvByteOffset = task.kvByteOffset;
                item.vScaleOffset = task.vScaleOffset;
                item.kvTokens = task.kvTokens;
                item.vScalePairs = task.vScalePairs;
#endif
#ifdef __DAV_VEC__
                item.outputOffset = task.outputOffset;
                item.lseOffset = task.lseOffset;
                item.partialId = partialId;
#endif
                item.tileCount = task.tileCount;
                item.tileId = tileId;
                if (task.tileCount != 0U) {
                    const uint32_t sSlot = sequence % 2U;
                    const uint32_t pSlot = sequence % Mxfp4::P_STAGES;
                    const uint32_t descriptorSlot = sequence % Mxfp4::DESCRIPTOR_STAGES;
                    const uint32_t tokens = min(KV, task.kvTokens - tileId * KV);
#ifdef __DAV_CUBE__
                    auto key = tla::MakeTensor(gK[task.kvByteOffset], gKVLayout, NpuArch::Arch::PositionGM{});
                    auto descriptor = gKvOffsets[static_cast<uint64_t>(descriptorSlot) * tiling.kvOffsetTileWords];
                    if (tileId == 0U) {
                        Mxfp4::CubeWaitBoth<PIPE_S>(static_cast<uint32_t>(Mxfp4::Flag::DESCRIPTOR_BASE) +
                                                    descriptorSlot);
                        blockMmadQK.LoadKey(key, gKScale[task.kvByteOffset / 16U], descriptor, groupWords, tokens,
                                            sequence);
                    }
                    Mxfp4::CubeWaitBoth<PIPE_FIX>(sSlot);
                    auto score = tla::MakeTensor(ubSTensor[sSlot], ubSLayout, NpuArch::Arch::PositionUB{});
                    blockMmadQK(key, gKScale[task.kvByteOffset / 16U], descriptor, groupWords, tokens, score, sequence);
                    Mxfp4::CubePublishBoth<PIPE_FIX>(sSlot);
                    if (!DirectTask && tileId + 1U < task.tileCount) {
                        const uint32_t nextSlot = (sequence + 1U) % Mxfp4::DESCRIPTOR_STAGES;
                        Mxfp4::CubeWaitBoth<PIPE_S>(static_cast<uint32_t>(Mxfp4::Flag::DESCRIPTOR_BASE) + nextSlot);
                        blockMmadQK.LoadKey(key, gKScale[task.kvByteOffset / 16U],
                                            gKvOffsets[static_cast<uint64_t>(nextSlot) * tiling.kvOffsetTileWords],
                                            groupWords, min(KV, task.kvTokens - (tileId + 1U) * KV), sequence + 1U);
                    }
#endif
#ifdef __DAV_VEC__
                    if (PreferScaleLookahead(HeadDim, tiling.blockShapeY)) {
                        // Keep the first softmax on the critical path for a single
                        // task. A stream of tasks can prepare the next tile's scale
                        // before softmax and overlap it with Cube consumption.
                        if (scaleLookahead != 0U && tileId == 0U) {
                            prepareKvScale.PrepareTile(tokens, bank, tiling.blockShapeY, sequence);
                        }
                        const uint32_t scaleTile = tileId + scaleLookahead;
                        if (scaleTile < task.tileCount) {
                            prepareKvScale.PrepareTile(min(KV, task.kvTokens - scaleTile * KV), bank,
                                                       tiling.blockShapeY, sequence + scaleLookahead);
                        }
                    } else {
                        prepareKvScale.PrepareTile(tokens, bank, tiling.blockShapeY, sequence);
                    }
                    if (!DirectTask && tileId + Mxfp4::DESCRIPTOR_PRE_LAUNCH < task.tileCount) {
                        PrepareTile(prepareOffsets, prepareKvScale, gKScale, gKvOffsets, gSparseIdx, gBlockTable,
                                    tiling, task, tileId + Mxfp4::DESCRIPTOR_PRE_LAUNCH, taskBegin);
                    }
                    AscendC::CrossCoreWaitFlag<4, PIPE_V>(sSlot);
                    AscendC::CrossCoreWaitFlag<4, PIPE_V>(static_cast<uint32_t>(Mxfp4::Flag::P_BASE) + pSlot);
                    const uint32_t statsSlot = sequence % Mxfp4::STATS_STAGES;
                    const auto& previousMax = epilogueComputePScale.BeginTile(statsSlot, tileId == 0U);
                    const uint32_t pRowOffset = bank * ((rows0 + 15U) / 16U * 16U);
                    epilogueOnlineSoftmax(sSlot, sSlot, statsSlot, tokens, l1PTensor[pSlot], pRowOffset,
                                          tiling.groupSize, previousMax);
                    AscendC::CrossCoreSetFlag<4, PIPE_V>(sSlot);
                    epilogueComputePScale(statsSlot, l1PScaleTensor[pSlot], bank, validRows, tiling.groupSize, tokens);
                    AscendC::CrossCoreSetFlag<4, PIPE_MTE3>(static_cast<uint32_t>(Mxfp4::Flag::P_BASE) + pSlot);
#endif
                }
                if constexpr (DirectTask) {
#ifdef __DAV_CUBE__
                    PrefetchValue(item, 0U, blockMmadPV, gV, gKvOffsets, tiling, groupWords);
                    ConsumeTile(item, 0U, blockMmadPV, gV, gVScale, gKvOffsets, tiling, groupWords, ubPVLayout);
#endif
#ifdef __DAV_VEC__
                    ConsumeTile(item, 0U, epilogueRescaleO, epilogueOnlineSoftmax, prepareKvScale, gO, gLse, gPartialN,
                                gPartialStats, tiling, validRows, rowOffset, false);
#endif
                    break;
                }
#ifdef __DAV_CUBE__
                // The first PV only needs V0. Queue its L0/MMAD before
                // parsing and issuing the future V1/V2 descriptor lists.
                if (sequence == Mxfp4::PRE_LAUNCH) {
                    PrefetchValue(pending[0], 0U, blockMmadPV, gV, gKvOffsets, tiling, groupWords);
                }
#endif
                if (sequence >= Mxfp4::PRE_LAUNCH) {
                    const uint32_t consumer = sequence - Mxfp4::PRE_LAUNCH;
#ifdef __DAV_CUBE__
                    ConsumeTile(pending[consumer % Mxfp4::P_STAGES], consumer, blockMmadPV, gV, gVScale, gKvOffsets,
                                tiling, groupWords, ubPVLayout);
#endif
#ifdef __DAV_VEC__
                    ConsumeTile(pending[consumer % Mxfp4::P_STAGES], consumer, epilogueRescaleO, epilogueOnlineSoftmax,
                                prepareKvScale, gO, gLse, gPartialN, gPartialStats, tiling, validRows, rowOffset,
                                flashDecode);
#endif
                }
#ifdef __DAV_CUBE__
                if (sequence == Mxfp4::PRE_LAUNCH) {
                    PrefetchValue(pending[1], 1U, blockMmadPV, gV, gKvOffsets, tiling, groupWords);
                }
                if (sequence >= Mxfp4::PRE_LAUNCH) {
                    PrefetchValue(pending[sequence % Mxfp4::P_STAGES], sequence, blockMmadPV, gV, gKvOffsets, tiling,
                                  groupWords);
                }
#endif
                // UB reuse is protected at each consumer; future descriptor
                // work need not wait for this iteration's entire Vector queue.
            }
            if constexpr (DirectTask) {
                break;
            }
        }
        // Only the end of the whole core stream drains the remaining tiles.
        for (uint32_t consumer = sequence > Mxfp4::PRE_LAUNCH ? sequence - Mxfp4::PRE_LAUNCH : 0U;
             !DirectTask && consumer < sequence; ++consumer) {
#ifdef __DAV_CUBE__
            if (sequence <= Mxfp4::PRE_LAUNCH) {
                PrefetchValue(pending[consumer % Mxfp4::P_STAGES], consumer, blockMmadPV, gV, gKvOffsets, tiling,
                              groupWords);
            }
            ConsumeTile(pending[consumer % Mxfp4::P_STAGES], consumer, blockMmadPV, gV, gVScale, gKvOffsets, tiling,
                        groupWords, ubPVLayout);
#endif
#ifdef __DAV_VEC__
            ConsumeTile(pending[consumer % Mxfp4::P_STAGES], consumer, epilogueRescaleO, epilogueOnlineSoftmax,
                        prepareKvScale, gO, gLse, gPartialN, gPartialStats, tiling, validRows, rowOffset, flashDecode);
#endif
        }
#ifdef __DAV_CUBE__
        Mxfp4::CubeWaitBoth<PIPE_FIX>(0U);
        Mxfp4::CubeWaitBoth<PIPE_FIX>(1U);
        for (uint32_t slot = 0; slot < ub.PV_SLOTS; ++slot) {
            Mxfp4::CubeWaitBoth<PIPE_FIX>(static_cast<uint32_t>(Mxfp4::Flag::PV) + slot);
        }
        blockMmadQK.FreeEventID();
        blockMmadPV.FreeEventID();
#endif
#ifdef __DAV_VEC__
        for (uint32_t slot = 0; slot < Mxfp4::DESCRIPTOR_STAGES; ++slot) {
            AscendC::CrossCoreWaitFlag<4, PIPE_S>(static_cast<uint32_t>(Mxfp4::Flag::DESCRIPTOR_BASE) + slot);
        }
        for (uint32_t slot = 0; slot < Mxfp4::P_STAGES; ++slot) {
            AscendC::CrossCoreWaitFlag<4, PIPE_V>(static_cast<uint32_t>(Mxfp4::Flag::P_BASE) + slot);
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID5);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID6);
#endif
        AscendC::PipeBarrier<PIPE_ALL>();
    }

#ifdef __DAV_CUBE__
    template <class Tile>
    __aicore__ inline void PrefetchValue(const Tile& item, uint32_t sequence, BlockMmadPV& block,
                                         const AscendC::GlobalTensor<ElementStorage>& value,
                                         const AscendC::GlobalTensor<uint64_t>& descriptors,
                                         const GenericBlockSparseAttn::GenericBlockSparseAttentionTilingData& tiling,
                                         uint32_t groupWords)
    {
        if (item.tileCount == 0U) {
            return;
        }
        const uint32_t slot = sequence % Mxfp4::DESCRIPTOR_STAGES;
        auto layout = MakeKvLayout(tiling);
        auto tensor = tla::MakeTensor(value[item.kvByteOffset], layout, NpuArch::Arch::PositionGM{});
        block.LoadValue(tensor, descriptors[static_cast<uint64_t>(slot) * tiling.kvOffsetTileWords], groupWords,
                        min(KV, item.kvTokens - item.tileId * KV), sequence);
    }

    template <class PVLayout, class Tile>
    __aicore__ inline void ConsumeTile(const Tile& item, uint32_t sequence, BlockMmadPV& block,
                                       const AscendC::GlobalTensor<ElementStorage>& value,
                                       const AscendC::GlobalTensor<uint8_t>& scale,
                                       const AscendC::GlobalTensor<uint64_t>& descriptors,
                                       const GenericBlockSparseAttn::GenericBlockSparseAttentionTilingData& tiling,
                                       uint32_t groupWords, const PVLayout& pvLayout)
    {
        const auto& task = item;
        if (task.tileCount == 0U) {
            return;
        }
        const uint32_t pSlot = sequence % Mxfp4::P_STAGES;
        const uint32_t descriptorSlot = sequence % Mxfp4::DESCRIPTOR_STAGES;
        const uint32_t pvSlot = sequence % ub.PV_SLOTS;
        const uint32_t pvFlag = static_cast<uint32_t>(Mxfp4::Flag::PV) + pvSlot;
        if (item.tileId == 0U) {
            block.LoadValueScale(scale, task.vScaleOffset, task.vScalePairs,
                                 KvFormat == Format::BNSD ? HeadDim * 2U : tiling.kvHeads * HeadDim * 2U);
        }
        Mxfp4::CubeWaitBoth<PIPE_MTE1>(static_cast<uint32_t>(Mxfp4::Flag::P_BASE) + pSlot);
        auto valueLayout = MakeKvLayout(tiling);
        auto tensor = tla::MakeTensor(value[task.kvByteOffset], valueLayout, NpuArch::Arch::PositionGM{});
        auto output = tla::MakeTensor(ubOTmpTensor[pvSlot], pvLayout, NpuArch::Arch::PositionUB{});
        block(tensor, descriptors[static_cast<uint64_t>(descriptorSlot) * tiling.kvOffsetTileWords], groupWords,
              min(KV, task.kvTokens - item.tileId * KV), sequence, tiling.groupSize, output, pvFlag);
        Mxfp4::CubePublishBoth<PIPE_MTE1>(static_cast<uint32_t>(Mxfp4::Flag::DESCRIPTOR_BASE) + descriptorSlot);
        Mxfp4::CubePublishBoth<PIPE_MTE1>(static_cast<uint32_t>(Mxfp4::Flag::P_BASE) + pSlot);
        Mxfp4::CubePublishBoth<PIPE_FIX>(pvFlag);
    }
#endif
#ifdef __DAV_VEC__
    template <class Tile>
    __aicore__ inline void ConsumeTile(const Tile& item, uint32_t sequence, EpilogueRescaleO& epilogue,
                                       EpilogueOnlineSoftmax& frontend, EpiloguePrepareKvScale& prepareScale,
                                       const AscendC::GlobalTensor<ElementO>& output,
                                       const AscendC::GlobalTensor<ElementLse>& lse,
                                       const AscendC::GlobalTensor<float>& partialN,
                                       const AscendC::GlobalTensor<float>& partialStats,
                                       const GenericBlockSparseAttn::GenericBlockSparseAttentionTilingData& tiling,
                                       uint32_t rows, uint32_t rowOffset, bool flashDecode)
    {
        const auto& task = item;
        if (task.tileCount <= 1U && (!flashDecode || item.partialId < 0)) {
            const uint32_t headStride = QFormat == Format::BNSD ? tiling.qSeqCapacity : 1U;
            const uint32_t lseStride = QFormat == Format::TND ? 1U : tiling.qSeqCapacity;
            auto taskOutput = output[task.outputOffset + rowOffset * headStride * HeadDim];
            auto taskLse = lse;
            if (writeLse) {
                taskLse = lse[task.lseOffset + rowOffset * lseStride];
            }
            if (task.tileCount == 0U) {
                epilogue.StoreEmpty(taskOutput, taskLse, rows, writeLse, headStride, lseStride);
            } else {
                const uint32_t pvSlot = sequence % ub.PV_SLOTS;
                const uint32_t flag = static_cast<uint32_t>(Mxfp4::Flag::PV) + pvSlot;
                AscendC::CrossCoreWaitFlag<4, PIPE_V>(flag);
                epilogue.StoreTile(taskOutput, taskLse, rows, writeLse, headStride, lseStride,
                                   sequence % Mxfp4::STATS_STAGES, pvSlot);
                AscendC::CrossCoreSetFlag<4, PIPE_V>(flag);
            }
            frontend.InvalidatePadding();
            return;
        }
        if (item.tileId == 0U) {
            epilogue.BeginTask(rows, flashDecode);
        }
        if (task.tileCount != 0U) {
            const uint32_t pvSlot = sequence % ub.PV_SLOTS;
            const uint32_t flag = static_cast<uint32_t>(Mxfp4::Flag::PV) + pvSlot;
            AscendC::CrossCoreWaitFlag<4, PIPE_V>(flag);
            const bool saveMax =
                item.tileId + 1U == task.tileCount && (writeLse || (flashDecode && item.partialId >= 0));
            epilogue.ConsumeTile(sequence % Mxfp4::STATS_STAGES, pvSlot, rows, saveMax);
            AscendC::CrossCoreSetFlag<4, PIPE_V>(flag);
        }
        if (item.tileId + 1U >= task.tileCount) {
            if (flashDecode) {
                if (item.partialId >= 0) {
                    const uint64_t partial = static_cast<uint32_t>(item.partialId);
                    const uint32_t bank = AscendC::GetSubBlockIdx();
                    epilogue.StorePartial(partialN[(partial * tiling.groupSize + rowOffset) * HeadDim],
                                          partialStats[(partial * 4U + bank) * tiling.fdLseSubStride], rows,
                                          tiling.fdLseSubStride);
                    return;
                }
            }
            // LSE has its own source buffer; output conversion retains the
            // frontend's MTE3->V credits while reusing P storage.
            auto taskLse = lse;
            if (writeLse) {
                taskLse = lse[task.lseOffset + rowOffset * (QFormat == Format::TND ? 1U : tiling.qSeqCapacity)];
            }
            const uint32_t headStride = QFormat == Format::BNSD ? tiling.qSeqCapacity : 1U;
            epilogue.Store(output[task.outputOffset + rowOffset * headStride * HeadDim], taskLse, rows, writeLse,
                           headStride, QFormat == Format::TND ? 1U : tiling.qSeqCapacity);
            // NormalizeCallVF has overwritten P storage, including padding.
            frontend.InvalidatePadding();
        }
    }
#endif

    bool allQStored = false;
    bool reuseQuerySeq = false;
    bool writeLse = false;
    __aicore__ inline bool DecodeTask(uint32_t taskId, const Mxfp4::KernelParams& params,
                                      const GenericBlockSparseAttn::GenericBlockSparseAttentionTilingData& tiling,
                                      AscendC::GlobalTensor<int64_t>& cuQ, AscendC::GlobalTensor<int64_t>& cuKv,
                                      AscendC::GlobalTensor<int32_t>& gSequsedQ,
                                      AscendC::GlobalTensor<int32_t>& gSequsedKv,
                                      AscendC::GlobalTensor<int32_t>& gSparseIdx,
                                      AscendC::GlobalTensor<int32_t>& gSparseCount,
                                      AscendC::GlobalTensor<int32_t>& blockTable, Mxfp4::TaskInfo& task,
                                      uint32_t& visible)
    {
        BatchSeqInfo qSeq{}, kvSeq{};
        task.kvHead = taskId % tiling.kvHeads;
        const uint32_t qToken = taskId / tiling.kvHeads;
        if constexpr (QFormat == Format::TND) {
            if (reuseQuerySeq) {
                const uint32_t average = max(static_cast<uint32_t>(tiling.totalQTokens / tiling.batch), 1U);
                uint32_t b = min(qToken / average, static_cast<uint32_t>(tiling.batch - 1U));
                ReadBatchSeqInfo<QFormat>(cuQ, gSequsedQ, false, b, qSeq);
                if (qToken < qSeq.storageStart || qToken >= qSeq.storageStart + qSeq.storageLen) {
                    uint32_t left = 0U;
                    uint32_t right = tiling.batch;
                    while (left + 1 < right) {
                        const uint32_t mid = left + (right - left) / 2U;
                        if (static_cast<uint64_t>(cuQ.GetValue(mid)) <= qToken) {
                            left = mid;
                        } else {
                            right = mid;
                        }
                    }
                    b = left;
                    ReadBatchSeqInfo<QFormat>(cuQ, gSequsedQ, false, b, qSeq);
                }
                task.batch = b;
                task.qTokenInBatch = qToken - qSeq.storageStart;
            } else {
                uint32_t prefix = 0U;
                for (uint32_t batch = 0U; batch < tiling.batch; ++batch) {
                    ReadBatchSeqInfo<QFormat>(cuQ, gSequsedQ, params.sequsedQ != nullptr, batch, qSeq);
                    if (qToken < prefix + qSeq.actualLen) {
                        task.batch = batch;
                        task.qTokenInBatch = qToken - prefix;
                        break;
                    }
                    prefix += qSeq.actualLen;
                }
            }
        } else {
            DecodeBatchedQToken(qToken, tiling.batch, tiling.qSeqCapacity, gSequsedQ, task.batch, task.qTokenInBatch,
                                qSeq);
        }
        task.qStorageToken = qSeq.storageStart + task.qTokenInBatch;
        if constexpr (IsPaged) {
            kvSeq.actualLen = static_cast<uint32_t>(gSequsedKv.GetValue(task.batch));
        } else {
            ReadBatchSeqInfo<KvFormat>(cuKv, gSequsedKv, params.sequsedKv != nullptr, task.batch, kvSeq,
                                       tiling.kvSeqCapacity);
        }
        const uint32_t qHead = task.kvHead * tiling.groupSize;
        task.qByteOffset = GetTensorOffset<QFormat>(qSeq, task.qTokenInBatch, qHead, tiling.numHeads, HeadDim) / 2;
        task.outputOffset = task.qByteOffset * 2;
        task.lseOffset = GetLseOffset<QFormat>(qSeq, task.qTokenInBatch, qHead, tiling.numHeads);
        if constexpr (IsPaged) {
            task.kvByteOffset = task.kvHead * GbsaLayoutHelper<KvFormat>::HeadStride(tiling.blockSize, HeadDim / 2);
            task.vScalePairs = tiling.blockSize / 64U;
            if (kvSeq.actualLen != 0U) {
                const uint64_t page = blockTable.GetValue(static_cast<uint64_t>(task.batch) * tiling.maxBlocksPerBatch);
                task.vScaleOffset = (page * task.vScalePairs * tiling.kvHeads +
                                     task.kvHead * (KvFormat == Format::BNSD ? task.vScalePairs : 1U)) *
                                    HeadDim * 2U;
            }
        } else {
            task.kvByteOffset = GetTensorOffset<KvFormat>(kvSeq, 0, task.kvHead, tiling.kvHeads, HeadDim) / 2;
            task.vScalePairs = (kvSeq.storageLen + 63U) / 64U;
            uint64_t pairBase = static_cast<uint64_t>(task.batch) * task.vScalePairs;
            if constexpr (KvFormat == Format::TND) {
                pairBase = 0U;
                for (uint32_t batch = 0U; batch < task.batch; ++batch) {
                    const uint64_t length = cuKv.GetValue(batch + 1U) - cuKv.GetValue(batch);
                    pairBase += (length + 63U) / 64U;
                }
            }
            task.vScaleOffset =
                (pairBase * tiling.kvHeads + task.kvHead * (KvFormat == Format::BNSD ? task.vScalePairs : 1U)) *
                HeadDim * 2U;
        }
        task.sparseOffset = (static_cast<uint64_t>(task.kvHead) * tiling.qBlockNum + task.qStorageToken) * tiling.topK;
        task.explicitBlocks = static_cast<uint32_t>(
            gSparseCount.GetValue(static_cast<uint64_t>(task.kvHead) * tiling.qBlockNum + task.qStorageToken));
        visible = kvSeq.actualLen;
        if (visible == 0U || (IsCausal && visible < qSeq.actualLen)) {
            visible = 0U;
            task.explicitBlocks = 0U;
            return true;
        }
        if constexpr (IsCausal) {
            visible = visible - qSeq.actualLen + task.qTokenInBatch + 1U;
        }
        task.residualBlock = visible / tiling.blockShapeY;
        uint32_t blocks = task.explicitBlocks;
        if constexpr (HasResidualBlock) {
            blocks += static_cast<uint32_t>(visible % tiling.blockShapeY != 0U);
        }
        auto indices = gSparseIdx[task.sparseOffset];
        CalcSparseKvSize<HasResidualBlock>(indices, task.explicitBlocks, 0U, blocks, tiling.blockShapeY, visible,
                                           task.kvTokens);
        task.tileCount = (task.kvTokens + KV - 1U) / KV;
        return true;
    }

    __aicore__ inline auto MakeKvLayout(const GenericBlockSparseAttn::GenericBlockSparseAttentionTilingData& tiling)
    {
        return tla::MakeLayout(
            tla::MakeShape(tiling.kvSeqCapacity, HeadDim / 2),
            tla::MakeStride(GbsaLayoutHelper<KvFormat>::TokenStride(tiling.kvHeads, HeadDim / 2), tla::Int<1>{}));
    }

    __aicore__ inline void InitCrossCoreDstBuf()
    {
        for (uint32_t i = 0; i < 2; ++i) {
            ubSTensor[i] = resource.ubBuf.template GetBufferByByte<float>(Ub::S_OFFSET + i * ub.S_SLOT_BYTES);
        }
        for (uint32_t i = 0; i < Mxfp4::P_STAGES; ++i) {
            l1PTensor[i] = resource.l1Buf.template GetBufferByByte<uint8_t>(l1.P_OFFSET + i * l1.P_SLOT_BYTES);
            l1PScaleTensor[i] =
                resource.l1Buf.template GetBufferByByte<uint8_t>(l1.PSCALE_OFFSET + i * l1.PSCALE_SLOT_BYTES);
        }
        for (uint32_t slot = 0; slot < ub.PV_SLOTS; ++slot) {
            ubOTmpTensor[slot] = resource.ubBuf.template GetBufferByByte<float>(ub.PV_OFFSET + slot * ub.PV_SLOT_BYTES);
        }
    }

#ifdef __DAV_VEC__
    __aicore__ inline void PrepareTile(EpiloguePrepareOffsets& prepareOffsets, EpiloguePrepareKvScale& prepareScale,
                                       const AscendC::GlobalTensor<uint8_t>& keyScale,
                                       const AscendC::GlobalTensor<uint64_t>& descriptors,
                                       const AscendC::GlobalTensor<int32_t>& sparseIndices,
                                       const AscendC::GlobalTensor<int32_t>& blockTable,
                                       const GenericBlockSparseAttn::GenericBlockSparseAttentionTilingData& tiling,
                                       const Mxfp4::TaskInfo& task, uint32_t tileId, uint32_t taskBegin)
    {
        const uint32_t slot = (taskBegin + tileId) % Mxfp4::DESCRIPTOR_STAGES;
        // The delayed PV also reads these records. A scalar wait protects GM
        // writes even when Vector instructions have not reached their waits.
        AscendC::CrossCoreWaitFlag<4, PIPE_S>(static_cast<uint32_t>(Mxfp4::Flag::DESCRIPTOR_BASE) + slot);
        const uint32_t tokens = min(KV, task.kvTokens - tileId * KV);
        prepareScale.BeginTile(keyScale[task.kvByteOffset / 16U], taskBegin + tileId, tiling.blockShapeY,
                               IsPaged ? tiling.blockSize : tiling.kvSeqCapacity,
                               GbsaLayoutHelper<KvFormat>::TokenStride(tiling.kvHeads, HeadDim / 2U));
        prepareOffsets(
            descriptors[static_cast<uint64_t>(slot) * tiling.kvOffsetTileWords],
            sparseIndices[task.sparseOffset].GetPhyAddr(),
            IsPaged ? blockTable[static_cast<uint64_t>(task.batch) * tiling.maxBlocksPerBatch].GetPhyAddr() : nullptr,
            tileId, tokens, task.explicitBlocks, task.residualBlock, prepareScale);
        AscendC::CrossCoreSetFlag<4, PIPE_MTE3>(static_cast<uint32_t>(Mxfp4::Flag::DESCRIPTOR_BASE) + slot);
    }

    __aicore__ inline void InitializeVectorMailboxes()
    {
        AscendC::CrossCoreSetFlag<4, PIPE_V>(0U);
        AscendC::CrossCoreSetFlag<4, PIPE_V>(1U);
        for (uint32_t slot = 0; slot < ub.PV_SLOTS; ++slot) {
            AscendC::CrossCoreSetFlag<4, PIPE_V>(static_cast<uint32_t>(Mxfp4::Flag::PV) + slot);
        }
    }

#endif
    Resource resource;
    AscendC::LocalTensor<float> ubSTensor[2], ubOTmpTensor[2];
    AscendC::LocalTensor<uint8_t> l1PTensor[Mxfp4::P_STAGES], l1PScaleTensor[Mxfp4::P_STAGES];
};

} // namespace GbsaKernelArch35
#endif // GENERIC_BLOCK_SPARSE_ATTENTION_KERNEL_ARCH35_MXFP4_FULL_QUANT_H
