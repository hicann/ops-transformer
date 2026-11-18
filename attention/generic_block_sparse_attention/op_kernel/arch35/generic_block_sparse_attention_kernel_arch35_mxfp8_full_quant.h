/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GENERIC_BLOCK_SPARSE_ATTENTION_KERNEL_ARCH35_MXFP8_FULL_QUANT_H
#define GENERIC_BLOCK_SPARSE_ATTENTION_KERNEL_ARCH35_MXFP8_FULL_QUANT_H

#include "kernel_utils.hpp"
#include "../generic_block_sparse_attention_fd_utils.h"
#include "gbsa_layout_utils.hpp"
#include "generic_block_sparse_attention_fd_combine_arch35.h"

#include "../tla/gbsa_tla_tensor.hpp"

namespace GbsaKernelArch35 {

template <class BlockMmadQK, class BlockMmadPV, class EpilogueOnlineSoftmax, class EpilogueRescaleO,
          class EpiloguePrepareOffsets, class EpiloguePrepareKScale, Format QFormat, Format KvFormat,
          bool HasResidualBlock, bool OutputLse, uint32_t QueryCapacity = 32U>
class GbsaMxfp8FullTileKernelArch35 {
private:
    // Overall allocation and cross-component aliasing belong to the Kernel.
    static constexpr uint32_t SHARED_L0B_EVENT_BASE = 2U;
    static constexpr uint32_t QUANT_TYPE = 6U;
    static constexpr uint32_t PRE_LAUNCH = 2U;
    static constexpr uint32_t DESCRIPTOR_PRE_LAUNCH = 2U;
    static constexpr uint32_t KSCALE_STAGES = 2U;
    static constexpr uint32_t P_STAGES = PRE_LAUNCH + 1U;
    static constexpr uint32_t STATS_STAGES = PRE_LAUNCH + 2U;
    static constexpr uint32_t DESCRIPTOR_STAGES = PRE_LAUNCH + DESCRIPTOR_PRE_LAUNCH + 1U;
    // Keep the host workspace contract; only DESCRIPTOR_STAGES slots are active.
    static constexpr uint32_t DESCRIPTOR_WORKSPACE_SLOTS = Mxfp8Descriptor::WORKSPACE_SLOTS;

    // Mode 4 maps each AIV's local flag to id / id+16 on the paired AIC.
    // S and PV flags are bidirectional mailboxes: AIV seeds free, AIC publishes
    // ready after FIX, AIV returns free after V. Descriptor and P mailboxes start
    // free on AIV; AIV publishes ready and PV returns free after its L1 loads.
    // KScale ready/free mailboxes identify two descriptor pairs in a full KV tile.
    enum class Flag : uint32_t {
        S0 = 0U,
        S1 = 1U,
        DESCRIPTOR_BASE = 2U,  // 2..6
        P_BASE = 7U,           // 7..9
        PV = 10U,              // 10..11
        KSCALE_BASE = 12U,     // 12..13
        KSCALE_FREE_BASE = 14U // 14..15
    };
    static_assert(static_cast<uint32_t>(Flag::KSCALE_FREE_BASE) + KSCALE_STAGES == 16U,
                  "Cross-core flags must fit one AIV bank");

    struct TaskInfo {
        uint32_t batch;
        uint32_t kvHead;
        uint32_t qStorageToken;
        uint32_t qTokenInBatch;
        uint32_t explicitBlocks;
        uint32_t residualBlock;
        uint32_t kvTokens;
        uint32_t tileCount;
        uint64_t vScaleOffset;
        uint32_t vScalePairs;
        uint64_t qByteOffset;
        uint64_t kvByteOffset;
        uint64_t outputOffset;
        uint64_t lseOffset;
        uint64_t sparseOffset;
    };

    struct CubeLayout {
        static constexpr uint32_t QUERY_TILE = QueryCapacity, R = QueryCapacity / 2U, QUERY_ROWS = QueryCapacity;
        static constexpr uint32_t HEAD_CAPACITY = 256, KV_ROWS = QueryCapacity == 64U ? 256U : 512U, K_ROWS = KV_ROWS;
        static constexpr uint32_t Q_SLOTS = 1, K_SLOTS = 1, V_SLOTS = 2, P_SLOTS = 3, KSCALE_SLOTS = 3;
        static constexpr uint32_t Q_OFFSET = 0, Q_SLOT_BYTES = QUERY_TILE * HEAD_CAPACITY;
        static constexpr uint32_t K_OFFSET = Q_OFFSET + Q_SLOT_BYTES, K_SLOT_BYTES = KV_ROWS * HEAD_CAPACITY;
        static constexpr uint32_t P_OFFSET = K_OFFSET + K_SLOTS * K_SLOT_BYTES, P_SLOT_BYTES = QUERY_ROWS * KV_ROWS;
        static constexpr uint32_t V_OFFSET = P_OFFSET + P_SLOTS * P_SLOT_BYTES, V_SLOT_BYTES = KV_ROWS * HEAD_CAPACITY;
        static constexpr uint32_t QSCALE_OFFSET = V_OFFSET + V_SLOTS * V_SLOT_BYTES,
                                  QSCALE_SLOT_BYTES = QUERY_TILE * HEAD_CAPACITY / 32U;
        static constexpr uint32_t KSCALE_OFFSET = QSCALE_OFFSET + QSCALE_SLOT_BYTES,
                                  KSCALE_SLOT_BYTES = KV_ROWS * (HEAD_CAPACITY / 32U);
        static constexpr uint32_t TOTAL_BYTES = KSCALE_OFFSET + KSCALE_SLOTS * KSCALE_SLOT_BYTES;
        static constexpr uint32_t QK_A_OFFSET = 0, QK_A_SLOT_BYTES = 16 * 1024;
        static constexpr uint32_t PV_A_OFFSET = 32 * 1024, PV_A_SLOT_BYTES = 16 * 1024;
        // QK and PV share the physical B slots and M_MTE1 events 2/3.
        static constexpr uint32_t QK_B_OFFSET = 0, QK_B_SLOT_BYTES = 32 * 1024;
        static constexpr uint32_t PV_B_OFFSET = 0, PV_B_SLOT_BYTES = 32 * 1024;
        static constexpr uint32_t QK_C_OFFSET = 0, QK_C_SLOT_BYTES = 64 * 1024;
        static constexpr uint32_t PV_C_OFFSET = 128 * 1024, PV_C_SLOT_BYTES = 64 * 1024;
        static constexpr uint32_t QK_COLUMNS = 128, PV_REDUCTION = 256, PV_COLUMNS = 128;
    };
    struct Ub {
        static constexpr uint32_t R = QueryCapacity / 2U, KV = CubeLayout::KV_ROWS;
        static constexpr uint32_t STATS_SLOTS = 4, VECTOR_BYTES = 256, RAW_STAGES = 3;
        static constexpr uint32_t S_OFFSET = 0, S_SLOT_BYTES = R * KV * sizeof(float);
        static constexpr uint32_t PV_OFFSET = S_OFFSET + 2 * S_SLOT_BYTES, PV_SLOTS = 2;
        static constexpr uint32_t PV_SLOT_BYTES = R * 256 * sizeof(float);
        static constexpr uint32_t O_OFFSET = PV_OFFSET + PV_SLOTS * PV_SLOT_BYTES;
        static constexpr uint32_t P_OFFSET = O_OFFSET + R * 256 * sizeof(float), P_SLOT_BYTES = R * KV;
        static constexpr uint32_t TILE_G_OFFSET = P_OFFSET + 2 * P_SLOT_BYTES;
        static constexpr uint32_t TILE_DM_OFFSET = TILE_G_OFFSET + STATS_SLOTS * VECTOR_BYTES;
        static constexpr uint32_t LOCAL_ROW_SUM_OFFSET = TILE_DM_OFFSET + STATS_SLOTS * VECTOR_BYTES;
        static constexpr uint32_t OUT_G_OFFSET = LOCAL_ROW_SUM_OFFSET + STATS_SLOTS * VECTOR_BYTES;
        static constexpr uint32_t OUT_L_OFFSET = OUT_G_OFFSET + VECTOR_BYTES;
        static constexpr uint32_t TILE_MAX_OFFSET = OUT_L_OFFSET + VECTOR_BYTES;
        static constexpr uint32_t VSCALE_OFFSET = TILE_MAX_OFFSET + VECTOR_BYTES, VSCALE_BYTES = 512;
        static constexpr uint32_t LSE_OFFSET = VSCALE_OFFSET + VSCALE_BYTES;
        static constexpr uint32_t LSE_PACKED_OFFSET = LSE_OFFSET + VECTOR_BYTES;
        static constexpr uint32_t DESC_OFFSET = LSE_PACKED_OFFSET + R * 32;
        static constexpr uint32_t KSCALE_OUTPUT_OFFSET = DESC_OFFSET + RAW_STAGES * 256 * 32;
        static constexpr uint32_t KSCALE_OUTPUT_BYTES = 256 * 8;
        static constexpr uint32_t LOCAL_DESC_OFFSET = KSCALE_OUTPUT_OFFSET + KSCALE_OUTPUT_BYTES;
        static constexpr uint32_t TOTAL_BYTES = LOCAL_DESC_OFFSET + 2 * 4160;
        // Output conversion borrows P only after both MTE3 readers release it.
        static constexpr uint32_t OUTPUT_SCRATCH_OFFSET = P_OFFSET, OUTPUT_SCRATCH_BYTES = 2 * P_SLOT_BYTES;
    };
    static_assert(QueryCapacity == 32U || QueryCapacity == 64U);
    static_assert(CubeLayout::TOTAL_BYTES == (QueryCapacity == 64U ? 268800U : 463104U));
    static_assert(CubeLayout::TOTAL_BYTES <= 512U * 1024U);
    static_assert(Ub::TOTAL_BYTES == (QueryCapacity == 64U ? 220800U : 171136U));
    static_assert(Ub::TOTAL_BYTES <= 256U * 1024U);
    static_assert(CubeLayout::QUERY_TILE * CubeLayout::HEAD_CAPACITY / 32U <= CubeLayout::QSCALE_SLOT_BYTES);
    static_assert(Ub::R * 2U == CubeLayout::QUERY_TILE);
    static_assert(CubeLayout::QK_B_OFFSET == CubeLayout::PV_B_OFFSET &&
                  CubeLayout::QK_B_SLOT_BYTES == CubeLayout::PV_B_SLOT_BYTES);
    static_assert(Ub::OUTPUT_SCRATCH_OFFSET == Ub::P_OFFSET);
    static_assert(CubeLayout::P_SLOTS == P_STAGES && Ub::STATS_SLOTS == STATS_STAGES);
    static_assert(DESCRIPTOR_STAGES <= DESCRIPTOR_WORKSPACE_SLOTS);
    static_assert(CubeLayout::K_SLOTS <= BlockMmadQK::MAX_K_SLOTS);
    static_assert(CubeLayout::KSCALE_SLOTS <= BlockMmadQK::MAX_KSCALE_SLOTS);
    static_assert(CubeLayout::KSCALE_SLOTS <= EpiloguePrepareKScale::MAX_L1_SLOTS);
    static_assert(CubeLayout::V_SLOTS <= BlockMmadPV::MAX_V_SLOTS);
    static_assert(CubeLayout::P_SLOTS <= BlockMmadPV::MAX_P_SLOTS);
    static_assert(CubeLayout::PV_A_OFFSET + 2U * CubeLayout::PV_A_SLOT_BYTES <= 64U * 1024U);
    static_assert(CubeLayout::QK_B_OFFSET + 2U * CubeLayout::QK_B_SLOT_BYTES <= 64U * 1024U);
    static_assert(CubeLayout::PV_C_OFFSET + 2U * CubeLayout::PV_C_SLOT_BYTES <= 256U * 1024U);

    inline static constexpr typename BlockMmadQK::Buffers qkBuffers{CubeLayout::KSCALE_OFFSET,
                                                                    CubeLayout::KSCALE_SLOTS,
                                                                    CubeLayout::KSCALE_SLOT_BYTES,
                                                                    CubeLayout::KV_ROWS,
                                                                    CubeLayout::K_OFFSET,
                                                                    CubeLayout::K_ROWS,
                                                                    CubeLayout::K_SLOTS,
                                                                    CubeLayout::K_SLOT_BYTES,
                                                                    CubeLayout::QK_A_OFFSET,
                                                                    CubeLayout::QK_A_SLOT_BYTES,
                                                                    CubeLayout::QK_B_OFFSET,
                                                                    CubeLayout::QK_B_SLOT_BYTES,
                                                                    CubeLayout::QK_C_OFFSET,
                                                                    CubeLayout::QK_C_SLOT_BYTES,
                                                                    CubeLayout::QSCALE_OFFSET,
                                                                    CubeLayout::Q_OFFSET,
                                                                    SHARED_L0B_EVENT_BASE,
                                                                    static_cast<uint32_t>(Flag::KSCALE_BASE),
                                                                    static_cast<uint32_t>(Flag::KSCALE_FREE_BASE)};
    static_assert(qkBuffers.K_SLOTS == CubeLayout::K_SLOTS && qkBuffers.KSCALE_SLOTS == CubeLayout::KSCALE_SLOTS &&
                  qkBuffers.Q_OFFSET == CubeLayout::Q_OFFSET && qkBuffers.KV_ROWS == CubeLayout::KV_ROWS);
    inline static constexpr typename BlockMmadPV::Buffers pvBuffers{
        CubeLayout::KV_ROWS,         CubeLayout::PV_A_OFFSET, CubeLayout::PV_A_SLOT_BYTES, CubeLayout::PV_B_OFFSET,
        CubeLayout::PV_B_SLOT_BYTES, CubeLayout::PV_COLUMNS,  CubeLayout::PV_C_OFFSET,     CubeLayout::PV_C_SLOT_BYTES,
        CubeLayout::PV_REDUCTION,    CubeLayout::P_OFFSET,    CubeLayout::P_SLOTS,         CubeLayout::P_SLOT_BYTES,
        CubeLayout::V_OFFSET,        CubeLayout::V_SLOTS,     CubeLayout::V_SLOT_BYTES,    SHARED_L0B_EVENT_BASE};
    static_assert(pvBuffers.P_SLOTS == P_STAGES && pvBuffers.V_SLOTS == CubeLayout::V_SLOTS &&
                  pvBuffers.PV_REDUCTION == CubeLayout::PV_REDUCTION);
    using BoundQK = typename BlockMmadQK::template WithBuffers<&qkBuffers>;
    using BoundPV = typename BlockMmadPV::template WithBuffers<&pvBuffers>;

    struct Descriptor {
        uint32_t ROWS, GROUPS, LOCAL_DESC_SLOTS;
        __aicore__ explicit Descriptor(uint32_t d)
            : ROWS(Mxfp8Descriptor::Rows(d, BlockMmadQK::DispatchPolicy::PAGED_CACHE_FLAG, CubeLayout::KV_ROWS)),
              GROUPS(CubeLayout::KV_ROWS / ROWS),
              LOCAL_DESC_SLOTS(GROUPS / 2U)
        {}
    };

public:
    using ArchTag = typename BlockMmadQK::ArchTag;
    using Resource = NpuArch::Arch::Resource<ArchTag>;
    // FP8 storage and descriptor offsets use one byte per logical element.
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
    static constexpr uint32_t QUERY_TILE = CubeLayout::QUERY_TILE;
    Ub ub;
    CubeLayout l1;
    Descriptor descriptor;
    __aicore__ GbsaMxfp8FullTileKernelArch35(uint32_t d, uint32_t q = QUERY_TILE)
        : HeadDim(d),
          VecRows(Ub::R),
          QueryTile(q),
          KV(CubeLayout::KV_ROWS),
          descriptor(d)
    {}

    __aicore__ inline void operator()(const Mxfp8Full::KernelParams& params)
    {
        GET_TILING_DATA_WITH_STRUCT(GenericBlockSparseAttn::GenericBlockSparseAttentionTilingData, tiling,
                                    params.tiling);
        const auto* meta = reinterpret_cast<__gm__ const GbsaMetadata::Metadata*>(params.metaData);
        isCausal_ = tiling.maskType != 0U;
        const bool validMetadata =
            isCausal_ ? GsaFd::ValidateMetadata<true, QUANT_TYPE>(meta, &tiling, AscendC::GetBlockNum()) :
                        GsaFd::ValidateMetadata<false, QUANT_TYPE>(meta, &tiling, AscendC::GetBlockNum());
        if (!validMetadata || tiling.embeddingSize != HeadDim || tiling.groupSize == 0U ||
            QueryTile != CubeLayout::QUERY_TILE || tiling.blockShapeX != 1U || tiling.baseTileInfo.kvBaseTile != KV ||
            tiling.groupSize > QueryTile || tiling.baseTileInfo.qBaseTile != QueryTile) {
            return;
        }
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
        Run(params, tiling, meta, flashDecode);
        if (flashDecode) {
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::SyncAll<false>();
#ifdef __DAV_VEC__
            GenericBlockSparseAttentionFdCombineArch35<ElementO, Resource, OutputLse, QFormat> combine(resource);
            AscendC::GlobalTensor<float> partialO, partialLse, lse;
            AscendC::GlobalTensor<ElementO> output;
            AscendC::GlobalTensor<int64_t> cuQ;
            AscendC::GlobalTensor<int32_t> usedQ;
            partialO.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(params.workspace + tiling.fdPartialOOffset));
            partialLse.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(params.workspace + tiling.fdPartialLseOffset));
            output.SetGlobalBuffer(reinterpret_cast<__gm__ ElementO*>(params.o));
            lse.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(params.softmaxLse));
            cuQ.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(params.cuSeqLengths));
            usedQ.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(params.sequsedQ));
            combine(meta, &tiling, partialLse, partialO, output, lse, cuQ, usedQ, params.sequsedQ != nullptr);
#endif
        }
    }

private:
#ifdef __DAV_VEC__
    __aicore__ inline void StoreEmptyTensor(const Mxfp8Full::KernelParams& params,
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
        if constexpr (OutputLse) {
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
        if constexpr (OutputLse) {
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
        uint32_t kvTokens;
#endif
#ifdef __DAV_VEC__
        uint64_t vScaleOffset;
        uint64_t outputOffset;
        uint64_t lseOffset;
        int32_t partialId;
#endif
        uint32_t tileCount;
        uint32_t tileId;
    };

    __aicore__ inline void Run(const Mxfp8Full::KernelParams& params,
                               const GenericBlockSparseAttn::GenericBlockSparseAttentionTilingData& tiling,
                               __gm__ const GbsaMetadata::Metadata* meta, bool flashDecode)
    {
        const uint32_t taskCount = meta->saTotalTaskNum;
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
        if (flashDecode) {
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
        const uint32_t scaleLookahead = HeadDim == 192U && taskStart + taskStep < taskEnd ? 1U : 0U;
#endif
        const uint32_t rows0 = (tiling.groupSize + 1U) / 2U;
        const uint32_t validRows = bank == 0U ? rows0 : tiling.groupSize - rows0;
        const uint32_t rowOffset = bank * rows0;
        const uint32_t groupWords = Mxfp8Descriptor::GroupWords(descriptor.ROWS, tiling.blockShapeY, IsPaged);
        const uint64_t tokenStrideBytes = GbsaLayoutHelper<KvFormat>::TokenStride(tiling.kvHeads, HeadDim);
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
                                   static_cast<uint64_t>(core) * DESCRIPTOR_WORKSPACE_SLOTS * tiling.kvOffsetTileWords);
        auto ubSLayout = tla::MakeLayout<float, NpuArch::layout::RowMajor>(VecRows, KV);
        auto ubPVLayout = tla::MakeLayout<float, NpuArch::layout::RowMajor>(VecRows, HeadDim);
        auto gQLayout = tla::MakeLayout(tla::MakeShape(tiling.groupSize, HeadDim),
                                        tla::MakeStride(qHeadStride * HeadDim, tla::Int<1>{}));
        auto gKVLayout = MakeKvLayout(tiling);
#ifdef __DAV_CUBE__
        BoundQK blockMmadQK(resource, HeadDim, QueryTile, VecRows, descriptor.ROWS);
        BoundPV blockMmadPV(resource, HeadDim, QueryTile, VecRows, descriptor.ROWS);
        for (uint32_t slot = 0; slot < DESCRIPTOR_STAGES; ++slot) {
            NpuArch::Arch::CubePublishBoth<PIPE_MTE1>(static_cast<uint32_t>(Flag::DESCRIPTOR_BASE) + slot);
        }
        for (uint32_t slot = 0; slot < P_STAGES; ++slot) {
            NpuArch::Arch::CubePublishBoth<PIPE_MTE1>(static_cast<uint32_t>(Flag::P_BASE) + slot);
        }
#endif
#ifdef __DAV_VEC__
        float pScale = 16.0f;
        if (params.pQuantScale != nullptr) {
            AscendC::GlobalTensor<float> pScaleGm;
            pScaleGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(params.pQuantScale));
            pScale = pScaleGm.GetValue(0);
        }
        constexpr typename EpilogueOnlineSoftmax::Buffers softmaxBuffers{
            Ub::KV,           Ub::LOCAL_ROW_SUM_OFFSET, Ub::P_OFFSET,      Ub::P_SLOT_BYTES,   Ub::S_OFFSET,
            Ub::S_SLOT_BYTES, Ub::TILE_DM_OFFSET,       Ub::TILE_G_OFFSET, Ub::TILE_MAX_OFFSET};
        EpilogueOnlineSoftmax epilogueOnlineSoftmax(resource, HeadDim, VecRows, softmaxBuffers, pScale);
        constexpr typename EpilogueRescaleO::Buffers rescaleBuffers{Ub::LOCAL_ROW_SUM_OFFSET,
                                                                    Ub::LSE_OFFSET,
                                                                    Ub::LSE_PACKED_OFFSET,
                                                                    Ub::OUTPUT_SCRATCH_BYTES,
                                                                    Ub::OUTPUT_SCRATCH_OFFSET,
                                                                    Ub::OUT_G_OFFSET,
                                                                    Ub::OUT_L_OFFSET,
                                                                    Ub::O_OFFSET,
                                                                    Ub::PV_OFFSET,
                                                                    Ub::PV_SLOTS,
                                                                    Ub::PV_SLOT_BYTES,
                                                                    Ub::TILE_DM_OFFSET,
                                                                    Ub::TILE_G_OFFSET,
                                                                    Ub::VSCALE_OFFSET};
        EpilogueRescaleO epilogueRescaleO(resource, HeadDim, VecRows, rescaleBuffers, pScale);
        constexpr typename EpiloguePrepareKScale::Buffers scaleBuffers{Ub::DESC_OFFSET,
                                                                       Ub::KSCALE_OUTPUT_OFFSET,
                                                                       Ub::RAW_STAGES,
                                                                       CubeLayout::KSCALE_OFFSET,
                                                                       CubeLayout::KSCALE_SLOT_BYTES,
                                                                       CubeLayout::KSCALE_SLOTS,
                                                                       static_cast<uint32_t>(Flag::KSCALE_BASE),
                                                                       static_cast<uint32_t>(Flag::KSCALE_FREE_BASE)};
        static_assert(scaleBuffers.rawStages == Ub::RAW_STAGES && scaleBuffers.l1Offset == CubeLayout::KSCALE_OFFSET &&
                      scaleBuffers.l1SlotBytes == CubeLayout::KSCALE_SLOT_BYTES &&
                      scaleBuffers.l1Slots == CubeLayout::KSCALE_SLOTS &&
                      scaleBuffers.scaleReadyFlag == static_cast<uint32_t>(Flag::KSCALE_BASE) &&
                      scaleBuffers.scaleFreeFlag == static_cast<uint32_t>(Flag::KSCALE_FREE_BASE));
        EpiloguePrepareKScale prepareKScale(resource, HeadDim, descriptor.ROWS, scaleBuffers);
        NpuArch::Epilogue::Block::UBufTileHelper offsetsLayout{};
        offsetsLayout.maskStartOffset = ub.LOCAL_DESC_OFFSET;
        // EndGroup consumes local records before publication. A single group
        // per bank can reuse one buffer: MTE3_S event 5 protects its DMA source.
        // QK and delayed PV retain the separately protected GM descriptor slots.
        offsetsLayout.kvOffsetUbBufNum = descriptor.LOCAL_DESC_SLOTS;
        EpiloguePrepareOffsets prepareOffsets(resource, offsetsLayout, KV, descriptor.ROWS, tiling.blockShapeY,
                                              tiling.blockSize, tokenStrideBytes, tiling.kStride0, tiling.vStride0,
                                              HeadDim);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID5);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID6);
        InitializeVectorMailboxes();
        // sequsedQ leaves storage padding outside the metadata task space.
        if (params.sequsedQ != nullptr) {
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
                        if constexpr (OutputLse) {
                            taskLse = gLse[lseOffset];
                        }
                        epilogueRescaleO.StoreEmpty(gO[outputOffset], taskLse, validRows, OutputLse, qHeadStride,
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
        Tile pending[P_STAGES];
        uint32_t sequence = 0U;
        for (uint32_t taskId = taskStart; taskId < taskEnd; taskId += taskStep) {
            TaskInfo task{};
            uint32_t visible = 0U;
            if (!DecodeTask(taskId, params, tiling, cuQ, cuKv, gSequsedQ, gSequsedKv, gSparseIdx, gSparseCount,
                            gBlockTable, task, visible, flashDecode)) {
                continue;
            }
            int32_t partialId = -1;
            if (flashDecode) {
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
                blockMmadQK.LoadQuery(query, gQScale[task.qByteOffset / 32U], tiling.groupSize);
#endif
#ifdef __DAV_VEC__
                for (uint32_t tile = 0; tile < min(task.tileCount, DESCRIPTOR_PRE_LAUNCH); ++tile) {
                    PrepareTile(prepareOffsets, prepareKScale, gKScale, gKvOffsets, gSparseIdx, gBlockTable, tiling,
                                task, tile, taskBegin);
                }
#endif
            }
            // Empty tasks occupy one queue position so their output is cleared
            // only after all preceding PV consumers have finished.
            for (uint32_t tileId = 0; tileId < max(task.tileCount, 1U); ++tileId, ++sequence) {
                auto& item = pending[sequence % P_STAGES];
                // Keep only the fields needed by the delayed PV consumer.
                // Query addresses, sparse-list cursors and batch decoding
                // state belong to the producer and need no queue copy.
#ifdef __DAV_CUBE__
                item.kvByteOffset = task.kvByteOffset;
                item.kvTokens = task.kvTokens;
#endif
#ifdef __DAV_VEC__
                item.vScaleOffset = task.vScaleOffset;
                item.outputOffset = task.outputOffset;
                item.lseOffset = task.lseOffset;
                item.partialId = partialId;
#endif
                item.tileCount = task.tileCount;
                item.tileId = tileId;
                if (task.tileCount != 0U) {
                    const uint32_t sSlot = sequence % 2U;
                    const uint32_t pSlot = sequence % P_STAGES;
                    const uint32_t descriptorSlot = sequence % DESCRIPTOR_STAGES;
                    const uint32_t tokens = min(KV, task.kvTokens - tileId * KV);
#ifdef __DAV_CUBE__
                    auto key = tla::MakeTensor(gK[task.kvByteOffset], gKVLayout, NpuArch::Arch::PositionGM{});
                    auto descriptor = gKvOffsets[static_cast<uint64_t>(descriptorSlot) * tiling.kvOffsetTileWords];
                    if (tileId == 0U) {
                        NpuArch::Arch::CubeWaitBoth<PIPE_S>(static_cast<uint32_t>(Flag::DESCRIPTOR_BASE) +
                                                            descriptorSlot);
                        blockMmadQK.LoadKey(key, gKScale[task.kvByteOffset / 32U], descriptor, groupWords, tokens,
                                            sequence);
                    }
                    NpuArch::Arch::CubeWaitBoth<PIPE_FIX>(sSlot);
                    auto score = tla::MakeTensor(ubSTensor[sSlot], ubSLayout, NpuArch::Arch::PositionUB{});
                    blockMmadQK(key, gKScale[task.kvByteOffset / 32U], descriptor, groupWords, tokens, score, sequence);
                    NpuArch::Arch::CubePublishBoth<PIPE_FIX>(sSlot);
                    if (tileId + 1U < task.tileCount) {
                        const uint32_t nextSlot = (sequence + 1U) % DESCRIPTOR_STAGES;
                        NpuArch::Arch::CubeWaitBoth<PIPE_S>(static_cast<uint32_t>(Flag::DESCRIPTOR_BASE) + nextSlot);
                        blockMmadQK.LoadKey(key, gKScale[task.kvByteOffset / 32U],
                                            gKvOffsets[static_cast<uint64_t>(nextSlot) * tiling.kvOffsetTileWords],
                                            groupWords, min(KV, task.kvTokens - (tileId + 1U) * KV), sequence + 1U);
                    }
#endif
#ifdef __DAV_VEC__
                    if (HeadDim == 192U) {
                        // Keep the first softmax on the critical path for a single
                        // task. A stream of tasks can prepare the next tile's scale
                        // before softmax and overlap it with Cube consumption.
                        if (scaleLookahead != 0U && tileId == 0U) {
                            prepareKScale.PrepareTile(tokens, bank, tiling.blockShapeY, sequence);
                        }
                        const uint32_t scaleTile = tileId + scaleLookahead;
                        if (scaleTile < task.tileCount) {
                            prepareKScale.PrepareTile(min(KV, task.kvTokens - scaleTile * KV), bank, tiling.blockShapeY,
                                                      sequence + scaleLookahead);
                        }
                    } else {
                        prepareKScale.PrepareTile(tokens, bank, tiling.blockShapeY, sequence);
                    }
                    if (tileId + DESCRIPTOR_PRE_LAUNCH < task.tileCount) {
                        PrepareTile(prepareOffsets, prepareKScale, gKScale, gKvOffsets, gSparseIdx, gBlockTable, tiling,
                                    task, tileId + DESCRIPTOR_PRE_LAUNCH, taskBegin);
                    }
                    AscendC::CrossCoreWaitFlag<4, PIPE_V>(sSlot);
                    AscendC::CrossCoreWaitFlag<4, PIPE_V>(static_cast<uint32_t>(Flag::P_BASE) + pSlot);
                    const uint32_t statsSlot = sequence % STATS_STAGES;
                    const uint32_t pRowOffset = bank * ((rows0 + 15U) / 16U * 16U);
                    epilogueOnlineSoftmax(sSlot, sSlot, statsSlot, tokens, l1PTensor[pSlot], pRowOffset,
                                          tiling.groupSize, tileId == 0U);
                    AscendC::CrossCoreSetFlag<4, PIPE_V>(sSlot);
                    AscendC::CrossCoreSetFlag<4, PIPE_MTE3>(static_cast<uint32_t>(Flag::P_BASE) + pSlot);
#endif
                }
#ifdef __DAV_CUBE__
                // The first PV only needs V0. Queue its L0/MMAD before
                // parsing and issuing the future V1/V2 descriptor lists.
                if (sequence == PRE_LAUNCH) {
                    PrefetchValue(pending[0], 0U, blockMmadPV, gV, gKvOffsets, tiling, groupWords);
                }
#endif
                if (sequence >= PRE_LAUNCH) {
                    const uint32_t consumer = sequence - PRE_LAUNCH;
#ifdef __DAV_CUBE__
                    ConsumeTile(pending[consumer % P_STAGES], consumer, blockMmadPV, gV, gKvOffsets, tiling, groupWords,
                                ubPVLayout);
#endif
#ifdef __DAV_VEC__
                    ConsumeTile(pending[consumer % P_STAGES], consumer, epilogueRescaleO, epilogueOnlineSoftmax,
                                prepareKScale, gO, gLse, gVScale, gPartialN, gPartialStats, tiling, validRows,
                                rowOffset, flashDecode);
#endif
                }
#ifdef __DAV_CUBE__
                if (sequence == PRE_LAUNCH) {
                    PrefetchValue(pending[1], 1U, blockMmadPV, gV, gKvOffsets, tiling, groupWords);
                }
                if (sequence >= PRE_LAUNCH) {
                    PrefetchValue(pending[sequence % P_STAGES], sequence, blockMmadPV, gV, gKvOffsets, tiling,
                                  groupWords);
                }
#endif
                // UB reuse is protected at each consumer; future descriptor
                // work need not wait for this iteration's entire Vector queue.
            }
        }
        // Only the end of the whole core stream drains the remaining tiles.
        for (uint32_t consumer = sequence > PRE_LAUNCH ? sequence - PRE_LAUNCH : 0U; consumer < sequence; ++consumer) {
#ifdef __DAV_CUBE__
            if (sequence <= PRE_LAUNCH) {
                PrefetchValue(pending[consumer % P_STAGES], consumer, blockMmadPV, gV, gKvOffsets, tiling, groupWords);
            }
            ConsumeTile(pending[consumer % P_STAGES], consumer, blockMmadPV, gV, gKvOffsets, tiling, groupWords,
                        ubPVLayout);
#endif
#ifdef __DAV_VEC__
            ConsumeTile(pending[consumer % P_STAGES], consumer, epilogueRescaleO, epilogueOnlineSoftmax, prepareKScale,
                        gO, gLse, gVScale, gPartialN, gPartialStats, tiling, validRows, rowOffset, flashDecode);
#endif
        }
#ifdef __DAV_CUBE__
        NpuArch::Arch::CubeWaitBoth<PIPE_FIX>(0U);
        NpuArch::Arch::CubeWaitBoth<PIPE_FIX>(1U);
        for (uint32_t slot = 0; slot < ub.PV_SLOTS; ++slot) {
            NpuArch::Arch::CubeWaitBoth<PIPE_FIX>(static_cast<uint32_t>(Flag::PV) + slot);
        }
        blockMmadQK.FreeEventID();
        blockMmadPV.FreeEventID();
#endif
#ifdef __DAV_VEC__
        for (uint32_t slot = 0; slot < DESCRIPTOR_STAGES; ++slot) {
            AscendC::CrossCoreWaitFlag<4, PIPE_S>(static_cast<uint32_t>(Flag::DESCRIPTOR_BASE) + slot);
        }
        for (uint32_t slot = 0; slot < P_STAGES; ++slot) {
            AscendC::CrossCoreWaitFlag<4, PIPE_V>(static_cast<uint32_t>(Flag::P_BASE) + slot);
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID5);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID6);
#endif
        AscendC::PipeBarrier<PIPE_ALL>();
    }

#ifdef __DAV_CUBE__
    template <class Tile>
    __aicore__ inline void PrefetchValue(const Tile& item, uint32_t sequence, BoundPV& block,
                                         const AscendC::GlobalTensor<ElementStorage>& value,
                                         const AscendC::GlobalTensor<uint64_t>& descriptors,
                                         const GenericBlockSparseAttn::GenericBlockSparseAttentionTilingData& tiling,
                                         uint32_t groupWords)
    {
        if (item.tileCount == 0U) {
            return;
        }
        const uint32_t slot = sequence % DESCRIPTOR_STAGES;
        auto layout = MakeKvLayout(tiling);
        auto tensor = tla::MakeTensor(value[item.kvByteOffset], layout, NpuArch::Arch::PositionGM{});
        block.LoadValue(tensor, descriptors[static_cast<uint64_t>(slot) * tiling.kvOffsetTileWords], groupWords,
                        min(KV, item.kvTokens - item.tileId * KV), sequence);
    }

    template <class PVLayout, class Tile>
    __aicore__ inline void ConsumeTile(const Tile& item, uint32_t sequence, BoundPV& block,
                                       const AscendC::GlobalTensor<ElementStorage>& value,
                                       const AscendC::GlobalTensor<uint64_t>& descriptors,
                                       const GenericBlockSparseAttn::GenericBlockSparseAttentionTilingData& tiling,
                                       uint32_t groupWords, const PVLayout& pvLayout)
    {
        const auto& task = item;
        if (task.tileCount == 0U) {
            return;
        }
        const uint32_t pSlot = sequence % P_STAGES;
        const uint32_t descriptorSlot = sequence % DESCRIPTOR_STAGES;
        const uint32_t pvSlot = sequence % ub.PV_SLOTS;
        const uint32_t pvFlag = static_cast<uint32_t>(Flag::PV) + pvSlot;
        NpuArch::Arch::CubeWaitBoth<PIPE_MTE1>(static_cast<uint32_t>(Flag::P_BASE) + pSlot);
        auto valueLayout = MakeKvLayout(tiling);
        auto tensor = tla::MakeTensor(value[task.kvByteOffset], valueLayout, NpuArch::Arch::PositionGM{});
        auto output = tla::MakeTensor(ubOTmpTensor[pvSlot], pvLayout, NpuArch::Arch::PositionUB{});
        block(tensor, descriptors[static_cast<uint64_t>(descriptorSlot) * tiling.kvOffsetTileWords], groupWords,
              min(KV, task.kvTokens - item.tileId * KV), sequence, tiling.groupSize, output, pvFlag);
        NpuArch::Arch::CubePublishBoth<PIPE_MTE1>(static_cast<uint32_t>(Flag::DESCRIPTOR_BASE) + descriptorSlot);
        NpuArch::Arch::CubePublishBoth<PIPE_MTE1>(static_cast<uint32_t>(Flag::P_BASE) + pSlot);
        NpuArch::Arch::CubePublishBoth<PIPE_FIX>(pvFlag);
    }
#endif
#ifdef __DAV_VEC__
    template <class Tile>
    __aicore__ inline void ConsumeTile(const Tile& item, uint32_t sequence, EpilogueRescaleO& epilogue,
                                       EpilogueOnlineSoftmax& frontend, EpiloguePrepareKScale& prepareScale,
                                       const AscendC::GlobalTensor<ElementO>& output,
                                       const AscendC::GlobalTensor<ElementLse>& lse,
                                       const AscendC::GlobalTensor<uint8_t>& valueScale,
                                       const AscendC::GlobalTensor<float>& partialN,
                                       const AscendC::GlobalTensor<float>& partialStats,
                                       const GenericBlockSparseAttn::GenericBlockSparseAttentionTilingData& tiling,
                                       uint32_t rows, uint32_t rowOffset, bool flashDecode)
    {
        const auto& task = item;
        if (item.tileId == 0U && task.tileCount != 0U) {
            epilogue.SetValueScale(valueScale[item.vScaleOffset]);
        }
        if (task.tileCount <= 1U && (!flashDecode || item.partialId < 0)) {
            const uint32_t headStride = QFormat == Format::BNSD ? tiling.qSeqCapacity : 1U;
            const uint32_t lseStride = QFormat == Format::TND ? 1U : tiling.qSeqCapacity;
            auto taskOutput = output[task.outputOffset + rowOffset * headStride * HeadDim];
            auto taskLse = lse;
            if constexpr (OutputLse) {
                taskLse = lse[task.lseOffset + rowOffset * lseStride];
            }
            if (task.tileCount == 0U) {
                epilogue.StoreEmpty(taskOutput, taskLse, rows, OutputLse, headStride, lseStride);
            } else {
                const uint32_t pvSlot = sequence % ub.PV_SLOTS;
                const uint32_t flag = static_cast<uint32_t>(Flag::PV) + pvSlot;
                AscendC::CrossCoreWaitFlag<4, PIPE_V>(flag);
                epilogue.StoreTile(taskOutput, taskLse, rows, OutputLse, headStride, lseStride, sequence % STATS_STAGES,
                                   pvSlot);
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
            const uint32_t flag = static_cast<uint32_t>(Flag::PV) + pvSlot;
            AscendC::CrossCoreWaitFlag<4, PIPE_V>(flag);
            const bool saveMax =
                item.tileId + 1U == task.tileCount && (OutputLse || (flashDecode && item.partialId >= 0));
            epilogue.ConsumeTile(sequence % STATS_STAGES, pvSlot, rows, saveMax);
            AscendC::CrossCoreSetFlag<4, PIPE_V>(flag);
        }
        if (item.tileId + 1U >= task.tileCount) {
            if (flashDecode) {
                if (item.partialId >= 0) {
                    const uint64_t partial = static_cast<uint32_t>(item.partialId);
                    const uint32_t bank = AscendC::GetSubBlockIdx();
                    epilogue.StorePartial(partialN[(partial * tiling.groupSize + rowOffset) * HeadDim],
                                          partialStats[(partial * 2U + bank) * tiling.fdLseSubStride], rows,
                                          tiling.fdLseSubStride);
                    return;
                }
            }
            // LSE has its own source buffer; output conversion retains the
            // frontend's MTE3->V credits while reusing P storage.
            auto taskLse = lse;
            if constexpr (OutputLse) {
                taskLse = lse[task.lseOffset + rowOffset * (QFormat == Format::TND ? 1U : tiling.qSeqCapacity)];
            }
            const uint32_t headStride = QFormat == Format::BNSD ? tiling.qSeqCapacity : 1U;
            epilogue.Store(output[task.outputOffset + rowOffset * headStride * HeadDim], taskLse, rows, OutputLse,
                           headStride, QFormat == Format::TND ? 1U : tiling.qSeqCapacity);
            // NormalizeCallVF has overwritten P storage, including padding.
            frontend.InvalidatePadding();
        }
    }
#endif

    __aicore__ inline bool DecodeTask(uint32_t taskId, const Mxfp8Full::KernelParams& params,
                                      const GenericBlockSparseAttn::GenericBlockSparseAttentionTilingData& tiling,
                                      AscendC::GlobalTensor<int64_t>& cuQ, AscendC::GlobalTensor<int64_t>& cuKv,
                                      AscendC::GlobalTensor<int32_t>& gSequsedQ,
                                      AscendC::GlobalTensor<int32_t>& gSequsedKv,
                                      AscendC::GlobalTensor<int32_t>& gSparseIdx,
                                      AscendC::GlobalTensor<int32_t>& gSparseCount,
                                      AscendC::GlobalTensor<int32_t>& blockTable, TaskInfo& task, uint32_t& visible,
                                      bool flashDecode)
    {
        BatchSeqInfo qSeq{}, kvSeq{};
        task.kvHead = taskId % tiling.kvHeads;
        const uint32_t qToken = taskId / tiling.kvHeads;
        if constexpr (QFormat == Format::TND) {
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
        } else {
            DecodeBatchedQToken(qToken, tiling.batch, tiling.qSeqCapacity, gSequsedQ, task.batch, task.qTokenInBatch,
                                qSeq);
        }
        if (!flashDecode && tiling.kvHeads > 1U) {
            // Batch task intervals still contain actualLen * kvHeads entries.
            // Reinterpret the batch-local task as [KV head, Q token] before
            // deriving tensor offsets. FD retains metadata's original IDs.
            const uint64_t localTask = static_cast<uint64_t>(task.qTokenInBatch) * tiling.kvHeads + task.kvHead;
            task.qTokenInBatch = static_cast<uint32_t>(localTask % qSeq.actualLen);
            task.kvHead = static_cast<uint32_t>(localTask / qSeq.actualLen);
        }
        task.qStorageToken = qSeq.storageStart + task.qTokenInBatch;
        if constexpr (IsPaged) {
            kvSeq.actualLen = static_cast<uint32_t>(gSequsedKv.GetValue(task.batch));
        } else {
            ReadBatchSeqInfo<KvFormat>(cuKv, gSequsedKv, params.sequsedKv != nullptr, task.batch, kvSeq,
                                       tiling.kvSeqCapacity);
        }
        const uint32_t qHead = task.kvHead * tiling.groupSize;
        task.qByteOffset = GetTensorOffset<QFormat>(qSeq, task.qTokenInBatch, qHead, tiling.numHeads, HeadDim);
        task.outputOffset = task.qByteOffset;
        task.lseOffset = GetLseOffset<QFormat>(qSeq, task.qTokenInBatch, qHead, tiling.numHeads);
        if constexpr (IsPaged) {
            task.kvByteOffset = task.kvHead * GbsaLayoutHelper<KvFormat>::HeadStride(tiling.blockSize, HeadDim);
            task.vScalePairs = tiling.blockSize / 64U;
            if (kvSeq.actualLen != 0U) {
                const uint64_t page = blockTable.GetValue(static_cast<uint64_t>(task.batch) * tiling.maxBlocksPerBatch);
                task.vScaleOffset = (page * task.vScalePairs * tiling.kvHeads +
                                     task.kvHead * (KvFormat == Format::BNSD ? task.vScalePairs : 1U)) *
                                    HeadDim * 2U;
            }
        } else {
            task.kvByteOffset = GetTensorOffset<KvFormat>(kvSeq, 0, task.kvHead, tiling.kvHeads, HeadDim);
            task.vScaleOffset = (static_cast<uint64_t>(task.batch) * tiling.kvHeads + task.kvHead) * HeadDim * 2U;
        }
        task.sparseOffset = (static_cast<uint64_t>(task.kvHead) * tiling.qBlockNum + task.qStorageToken) * tiling.topK;
        task.explicitBlocks = static_cast<uint32_t>(
            gSparseCount.GetValue(static_cast<uint64_t>(task.kvHead) * tiling.qBlockNum + task.qStorageToken));
        visible = kvSeq.actualLen;
        if (visible == 0U || (isCausal_ && visible < qSeq.actualLen)) {
            visible = 0U;
            task.explicitBlocks = 0U;
            return true;
        }
        if (isCausal_) {
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
            tla::MakeShape(tiling.kvSeqCapacity, HeadDim),
            tla::MakeStride(GbsaLayoutHelper<KvFormat>::TokenStride(tiling.kvHeads, HeadDim), tla::Int<1>{}));
    }

    __aicore__ inline void InitCrossCoreDstBuf()
    {
        for (uint32_t i = 0; i < 2; ++i) {
            ubSTensor[i] = resource.ubBuf.template GetBufferByByte<float>(Ub::S_OFFSET + i * ub.S_SLOT_BYTES);
        }
        for (uint32_t i = 0; i < P_STAGES; ++i) {
            l1PTensor[i] = resource.l1Buf.template GetBufferByByte<uint8_t>(l1.P_OFFSET + i * l1.P_SLOT_BYTES);
        }
        for (uint32_t slot = 0; slot < ub.PV_SLOTS; ++slot) {
            ubOTmpTensor[slot] = resource.ubBuf.template GetBufferByByte<float>(ub.PV_OFFSET + slot * ub.PV_SLOT_BYTES);
        }
    }

#ifdef __DAV_VEC__
    __aicore__ inline void PrepareTile(EpiloguePrepareOffsets& prepareOffsets, EpiloguePrepareKScale& prepareScale,
                                       const AscendC::GlobalTensor<uint8_t>& keyScale,
                                       const AscendC::GlobalTensor<uint64_t>& descriptors,
                                       const AscendC::GlobalTensor<int32_t>& sparseIndices,
                                       const AscendC::GlobalTensor<int32_t>& blockTable,
                                       const GenericBlockSparseAttn::GenericBlockSparseAttentionTilingData& tiling,
                                       const TaskInfo& task, uint32_t tileId, uint32_t taskBegin)
    {
        const uint32_t slot = (taskBegin + tileId) % DESCRIPTOR_STAGES;
        // The delayed PV also reads these records. A scalar wait protects GM
        // writes even when Vector instructions have not reached their waits.
        AscendC::CrossCoreWaitFlag<4, PIPE_S>(static_cast<uint32_t>(Flag::DESCRIPTOR_BASE) + slot);
        const uint32_t tokens = min(KV, task.kvTokens - tileId * KV);
        prepareScale.BeginTile(keyScale[task.kvByteOffset / 32U], taskBegin + tileId, tiling.blockShapeY,
                               IsPaged ? tiling.blockSize : tiling.kvSeqCapacity,
                               GbsaLayoutHelper<KvFormat>::TokenStride(tiling.kvHeads, HeadDim));
        prepareOffsets(
            descriptors[static_cast<uint64_t>(slot) * tiling.kvOffsetTileWords],
            sparseIndices[task.sparseOffset].GetPhyAddr(),
            IsPaged ? blockTable[static_cast<uint64_t>(task.batch) * tiling.maxBlocksPerBatch].GetPhyAddr() : nullptr,
            tileId, tokens, task.explicitBlocks, task.residualBlock, prepareScale);
        AscendC::CrossCoreSetFlag<4, PIPE_MTE3>(static_cast<uint32_t>(Flag::DESCRIPTOR_BASE) + slot);
    }

    __aicore__ inline void InitializeVectorMailboxes()
    {
        AscendC::CrossCoreSetFlag<4, PIPE_V>(0U);
        AscendC::CrossCoreSetFlag<4, PIPE_V>(1U);
        for (uint32_t slot = 0; slot < ub.PV_SLOTS; ++slot) {
            AscendC::CrossCoreSetFlag<4, PIPE_V>(static_cast<uint32_t>(Flag::PV) + slot);
        }
    }

#endif
    bool isCausal_ = false;
    Resource resource;
    AscendC::LocalTensor<float> ubSTensor[2], ubOTmpTensor[2];
    AscendC::LocalTensor<uint8_t> l1PTensor[P_STAGES];
};

} // namespace GbsaKernelArch35
#endif // GENERIC_BLOCK_SPARSE_ATTENTION_KERNEL_ARCH35_MXFP8_FULL_QUANT_H
