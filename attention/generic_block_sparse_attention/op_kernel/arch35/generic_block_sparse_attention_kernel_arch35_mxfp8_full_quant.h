/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GBSA_KERNEL_ARCH35_MXFP8_FULL_QUANT_H
#define GBSA_KERNEL_ARCH35_MXFP8_FULL_QUANT_H

#include "kernel_utils.hpp"
#include "../kernel_common.hpp"
#include "../generic_block_sparse_attention_metadata_kernel.h"
#include "../generic_block_sparse_attention_fd_utils.h"
#include "generic_block_sparse_attention_fd_combine_arch35.h"
#include "../attn_infra/gemm/block/gbsa_block_mmad_qk_arch35_mxfp8.hpp"
#include "../attn_infra/gemm/block/gbsa_block_mmad_pv_arch35_mxfp8.hpp"
#include "../attn_infra/epilogue/block/gbsa_block_epilogue_online_softmax_arch35_mxfp8.hpp"
#include "../attn_infra/epilogue/block/gbsa_block_epilogue_rescale_o_arch35_mxfp8.hpp"
#include "../attn_infra/gemm/tile_common/gbsa_mxfp8_packed_qk_mmad.hpp"
#include "../generic_block_sparse_attention_tilingkey.h"
#include "gbsa_layout_utils.hpp"
#include "../attn_infra/epilogue/block/gbsa_block_epilogue_prepare_kv_mxfp8_arch35.hpp"

using namespace NpuArch;
using namespace tla;

namespace GbsaKernelArch35 {

namespace Mxfp8Task {

constexpr uint32_t DIRECT_INDEX_COUNT_LIMIT = 8192U;

__aicore__ inline bool DirectIndexMapFits(uint64_t count)
{
    return count <= DIRECT_INDEX_COUNT_LIMIT;
}

// Scalar task decoding only. These helpers neither own pipeline buffers nor
// publish/wait for events; Cube and both Vector cores decode immutable inputs.
struct Geometry {
    uint32_t qHeads;
    uint32_t kvHeads;
    uint32_t dim;
    uint32_t blockY;
    uint32_t qBlockNum;
    uint32_t indexCapacity;
    uint32_t groupSize;
};

struct BatchContext {
    uint32_t index = 0U;
    uint64_t logicalQStart = 0U;
    BatchSeqInfo q{};
    BatchSeqInfo kv{};
};

struct TaskSchedule {
    uint64_t begin;
    uint64_t end;
    uint32_t step;
    uint32_t firstBlock = 0U;
    uint32_t lastBlock = 0U;
};

struct TaskInfo {
    uint32_t head;
    uint64_t qHeadOffset;
    uint64_t qOffset;
    uint64_t kvHeadOffset;
    uint64_t vScaleOffset;
    uint64_t scaleBase; // KScale offset in uint16_t E8M0 pairs.
    uint64_t indexOffset;
    uint32_t visibleEnd;
    uint32_t count;
    uint64_t totalRows;
    uint32_t firstToken;
    bool directIndexMap;
    bool contiguousIndexMap;
    uint32_t fdPartialTaskId;
    uint32_t fdPartialCount;
    bool isFdPartial;
};

template <bool fdEnabled, class MetadataPtr>
__aicore__ inline TaskSchedule DecodeSchedule(MetadataPtr meta, uint64_t taskCount, uint32_t coreNum, uint32_t coreIdx)
{
    TaskSchedule result{coreIdx, taskCount, coreNum};
    if (fdEnabled) {
        result.begin = taskCount;
        result.step = 1U;
        if (coreIdx < static_cast<uint32_t>(meta->fdActiveCoreNum)) {
            const auto& schedule = meta->decodeSchedules[coreIdx];
            result.begin = static_cast<uint32_t>(schedule.baseTaskStart);
            result.end = static_cast<uint32_t>(schedule.baseTaskEnd);
            result.firstBlock = static_cast<uint32_t>(schedule.firstBlockStart);
            result.lastBlock = static_cast<uint32_t>(schedule.lastBlockEnd);
        }
    }
    return result;
}

template <Format qFormat, Format kvFormat, class Params, class Tiling, class CuTensor, class UsedTensor>
__aicore__ inline void LoadBatch(const Params& params, const Tiling* td, CuTensor& cuQ, UsedTensor& usedQ,
                                 CuTensor& cuKv, UsedTensor& usedKv, BatchContext& batch)
{
    ReadBatchSeqInfo<qFormat>(cuQ, usedQ, params.sequsedQ != nullptr, batch.index, batch.q, td->qSeqCapacity);
    ReadBatchSeqInfo<kvFormat>(cuKv, usedKv, params.sequsedKv != nullptr, batch.index, batch.kv, td->kvSeqCapacity);
}

template <Format qFormat, Format kvFormat, class Params, class Tiling, class CuTensor, class UsedTensor>
__aicore__ inline bool AdvanceBatch(uint64_t globalQToken, const Params& params, const Tiling* td, CuTensor& cuQ,
                                    UsedTensor& usedQ, CuTensor& cuKv, UsedTensor& usedKv, BatchContext& batch)
{
    // Monotonic task IDs visit each actual-length prefix once, skipping empty
    // batches. Tensor addresses retain the separate physical storage prefix.
    while (batch.index < td->batch && globalQToken >= batch.logicalQStart + batch.q.actualLen) {
        batch.logicalQStart += batch.q.actualLen;
        ++batch.index;
        if (batch.index < td->batch) {
            LoadBatch<qFormat, kvFormat>(params, td, cuQ, usedQ, cuKv, usedKv, batch);
        }
    }
    return batch.index < td->batch;
}

template <Format qFormat, Format kvFormat, bool isPaged, bool fdEnabled, class Tiling, class IndexTensor,
          class CountTensor>
__aicore__ inline void DecodeTask(uint64_t taskId, const Geometry& geometry, const Tiling* td,
                                  const TaskSchedule& schedule, const BatchContext& batch, bool causal,
                                  IndexTensor& indices, CountTensor& counts, TaskInfo& task)
{
    constexpr bool kvHeadMajor = kvFormat == Format::BNSD;
    const uint32_t groups64 = geometry.dim / 64U;
    const uint32_t token = static_cast<uint32_t>(taskId / geometry.kvHeads - batch.logicalQStart);
    task.head = taskId % geometry.kvHeads;
    const uint32_t qLen = batch.q.actualLen;
    const uint32_t kvLen = batch.kv.actualLen;
    const uint64_t qStorageToken = batch.q.storageStart + token;
    task.qHeadOffset = GetTensorOffset<qFormat>(batch.q, token, task.head * geometry.groupSize, geometry.qHeads, 1U);
    task.qOffset = task.qHeadOffset * geometry.dim;
    task.kvHeadOffset = isPaged ? static_cast<uint64_t>(task.head) * (kvHeadMajor ? td->blockSize : 1U) * geometry.dim :
                                  GetTensorOffset<kvFormat>(batch.kv, 0U, task.head, geometry.kvHeads, geometry.dim);
    // Dense TND VScale is [B,Nkv,D,2], independent of cuKv. A live paged
    // task resolves its first physical page after the empty-task check.
    task.vScaleOffset = (static_cast<uint64_t>(batch.index) * geometry.kvHeads + task.head) * geometry.dim * 2U;
    task.scaleBase = isPaged     ? (kvHeadMajor ? static_cast<uint64_t>(task.head) * td->blockSize * groups64 : 0U) :
                     kvHeadMajor ? GetTensorOffset<kvFormat>(batch.kv, 0U, task.head, geometry.kvHeads, groups64) :
                                   batch.kv.storageStart * geometry.kvHeads * groups64;
    const uint64_t sparseRow = static_cast<uint64_t>(task.head) * geometry.qBlockNum + qStorageToken;
    task.visibleEnd = causal ? (kvLen < qLen ? 0U : kvLen - qLen + token + 1U) : kvLen;
    const uint32_t inputCount = static_cast<uint32_t>(counts.GetValue(sparseRow));
    const uint32_t fullCount = inputCount < geometry.indexCapacity ? inputCount : geometry.indexCapacity;
    const uint32_t rawBegin = fdEnabled && taskId == schedule.begin ?
                                  (schedule.firstBlock < fullCount ? schedule.firstBlock : fullCount) :
                                  0U;
    const uint32_t rawEnd = fdEnabled && taskId + 1U == schedule.end ?
                                (schedule.lastBlock < fullCount ? schedule.lastBlock : fullCount) :
                                fullCount;
    task.count = rawEnd - rawBegin;
    // Capacity is the physical sparse-row stride, not the amount of work.
    task.indexOffset = sparseRow * geometry.indexCapacity + rawBegin;
    task.totalRows = 0U;
    task.directIndexMap = false;
    task.contiguousIndexMap = false;
    task.firstToken = 0U;
    if (task.count != 0U && task.visibleEnd != 0U) {
        auto taskIndices = indices[task.indexOffset];
        const uint32_t first = static_cast<uint32_t>(taskIndices.GetValue(0U));
        const uint32_t last = static_cast<uint32_t>(taskIndices.GetValue(task.count - 1U));
        // The count entries are sorted, unique and visible. Only the last
        // selected block can be partial; no scan or temporary task header.
        task.totalRows = static_cast<uint64_t>(task.count) * geometry.blockY;
        if (task.visibleEnd % geometry.blockY != 0U) {
            const uint64_t lastStart = static_cast<uint64_t>(last) * geometry.blockY;
            const uint64_t remaining = static_cast<uint64_t>(task.visibleEnd) - lastStart;
            const uint64_t lastRows = remaining < geometry.blockY ? remaining : geometry.blockY;
            task.totalRows -= geometry.blockY - lastRows;
        }
        task.directIndexMap = Mxfp8Task::DirectIndexMapFits(task.count);
        task.contiguousIndexMap = task.directIndexMap && last - first == task.count - 1U;
        task.firstToken = static_cast<uint64_t>(first) * geometry.blockY;
    }
}

template <Format kvFormat, class PageTensor>
__aicore__ inline void ResolvePagedValueScale(PageTensor& pages, const Geometry& geometry, TaskInfo& task)
{
    // Call only for live tasks. Empty tasks must not dereference their page table.
    constexpr bool kvHeadMajor = kvFormat == Format::BNSD;
    const uint64_t firstPage = static_cast<uint32_t>(pages.GetValue(0U));
    task.vScaleOffset = (firstPage * geometry.kvHeads * 2U + task.head * (kvHeadMajor ? 2U : 1U)) * geometry.dim * 2U;
}

} // namespace Mxfp8Task

// Small dense sparse blocks are compacted into a full attention tile. K/V
// stay in FP8; both AIVs gather their groups' E8M0 pairs into scale NZ, avoiding
// DN2NZ's 16-row padding between 4/8-token blocks. Layouts have independent keys.
template <class BlockMmadQK, class EpilogueOnlineSoftmax, class BlockMmadPV, class EpilogueRescaleO,
          GbsaMxQkVariant QkVariant = GbsaMxQkVariant::D192, Format QFormat = Format::BSND, Format KvFormat = QFormat,
          bool IsPaged = false>
class GbsaMxfp8PackedKernelArch35 {
private:
    static_assert(QkVariant == GbsaMxQkVariant::D192 || QkVariant == GbsaMxQkVariant::D192_PREFILL ||
                      QkVariant == GbsaMxQkVariant::D256,
                  "Packed MXFP8 requires a D192 or D256 QK variant");
    static constexpr bool QHeadMajor = QFormat == Format::BNSD;
    static constexpr bool KvHeadMajor = KvFormat == Format::BNSD;
    static_assert(IsPaged || QFormat == KvFormat, "Dense MXFP8 Q/KV layouts must match");
    static_assert(!IsPaged || KvFormat != Format::TND, "Paged MXFP8 uses a batched page layout");
    using ArchTag = typename BlockMmadPV::ArchTag;

    using ElementQ = typename BlockMmadQK::ElementA;
    using ElementK = typename BlockMmadQK::ElementB;
    using ElementS = typename EpilogueOnlineSoftmax::ElementInput;
    using ElementP = typename BlockMmadPV::ElementA;
    using ElementV = typename BlockMmadPV::ElementB;
    using ElementOTmp = typename BlockMmadPV::ElementC;
    using ElementO = typename EpilogueRescaleO::ElementO;
    using ElementLse = typename EpilogueRescaleO::ElementLse;

    using LayoutQ = layout::RowMajor;
    using LayoutK = layout::ColumnMajor;
    using LayoutS = layout::RowMajor;
    using LayoutV = layout::RowMajor;
    using LayoutO = layout::RowMajor;
    using LayoutOTmp = layout::RowMajor;
    using LayoutLse = layout::RowMajor;

    using Cursor = NpuArch::Gemm::Tile::GbsaMxPackedCursor;
    // Preserve each kernel variant's Vector-function instantiations.
    struct KvProducerConfig {
        static constexpr bool KV_HEAD_MAJOR = KvHeadMajor;
        static constexpr bool IS_PAGED = IsPaged;
    };
    using KvProducer = Epilogue::Block::GbsaMxfp8KvProducer<KvProducerConfig>;
    static constexpr uint32_t PRE_LAUNCH = 2U;
    static constexpr uint32_t RING = PRE_LAUNCH + 1U;
    static constexpr uint32_t UB_STAGES = 2U;
    static constexpr uint32_t DESCRIPTOR_STAGES = KvProducer::DESCRIPTOR_STAGES;
    static constexpr uint32_t DESCRIPTOR_AHEAD = KvProducer::DESCRIPTOR_AHEAD;
    static constexpr uint32_t GROUPED_RECORD_UB_STAGES = KvProducer::GROUPED_RECORD_UB_STAGES;
    static constexpr uint32_t SCALE_FLAG = KvProducer::SCALE_FLAG;
    static constexpr uint32_t SCALE_L1_LARGE_BASE = NpuArch::Gemm::Tile::GBSA_MX_SCALE_BASE - RING * 4096U;

    struct PendingTile {
        Cursor cursor;
        uint64_t outputOffset;
        uint32_t rows;
        uint32_t computeRows;
        uint64_t valueOffset;
        uint64_t valueScaleOffset;
        bool first;
        bool last;
    };

    __aicore__ inline uint64_t QueryRowOffset(const BatchSeqInfo& seq, uint32_t token, uint32_t head)
    {
        return GetTensorOffset<QFormat>(seq, token, head, qHeads_, 1U);
    }

    template <class Consumer>
    __aicore__ inline void ConsumePendingTile(const PendingTile& item, uint64_t sequence, uint64_t produced,
                                              Consumer& consumer, AscendC::GlobalTensor<ElementV>& values,
                                              AscendC::GlobalTensor<ElementO>& output,
                                              AscendC::GlobalTensor<ElementLse>& lse,
                                              AscendC::GlobalTensor<uint8_t>& valueScale,
                                              AscendC::LocalTensor<ElementOTmp> (&partialBuffers)[UB_STAGES],
                                              uint64_t tokenStride, uint64_t queryHeadStride)
    {
        const uint32_t pSlot = sequence % RING;
        const uint32_t oSlot = sequence % uBufTileHelper_.loUbBufNum;
        Arch::CrossCoreFlag pReady(pSlot + UB_STAGES), oReady(oSlot + UB_STAGES + RING);
        const GemmCoord shape{item.computeRows, embed_, item.rows};
#ifdef __DAV_CUBE__
        consumer.SetMxPacked(item.cursor);
        auto value = tla::MakeTensor(values[item.valueOffset],
                                     tla::MakeLayout<ElementV, LayoutV>(kvBaseTile_, tokenStride), Arch::PositionGM{});
        auto partial = tla::MakeTensor(partialBuffers[oSlot],
                                       tla::MakeLayout<ElementOTmp, LayoutOTmp>(RoundUp(item.computeRows, 16U), embed_),
                                       Arch::PositionUB{});
        AscendC::GlobalTensor<int32_t> unused;
        consumer(value, partial, unused, shape, sequence % 6U, kvBaseTile_, blockShapeY_,
                 produced * mm1L0ATotalStages_ + sequence * mm2L0ATotalStages_,
                 produced * mm1L0BTotalStages_ + sequence * mm2L0BTotalStages_, pReady, oReady, tokenStride);
        AscendC::CrossCoreSetFlag<4, PIPE_S>(KvProducer::DescriptorFlag(sequence % DESCRIPTOR_STAGES));
        AscendC::CrossCoreSetFlag<4, PIPE_S>(KvProducer::DescriptorFlag(sequence % DESCRIPTOR_STAGES) - 16U);
#endif
#ifdef __DAV_VEC__
        consumer.SetMxVScale(valueScale[item.valueScaleOffset]);
        auto out = tla::MakeTensor(output[item.outputOffset],
                                   tla::MakeLayout<ElementO, LayoutO>(qBaseTile_, queryHeadStride), Arch::PositionGM{});
        auto outLse = tla::MakeTensor(lse, tla::MakeLayout<ElementLse, LayoutLse>(qBaseTile_, 1), Arch::PositionGM{});
        consumer(out, outLse, shape, pSlot, sequence % 6U, item.first, item.last, oReady);
#endif
    }

    __aicore__ inline void CalcPackedUb()
    {
        static_assert(AscendC::IsSameType<ElementS, float>::value, "Packed MXFP8 requires FP32 scores");
        static_assert(sizeof(ElementS) == Epilogue::Block::Mxfp8UbLayout::SCORE_ELEMENT_BYTES &&
                          sizeof(ElementP) == Epilogue::Block::Mxfp8UbLayout::PROBABILITY_ELEMENT_BYTES &&
                          sizeof(ElementOTmp) == sizeof(float),
                      "MXFP8 compute element sizes must agree with the epilogue layout");
        static_assert(
            UB_STAGES == Epilogue::Block::Mxfp8UbLayout::UB_COMPUTE_STAGES &&
                RING == Epilogue::Block::Mxfp8UbLayout::UB_DELTA_STAGES &&
                Epilogue::Block::UBufTileHelper::STATS_ROW_NUM == Epilogue::Block::Mxfp8UbLayout::UB_STATS_ROWS,
            "MXFP8 compute buffer counts must agree with the epilogue layout");
        // Head groups are split between the two AIVs. Statistics retain their
        // 64-row stride; the largest tier gives PV only one UB result slot.
        const uint32_t rows = Epilogue::Block::Mxfp8UbLayout::VectorRows(qBaseTile_);
        const uint32_t pvStages = Epilogue::Block::Mxfp8UbLayout::PvUbStages(qBaseTile_);
        constexpr uint32_t stats = Epilogue::Block::Mxfp8UbLayout::UB_STATS_ROWS * sizeof(float);
        const uint32_t s = 0;
        const uint32_t p = static_cast<uint32_t>(
            Epilogue::Block::Mxfp8UbLayout::ScoreUbSlotBytes(qBaseTile_, kvBaseTile_) * UB_STAGES);
        const uint32_t lo = p + rows * kvBaseTile_ * sizeof(ElementP) * UB_STAGES;
        const uint32_t outputSlotBytes = Epilogue::Block::Mxfp8UbLayout::OutputUbSlotBytes(embed_, qBaseTile_);
        const uint32_t go = lo + outputSlotBytes * pvStages;
        const uint32_t lm = go + outputSlotBytes;
        const uint32_t gm = lm + stats;
        const uint32_t dm = gm + stats;
        const uint32_t ll = dm + stats * RING;
        const uint32_t gl = ll + stats;
        const uint32_t lse = gl + stats;
        const uint32_t mask = lse + 64U * 32U;
        uBufTileHelper_ = Epilogue::Block::UBufTileHelper(rows, kvBaseTile_, embed_, s, p, lo, go, lm, gm, dm, ll, gl,
                                                          lse, mask, pvStages);
        // Keep D192 rows compact within each D256-capacity slot. Cube Fixpipe
        // and Vector rescale must use the same reserved stride between slots.
        uBufTileHelper_.loUbColCapacity = Epilogue::Block::Mxfp8UbLayout::BufferDim(embed_);
        uBufTileHelper_.mxVScaleStartOffset = mask;
    }

    __aicore__ inline static uint64_t Prefix(uint64_t iteration, uint64_t total, uint32_t qk, uint32_t pv, bool isQk)
    {
        if (isQk) {
            return iteration * qk + (iteration > PRE_LAUNCH ? iteration - PRE_LAUNCH : 0U) * pv;
        }
        const uint64_t submitted = iteration + PRE_LAUNCH + 1U < total ? iteration + PRE_LAUNCH + 1U : total;
        return submitted * qk + iteration * pv;
    }

public:
    __aicore__ inline void operator()(GbsaFullQuantKernelParamsArch35 const& params)
    {
        GET_TILING_DATA_WITH_STRUCT(GenericBlockSparseAttn::GenericBlockSparseAttentionTilingData, data, params.tiling);
        const auto* td = &data;
        FetchBaseShapeInfo(td);
        CalcOnChipBufTileInfo(td);
        CalcPackedUb();
        const uint32_t coreNum = AscendC::GetBlockNum();
        const bool causal =
            (td->tilingKey / GbsaTilingKeyEncoding::MASK_DIVISOR) % GbsaTilingKeyEncoding::DECIMAL_RADIX != 0U;
        uint32_t coreIdx = AscendC::GetBlockIdx();
#ifdef __DAV_VEC__
        coreIdx /= AscendC::GetSubBlockNum();
#endif
        const auto* meta = reinterpret_cast<__gm__ const GbsaMetadata::Metadata*>(params.metaData);
        uint32_t fdScheduleFlags = 0U;
        if (!(causal ? GsaFd::ValidateMetadata<true>(meta, td, coreNum, &fdScheduleFlags) :
                       GsaFd::ValidateMetadata<false>(meta, td, coreNum, &fdScheduleFlags))) {
            return;
        }
        if (meta->saTotalTaskNum == 0 && params.sequsedQ != nullptr) {
            // There are no scheduled Q tokens. Confirm the actual lengths before
            // clearing all physical output, including padding, without a pipeline.
            AscendC::GlobalTensor<int32_t> actualQ;
            actualQ.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(params.sequsedQ));
            bool allQueriesEmpty = true;
            for (uint32_t batch = 0U; batch < td->batch; ++batch) {
                allQueriesEmpty = allQueriesEmpty && actualQ.GetValue(batch) == 0;
            }
            if (allQueriesEmpty) {
#ifdef __DAV_VEC__
                AscendC::TPipe emptyPipe;
                const uint64_t elements = td->totalQTokens * qHeads_ * embed_;
                const uint32_t vectorCores = coreNum * AscendC::GetSubBlockNum();
                const uint64_t coreElements = ((elements + vectorCores - 1U) / vectorCores + 15U) / 16U * 16U;
                const uint64_t begin = AscendC::GetBlockIdx() * coreElements;
                const uint64_t end = min(elements, begin + coreElements);
                AscendC::GlobalTensor<ElementO> output;
                output.SetGlobalBuffer(reinterpret_cast<__gm__ ElementO*>(params.o));
                for (uint64_t offset = begin; offset < end;) {
                    const uint64_t size = min(end - offset, uint64_t{1U << 30U});
                    auto target = output[offset];
                    AscendC::InitGlobalMemory(target, size, static_cast<ElementO>(0));
                    offset += size;
                }
#endif
                return;
            }
        }
        // The validated metadata flag is already local. Ordinary packed tasks
        // need not fetch the optional FD tiling fields to select their path.
        const bool fdEnabled = (fdScheduleFlags & GbsaMetadata::FD_SCHEDULE_ENABLED) != 0U && td->fdStaticEnabled != 0U;
        if (fdEnabled) {
            Run<true>(params, td, meta, coreNum, coreIdx, causal);
        } else {
            Run<false>(params, td, meta, coreNum, coreIdx, causal);
        }
    }

private:
    // Keep FD partial/combine state out of the ordinary packed loop. Both
    // specializations retain the same layout, buffers and runtime batch walk.
    template <bool fdEnabled>
    __aicore__ inline void Run(const GbsaFullQuantKernelParamsArch35& params,
                               const GenericBlockSparseAttn::GenericBlockSparseAttentionTilingData* td,
                               __gm__ const GbsaMetadata::Metadata* meta, uint32_t coreNum, uint32_t coreIdx,
                               bool causal)
    {
        const uint32_t scaleCredits = Epilogue::Block::Mxfp8ScaleLayout::GroupsPerAiv(kvBaseTile_);
        const uint32_t groups64 = embed_ / 64U;
        AscendC::GlobalTensor<ElementLse> gLse;
        gLse.SetGlobalBuffer(reinterpret_cast<__gm__ ElementLse*>(params.softmaxLse));
        AscendC::GlobalTensor<float> gPartialO, gPartialLse;
        if (fdEnabled) {
            gPartialO.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(params.workSpace + td->fdPartialOOffset));
            gPartialLse.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(params.workSpace + td->fdPartialLseOffset));
        }
        AscendC::GlobalTensor<uint16_t> kScale;
        kScale.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(params.kDequantScale));
        AscendC::GlobalTensor<int32_t> usedQ, usedKv, unusedIndices;
        usedQ.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(params.sequsedQ));
        usedKv.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(params.sequsedKv));
        AscendC::GlobalTensor<int64_t> cuQ, cuKv;
        cuQ.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(params.cuSeqLengths));
        cuKv.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(params.cuSeqLengthsKv));
        const uint32_t batchCount = td->batch;
        const uint64_t originalTokenStride = KvHeadMajor ? embed_ : static_cast<uint64_t>(kvHeads_) * embed_;
        const uint64_t queryHeadStride = QHeadMajor ? static_cast<uint64_t>(td->qSeqCapacity) * embed_ : embed_;
        const uint64_t queryScaleHeadStride =
            QHeadMajor ? static_cast<uint64_t>(td->qSeqCapacity) * groups64 : groups64;
        // Each core owns four independent descriptor slots. Their group headers
        // and records stay live until PV consumption; all sizes align to 64 bytes.
        const uint32_t descriptorTileWords = Gemm::Tile::Mxfp8DescriptorLayout::DescriptorTileWords(kvBaseTile_);
        const uint64_t workspaceCoreBytes = DESCRIPTOR_STAGES * descriptorTileWords * sizeof(uint64_t);
        const uint64_t workspaceCoreOffset = static_cast<uint64_t>(coreIdx) * workspaceCoreBytes;
        AscendC::GlobalTensor<uint64_t> descriptors;
        descriptors.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(params.workSpace + workspaceCoreOffset));
        AscendC::LocalTensor<ElementP> l1P[RING];
        AscendC::LocalTensor<ElementS> ubS[UB_STAGES];
        AscendC::LocalTensor<ElementOTmp> ubO[UB_STAGES];
        AscendC::LocalTensor<uint8_t> l1Scale[RING];
        InitCrossCoreDstBuf(l1P, ubS, ubO);
        const uint32_t scaleL1SlotBytes = 4096U;
        const uint32_t scaleL1Base = SCALE_L1_LARGE_BASE;
        for (uint32_t i = 0; i < RING; ++i) {
            l1Scale[i] = resource.l1Buf.template GetBufferByByte<uint8_t>(scaleL1Base + i * scaleL1SlotBytes);
        }
        InitSyncFlags<4, 4, 4>();
#ifdef __DAV_CUBE__
        NpuArch::Gemm::Block::GbsaMxPackedLookaheadQK<BlockMmadQK, QkVariant> qk(resource, mm1L1TileHelper_);
        BlockMmadPV pv(resource, mm2L1AddrStart_, mm2L1TileHelper_);
        for (uint32_t i = 0; i < scaleCredits; ++i) {
            AscendC::CrossCoreSetFlag<4, PIPE_MTE1>(SCALE_FLAG + i);
            // Both AIVs receive independent credits for their two groups.
            AscendC::CrossCoreSetFlag<4, PIPE_MTE1>(SCALE_FLAG + 16U + i);
        }
        for (uint32_t i = 0; i < DESCRIPTOR_STAGES; ++i) {
            AscendC::CrossCoreSetFlag<4, PIPE_S>(KvProducer::DescriptorFlag(i));
            AscendC::CrossCoreSetFlag<4, PIPE_S>(KvProducer::DescriptorFlag(i) - 16U);
        }
#endif
#ifdef __DAV_VEC__
        EpilogueOnlineSoftmax softmax(resource, scaleValue_, uBufTileHelper_);
        EpilogueRescaleO rescale(resource, uBufTileHelper_);
        float alpha = 16.0f;
        if (params.pQuantScale != nullptr) {
            AscendC::GlobalTensor<float> pScale;
            pScale.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(params.pQuantScale));
            alpha = pScale.GetValue(0);
        }
        softmax.SetPQuantScale(alpha);
        rescale.SetPQuantScale(alpha);
        KvProducer kvProducer(embed_, kvHeads_, blockShapeY_, kvBaseTile_);
        const uint32_t cacheOffset = RoundUp(
            uBufTileHelper_.mxVScaleStartOffset + Epilogue::Block::Mxfp8UbLayout::ValueScaleUbBytes(embed_), 256U);
        // Raw -> NZ -> descriptor scratch. Grouped indices are read directly from GM.
        const uint32_t rawTileTokens = Epilogue::Block::Mxfp8ScaleLayout::RawTileTokens(kvBaseTile_);
        const uint32_t rawBytes = Epilogue::Block::Mxfp8ScaleLayout::RawCapacityBytes(rawTileTokens);
        const uint32_t packedScaleBytes = Epilogue::Block::Mxfp8ScaleLayout::NzScaleUbSlotBytes(embed_);
        const uint32_t auxBytes = Epilogue::Block::Mxfp8ScaleLayout::PipelineAuxBytes(embed_);
        // The kernel owns the complete UB partition. Prefer three Raw tiles only
        // when S/P/output, statistics, Raw, NZ and descriptor scratch all fit.
        const uint32_t rawStages = cacheOffset + Epilogue::Block::Mxfp8ScaleLayout::GROUPED_RAW_STAGES * rawBytes +
                                               auxBytes + descriptorTileWords * sizeof(uint64_t) <=
                                           ArchTag::UB_SIZE ?
                                       Epilogue::Block::Mxfp8ScaleLayout::GROUPED_RAW_STAGES :
                                       Epilogue::Block::Mxfp8ScaleLayout::RAW_STAGES;
        const uint32_t cacheBytes = rawStages * rawBytes;

        auto recordsUb = resource.ubBuf.template GetBufferByByte<uint64_t>(cacheOffset + cacheBytes + auxBytes);
        AscendC::LocalTensor<uint16_t> streamRaw[Epilogue::Block::Mxfp8ScaleLayout::GROUPED_RAW_STAGES];
        AscendC::LocalTensor<uint16_t> streamPacked[Epilogue::Block::Mxfp8ScaleLayout::NZ_STAGES];
        {
            for (uint32_t slot = 0U; slot < rawStages; ++slot) {
                streamRaw[slot] = resource.ubBuf.template GetBufferByByte<uint16_t>(cacheOffset + slot * rawBytes);
            }
            for (uint32_t slot = 0U; slot < Epilogue::Block::Mxfp8ScaleLayout::NZ_STAGES; ++slot) {
                streamPacked[slot] = resource.ubBuf.template GetBufferByByte<uint16_t>(cacheOffset + cacheBytes +
                                                                                       slot * packedScaleBytes);
            }

            kvProducer.InitStreamScalePipeline(rawStages);
        }
        kvProducer.InitRecords();
#endif
        // AIV prepares the next task while AIC consumes the last two PV tiles.
        // Descriptor/P slots follow a continuous stream. AIV drains the old
        // output before starting the next softmax. Batch changes keep this stream live.
#ifdef __DAV_CUBE__
        auto& taskConsumer = pv;
#else
        auto& taskConsumer = rescale;
#endif
        // Metadata numbers every (batch-local Q token, KV head) in one
        // compact stream over all actual Q lengths. Each core owns task IDs
        // coreIdx, coreIdx + coreNum, ...; query tokens are never paired.
        const uint64_t taskCount = static_cast<uint32_t>(meta->saTotalTaskNum);
        const Mxfp8Task::Geometry taskGeometry{qHeads_, kvHeads_, embed_, blockShapeY_, qBlockNum_, topK_, groupSize_};
        const uint64_t tokenStride = originalTokenStride;
        AscendC::GlobalTensor<ElementQ> gQ;
        AscendC::GlobalTensor<ElementK> gK;
        AscendC::GlobalTensor<ElementV> gV;
        AscendC::GlobalTensor<ElementO> gO;
        AscendC::GlobalTensor<uint8_t> qScale, vScale;
        AscendC::GlobalTensor<int32_t> indices, counts;
        gQ.SetGlobalBuffer(reinterpret_cast<__gm__ ElementQ*>(params.q));
        gK.SetGlobalBuffer(reinterpret_cast<__gm__ ElementK*>(params.k));
        gV.SetGlobalBuffer(reinterpret_cast<__gm__ ElementV*>(params.v));
        gO.SetGlobalBuffer(reinterpret_cast<__gm__ ElementO*>(params.o));
        qScale.SetGlobalBuffer(params.qDequantScale);
        vScale.SetGlobalBuffer(params.vDequantScale);
        indices.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(params.sparseBlockIdx));
        counts.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(params.sparseBlockCount));
#ifdef __DAV_VEC__
        // Padding does not belong to the compact task stream. Clear it before
        // any live output accumulator can alias WriteEmptyOutput's scratch.
        if (params.sequsedQ != nullptr) {
            for (uint32_t batch = 0U; batch < batchCount; ++batch) {
                BatchSeqInfo seq{};
                ReadBatchSeqInfo<QFormat>(cuQ, usedQ, true, batch, seq, td->qSeqCapacity);
                for (uint64_t pad = static_cast<uint64_t>(seq.actualLen) * kvHeads_ + coreIdx;
                     pad < static_cast<uint64_t>(seq.storageLen) * kvHeads_; pad += coreNum) {
                    const uint64_t offset = QueryRowOffset(seq, pad / kvHeads_, (pad % kvHeads_) * groupSize_) * embed_;
                    auto out =
                        tla::MakeTensor(gO[offset], tla::MakeLayout<ElementO, LayoutO>(qBaseTile_, queryHeadStride),
                                        Arch::PositionGM{});
                    auto lse = tla::MakeTensor(gLse, tla::MakeLayout<ElementLse, LayoutLse>(qBaseTile_, 1),
                                               Arch::PositionGM{});
                    rescale.WriteEmptyOutput(out, lse, GemmCoord{groupSize_, embed_, 0});
                }
            }
        }
#endif
        PendingTile pending[RING];
        uint64_t produced = 0U;
        uint64_t consumed = 0U;
        // FD owns contiguous sparse-list ranges, possibly crossing base tasks.
        // Drain each partial before reusing its accumulator. Keep the prefill
        // cross-task pipeline unchanged when metadata does not select FD.
        const bool taskPipeline = !fdEnabled && static_cast<uint64_t>(coreIdx) + coreNum < taskCount;
        const auto schedule = Mxfp8Task::DecodeSchedule<fdEnabled>(meta, taskCount, coreNum, coreIdx);
        Mxfp8Task::BatchContext batchContext;
        if (batchCount != 0U) {
            Mxfp8Task::LoadBatch<QFormat, KvFormat>(params, td, cuQ, usedQ, cuKv, usedKv, batchContext);
        }
        for (uint64_t task = schedule.begin; task < schedule.end; task += schedule.step) {
            Mxfp8Task::TaskInfo taskInfo;
            taskInfo.fdPartialTaskId = 0U;
            taskInfo.fdPartialCount = 0U;
            taskInfo.isFdPartial = fdEnabled && GsaFd::FindPartialTask(meta, task, coreIdx, taskInfo.fdPartialTaskId,
                                                                       taskInfo.fdPartialCount);
#ifdef __DAV_CUBE__
            while (taskPipeline && consumed < produced) {
                // Old PVs use the absolute addresses saved in pending, while
                // AIV prepares a new task that may belong to another batch.
                ConsumePendingTile(pending[consumed % RING], consumed, produced, taskConsumer, gV, gO, gLse, vScale,
                                   ubO, tokenStride, queryHeadStride);
                ++consumed;
            }
#endif
            const uint64_t globalQToken = task / kvHeads_;
            if (!Mxfp8Task::AdvanceBatch<QFormat, KvFormat>(globalQToken, params, td, cuQ, usedQ, cuKv, usedKv,
                                                            batchContext)) {
                break;
            }
            Mxfp8Task::DecodeTask<QFormat, KvFormat, IsPaged, fdEnabled>(task, taskGeometry, td, schedule, batchContext,
                                                                         causal, indices, counts, taskInfo);
            Cursor cursor{indices[taskInfo.indexOffset], taskInfo.count, blockShapeY_, taskInfo.visibleEnd, 0, 0};
#ifdef __DAV_VEC__
            auto out =
                tla::MakeTensor(gO[taskInfo.qOffset], tla::MakeLayout<ElementO, LayoutO>(qBaseTile_, queryHeadStride),
                                Arch::PositionGM{});
            auto lse = tla::MakeTensor(gLse, tla::MakeLayout<ElementLse, LayoutLse>(qBaseTile_, 1), Arch::PositionGM{});
#endif
            if (taskInfo.totalRows == 0U) {
                if (taskPipeline) {
                    // Empty-output scratch aliases the ordered O accumulator.
                    // Finish preceding consumers before clearing another task.
                    while (consumed < produced) {
                        ConsumePendingTile(pending[consumed % RING], consumed, produced, taskConsumer, gV, gO, gLse,
                                           vScale, ubO, tokenStride, queryHeadStride);
                        ++consumed;
                    }
                }
#ifdef __DAV_VEC__
                if (taskInfo.isFdPartial) {
                    rescale.WriteNeutralPartial(gPartialO, gPartialLse, taskInfo.fdPartialTaskId, groupSize_, embed_,
                                                td->fdLseSubStride);
                } else {
                    rescale.WriteEmptyOutput(out, lse, GemmCoord{groupSize_, embed_, 0});
                }
#endif
                continue;
            }
            if constexpr (IsPaged) {
                // V is per-channel: every page/group in this logical sequence
                // repeats the same scale. Resolve its first page only for live tasks.
                AscendC::GlobalTensor<int32_t> pages;
                pages.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(params.blockTable) +
                                      static_cast<uint64_t>(batchContext.index) * td->maxBlocksPerBatch);
                Mxfp8Task::ResolvePagedValueScale<KvFormat>(pages, taskGeometry, taskInfo);
#ifdef __DAV_VEC__
                kvProducer.SetPageTable(pages, td->kStride0);
#endif
            }
            if (taskInfo.contiguousIndexMap) {
                cursor.contiguousBegin = taskInfo.firstToken;
                cursor.contiguousRows = static_cast<uint32_t>(taskInfo.totalRows);
            }
            const uint32_t computeRows = groupSize_;
            const uint32_t qkRows = groupSize_;
            const uint64_t tiles = (taskInfo.totalRows + kvBaseTile_ - 1U) / kvBaseTile_;
            const uint64_t taskBegin = taskPipeline ? produced : 0U;
#ifdef __DAV_VEC__
            rescale.SetMxVScale(vScale[taskInfo.vScaleOffset]);
            for (uint32_t prepare = 0U; prepare < DESCRIPTOR_AHEAD && prepare < tiles; ++prepare) {
                const uint32_t descriptorSlot = (taskBegin + prepare) % DESCRIPTOR_STAGES;
                const uint32_t futureRows =
                    static_cast<uint32_t>(taskInfo.totalRows - prepare * kvBaseTile_ < kvBaseTile_ ?
                                              taskInfo.totalRows - prepare * kvBaseTile_ :
                                              kvBaseTile_);
                const uint32_t rawSlot = KvProducer::GroupedRawSlot(taskBegin + prepare, rawStages);
                Cursor future = cursor;
                if (taskInfo.directIndexMap) {
                    future.raw = prepare * kvBaseTile_ / blockShapeY_;
                    future.inner = 0U;
                } else {
                    future.Advance(prepare * kvBaseTile_);
                }
                kvProducer.PrepareGroupedTile(future, recordsUb, descriptors[descriptorSlot * descriptorTileWords],
                                              kScale[taskInfo.scaleBase], streamRaw[rawSlot], futureRows, taskInfo.head,
                                              prepare * kvBaseTile_, tokenStride, descriptorSlot, rawSlot,
                                              taskInfo.directIndexMap, streamPacked,
                                              l1Scale[(taskBegin + prepare) % RING], prepare == 0U);
            }
#endif
#ifdef __DAV_CUBE__
            qk.SetQueryTileCount(max(tiles, static_cast<uint64_t>(2U)));
            qk.SetMxGeometry(embed_, queryScaleHeadStride, KvHeadMajor ? groups64 : kvHeads_ * groups64);
            qk.SetMxQScale(qScale[taskInfo.qHeadOffset * groups64 * 2U]);
            auto query =
                tla::MakeTensor(gQ[taskInfo.qOffset], tla::MakeLayout<ElementQ, LayoutQ>(qBaseTile_, queryHeadStride),
                                Arch::PositionGM{});
            qk.loadQGM(query, GemmCoord{groupSize_, embed_, 0});
            auto key =
                tla::MakeTensor(gK[taskInfo.kvHeadOffset], tla::MakeLayout<ElementK, LayoutK>(tokenStride, kvBaseTile_),
                                Arch::PositionGM{});
            auto value =
                tla::MakeTensor(gV[taskInfo.kvHeadOffset], tla::MakeLayout<ElementV, LayoutV>(kvBaseTile_, tokenStride),
                                Arch::PositionGM{});
#endif
#ifdef __DAV_CUBE__
            Cursor nextCursor;
#endif
#ifdef __DAV_VEC__

            while (taskPipeline && consumed < taskBegin) {
                // Finish the preceding output only after submitting the
                // next task's descriptor/scale warmup. This also protects
                // the original online-softmax statistics before their reset.
                ConsumePendingTile(pending[consumed % RING], consumed, produced, taskConsumer, gV, gO, gLse, vScale,
                                   ubO, tokenStride, queryHeadStride);
                ++consumed;
            }
#endif
            for (uint64_t i = 0; i < tiles + (taskPipeline ? 0U : PRE_LAUNCH); ++i) {
                const uint64_t sequence = taskBegin + i;
                if (i < tiles) {
                    const uint32_t ringSlot = sequence % RING;
                    const uint32_t rows = static_cast<uint32_t>(taskInfo.totalRows - i * kvBaseTile_ < kvBaseTile_ ?
                                                                    taskInfo.totalRows - i * kvBaseTile_ :
                                                                    kvBaseTile_);
#ifdef __DAV_CUBE__
                    pending[ringSlot].cursor = qk.HasPrefetched() ? nextCursor : cursor;
#else
                    pending[ringSlot].cursor = cursor;
#endif
                    pending[ringSlot].rows = rows;
                    if (taskInfo.directIndexMap) {
                        // Every grouped tile is divisible by the supported sparse block sizes.
                        cursor.raw = static_cast<uint32_t>((i + 1U) * kvBaseTile_ / blockShapeY_);
                        cursor.inner = 0U;
                    } else {
                        cursor.Advance(rows);
                    }
                    const uint32_t pSlot = sequence % pL1BufNum_;
                    const uint32_t sSlot = sequence % UB_STAGES;
                    Arch::CrossCoreFlag qkReady(sSlot), pReady(pSlot + UB_STAGES);
                    GemmCoord qkShape{qkRows, rows, embed_};
                    auto score = tla::MakeTensor(
                        ubS[sSlot], tla::MakeLayout<ElementS, LayoutS>(RoundUp(qkRows, 16U), RoundUp(rows, 16U)),
                        Arch::PositionUB{});
#ifdef __DAV_VEC__
                    {
                        // Fused descriptors have already submitted this Raw tile.
                        // Three Raw slots let prepare(i+2) use a different slot
                        // while the vector gathers of tile i are still in flight.
                        const uint32_t rawSlot = KvProducer::GroupedRawSlot(sequence, rawStages);
                        // Tile 0 was published during descriptor warmup.
                        if (i != 0U) {
                            kvProducer.PublishGroupedTile(rows, streamRaw[rawSlot], streamPacked, rawSlot,
                                                          l1Scale[ringSlot]);
                        }
                    }
                    // Publish before either AIV waits for descriptor(i+2).
                    // Four slots let Cube finish QK(i), PV(i-2) and release the
                    // descriptor(i+2) slot without requiring softmax(i) on V1.
                    if (i + DESCRIPTOR_AHEAD < tiles) {
                        const uint64_t prepare = i + DESCRIPTOR_AHEAD;
                        const uint32_t descriptorSlot = (taskBegin + prepare) % DESCRIPTOR_STAGES;
                        const uint32_t futureRows =
                            static_cast<uint32_t>(taskInfo.totalRows - prepare * kvBaseTile_ < kvBaseTile_ ?
                                                      taskInfo.totalRows - prepare * kvBaseTile_ :
                                                      kvBaseTile_);
                        const uint32_t rawSlot = KvProducer::GroupedRawSlot(taskBegin + prepare, rawStages);
                        Cursor future = cursor;
                        if (taskInfo.directIndexMap) {
                            future.raw = prepare * kvBaseTile_ / blockShapeY_;
                            future.inner = 0U;
                        } else {
                            future.Advance(kvBaseTile_);
                        }
                        kvProducer.PrepareGroupedTile(
                            future, recordsUb, descriptors[descriptorSlot * descriptorTileWords],
                            kScale[taskInfo.scaleBase], streamRaw[rawSlot], futureRows, taskInfo.head,
                            prepare * kvBaseTile_, tokenStride, descriptorSlot, rawSlot, taskInfo.directIndexMap,
                            streamPacked, l1Scale[(taskBegin + prepare) % RING], false);
                    }
#endif
#ifdef __DAV_CUBE__
                    if (!qk.HasPrefetched()) {
                        const uint32_t descriptorSlot = sequence % DESCRIPTOR_STAGES;
                        AscendC::CrossCoreWaitFlag<4, PIPE_S>(KvProducer::DescriptorFlag(descriptorSlot));
                        AscendC::CrossCoreWaitFlag<4, PIPE_S>(KvProducer::DescriptorFlag(descriptorSlot) - 16U);
                        auto info = descriptors[descriptorSlot * descriptorTileWords];
                        pending[ringSlot].cursor.packedRecords = info.GetPhyAddr();
                        pending[ringSlot].cursor.packedRecordCount = 0U;
                        pending[ringSlot].cursor.packedRows = rows;
                        pending[ringSlot].cursor.packedGroupWords =
                            Gemm::Tile::Mxfp8DescriptorLayout::DescriptorGroupWords();
                    }

                    qk.SetMxPacked(pending[ringSlot].cursor, l1Scale[ringSlot]);
                    qk(key, score, unusedIndices, qkShape, i % 6U, kvBaseTile_, blockShapeY_,
                       Prefix(i, tiles, mm1L0ATotalStages_, mm2L0ATotalStages_, true),
                       Prefix(i, tiles, mm1L0BTotalStages_, mm2L0BTotalStages_, true), qkReady, tokenStride);

                    if (i + 1U == tiles) {
                        qk.ReleaseQuery();
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
                    }
#endif
                    auto prob = tla::MakeTensor(l1P[pSlot],
                                                tla::MakeLayout<ElementP, layout::zN>(computeRows, RoundUp(rows, 64U)),
                                                Arch::PositionL1{});
#ifdef __DAV_VEC__
                    softmax(prob, GemmCoord{computeRows, rows, embed_}, i == 0U, sSlot, pSlot, qkReady, pReady,
                            RoundUp(rows, 64U));
#endif
                    if (taskPipeline) {
                        auto& item = pending[ringSlot];
                        item.outputOffset = taskInfo.qOffset;
                        item.rows = rows;
                        item.computeRows = computeRows;
                        item.valueOffset = taskInfo.kvHeadOffset;
                        item.valueScaleOffset = taskInfo.vScaleOffset;
                        item.first = i == 0U;
                        item.last = i + 1U == tiles;
                        ++produced;
                    }
                }
                if (taskPipeline && produced - consumed > PRE_LAUNCH) {
                    ConsumePendingTile(pending[consumed % RING], consumed, produced, taskConsumer, gV, gO, gLse, vScale,
                                       ubO, tokenStride, queryHeadStride);
                    ++consumed;
                } else if (!taskPipeline && i >= PRE_LAUNCH) {
                    const uint64_t done = i - PRE_LAUNCH;
                    const uint32_t pSlot = done % pL1BufNum_;
                    const uint32_t oSlot = done % uBufTileHelper_.loUbBufNum;
                    Arch::CrossCoreFlag pReady(pSlot + UB_STAGES), oReady(oSlot + UB_STAGES + pL1BufNum_);
                    GemmCoord pvShape{computeRows, embed_, pending[done % RING].rows};
#ifdef __DAV_CUBE__
                    pv.SetMxPacked(pending[done % RING].cursor);
                    auto partial = tla::MakeTensor(
                        ubO[oSlot], tla::MakeLayout<ElementOTmp, LayoutOTmp>(RoundUp(computeRows, 16U), embed_),
                        Arch::PositionUB{});
                    pv(value, partial, unusedIndices, pvShape, done % 6U, kvBaseTile_, blockShapeY_,
                       Prefix(done, tiles, mm1L0ATotalStages_, mm2L0ATotalStages_, false),
                       Prefix(done, tiles, mm1L0BTotalStages_, mm2L0BTotalStages_, false), pReady, oReady, tokenStride);
                    // PV has now consumed every scalar descriptor field.
                    // K/V data DMA no longer reads descriptor workspace.
                    AscendC::CrossCoreSetFlag<4, PIPE_S>(KvProducer::DescriptorFlag(done % DESCRIPTOR_STAGES));
                    AscendC::CrossCoreSetFlag<4, PIPE_S>(KvProducer::DescriptorFlag(done % DESCRIPTOR_STAGES) - 16U);
#endif
#ifdef __DAV_VEC__
                    if (taskInfo.isFdPartial) {
                        rescale.ProcessPartial(out, lse, pvShape, done % RING, done % 6U, done == 0U,
                                               done + 1U == tiles, oReady, gPartialO, gPartialLse,
                                               taskInfo.fdPartialTaskId, td->fdLseSubStride);
                    } else {
                        rescale(out, lse, pvShape, done % RING, done % 6U, done == 0U, done + 1U == tiles, oReady);
                    }
#endif
                }
#ifdef __DAV_CUBE__
                // Submit K(i+1) after current pending PV has submitted its V
                // copy. The descriptor is already produced by both AIVs at i-1, so
                // this wait never depends on softmax(i+1). PV keeps ownership
                // until its existing descriptor release; next QK skips wait.
                if (i + 1U < tiles) {
                    const uint64_t next = i + 1U;
                    const uint32_t nextRows =
                        min(static_cast<uint64_t>(kvBaseTile_), taskInfo.totalRows - next * kvBaseTile_);
                    nextCursor = cursor;
                    const uint32_t slot = (taskBegin + next) % DESCRIPTOR_STAGES;
                    AscendC::CrossCoreWaitFlag<4, PIPE_S>(KvProducer::DescriptorFlag(slot));
                    AscendC::CrossCoreWaitFlag<4, PIPE_S>(KvProducer::DescriptorFlag(slot) - 16U);
                    auto info = descriptors[slot * descriptorTileWords];
                    nextCursor.packedRecords = info.GetPhyAddr();
                    nextCursor.packedRecordCount = 0U;
                    nextCursor.packedRows = nextRows;
                    nextCursor.packedGroupWords = Gemm::Tile::Mxfp8DescriptorLayout::DescriptorGroupWords();
                    qk.Preload(key, nextCursor, nextRows, embed_, next % 6U, kvBaseTile_, blockShapeY_, tokenStride);
                }
#endif
            }
        }
        // Only the end of this core's complete task stream drains its tail.
        if (taskPipeline) {
            while (consumed < produced) {
                ConsumePendingTile(pending[consumed % RING], consumed, produced, taskConsumer, gV, gO, gLse, vScale,
                                   ubO, tokenStride, queryHeadStride);
                ++consumed;
            }
        }
#ifdef __DAV_VEC__
        for (uint32_t i = 0; i < scaleCredits; ++i) {
            AscendC::CrossCoreWaitFlag<4, PIPE_MTE3>(SCALE_FLAG + i);
        }
        kvProducer.ReleaseStreamScalePipeline(rawStages);
        for (uint32_t i = 0; i < DESCRIPTOR_STAGES; ++i) {
            AscendC::CrossCoreWaitFlag<4, PIPE_S>(KvProducer::DescriptorFlag(i));
        }
        const uint32_t recordSlots = GROUPED_RECORD_UB_STAGES;
        for (uint32_t slot = 0U; slot < recordSlots; ++slot) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID5 + slot);
        }
#endif
        ReleaseSyncFlags<4, 4, 4>();
        if (fdEnabled) {
            // Inactive FD cores must also reach this barrier. All partial GM
            // writes finish before the combine reuses the drained UB buffers.
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::SyncAll<false>();
#ifdef __DAV_VEC__
            GenericBlockSparseAttentionFdCombineArch35<ElementO, Arch::Resource<ArchTag>, false, QFormat> combine(
                resource);
            combine(meta, td, gPartialLse, gPartialO, gO, gLse, cuQ, usedQ, params.sequsedQ != nullptr);
#endif
        }
    }

    // Resource initialization and synchronization use the same buffers as Run().
    __aicore__ inline void FetchBaseShapeInfo(
        const GenericBlockSparseAttn::GenericBlockSparseAttentionTilingData* tilingData)
    {
        qHeads_ = tilingData->numHeads;
        kvHeads_ = tilingData->kvHeads;
        embed_ = tilingData->embeddingSize;
        blockShapeY_ = tilingData->blockShapeY;
        qBlockNum_ = tilingData->qBlockNum;
        topK_ = tilingData->topK;
        scaleValue_ = tilingData->scaleValue;
        groupSize_ = tilingData->groupSize;
        qBaseTile_ = tilingData->baseTileInfo.qBaseTile;
        kvBaseTile_ = tilingData->baseTileInfo.kvBaseTile;
    }

    __aicore__ inline void CalcOnChipBufTileInfo(
        const GenericBlockSparseAttn::GenericBlockSparseAttentionTilingData* tilingData)
    {
        mm1L1TileM_ = tilingData->mmPhaseL1TileInfo.mm1L1TileM;
        mm1L1TileN_ = tilingData->mmPhaseL1TileInfo.mm1L1TileN;
        mm1L1TileKLeft_ = tilingData->mmPhaseL1TileInfo.mm1L1TileKLeft;
        mm1L1TileKRight_ = tilingData->mmPhaseL1TileInfo.mm1L1TileKRight;
        mm2L1TileM_ = tilingData->mmPhaseL1TileInfo.mm2L1TileM;
        mm2L1TileN_ = tilingData->mmPhaseL1TileInfo.mm2L1TileN;
        mm2L1TileKLeft_ = tilingData->mmPhaseL1TileInfo.mm2L1TileKLeft;
        mm2L1TileKRight_ = tilingData->mmPhaseL1TileInfo.mm2L1TileKRight;
        qL1BufNum_ = tilingData->mmPhaseL1TileInfo.qL1BufNum;
        kL1BufNum_ = tilingData->mmPhaseL1TileInfo.kL1BufNum;
        vL1BufNum_ = tilingData->mmPhaseL1TileInfo.vL1BufNum;
        pL1BufNum_ = tilingData->mmPhaseL1TileInfo.pL1BufNum;
        Gemm::Block::Mxfp8Mm1L1TileHelper mm1L1TileHelper(mm1L1TileM_, mm1L1TileN_, mm1L1TileKLeft_, mm1L1TileKRight_,
                                                          qL1BufNum_, kL1BufNum_);
        mm1L1TileHelper_ = mm1L1TileHelper;
        Gemm::Block::Mxfp8Mm2L1TileHelper mm2L1TileHelper(mm2L1TileM_, mm2L1TileN_, mm2L1TileKLeft_, mm2L1TileKRight_,
                                                          pL1BufNum_, vL1BufNum_);
        mm2L1TileHelper_ = mm2L1TileHelper;
        mm2L1AddrStart_ = mm1L1TileM_ * mm1L1TileKLeft_ * qL1BufNum_ * sizeof(ElementQ) +
                          mm1L1TileKRight_ * mm1L1TileN_ * kL1BufNum_ * sizeof(ElementK);
        uint32_t mL0LoopQK = CeilDiv(groupSize_, static_cast<uint32_t>(BlockMmadQK::L0_TILE_M));
        uint32_t mL0LoopPV = CeilDiv(groupSize_, static_cast<uint32_t>(BlockMmadPV::L0_TILE_M));
        const uint32_t qkKStages = CeilDiv(embed_, static_cast<uint32_t>(BlockMmadQK::L0_TILE_K));
        const uint32_t qkNStages = CeilDiv(kvBaseTile_, static_cast<uint32_t>(BlockMmadQK::L0_TILE_N));
        const uint32_t pvKStages = CeilDiv(kvBaseTile_, static_cast<uint32_t>(BlockMmadPV::L0_TILE_K));
        const uint32_t pvNStages = CeilDiv(embed_, static_cast<uint32_t>(BlockMmadPV::L0_TILE_N));
        mm1L0ATotalStages_ = mL0LoopQK * qkKStages;
        mm1L0BTotalStages_ = qkNStages * qkKStages;
        mm2L0ATotalStages_ = mL0LoopPV * pvKStages;
        mm2L0BTotalStages_ = pvKStages * pvNStages;
    }

    __aicore__ inline void InitCrossCoreDstBuf(AscendC::LocalTensor<ElementP> (&l1PTensor)[RING],
                                               AscendC::LocalTensor<ElementS> (&ubSTensor)[UB_STAGES],
                                               AscendC::LocalTensor<ElementOTmp> (&ubOTmpTensor)[UB_STAGES])
    {
        for (uint32_t i = 0; i < pL1BufNum_; i++) {
            l1PTensor[i] = resource.l1Buf.template GetBufferByByte<ElementP>(
                mm2L1AddrStart_ + mm2L1TileM_ * mm2L1TileKLeft_ * sizeof(ElementP) * i);
        }
        for (uint32_t i = 0; i < UB_STAGES; i++) {
            ubSTensor[i] = resource.ubBuf.template GetBufferByByte<ElementS>(
                uBufTileHelper_.sStartOffset +
                uBufTileHelper_.qBaseTilePerSubCore * uBufTileHelper_.kvBaseTilePerSubCore * sizeof(ElementS) * i);
            ubOTmpTensor[i] = resource.ubBuf.template GetBufferByByte<ElementOTmp>(
                uBufTileHelper_.loStartOffset + uBufTileHelper_.qBaseTilePerSubCore * uBufTileHelper_.loUbColCapacity *
                                                    sizeof(ElementOTmp) * (i % uBufTileHelper_.loUbBufNum));
        }
    }

    template <uint32_t MM1_SM_MODE, uint32_t MM2_RE_MODE, uint32_t SM_MM2_MODE>
    __aicore__ inline void InitSyncFlags()
    {
#ifdef __DAV_CUBE__
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID3);
        if constexpr (SM_MM2_MODE == 4U) {
            AscendC::CrossCoreSetFlag<SM_MM2_MODE, PIPE_MTE1>(2);
            AscendC::CrossCoreSetFlag<SM_MM2_MODE, PIPE_MTE1>(18);
            AscendC::CrossCoreSetFlag<SM_MM2_MODE, PIPE_MTE1>(3);
            AscendC::CrossCoreSetFlag<SM_MM2_MODE, PIPE_MTE1>(19);
            AscendC::CrossCoreSetFlag<SM_MM2_MODE, PIPE_MTE1>(4);
            AscendC::CrossCoreSetFlag<SM_MM2_MODE, PIPE_MTE1>(20);
        }
#endif
#ifdef __DAV_VEC__
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID1);
        if constexpr (MM1_SM_MODE == 4U) {
            AscendC::CrossCoreSetFlag<MM1_SM_MODE, PIPE_V>(0);
            AscendC::CrossCoreSetFlag<MM1_SM_MODE, PIPE_V>(1);
        }
        if constexpr (MM2_RE_MODE == 4U) {
            const uint32_t pvStages = uBufTileHelper_.loUbBufNum;
            for (uint32_t slot = 0U; slot < pvStages; ++slot) {
                AscendC::CrossCoreSetFlag<MM2_RE_MODE, PIPE_V>(5U + slot);
            }
        }
#endif
    }

    template <uint32_t MM1_SM_MODE, uint32_t MM2_RE_MODE, uint32_t SM_MM2_MODE>
    __aicore__ inline void ReleaseSyncFlags()
    {
#ifdef __DAV_CUBE__
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID3);
        if constexpr (MM1_SM_MODE == 4U) {
            AscendC::CrossCoreWaitFlag<MM1_SM_MODE, PIPE_FIX>(0);
            AscendC::CrossCoreWaitFlag<MM1_SM_MODE, PIPE_FIX>(1);
            AscendC::CrossCoreWaitFlag<MM1_SM_MODE, PIPE_FIX>(16);
            AscendC::CrossCoreWaitFlag<MM1_SM_MODE, PIPE_FIX>(17);
        }
        if constexpr (MM2_RE_MODE == 4U) {
            const uint32_t pvStages = uBufTileHelper_.loUbBufNum;
            for (uint32_t slot = 0U; slot < pvStages; ++slot) {
                AscendC::CrossCoreWaitFlag<MM2_RE_MODE, PIPE_FIX>(5U + slot);
                AscendC::CrossCoreWaitFlag<MM2_RE_MODE, PIPE_FIX>(21U + slot);
            }
        }
#endif
#ifdef __DAV_VEC__
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID1);
        if constexpr (SM_MM2_MODE == 4U) {
            AscendC::CrossCoreWaitFlag<SM_MM2_MODE, PIPE_MTE3>(2);
            AscendC::CrossCoreWaitFlag<SM_MM2_MODE, PIPE_MTE3>(3);
            AscendC::CrossCoreWaitFlag<SM_MM2_MODE, PIPE_MTE3>(4);
        }
#endif
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    // Kernel-owned resources. Preserve member order and buffer geometry.
    Arch::Resource<ArchTag> resource;
    uint32_t qHeads_;
    uint32_t kvHeads_;
    uint32_t embed_;
    uint32_t blockShapeY_;
    uint32_t qBlockNum_;
    uint32_t topK_;
    float scaleValue_;
    uint32_t groupSize_;
    // base tile info
    uint32_t qBaseTile_;
    uint32_t kvBaseTile_;
    // L1 tile info
    uint32_t mm1L1TileM_;
    uint32_t mm1L1TileN_;
    uint32_t mm1L1TileKLeft_;
    uint32_t mm1L1TileKRight_;
    uint32_t mm2L1TileM_;
    uint32_t mm2L1TileN_;
    uint32_t mm2L1TileKLeft_;
    uint32_t mm2L1TileKRight_;
    uint32_t qL1BufNum_;
    uint32_t kL1BufNum_;
    uint32_t vL1BufNum_;
    uint32_t pL1BufNum_;
    uint32_t mm1L0ATotalStages_;
    uint32_t mm1L0BTotalStages_;
    uint32_t mm2L0ATotalStages_;
    uint32_t mm2L0BTotalStages_;
    uint32_t mm2L1AddrStart_ = 0;
    Gemm::Block::Mxfp8Mm1L1TileHelper mm1L1TileHelper_;
    Gemm::Block::Mxfp8Mm2L1TileHelper mm2L1TileHelper_;
    Epilogue::Block::UBufTileHelper uBufTileHelper_;
};
} // namespace GbsaKernelArch35
#endif
