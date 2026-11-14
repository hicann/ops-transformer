/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../arch35/kernel_utils.hpp"
#include "../kernel_common.hpp"
#include "../generic_block_sparse_attention_metadata_kernel.h"
#include "../generic_block_sparse_attention_fd_utils.h"
#include "generic_block_sparse_attention_fd_combine_arch35.h"
#include "../attn_infra/epilogue/block/gbsa_block_epilogue_online_softmax_arch35_reg_a8c4.hpp"
#include "../attn_infra/epilogue/block/gbsa_block_epilogue_kv_dequant_arch35_a8c4.hpp"

using namespace NpuArch;
using namespace tla;

namespace GbsaKernelArch35 {

template <class BlockMmadQK, class EpilogueOnlineSoftmax, class BlockMmadPV, class EpilogueRescaleO, Format qFormat,
          Format kvFormat, bool isPaged, bool hasResidualBlock, bool isCausal = true>
class GbsaA8C4KernelArch35 {
public:
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
    using LayoutP = layout::RowMajor;
    using LayoutV = layout::RowMajor;
    using LayoutO = layout::RowMajor;
    using LayoutOTmp = layout::RowMajor;
    using LayoutLse = layout::RowMajor;

    static constexpr uint32_t PRE_LAUNCH = 2;
    static constexpr uint32_t PV_CONSUME_DELAY = 1U;
    static constexpr uint32_t MAX_CROSS_CORE_BUF_STAGES = PRE_LAUNCH + 1;
    static constexpr uint32_t UB_S_OTMP_BUF_STAGES = 2;
    static constexpr uint32_t PENDING_STAGES = PRE_LAUNCH + 2U;
    static constexpr uint32_t STATS_BYTES = Epilogue::Block::UBufTileHelper::STATS_ROW_NUM * sizeof(float);
    static constexpr uint32_t DESCRIPTOR_RECORD_BYTES = 64U;
    static constexpr uint32_t MAX_SEGMENTS = 512U / 128U;
    static constexpr uint32_t DESCRIPTOR_WORDS_PER_TILE = MAX_SEGMENTS * DESCRIPTOR_RECORD_BYTES / sizeof(uint64_t);
    static constexpr uint32_t UB_LIMIT_BYTES = 248U * 1024U;
    static constexpr uint32_t L1_LIMIT_BYTES = 512U * 1024U;

    __aicore__ inline GbsaA8C4KernelArch35() {}

    __aicore__ inline void operator()(GbsaFullQuantKernelParamsArch35 const& params)
    {
        GET_TILING_DATA_WITH_STRUCT(GenericBlockSparseAttn::GenericBlockSparseAttentionTilingData, tilingDataLocal,
                                    params.tiling);
        const GenericBlockSparseAttn::GenericBlockSparseAttentionTilingData* tilingData = &tilingDataLocal;
        FetchBaseShapeInfo(tilingData, params.metaData);
        CalcOnChipBufTileInfo(tilingData);
        CalcUBufTileInfo();
        // Host and device use identical tile sizes; reject an invalid layout before
        // publishing any mailbox tokens (including the two AIV halves).
        if (embed_ != 128U || kvBaseTile_ != 512U || kL1BufNum_ < 2U || vL1BufNum_ < 2U ||
            dequantOffset_ + 2U * dequantBytes_ > UB_LIMIT_BYTES ||
            mm2L1AddrStart_ + mm2L1TileM_ * mm2L1TileKLeft_ * pL1BufNum_ + mm2L1TileN_ * mm2L1TileKRight_ * vL1BufNum_ >
                L1_LIMIT_BYTES) {
            return;
        }
        __gm__ const GbsaMetadata::Metadata* meta =
            reinterpret_cast<__gm__ const GbsaMetadata::Metadata*>(params.metaData);
        uint32_t coreIdx = AscendC::GetBlockIdx();
        const uint32_t coreNum = AscendC::GetBlockNum();
        if (!GsaFd::ValidateMetadata<isCausal>(meta, tilingData, coreNum)) {
            return;
        }
        const bool fdEnabled = tilingData->fdStaticEnabled != 0U &&
                               (static_cast<uint32_t>(meta->fdScheduleFlags) & GbsaMetadata::FD_SCHEDULE_ENABLED) != 0U;

        AscendC::GlobalTensor<ElementQ> gQ;
        gQ.SetGlobalBuffer((__gm__ ElementQ*)params.q);
        AscendC::GlobalTensor<ElementK> gK;
        gK.SetGlobalBuffer((__gm__ ElementK*)params.k);
        AscendC::GlobalTensor<ElementK> gV;
        gV.SetGlobalBuffer((__gm__ ElementK*)params.v);
        AscendC::GlobalTensor<int32_t> gSparseBlockIdx;
        gSparseBlockIdx.SetGlobalBuffer((__gm__ int32_t*)params.sparseBlockIdx);
        AscendC::GlobalTensor<int32_t> gBlockTable;
        if constexpr (isPaged) {
            gBlockTable.SetGlobalBuffer((__gm__ int32_t*)params.blockTable);
        }
        AscendC::GlobalTensor<int64_t> gCuSeqLengthsKv;
        if constexpr (!isPaged && kvFormat == Format::TND) {
            gCuSeqLengthsKv.SetGlobalBuffer((__gm__ int64_t*)params.cuSeqLengthsKv);
        }
        AscendC::GlobalTensor<int32_t> gSparseBlockCount;
        gSparseBlockCount.SetGlobalBuffer((__gm__ int32_t*)params.sparseBlockCount);
        AscendC::GlobalTensor<int64_t> gCuSeqLengths;
        if constexpr (qFormat == Format::TND) {
            gCuSeqLengths.SetGlobalBuffer((__gm__ int64_t*)params.cuSeqLengths);
        }
        AscendC::GlobalTensor<int32_t> gSequsedQ;
        const bool hasSequsedQ = qFormat != Format::TND || params.sequsedQ != nullptr;
        if (hasSequsedQ) {
            gSequsedQ.SetGlobalBuffer((__gm__ int32_t*)params.sequsedQ);
        }
        AscendC::GlobalTensor<int32_t> gSequsedKv;
        if (kvFormat != Format::TND || params.sequsedKv != nullptr) {
            gSequsedKv.SetGlobalBuffer((__gm__ int32_t*)params.sequsedKv);
        }
        const bool hasSequsedKv = kvFormat != Format::TND || params.sequsedKv != nullptr;
        AscendC::GlobalTensor<ElementO> gO;
        gO.SetGlobalBuffer((__gm__ ElementO*)params.o);
        AscendC::GlobalTensor<ElementLse> gLse;
        gLse.SetGlobalBuffer((__gm__ ElementLse*)params.softmaxLse);
        AscendC::GlobalTensor<float> gPartialLse;
        gPartialLse.SetGlobalBuffer((__gm__ float*)(params.workSpace + tilingData->fdPartialLseOffset));
        AscendC::GlobalTensor<float> gPartialO;
        gPartialO.SetGlobalBuffer((__gm__ float*)(params.workSpace + tilingData->fdPartialOOffset));
        AscendC::GlobalTensor<uint8_t> gPackedK, gPackedV;
        gPackedK.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(params.k));
        gPackedV.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(params.v));
        AscendC::GlobalTensor<uint8_t> gKScale, gVScale;
        gKScale.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(params.kDequantScale));
        gVScale.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(params.vDequantScale));

        AscendC::LocalTensor<ElementP> l1PTensor[MAX_CROSS_CORE_BUF_STAGES];
        AscendC::LocalTensor<ElementS> ubSTensor[UB_S_OTMP_BUF_STAGES];
        AscendC::LocalTensor<ElementOTmp> ubOTmpTensor[UB_S_OTMP_BUF_STAGES];
        InitCrossCoreDstBuf(l1PTensor, ubSTensor, ubOTmpTensor);

        const uint64_t strideKVRow = GbsaLayoutHelper<kvFormat>::TokenStride(kvHeads_, embed_);

        InitSyncFlags<4, 4, 4>();

#ifdef __DAV_CUBE__
        coreIdx = AscendC::GetBlockIdx();
#endif
#ifdef __DAV_CUBE__
        BlockMmadQK blockMmadQK(resource, mm1L1TileHelper_);
        BlockMmadPV blockMmadPV(resource, mm2L1AddrStart_, mm2L1TileHelper_);
        blockMmadQK.ConfigureA8C4(resource);
        blockMmadPV.ConfigureA8C4(resource);
#endif
#ifdef __DAV_VEC__
        coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        EpilogueOnlineSoftmax epilogueOnlineSoftmax(resource, scaleValue_, uBufTileHelper_);
        EpilogueRescaleO epilogueRescaleOPadding(resource, uBufTileHelper_);
#endif

        uint32_t groupSize = groupSize_;
        uint64_t strideQHead = GbsaLayoutHelper<qFormat>::HeadStride(tilingData->qSeqCapacity, embed_);
        const uint32_t strideLseHead = qFormat == Format::TND ? 1U : tilingData->qSeqCapacity;
        uint32_t embedRound = RoundUp(embed_, 16);

#ifdef __DAV_VEC__
        if (hasSequsedQ) {
            uint32_t paddingTask = 0U;
            for (uint32_t batchIdx = 0U; batchIdx < batch_; ++batchIdx) {
                BatchSeqInfo qSeq;
                ReadBatchSeqInfo<qFormat>(gCuSeqLengths, gSequsedQ, hasSequsedQ, batchIdx, qSeq,
                                          tilingData->qSeqCapacity);
                const uint32_t storageLen = qSeq.storageLen;
                const uint32_t actualLen = qSeq.actualLen;
                for (uint32_t token = actualLen; token < storageLen; ++token) {
                    for (uint32_t kvHeadIdx = 0U; kvHeadIdx < kvHeads_; ++kvHeadIdx, ++paddingTask) {
                        if (paddingTask % coreNum != coreIdx) {
                            continue;
                        }
                        const uint32_t qHeadStart = kvHeadIdx * groupSize;
                        const uint64_t gmOffsetO = GetTensorOffset<qFormat>(qSeq, token, qHeadStart, qHeads_, embed_);
                        const uint64_t gmOffsetLse = GetLseOffset<qFormat>(qSeq, token, qHeadStart, qHeads_);
                        auto gmOLayout = tla::MakeLayout<ElementO, LayoutO>(qBaseTile_, strideQHead);
                        auto gmOTensor = tla::MakeTensor(gO[gmOffsetO], gmOLayout, Arch::PositionGM{});
                        auto gmLseLayout = tla::MakeLayout<ElementLse, LayoutLse>(qBaseTile_, strideLseHead);
                        auto gmLseTensor = tla::MakeTensor(gLse[gmOffsetLse], gmLseLayout, Arch::PositionGM{});
                        epilogueRescaleOPadding.WriteEmptyOutput(gmOTensor, gmLseTensor,
                                                                 GemmCoord{groupSize, embed_, 0});
                    }
                }
            }
        }
#endif

        uint32_t taskLoopStart = coreIdx;
        uint32_t taskLoopEnd = totalTaskNum_;
        uint32_t taskLoopStep = coreNum;
        uint32_t scheduleFirstBlock = 0U;
        uint32_t scheduleLastBlock = 0U;
        if (fdEnabled) {
            taskLoopStart = totalTaskNum_;
            taskLoopEnd = totalTaskNum_;
            taskLoopStep = 1U;
            if (coreIdx < static_cast<uint32_t>(meta->fdActiveCoreNum)) {
                const __gm__ GbsaMetadata::DecodeSchedule& schedule = meta->decodeSchedules[coreIdx];
                taskLoopStart = static_cast<uint32_t>(schedule.baseTaskStart);
                taskLoopEnd = static_cast<uint32_t>(schedule.baseTaskEnd);
                scheduleFirstBlock = static_cast<uint32_t>(schedule.firstBlockStart);
                scheduleLastBlock = static_cast<uint32_t>(schedule.lastBlockEnd);
            }
        }

        {
            const uint32_t storageHeads = GbsaLayoutHelper<kvFormat>::TokenStride(kvHeads_, embed_) / embed_;
            using KvDequant = Epilogue::Block::A8C4KvDequant;
            const bool preloadKv = isPaged && !fdEnabled;
            // With one scheduled task, the final max/sum are still live when its
            // last PV is consumed. Only cross-task lookahead needs a snapshot.
            const bool needsStatsSnapshot = !preloadKv && taskLoopEnd - taskLoopStart > taskLoopStep;
#ifdef __DAV_VEC__
            const bool singleFdTask = isCausal && fdEnabled && taskLoopEnd - taskLoopStart == 1U;
            constexpr uint32_t fdValueSlot = 1U;
            KvDequant dequantK(resource, dequantOffset_, storageHeads);
            // K and V use the same two slots in issue order. Each AIV owns its slots.
            uint32_t dequantSequence = 0U;
#endif
            // One descriptor of lookahead is retained across task boundaries. Slot rotation
            // uses the global sequence; sparse addressing always uses tile.tileId.
            struct PipelineTile {
                uint64_t qOffset = 0, oOffset = 0, lseOffset = 0, kvOffset = 0;
                BatchSeqInfo kvSeq{};
                AscendC::GlobalTensor<int32_t> sparse, pages;
                uint32_t head = 0, tileId = 0, tileCount = 0, tokens = 0;
                uint32_t explicitCount = 0, residual = 0, partialId = 0;
                bool partial = false;
            };
            PipelineTile pending[PENDING_STAGES];
            uint32_t sequence = 0U;
            uint64_t qkAStages = 0U, qkBStages = 0U, pvAStages = 0U, pvBStages = 0U;

            auto produceKey = [&](const PipelineTile& tile, uint32_t seq) __aicore__ {
#ifdef __DAV_VEC__
                if (tile.tokens != 0U) {
                    const uint32_t slot = dequantSequence++ % 2U;
                    ProduceKvTile<true>(dequantK, gPackedK, gKScale, tile.sparse, tile.pages, tile.kvSeq, tile.head,
                                        tile.tileId, tile.tokens, kStride0_, seq, slot, preloadKv);
                    if (singleFdTask && tile.partial && tile.tileCount == 1U && tile.tokens == kvBaseTile_) {
                        auto desc = resource.ubBuf.template GetBufferByByte<uint64_t>(descriptorOffset_);
                        const uint32_t base = (seq % PENDING_STAGES) * DESCRIPTOR_WORDS_PER_TILE;
                        dequantK.template LoadTile<DESCRIPTOR_RECORD_BYTES / sizeof(uint64_t), 1U, isPaged ? 3U : 2U>(
                            gPackedV, gVScale, desc[base], tile.tokens, fdValueSlot);
                    }
                }
#endif
            };
            auto produceValue = [&](const PipelineTile& tile, uint32_t seq) __aicore__ {
#ifdef __DAV_VEC__
                if (tile.tokens != 0U) {
                    const uint32_t slot = dequantSequence++ % 2U;
                    ProduceKvTile<false>(dequantK, gPackedV, gVScale, tile.sparse, tile.pages, tile.kvSeq, tile.head,
                                         tile.tileId, tile.tokens, vStride0_, seq, slot);
                }
#endif
            };
            auto consume = [&](uint32_t seq) __aicore__ {
                const auto& tile = pending[seq % PENDING_STAGES];
                const uint32_t pSlot = seq % pL1BufNum_;
                const uint32_t oSlot = seq % UB_S_OTMP_BUF_STAGES;
                Arch::CrossCoreFlag pFlag(pSlot + UB_S_OTMP_BUF_STAGES);
                Arch::CrossCoreFlag oFlag(oSlot + UB_S_OTMP_BUF_STAGES + pL1BufNum_);
#ifdef __DAV_CUBE__
                if (tile.tokens != 0U) {
                    WaitKvReady(11U + seq % vL1BufNum_);
                    auto outLayout = tla::MakeLayout<ElementOTmp, LayoutOTmp>(RoundUp(groupSize, 16U), embedRound);
                    auto out = tla::MakeTensor(ubOTmpTensor[oSlot], outLayout, Arch::PositionUB{});
                    auto vLayout = tla::MakeLayout<ElementV, LayoutV>(kvBaseTile_, strideKVRow);
                    auto value = tla::MakeTensor(gV[tile.kvOffset], vLayout, Arch::PositionGM{});
                    blockMmadPV(value, out, tile.sparse, GemmCoord{groupSize, embed_, tile.tokens}, seq, kvBaseTile_,
                                blockShapeY_, pvAStages, pvBStages, pFlag, oFlag, strideKVRow, tile.pages, blockSize_,
                                vStride0_, tile.explicitCount, tile.residual);
                    pvAStages += CeilDiv(groupSize, static_cast<uint32_t>(BlockMmadPV::L0_TILE_M)) *
                                 CeilDiv(tile.tokens, static_cast<uint32_t>(BlockMmadPV::L0_TILE_K));
                    pvBStages += CeilDiv(embed_, static_cast<uint32_t>(BlockMmadPV::L0_TILE_N)) *
                                 CeilDiv(tile.tokens, static_cast<uint32_t>(BlockMmadPV::L0_TILE_K));
                    ReleaseKv(11U + seq % vL1BufNum_);
                }
#endif
#ifdef __DAV_VEC__
                auto rescaleOLayout = uBufTileHelper_;
                // The preload path consumes i before Softmax(i+1) can overwrite max/sum.
                if (needsStatsSnapshot) {
                    rescaleOLayout.gmStartOffset = statsSnapshotOffset_ + pSlot * 2U * STATS_BYTES;
                    rescaleOLayout.glStartOffset = rescaleOLayout.gmStartOffset + STATS_BYTES;
                }
                EpilogueRescaleO epilogueRescaleO(resource, rescaleOLayout);
                auto oLayout = tla::MakeLayout<ElementO, LayoutO>(qBaseTile_, strideQHead);
                auto out = tla::MakeTensor(gO[tile.oOffset], oLayout, Arch::PositionGM{});
                auto lseLayout = tla::MakeLayout<ElementLse, LayoutLse>(qBaseTile_, strideLseHead);
                auto lse = tla::MakeTensor(gLse[tile.lseOffset], lseLayout, Arch::PositionGM{});
                if (tile.tokens == 0U) {
                    if (tile.partial) {
                        epilogueRescaleO.WriteNeutralPartial(gPartialO, gPartialLse, tile.partialId, groupSize, embed_,
                                                             tilingData->fdLseSubStride);
                    } else {
                        epilogueRescaleO.WriteEmptyOutput(out, lse, GemmCoord{groupSize, embed_, 0});
                    }
                } else if (tile.partial) {
                    epilogueRescaleO.ProcessPartial(out, lse, GemmCoord{groupSize, embed_, tile.tokens}, pSlot, seq,
                                                    tile.tileId == 0U, tile.tileId + 1U == tile.tileCount, oFlag,
                                                    gPartialO, gPartialLse, tile.partialId, tilingData->fdLseSubStride);
                } else {
                    epilogueRescaleO(out, lse, GemmCoord{groupSize, embed_, tile.tokens}, pSlot, seq, tile.tileId == 0U,
                                     tile.tileId + 1U == tile.tileCount, oFlag);
                }
#endif
            };
            auto step = [&](uint32_t seq) __aicore__ {
                const auto& tile = pending[seq % PENDING_STAGES];
                if (tile.tokens != 0U) {
                    const uint32_t sSlot = seq % UB_S_OTMP_BUF_STAGES;
                    const uint32_t pSlot = seq % pL1BufNum_;
                    Arch::CrossCoreFlag sFlag(sSlot), pFlag(pSlot + UB_S_OTMP_BUF_STAGES);
                    GemmCoord shape{groupSize, tile.tokens, embed_};
#ifdef __DAV_CUBE__
                    if (tile.tileId == 0U) {
                        auto qLayout = tla::MakeLayout<ElementQ, LayoutQ>(qBaseTile_, strideQHead);
                        auto query = tla::MakeTensor(gQ[tile.qOffset], qLayout, Arch::PositionGM{});
                        blockMmadQK.loadQGM(query, GemmCoord{groupSize, embed_, 0});
                    }
                    WaitKvReady(8U + seq % kL1BufNum_);
                    auto sLayout =
                        tla::MakeLayout<ElementS, LayoutS>(RoundUp(groupSize, 16U), RoundUp(tile.tokens, 16U));
                    auto score = tla::MakeTensor(ubSTensor[sSlot], sLayout, Arch::PositionUB{});
                    auto kLayout = tla::MakeLayout<ElementK, LayoutK>(strideKVRow, kvBaseTile_);
                    auto key = tla::MakeTensor(gK[tile.kvOffset], kLayout, Arch::PositionGM{});
                    blockMmadQK(key, score, tile.sparse, shape, seq, kvBaseTile_, blockShapeY_, qkAStages, qkBStages,
                                sFlag, strideKVRow, tile.pages, blockSize_, kStride0_, tile.explicitCount,
                                tile.residual);
                    qkAStages += CeilDiv(groupSize, static_cast<uint32_t>(BlockMmadQK::L0_TILE_M)) *
                                 CeilDiv(embed_, static_cast<uint32_t>(BlockMmadQK::L0_TILE_K));
                    qkBStages += CeilDiv(tile.tokens, static_cast<uint32_t>(BlockMmadQK::L0_TILE_N)) *
                                 CeilDiv(embed_, static_cast<uint32_t>(BlockMmadQK::L0_TILE_K));
                    ReleaseKv(8U + seq % kL1BufNum_);
                    if (tile.tileId + 1U == tile.tileCount) {
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
                    }
#endif
#ifdef __DAV_VEC__
                    auto pLayout = tla::MakeLayout<ElementP, NpuArch::layout::zN>(groupSize, tile.tokens);
                    auto probability = tla::MakeTensor(l1PTensor[pSlot], pLayout, Arch::PositionL1{});
                    const bool issueValueWithSoftmax =
                        singleFdTask && tile.partial && tile.tileCount == 1U && tile.tokens == kvBaseTile_;
                    if (issueValueWithSoftmax) {
                        const auto prepared = dequantK.PrepareTile(tile.tokens, fdValueSlot);
                        const uint32_t buffer = seq % vL1BufNum_;
                        const uint32_t valueFlag = 11U + buffer;
                        const uint32_t valueOffset = mm2L1AddrStart_ + mm2L1TileM_ * mm2L1TileKLeft_ * pL1BufNum_ +
                                                     buffer * mm2L1TileN_ * mm2L1TileKRight_;
                        const uint32_t l1Tokens = RoundUp(tile.tokens, 16U);
                        auto dst = resource.l1Buf.template GetBufferByByte<fp8_e4m3fn_t>(valueOffset);
                        // Queue the V VF after Softmax and keep the P copy ahead of V on MTE3.
                        epilogueOnlineSoftmax(probability, shape, true, sSlot, pSlot, sFlag, pFlag,
                                              [&]() __aicore__ { dequantK.RunTileVf(prepared); });
                        AscendC::CrossCoreWaitFlag<4, PIPE_MTE3>(valueFlag);
                        dequantK.StoreTile(dst, l1Tokens, fdValueSlot, prepared);
                        AscendC::CrossCoreSetFlag<4, PIPE_MTE3>(valueFlag);
                    } else {
                        epilogueOnlineSoftmax(probability, shape, tile.tileId == 0U, sSlot, pSlot, sFlag, pFlag);
                    }
                    if (needsStatsSnapshot && tile.tileId + 1U == tile.tileCount) {
                        const uint32_t statsOffset = statsSnapshotOffset_ + pSlot * 2U * STATS_BYTES;
                        auto savedMax = resource.ubBuf.template GetBufferByByte<float>(statsOffset);
                        auto savedSum = resource.ubBuf.template GetBufferByByte<float>(statsOffset + STATS_BYTES);
                        auto max = resource.ubBuf.template GetBufferByByte<float>(uBufTileHelper_.gmStartOffset);
                        auto sum = resource.ubBuf.template GetBufferByByte<float>(uBufTileHelper_.glStartOffset);
                        AscendC::DataCopy(savedMax, max, 64U);
                        AscendC::DataCopy(savedSum, sum, 64U);
                        AscendC::PipeBarrier<PIPE_V>();
                    }
                    if (!preloadKv && !issueValueWithSoftmax) {
                        produceValue(tile, seq);
                    }
#endif
                }
                if (!preloadKv && seq >= PV_CONSUME_DELAY) {
                    consume(seq - PV_CONSUME_DELAY);
                }
            };
            auto enqueue = [&](const PipelineTile& tile) __aicore__ {
                pending[sequence % PENDING_STAGES] = tile;
                if (preloadKv) {
                    // AIV: Softmax(i) -> prepare K/V(i+1) -> RescaleO(i).
                    // AIC: QK(i) -> PV(i), using the already prepared V(i).
                    if (sequence != 0U) {
                        step(sequence - 1U);
                    }
                    produceKey(tile, sequence);
                    produceValue(tile, sequence);
                    if (sequence != 0U) {
                        consume(sequence - 1U);
                    }
                } else {
                    produceKey(tile, sequence);
                    if (sequence != 0U) {
                        step(sequence - 1U);
                    }
                }
                ++sequence;
            };
            for (uint32_t taskIdx = taskLoopStart; taskIdx < taskLoopEnd; taskIdx += taskLoopStep) {
                uint32_t rawBegin = 0U;
                const bool isLastScheduledTask = taskIdx + 1U == taskLoopEnd;
                uint32_t fdPartialTaskId = 0U;
                uint32_t fdPartialCount = 0U;
                const bool isFdPartial =
                    fdEnabled && GsaFd::FindPartialTask(meta, taskIdx, coreIdx, fdPartialTaskId, fdPartialCount);
                if (fdEnabled) {
                    rawBegin = taskIdx == taskLoopStart ? scheduleFirstBlock : 0U;
                }
                uint32_t qToken = taskIdx / kvHeads_;
                uint32_t kvHeadIdx = taskIdx % kvHeads_;
                uint32_t qHeadStart = kvHeadIdx * groupSize;
                uint32_t batchIdx = 0;
                uint32_t qTokenInBatch = qToken;
                BatchSeqInfo qSeq;
                if constexpr (qFormat == Format::TND) {
                    // Task space = packed actual Q tokens (seqused if present, else cu storage).
                    // GM / sparse index use cu storage offsets (pad at end of each batch segment).
                    uint32_t accum = 0;
                    for (uint32_t b = 0; b < batch_; ++b) {
                        uint32_t storageLen =
                            static_cast<uint32_t>(gCuSeqLengths.GetValue(static_cast<int64_t>(b + 1)) -
                                                  gCuSeqLengths.GetValue(static_cast<int64_t>(b)));
                        uint32_t batchLen = hasSequsedQ ?
                                                static_cast<uint32_t>(gSequsedQ.GetValue(static_cast<int64_t>(b))) :
                                                storageLen;
                        if (qToken < accum + batchLen) {
                            batchIdx = b;
                            qTokenInBatch = qToken - accum;
                            break;
                        }
                        accum += batchLen;
                    }

                    uint32_t qStorageLen =
                        static_cast<uint32_t>(gCuSeqLengths.GetValue(static_cast<int64_t>(batchIdx + 1)) -
                                              gCuSeqLengths.GetValue(static_cast<int64_t>(batchIdx)));
                    const uint32_t qSeqlen =
                        hasSequsedQ ? static_cast<uint32_t>(gSequsedQ.GetValue(batchIdx)) : qStorageLen;
                    qSeq = {static_cast<uint64_t>(gCuSeqLengths.GetValue(batchIdx)), qStorageLen, qSeqlen};
                } else {
                    DecodeBatchedQToken(qToken, batch_, tilingData->qSeqCapacity, gSequsedQ, batchIdx, qTokenInBatch,
                                        qSeq);
                }
                BatchSeqInfo kvSeq{};
                if constexpr (isPaged) {
                    kvSeq.actualLen = static_cast<uint32_t>(gSequsedKv.GetValue(batchIdx));
                } else {
                    // The caller supplies valid cu storage prefixes and optional seqused visibility lengths.
                    ReadBatchSeqInfo<kvFormat>(gCuSeqLengthsKv, gSequsedKv, hasSequsedKv, batchIdx, kvSeq,
                                               tilingData->kvSeqCapacity);
                }
                const uint32_t kvSeqlen = kvSeq.actualLen;
                const uint32_t qSeqlen = qSeq.actualLen;
                uint64_t gmOffsetQ = GetTensorOffset<qFormat>(qSeq, qTokenInBatch, qHeadStart, qHeads_, embed_);
                uint64_t gmOffsetO = gmOffsetQ;
                // LSE uses [T,N,1] for TND and [B,N,S,1] for batched layouts.
                uint64_t gmOffsetLse = GetLseOffset<qFormat>(qSeq, qTokenInBatch, qHeadStart, qHeads_);

                PipelineTile task;
                task.qOffset = gmOffsetQ;
                task.oOffset = gmOffsetO;
                task.lseOffset = gmOffsetLse;
                task.kvSeq = kvSeq;
                task.head = kvHeadIdx;
                task.partial = isFdPartial;
                task.partialId = fdPartialTaskId;
                if (qSeqlen == 0U || kvSeqlen == 0U || (isCausal && kvSeqlen < qSeqlen)) {
                    enqueue(task);
                    continue;
                }

                // layout_sparse_pattern=KVN_TotalQB_KB: sparseBlockIdx 3D [N_kv, totalQBlocks, topK]
                // totalQBlocks spans storage (cu) blocks; align with metadata qStorageBlockStarts.
                uint32_t globalQBlock;
                if constexpr (qFormat == Format::TND) {
                    globalQBlock = 0;
                    for (uint32_t b = 0; b < batchIdx; ++b) {
                        uint32_t qLen = static_cast<uint32_t>(gCuSeqLengths.GetValue(static_cast<int64_t>(b + 1)) -
                                                              gCuSeqLengths.GetValue(static_cast<int64_t>(b)));
                        globalQBlock += (qLen + blockShapeX_ - 1) / blockShapeX_;
                    }
                    globalQBlock += qTokenInBatch / blockShapeX_;
                } else {
                    globalQBlock = batchIdx * tilingData->qSeqCapacity + qTokenInBatch;
                }
                int64_t sparseIdxBase =
                    static_cast<int64_t>(kvHeadIdx) * qBlockNum_ * topK_ + static_cast<int64_t>(globalQBlock) * topK_;
                // sparseBlockCount 2D: [N_kv, totalQBlocks]
                const int64_t countOffset =
                    static_cast<int64_t>(kvHeadIdx) * qBlockNum_ + static_cast<int64_t>(globalQBlock);
                // The caller supplies a valid count and sorted, valid sparse indices.
                const uint32_t explicitBlockCount = static_cast<uint32_t>(gSparseBlockCount.GetValue(countOffset));
                uint32_t visibleKvEnd = kvSeqlen;
                if constexpr (isCausal) {
                    visibleKvEnd = kvSeqlen - qSeqlen + qTokenInBatch + 1U;
                }
                uint32_t virtualBlockCount = explicitBlockCount;
                uint32_t residualBlockId = 0U;
                if constexpr (hasResidualBlock) {
                    const uint32_t residualSize = visibleKvEnd % blockShapeY_;
                    residualBlockId = visibleKvEnd / blockShapeY_;
                    virtualBlockCount += static_cast<uint32_t>(residualSize != 0U);
                }
                uint32_t rawEnd = virtualBlockCount;
                if (fdEnabled) {
                    rawBegin = rawBegin < virtualBlockCount ? rawBegin : virtualBlockCount;
                    if (isLastScheduledTask) {
                        rawEnd = scheduleLastBlock < virtualBlockCount ? scheduleLastBlock : virtualBlockCount;
                    }
                }

                uint32_t validKvTokens = 0;
                // FD changes only the selected list range. All tile coordinates below
                // are local to this range, including ordinary tasks starting at zero.
                auto gSparseBlockIdxBase = gSparseBlockIdx[sparseIdxBase];
                AscendC::GlobalTensor<int32_t> gBatchBlockTable;
                if constexpr (isPaged) {
                    gBatchBlockTable = gBlockTable[static_cast<uint64_t>(batchIdx) * maxBlocksPerBatch_];
                }

                // Valid, sorted indices leave a visible prefix with at most one partial block.
                CalcSparseKvSize<hasResidualBlock>(gSparseBlockIdxBase, explicitBlockCount, rawBegin, rawEnd,
                                                   blockShapeY_, visibleKvEnd, validKvTokens);
                const uint32_t explicitBegin = rawBegin < explicitBlockCount ? rawBegin : explicitBlockCount;
                const uint32_t explicitEnd = rawEnd < explicitBlockCount ? rawEnd : explicitBlockCount;
                const uint32_t localExplicitBlockCount = explicitEnd - explicitBegin;
                // Rebase the explicit list to this FD slice. For a residual-only slice this points to the
                // end of the explicit list, but GetSparseBlockId resolves the implicit block before any GM read.
                auto gTaskSparseBlockIdx = gSparseBlockIdxBase[explicitBegin];
                const uint32_t kvSLoopNum = CeilDiv(validKvTokens, kvBaseTile_);

                task.sparse = gTaskSparseBlockIdx;
                task.pages = gBatchBlockTable;
                task.explicitCount = localExplicitBlockCount;
                task.residual = residualBlockId;
                task.tileCount = kvSLoopNum;
                task.kvOffset = isPaged ? static_cast<uint64_t>(kvHeadIdx) *
                                              GbsaLayoutHelper<kvFormat>::HeadStride(blockSize_, embed_) :
                                          GetTensorOffset<kvFormat>(kvSeq, 0U, kvHeadIdx, kvHeads_, embed_);
                for (uint32_t tile = 0U; tile < max(kvSLoopNum, 1U); ++tile) {
                    task.tileId = tile;
                    task.tokens = kvSLoopNum == 0U ? 0U : min(kvBaseTile_, validKvTokens - tile * kvBaseTile_);
                    enqueue(task);
                }
            }
            // Only the end of the core's entire task stream drains the pipeline.
            if (sequence != 0U) {
                step(sequence - 1U);
            }
            for (uint32_t seq = sequence > PV_CONSUME_DELAY ? sequence - PV_CONSUME_DELAY : 0U; seq < sequence; ++seq) {
                consume(seq);
            }
        }
        ReleaseSyncFlags<4, 4, 4>();
        if (fdEnabled) {
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::SyncAll<false>();
#ifdef __DAV_VEC__
            using RescaleDispatchPolicy = typename EpilogueRescaleO::DispatchPolicy;
            constexpr bool OUTPUT_LSE = RescaleDispatchPolicy::LSE_MODE == Epilogue::LseMode::OUT_ONLY;
            GenericBlockSparseAttentionFdCombineArch35<ElementO, Arch::Resource<ArchTag>, OUTPUT_LSE, qFormat> combine(
                resource);
            combine(meta, tilingData, gPartialLse, gPartialO, gO, gLse, gCuSeqLengths, gSequsedQ, hasSequsedQ);
#endif
        }
    }

private:
    __aicore__ inline void WaitKvReady(uint32_t flag)
    {
        AscendC::CrossCoreWaitFlag<4, PIPE_MTE1>(flag);
        AscendC::CrossCoreWaitFlag<4, PIPE_MTE1>(flag + 16U);
    }

    __aicore__ inline void ReleaseKv(uint32_t flag)
    {
        AscendC::CrossCoreSetFlag<4, PIPE_MTE1>(flag);
        AscendC::CrossCoreSetFlag<4, PIPE_MTE1>(flag + 16U);
    }

#ifdef __DAV_VEC__
    template <bool IS_K>
    __aicore__ inline void ProduceKvTile(Epilogue::Block::A8C4KvDequant& dequant, AscendC::GlobalTensor<uint8_t> data,
                                         AscendC::GlobalTensor<uint8_t> scales, AscendC::GlobalTensor<int32_t> sparse,
                                         AscendC::GlobalTensor<int32_t> pages, const BatchSeqInfo& sequence,
                                         uint32_t head, uint32_t tile, uint32_t rows, uint64_t pageStride,
                                         uint32_t globalSequence, uint32_t tileDequantSlot,
                                         bool localDescriptor = false)
    {
        const uint32_t buffer = globalSequence % (IS_K ? kL1BufNum_ : vL1BufNum_);
        const uint32_t freeFlag = (IS_K ? 8U : 11U) + buffer;
        const uint32_t readyFlag = freeFlag; // opposite directions have independent mailboxes
        AscendC::CrossCoreWaitFlag<4, PIPE_MTE3>(freeFlag);
        uint32_t l1Offset;
        if constexpr (IS_K) {
            l1Offset = mm1L1TileM_ * mm1L1TileKLeft_ + buffer * mm1L1TileKRight_ * mm1L1TileN_;
        } else {
            l1Offset =
                mm2L1AddrStart_ + mm2L1TileM_ * mm2L1TileKLeft_ * pL1BufNum_ + buffer * mm2L1TileN_ * mm2L1TileKRight_;
        }
        auto l1Kv = resource.l1Buf.template GetBufferByByte<fp8_e4m3fn_t>(l1Offset);
        const uint32_t tokenStride = GbsaLayoutHelper<kvFormat>::TokenStride(kvHeads_, embed_);
        const uint32_t scaleTokenStride =
            GbsaLayoutHelper<kvFormat>::TokenStride(kvHeads_, Epilogue::Block::A8C4KvDequant::SCALE_BYTES);
        const uint32_t segmentNum = CeilDiv(rows, 128U);
        const uint32_t l1Rows = RoundUp(rows, 16U);
        uint64_t dataBase;
        uint64_t scaleBase;
        uint64_t dataBlockStride;
        uint64_t scaleBlockStride;
        if constexpr (isPaged) {
            dataBase = head * GbsaLayoutHelper<kvFormat>::HeadStride(blockSize_, embed_) / 2U;
            scaleBase =
                head * GbsaLayoutHelper<kvFormat>::HeadStride(blockSize_, Epilogue::Block::A8C4KvDequant::SCALE_BYTES);
            dataBlockStride = pageStride / 2U;
            scaleBlockStride = kScalePageStrideBytes_;
        } else {
            dataBase = GetTensorOffset<kvFormat>(sequence, 0U, head, kvHeads_, embed_) / 2U;
            scaleBase =
                GetTensorOffset<kvFormat>(sequence, 0U, head, kvHeads_, Epilogue::Block::A8C4KvDequant::SCALE_BYTES);
            dataBlockStride = static_cast<uint64_t>(128U) * tokenStride / 2U;
            scaleBlockStride = static_cast<uint64_t>(128U) * scaleTokenStride;
        }
        auto desc = resource.ubBuf.template GetBufferByByte<uint64_t>(descriptorOffset_);
        const uint32_t descriptorBase = (globalSequence % PENDING_STAGES) * DESCRIPTOR_WORDS_PER_TILE;
        constexpr uint32_t RECORD_WORDS = DESCRIPTOR_RECORD_BYTES / sizeof(uint64_t);
        constexpr uint32_t DATA_WORD = IS_K ? 0U : 1U;
        constexpr uint32_t SCALE_WORD = isPaged && !IS_K ? 3U : 2U;
        uint32_t segmentBegin = 0U;
        uint32_t segmentEnd = IS_K ? segmentNum : 0U;
        if constexpr (IS_K && isPaged) {
            // A whole tile gives each AIV two complete pages, for both K and V.
            if (localDescriptor && rows == 2U * Epilogue::Block::A8C4KvDequant::HALF_TILE_TOKENS) {
                const uint32_t pagesPerSubBlock = segmentNum / 2U;
                segmentBegin = AscendC::GetSubBlockIdx() * pagesPerSubBlock;
                segmentEnd = segmentBegin + pagesPerSubBlock;
            }
        }
        for (uint32_t segment = segmentBegin; segment < segmentEnd; ++segment) {
            const uint32_t record = descriptorBase + segment * RECORD_WORDS;
            if constexpr (IS_K) {
                uint32_t block = static_cast<uint32_t>(sparse.GetValue(tile * (kvBaseTile_ / 128U) + segment));
                if constexpr (isPaged) {
                    block = static_cast<uint32_t>(pages.GetValue(block));
                }
                desc.SetValue(record, dataBase + block * dataBlockStride);
                const uint64_t valueStride = isPaged ? vStride0_ / 2U : dataBlockStride;
                desc.SetValue(record + 1U, dataBase + block * valueStride);
                desc.SetValue(record + 2U, scaleBase + block * scaleBlockStride);
                if constexpr (isPaged) {
                    desc.SetValue(record + 3U, scaleBase + block * vScalePageStrideBytes_);
                }
            }
        }
        dequant.ProduceTile<RECORD_WORDS, DATA_WORD, SCALE_WORD>(l1Kv, data, scales, desc[descriptorBase], rows, l1Rows,
                                                                 tileDequantSlot);
        // Both AIV halves publish even when one half has no effective rows.
        AscendC::CrossCoreSetFlag<4, PIPE_MTE3>(readyFlag);
    }

#endif

    __aicore__ inline void FetchBaseShapeInfo(
        const GenericBlockSparseAttn::GenericBlockSparseAttentionTilingData* tilingData, GM_ADDR metaData)
    {
        batch_ = tilingData->batch;
        qHeads_ = tilingData->numHeads;
        kvHeads_ = tilingData->kvHeads;
        embed_ = tilingData->embeddingSize;
        blockShapeY_ = tilingData->blockShapeY;
        blockShapeX_ = tilingData->blockShapeX;
        blockSize_ = tilingData->blockSize;
        qBlockNum_ = tilingData->qBlockNum;
        topK_ = tilingData->topK;
        maxBlocksPerBatch_ = tilingData->maxBlocksPerBatch;
        // Full AICPU metadata protocol overlay (no tiling fallback for task schedule).
        __gm__ GbsaMetadata::Metadata* meta = reinterpret_cast<__gm__ GbsaMetadata::Metadata*>(metaData);
        totalTaskNum_ = static_cast<uint32_t>(meta->saTotalTaskNum);
        scaleValue_ = tilingData->scaleValue;
        groupSize_ = tilingData->groupSize;
        qBaseTile_ = tilingData->baseTileInfo.qBaseTile;
        kvBaseTile_ = tilingData->baseTileInfo.kvBaseTile;
        kStride0_ = tilingData->kStride0;
        vStride0_ = tilingData->vStride0;
        if constexpr (isPaged) {
            kScalePageStrideBytes_ = tilingData->kScaleStride0 * sizeof(bfloat16_t);
            vScalePageStrideBytes_ = tilingData->vScaleStride0 * sizeof(bfloat16_t);
        }
        isConsistentTopk_ = tilingData->isConsistentTopk;
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
        Gemm::Block::Mm1L1TileHelper mm1L1TileHelper(mm1L1TileM_, mm1L1TileN_, mm1L1TileKLeft_, mm1L1TileKRight_,
                                                     qL1BufNum_, kL1BufNum_);
        mm1L1TileHelper_ = mm1L1TileHelper;
        Gemm::Block::Mm2L1TileHelper mm2L1TileHelper(mm2L1TileM_, mm2L1TileN_, mm2L1TileKLeft_, mm2L1TileKRight_,
                                                     pL1BufNum_, vL1BufNum_);
        mm2L1TileHelper_ = mm2L1TileHelper;
        mm2L1AddrStart_ = mm1L1TileM_ * mm1L1TileKLeft_ * qL1BufNum_ * sizeof(ElementQ) +
                          mm1L1TileKRight_ * mm1L1TileN_ * kL1BufNum_ * sizeof(ElementK);
        uint32_t mL0LoopQK = CeilDiv(groupSize_, static_cast<uint32_t>(BlockMmadQK::L0_TILE_M));
        uint32_t mL0LoopPV = CeilDiv(groupSize_, static_cast<uint32_t>(BlockMmadPV::L0_TILE_M));
        const uint32_t qkKLoop = CeilDiv(embed_, static_cast<uint32_t>(BlockMmadQK::L0_TILE_K));
        const uint32_t pvKLoop = CeilDiv(kvBaseTile_, static_cast<uint32_t>(BlockMmadPV::L0_TILE_K));
        mm1L0ATotalStages_ = mL0LoopQK * qkKLoop;
        mm1L0BTotalStages_ = CeilDiv(kvBaseTile_, static_cast<uint32_t>(BlockMmadQK::L0_TILE_N)) * qkKLoop;
        mm2L0ATotalStages_ = mL0LoopPV * pvKLoop;
        mm2L0BTotalStages_ = pvKLoop * CeilDiv(embed_, static_cast<uint32_t>(BlockMmadPV::L0_TILE_N));
    }

    __aicore__ inline void CalcUBufTileInfo()
    {
        constexpr uint32_t VECTOR_SUB_CORE_NUM = 2U;
        // The first sub-core takes the extra row of an odd Q tile. NZ P storage rounds up to 16 rows.
        const uint32_t qBaseTilePerSubCore = RoundUp(CeilDiv(qBaseTile_, VECTOR_SUB_CORE_NUM), 16U);
        constexpr uint32_t statsRows = Epilogue::Block::UBufTileHelper::STATS_ROW_NUM;
        uint32_t kvBaseTilePerSubCore = RoundUp(kvBaseTile_, 32);
        uint32_t embedPerSubCore = RoundUp(embed_, 32);
        constexpr uint32_t loUbBufNum = UB_S_OTMP_BUF_STAGES;
        uint32_t sStartOffset = 0;
        uint32_t pStartOffset =
            sStartOffset + qBaseTilePerSubCore * kvBaseTilePerSubCore * sizeof(ElementS) * UB_S_OTMP_BUF_STAGES;
        uint32_t loStartOffset = pStartOffset + (qBaseTilePerSubCore + EpilogueOnlineSoftmax::P_UB_EXTRA_ROWS) *
                                                    kvBaseTilePerSubCore * sizeof(ElementP) * UB_S_OTMP_BUF_STAGES;
        uint32_t goStartOffset =
            loStartOffset + qBaseTilePerSubCore * embedPerSubCore * sizeof(ElementOTmp) * loUbBufNum;
        uint32_t lmStartOffset = goStartOffset + qBaseTilePerSubCore * embedPerSubCore * sizeof(ElementOTmp);
        uint32_t gmStartOffset = lmStartOffset + statsRows * sizeof(float);
        uint32_t dmStartOffset = gmStartOffset + statsRows * sizeof(float);
        uint32_t llStartOffset = dmStartOffset + statsRows * MAX_CROSS_CORE_BUF_STAGES * sizeof(float);
        uint32_t glStartOffset = llStartOffset + statsRows * sizeof(float);
        uint32_t lseStartOffset = glStartOffset + statsRows * sizeof(float);
        uint32_t maskStartOffset = lseStartOffset + statsRows * 32U;
        uBufTileHelper_ = Epilogue::Block::UBufTileHelper(qBaseTilePerSubCore, kvBaseTilePerSubCore, embedPerSubCore,
                                                          sStartOffset, pStartOffset, loStartOffset, goStartOffset,
                                                          lmStartOffset, gmStartOffset, dmStartOffset, llStartOffset,
                                                          glStartOffset, lseStartOffset, maskStartOffset, loUbBufNum);
        uBufTileHelper_.kvOffsetUbBufNum = 2U;
        statsSnapshotOffset_ = maskStartOffset;
        descriptorOffset_ = statsSnapshotOffset_ + MAX_CROSS_CORE_BUF_STAGES * 2U * STATS_BYTES;
        dequantOffset_ = descriptorOffset_ + PENDING_STAGES * MAX_SEGMENTS * DESCRIPTOR_RECORD_BYTES;
        const uint32_t storageHeads = GbsaLayoutHelper<kvFormat>::TokenStride(kvHeads_, embed_) / embed_;
        using Dequant = Epilogue::Block::A8C4KvDequant;
        // K and V share two whole-tile slots. Each AIV holds at most 256
        // tokens of the current KV head, regardless of the head count.
        dequantBytes_ = Dequant::HalfTileScratchBytes(storageHeads);
    }

    __aicore__ inline void InitCrossCoreDstBuf(AscendC::LocalTensor<ElementP> (&l1PTensor)[MAX_CROSS_CORE_BUF_STAGES],
                                               AscendC::LocalTensor<ElementS> (&ubSTensor)[UB_S_OTMP_BUF_STAGES],
                                               AscendC::LocalTensor<ElementOTmp> (&ubOTmpTensor)[UB_S_OTMP_BUF_STAGES])
    {
        for (uint32_t i = 0; i < pL1BufNum_; i++) {
            l1PTensor[i] = resource.l1Buf.template GetBufferByByte<ElementP>(
                mm2L1AddrStart_ + mm2L1TileM_ * mm2L1TileKLeft_ * sizeof(ElementP) * i);
        }
        for (uint32_t i = 0; i < UB_S_OTMP_BUF_STAGES; i++) {
            ubSTensor[i] = resource.ubBuf.template GetBufferByByte<ElementS>(
                uBufTileHelper_.sStartOffset +
                uBufTileHelper_.qBaseTilePerSubCore * uBufTileHelper_.kvBaseTilePerSubCore * sizeof(ElementS) * i);
            ubOTmpTensor[i] = resource.ubBuf.template GetBufferByByte<ElementOTmp>(
                uBufTileHelper_.loStartOffset + uBufTileHelper_.loUbRowNum * uBufTileHelper_.embedPerSubCore *
                                                    sizeof(ElementOTmp) * (i % uBufTileHelper_.loUbBufNum));
        }
    }

    template <uint32_t MM1_SM_MODE, uint32_t MM2_RE_MODE, uint32_t SM_MM2_MODE>
    __aicore__ inline void InitSyncFlags()
    {
#ifdef __DAV_CUBE__
        for (uint32_t i = 0; i < kL1BufNum_; ++i) {
            ReleaseKv(8U + i);
        }
        for (uint32_t i = 0; i < vL1BufNum_; ++i) {
            ReleaseKv(11U + i);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID5);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID6);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID4);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID5);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID6);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID7);
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
            AscendC::CrossCoreSetFlag<MM2_RE_MODE, PIPE_V>(5);
            AscendC::CrossCoreSetFlag<MM2_RE_MODE, PIPE_V>(6);
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
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID5);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID6);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID5);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID6);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID7);
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
            AscendC::CrossCoreWaitFlag<MM2_RE_MODE, PIPE_FIX>(5);
            AscendC::CrossCoreWaitFlag<MM2_RE_MODE, PIPE_FIX>(21);
            AscendC::CrossCoreWaitFlag<MM2_RE_MODE, PIPE_FIX>(6);
            AscendC::CrossCoreWaitFlag<MM2_RE_MODE, PIPE_FIX>(22);
        }
#endif
#ifdef __DAV_VEC__
        for (uint32_t i = 0; i < kL1BufNum_; ++i) {
            AscendC::CrossCoreWaitFlag<4, PIPE_MTE3>(8U + i);
        }
        for (uint32_t i = 0; i < vL1BufNum_; ++i) {
            AscendC::CrossCoreWaitFlag<4, PIPE_MTE3>(11U + i);
        }
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

private:
    Arch::Resource<ArchTag> resource;
    // basic shape info
    uint32_t batch_;
    uint32_t qHeads_;
    uint32_t kvHeads_;
    uint32_t embed_;
    uint32_t blockShapeY_;
    uint32_t blockShapeX_;
    uint32_t blockSize_;
    uint32_t qBlockNum_;
    uint32_t topK_;
    uint32_t maxBlocksPerBatch_;
    uint32_t totalTaskNum_;
    float scaleValue_;
    uint32_t groupSize_;
    // PA_BBND page base strides (elements); may exceed blockSize*Nkv*D when dim0 is strided.
    uint64_t kStride0_;
    uint64_t vStride0_;
    uint64_t kScalePageStrideBytes_;
    uint64_t vScalePageStrideBytes_;
    uint32_t isConsistentTopk_;
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
    uint32_t statsSnapshotOffset_, descriptorOffset_, dequantOffset_, dequantBytes_;
    uint32_t mm2L1AddrStart_ = 0;
    Gemm::Block::Mm1L1TileHelper mm1L1TileHelper_;
    Gemm::Block::Mm2L1TileHelper mm2L1TileHelper_;
    Epilogue::Block::UBufTileHelper uBufTileHelper_;
};

} // namespace GbsaKernelArch35
