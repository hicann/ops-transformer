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

using namespace NpuArch;
using namespace tla;

namespace GbsaKernelArch35 {

// Isolated path for maskType=3 (union exactMask / DN). ND stays in
// generic_block_sparse_attention_kernel_arch35.h (GbsaRegularKernelArch35).
// isFp8 selects FP8 QKV/P physical-row / UB budget paths; O always comes from Rescale.
template <class EpiloguePrepareKvOffsets, class BlockMmadQK, class EpilogueOnlineSoftmax, class BlockMmadPV,
          class EpilogueRescaleO, Format kvFormat, bool isPaged, bool isFp8>
class GbsaUnionKernelArch35 {
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

    // DN: K * Q^T — Q as B, K as A.
    using LayoutQ = layout::ColumnMajor;
    using LayoutK = layout::RowMajor;
    using LayoutS = typename EpilogueOnlineSoftmax::LayoutSType;
    using LayoutP = layout::RowMajor;
    using LayoutV = layout::RowMajor;
    using LayoutO = layout::RowMajor;
    using LayoutOTmp = layout::RowMajor;
    using LayoutLse = layout::RowMajor;

    using LayoutTagL1P = typename BlockMmadPV::LayoutTagL1A;

    static constexpr uint32_t PRE_LAUNCH = 2;
    static constexpr uint32_t KV_OFFSET_PRE_LAUNCH = 2;
    static constexpr uint32_t UNION_SIZE = 8U;
    static constexpr uint32_t MAX_CROSS_CORE_BUF_STAGES = PRE_LAUNCH + 1;
    static constexpr uint32_t UB_S_OTMP_BUF_STAGES = 2;

    __aicore__ inline GbsaUnionKernelArch35() {}

    __aicore__ inline void operator()(GbsaKernelParamsArch35 const& params)
    {
        GET_TILING_DATA_WITH_STRUCT(GenericBlockSparseAttn::GenericBlockSparseAttentionTilingData, tilingDataLocal,
                                    params.tiling);
        const GenericBlockSparseAttn::GenericBlockSparseAttentionTilingData* tilingData = &tilingDataLocal;
        FetchBaseShapeInfo(tilingData, params.metaData);
        CalcOnChipBufTileInfo(tilingData);
        CalcUBufTileInfo();
        __gm__ const GbsaMetadata::Metadata* meta =
            reinterpret_cast<__gm__ const GbsaMetadata::Metadata*>(params.metaData);
        uint32_t coreIdx = AscendC::GetBlockIdx();
        const uint32_t coreNum = AscendC::GetBlockNum();
        if (!ValidateMetadata(meta, tilingData)) {
            return;
        }

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
        if constexpr (!isPaged) {
            if (qFormat_ == Format::TND) {
                gCuSeqLengthsKv.SetGlobalBuffer((__gm__ int64_t*)params.cuSeqLengthsKv);
            }
        }
        AscendC::GlobalTensor<int32_t> gSparseBlockCount;
        gSparseBlockCount.SetGlobalBuffer((__gm__ int32_t*)params.sparseBlockCount);
        AscendC::GlobalTensor<int32_t> gExactMask;

        gExactMask.SetGlobalBuffer((__gm__ int32_t*)params.attenMask);
        AscendC::GlobalTensor<int64_t> gCuSeqLengths;
        if (qFormat_ == Format::TND) {
            gCuSeqLengths.SetGlobalBuffer((__gm__ int64_t*)params.cuSeqLengths);
        }
        AscendC::GlobalTensor<int32_t> gSequsedQ;
        const bool hasSequsedQ = (params.sequsedQ != nullptr);
        if (hasSequsedQ) {
            gSequsedQ.SetGlobalBuffer((__gm__ int32_t*)params.sequsedQ);
        }
        AscendC::GlobalTensor<int32_t> gSequsedKv;
        if (params.sequsedKv != nullptr) {
            gSequsedKv.SetGlobalBuffer((__gm__ int32_t*)params.sequsedKv);
        }
        const bool hasSequsedKv = params.sequsedKv != nullptr;
        AscendC::GlobalTensor<ElementO> gO;
        gO.SetGlobalBuffer((__gm__ ElementO*)params.o);
        AscendC::GlobalTensor<ElementLse> gLse;
        gLse.SetGlobalBuffer((__gm__ ElementLse*)params.softmaxLse);
        // Both AIVs share their physical AIC's two full LO tiles.
        uint32_t loCoreIdx = AscendC::GetBlockIdx();
#ifdef __DAV_VEC__
        loCoreIdx /= AscendC::GetSubBlockNum();
#endif

        AscendC::GlobalTensor<uint64_t> gKvOffsetInfo;
        const uint32_t kvOffsetTileWords = tilingData->kvOffsetTileWords;
        gKvOffsetInfo.SetGlobalBuffer(
            reinterpret_cast<__gm__ uint64_t*>(params.workSpace + tilingData->kvOffsetInfoOffset) +
            static_cast<uint64_t>(loCoreIdx) * KV_OFFSET_STAGES * kvOffsetTileWords);

        AscendC::LocalTensor<ElementP> l1PTensor[MAX_CROSS_CORE_BUF_STAGES];
        AscendC::LocalTensor<ElementS> ubSTensor[UB_S_OTMP_BUF_STAGES];
        AscendC::LocalTensor<ElementOTmp> ubOTmpTensor[UB_S_OTMP_BUF_STAGES];
        InitCrossCoreDstBuf(l1PTensor, ubSTensor, ubOTmpTensor);

        const uint32_t kvOffsetGroupWords = CalcKvOffsetGroupWords<isPaged>(mm1L1TileN_, blockShapeY_);
        const uint64_t strideKVRow = isPaged ?
                                         GbsaLayoutHelper<kvFormat>::TokenStride(kvHeads_, embed_) :
                                         (qFormat_ == Format::BNSD ? embed_ : static_cast<uint64_t>(kvHeads_) * embed_);

        InitSyncFlags<4, 4, 4>();

#ifdef __DAV_CUBE__
        coreIdx = AscendC::GetBlockIdx();
#endif
        AscendC::SyncAll<false>();
#ifdef __DAV_CUBE__
        BlockMmadQK blockMmadQK(resource, mm1L1TileHelper_);
        BlockMmadPV blockMmadPV(resource, mm2L1AddrStart_, mm2L1TileHelper_);
#endif
#ifdef __DAV_VEC__
        coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        EpiloguePrepareKvOffsets epiloguePrepareKvOffsets(resource, uBufTileHelper_, kvBaseTile_, mm1L1TileN_,
                                                          blockShapeY_, blockSize_, strideKVRow, kStride0_, vStride0_);
        EpilogueOnlineSoftmax epilogueOnlineSoftmax(resource, scaleValue_, uBufTileHelper_);

        epilogueOnlineSoftmax.SetExactMask(gExactMask, groupSize_);
        // Gathered columns must stay in logical union order for the
        // exactMask; forbid the reversed-pair gather reordering.
        epiloguePrepareKvOffsets.SetForbidReversePair();
        EpilogueRescaleO epilogueRescaleO(resource, uBufTileHelper_);
#endif

        uint32_t groupSize = groupSize_;
        const uint64_t strideQHead =
            qFormat_ == Format::BNSD ? GbsaLayoutHelper<Format::BNSD>::HeadStride(qSeqCapacity_, embed_) : embed_;
        uint32_t embedRound = RoundUp(embed_, 16);

#ifdef __DAV_VEC__
        if (hasSequsedQ) {
            uint32_t paddingTask = 0U;
            for (uint32_t batchIdx = 0U; batchIdx < batch_; ++batchIdx) {
                BatchSeqInfo qSeq{};
                ReadUnionBatch(gCuSeqLengths, gSequsedQ, hasSequsedQ, batchIdx, qSeqCapacity_, qSeq);
                for (uint32_t token = qSeq.actualLen; token < qSeq.storageLen; ++token) {
                    for (uint32_t kvHeadIdx = 0U; kvHeadIdx < kvHeads_; ++kvHeadIdx, ++paddingTask) {
                        if (paddingTask % coreNum != coreIdx) {
                            continue;
                        }
                        const uint32_t qHeadStart = kvHeadIdx * groupSize;
                        WriteEmptyScattered(gO, gLse, qSeq, token, qHeadStart, 1U, groupSize);
                    }
                }
            }
        }
#endif

        for (uint32_t taskIdx = coreIdx; taskIdx < totalTaskNum_; taskIdx += coreNum) {
            // In union mode one task covers a group of UNION_SIZE Q tokens and
            // qBlockNum_/topK_ carry G (total groups) / U (union capacity).
            uint32_t qUnit = taskIdx / kvHeads_;
            uint32_t kvHeadIdx = taskIdx % kvHeads_;
            uint32_t qHeadStart = kvHeadIdx * groupSize;
            uint32_t batchIdx = 0;
            uint32_t unitInBatch = qUnit;
            // The sparse group axis concatenates actual groups in batch order.
            BatchSeqInfo qSeq{};
            uint32_t accum = 0;
            for (uint32_t b = 0; b < batch_; ++b) {
                ReadUnionBatch(gCuSeqLengths, gSequsedQ, hasSequsedQ, b, qSeqCapacity_, qSeq);
                const uint32_t unitLen = CeilDiv(qSeq.actualLen, UNION_SIZE);
                if (qUnit < accum + unitLen) {
                    batchIdx = b;
                    unitInBatch = qUnit - accum;
                    break;
                }
                accum += unitLen;
            }

            BatchSeqInfo kvSeq{};
            if constexpr (isPaged) {
                kvSeq.actualLen = static_cast<uint32_t>(gSequsedKv.GetValue(batchIdx));
            } else {
                ReadUnionBatch(gCuSeqLengthsKv, gSequsedKv, hasSequsedKv, batchIdx, kvSeqCapacity_, kvSeq);
            }
            const uint32_t kvSeqlen = kvSeq.actualLen;
            const uint32_t qSeqlen = qSeq.actualLen;
            // First Q token of this task and how many tokens it covers.
            // Batched layouts scatter LSE in [B,N,S,1] order. TND can write
            // O/LSE directly when the packed Q rows are contiguous.
            const uint32_t qTokenInBatch = unitInBatch * UNION_SIZE;
            const uint32_t rowsInUnit = min(UNION_SIZE, qSeqlen - qTokenInBatch);
            const uint32_t taskRows = rowsInUnit * groupSize;
            const bool contiguousIo = qFormat_ == Format::TND && (kvHeads_ == 1U || rowsInUnit <= 1U);
            uint64_t gmOffsetQ = UnionTensorOffset(qSeq, qTokenInBatch, qHeadStart, qHeads_, embed_);
            uint64_t gmOffsetO = gmOffsetQ;
            uint64_t gmOffsetLse = UnionLseOffset(qSeq, qTokenInBatch, qHeadStart, qHeads_);
            const uint32_t oPackRowStride = tilingData->unionIoPackORowStride;
            // Must match host AlignUp(oBytes + lseBytes, 512) in CalculateWorkSpace.
            constexpr uint64_t PACK_CORE_ALIGN = 512ULL;
            const uint64_t packRawBytes = static_cast<uint64_t>(qBaseTile_) * oPackRowStride * sizeof(ElementO) +
                                          static_cast<uint64_t>(RoundUp(qBaseTile_, 8U)) * sizeof(ElementLse);
            const uint64_t packCoreBytes = (packRawBytes + PACK_CORE_ALIGN - 1ULL) & ~(PACK_CORE_ALIGN - 1ULL);
            const uint64_t packBase = reinterpret_cast<uint64_t>(params.workSpace) + tilingData->unionIoPackOffset +
                                      static_cast<uint64_t>(loCoreIdx) * packCoreBytes;
            AscendC::GlobalTensor<ElementO> gPackO;
            AscendC::GlobalTensor<ElementLse> gPackLse;
            gPackO.SetGlobalBuffer(reinterpret_cast<__gm__ ElementO*>(packBase));
            gPackLse.SetGlobalBuffer(reinterpret_cast<__gm__ ElementLse*>(
                packBase + static_cast<uint64_t>(qBaseTile_) * oPackRowStride * sizeof(ElementO)));

#ifdef __DAV_VEC__
            // Contiguous path writes Q/O/LSE GM directly; otherwise Rescale
            // stages into the per-AIC pack buffer and VEC scatters afterward.
            const uint64_t gmOffsetOWrite = contiguousIo ? gmOffsetO : 0ULL;
            const uint64_t gmOffsetLseWrite = contiguousIo ? gmOffsetLse : 0ULL;
            auto gmOLayoutTla =
                tla::MakeLayout<ElementO, LayoutO>(qBaseTile_, contiguousIo ? strideQHead : oPackRowStride);
            auto gmOTensorTla = tla::MakeTensor(contiguousIo ? gO[gmOffsetOWrite] : gPackO[gmOffsetOWrite],
                                                gmOLayoutTla, Arch::PositionGM{});
            auto gmLseLayoutTla = tla::MakeLayout<ElementLse, LayoutLse>(qBaseTile_, 1);
            auto gmLseTensorTla = tla::MakeTensor(contiguousIo ? gLse[gmOffsetLseWrite] : gPackLse[gmOffsetLseWrite],
                                                  gmLseLayoutTla, Arch::PositionGM{});
#endif
            if (qSeqlen == 0U || kvSeqlen == 0U) {
#ifdef __DAV_VEC__
                if (contiguousIo) {
                    epilogueRescaleO.WriteEmptyOutput(gmOTensorTla, gmLseTensorTla, GemmCoord{taskRows, embed_, 0});
                } else {
                    WriteEmptyScattered(gO, gLse, qSeq, qTokenInBatch, qHeadStart, rowsInUnit, groupSize);
                }
#endif
                continue;
            }

            // layout_sparse_pattern=KVN_TotalQB_KB: sparseBlockIdx 3D [N_kv, totalQBlocks, topK]
            // totalQBlocks spans storage (cu) blocks; align with metadata qStorageBlockStarts.
            // Union mode: the index is [kvHeads, G, U] and the packed task unit
            // index already is the global group index.
            uint32_t globalQBlock = qUnit;
            int64_t sparseIdxBase =
                static_cast<int64_t>(kvHeadIdx) * qBlockNum_ * topK_ + static_cast<int64_t>(globalQBlock) * topK_;
            // sparseBlockCount 2D: [N_kv, totalQBlocks]
            const int64_t countOffset =
                static_cast<int64_t>(kvHeadIdx) * qBlockNum_ + static_cast<int64_t>(globalQBlock);
            // The caller supplies a valid count and sorted, valid sparse indices.
            const uint32_t explicitBlockCount = static_cast<uint32_t>(gSparseBlockCount.GetValue(countOffset));
            // Union consumes the complete sorted list. ExactMask alone
            // controls visibility; clip dense GM reads to the real KV length.
            auto gTaskSparseBlockIdx = gSparseBlockIdx[sparseIdxBase];
            const uint32_t localExplicitBlockCount = explicitBlockCount;
            // ExactMask alone controls visibility. Clip dense GM reads to the
            // real KV length. FP8 paged keeps the full explicit*blockY allocation
            // so m stays a multiple of 128 — that alignment is load-bearing for
            // the SoftmaxUnion FP8 epilogue (avoids the FP16/BF16 mRound tail path).
            uint32_t validKvTokens = explicitBlockCount * blockShapeY_;
            if constexpr (!(isFp8 && isPaged)) {
                CalcSparseKvSize<false>(gTaskSparseBlockIdx, explicitBlockCount, 0U, explicitBlockCount, blockShapeY_,
                                        kvSeqlen, validKvTokens);
            }
            AscendC::GlobalTensor<int32_t> gBatchBlockTable;
            if constexpr (isPaged) {
                gBatchBlockTable = gBlockTable[static_cast<uint64_t>(batchIdx) * maxBlocksPerBatch_];
            }
            const uint32_t kvSLoopNum = CeilDiv(validKvTokens, kvBaseTile_);
            uint32_t rowNum = taskRows;
            uint32_t rowNumRound = RoundUp(rowNum, 16);
            // FP8 nZ C0 groups are 32 rows; keep AIV half-ranges physically disjoint.
            const uint32_t pvRowsPerSubCore =
                isFp8 ? RoundUp(RoundUp(rowNum, 32U) / 2U, 32U) : RoundUp(rowNum, 16U) / 2U;
            const uint32_t pvPhysicalRows = isFp8 ? (2U * pvRowsPerSubCore) : rowNum;

            if (kvSLoopNum == 0U) {
#ifdef __DAV_VEC__
                if (contiguousIo) {
                    epilogueRescaleO.WriteEmptyOutput(gmOTensorTla, gmLseTensorTla, GemmCoord{taskRows, embed_, 0});
                } else {
                    WriteEmptyScattered(gO, gLse, qSeq, qTokenInBatch, qHeadStart, rowsInUnit, groupSize);
                }
#endif
                continue;
            }

            const uint32_t lastKvTileSize = validKvTokens - (kvSLoopNum - 1U) * kvBaseTile_;
            // In DN mode the KV-sized operand is A (not B), so the tail shrink
            // moves to the A stage count and B keeps the full count.
            const uint32_t lastMm1L0AStages = CeilDiv(lastKvTileSize, static_cast<uint32_t>(BlockMmadQK::L0_TILE_M)) *
                                              CeilDiv(embed_, static_cast<uint32_t>(BlockMmadQK::L0_TILE_K));
            const uint32_t lastMm1L0BStages = mm1L0BTotalStages_;

#ifdef __DAV_CUBE__
            // GM stays at batch/head (Dense) or cache/head (PA); the common loader
            // resolves each local tile through the selected sparse list range.
            // Under PA the head stride is page-local: D for BBND, pageSize*D for BNBD
            // (must match the non-union kernel / GbsaLayoutHelper).
            const uint64_t gmOffsetKv = isPaged ? static_cast<uint64_t>(kvHeadIdx) *
                                                      GbsaLayoutHelper<kvFormat>::HeadStride(blockSize_, embed_) :
                                                  UnionTensorOffset(kvSeq, 0, kvHeadIdx, kvHeads_, embed_);
            GemmCoord actualBlockShapeQ{rowNum, embed_, 0};

            if (contiguousIo) {
                // DN: Q is the mmad B operand, laid out [embed, rowNum].
                auto gmQTensorTla = tla::MakeTensor(
                    gQ[gmOffsetQ], tla::MakeLayout<ElementQ, LayoutQ>(strideQHead, qBaseTile_), Arch::PositionGM{});
                blockMmadQK.loadQGM(gmQTensorTla, actualBlockShapeQ);
            } else {
                uint64_t tokenOffsets[UNION_SIZE];
                for (uint32_t t = 0; t < rowsInUnit; ++t) {
                    tokenOffsets[t] = UnionTensorOffset(qSeq, qTokenInBatch + t, qHeadStart, qHeads_, embed_);
                }
                blockMmadQK.loadQGMTokenGroups(gQ, tokenOffsets, rowsInUnit, groupSize, embed_, strideQHead);
            }
#endif
#ifdef __DAV_VEC__
            // Prepare ahead of QK without waiting for softmax of the previous tile.
            for (uint32_t prepareIdx = 0; prepareIdx < min(kvSLoopNum, KV_OFFSET_PRE_LAUNCH); ++prepareIdx) {
                const uint32_t flag = KV_OFFSET_FLAG_BASE + prepareIdx % KV_OFFSET_STAGES;
                AscendC::CrossCoreWaitFlag<4, PIPE_S>(flag);
                epiloguePrepareKvOffsets(
                    gKvOffsetInfo[static_cast<uint64_t>(prepareIdx % KV_OFFSET_STAGES) * kvOffsetTileWords],
                    gTaskSparseBlockIdx.GetPhyAddr(), isPaged ? gBatchBlockTable.GetPhyAddr() : nullptr, prepareIdx,
                    min(kvBaseTile_, validKvTokens - prepareIdx * kvBaseTile_), localExplicitBlockCount, 0U);
                AscendC::CrossCoreSetFlag<4, PIPE_MTE3>(flag);
            }
#endif
            for (uint32_t kvBlockIdx = 0; kvBlockIdx < kvSLoopNum + PRE_LAUNCH; kvBlockIdx++) {
#ifdef __DAV_VEC__
                if (kvBlockIdx + KV_OFFSET_PRE_LAUNCH < kvSLoopNum) {
                    const uint32_t prepareIdx = kvBlockIdx + KV_OFFSET_PRE_LAUNCH;
                    const uint32_t flag = KV_OFFSET_FLAG_BASE + prepareIdx % KV_OFFSET_STAGES;
                    AscendC::CrossCoreWaitFlag<4, PIPE_S>(flag);
                    epiloguePrepareKvOffsets(
                        gKvOffsetInfo[static_cast<uint64_t>(prepareIdx % KV_OFFSET_STAGES) * kvOffsetTileWords],
                        gTaskSparseBlockIdx.GetPhyAddr(), isPaged ? gBatchBlockTable.GetPhyAddr() : nullptr, prepareIdx,
                        min(kvBaseTile_, validKvTokens - prepareIdx * kvBaseTile_), localExplicitBlockCount, 0U);
                    AscendC::CrossCoreSetFlag<4, PIPE_MTE3>(flag);
                }
#endif
                if (kvBlockIdx < kvSLoopNum) {
                    uint32_t kvSTileSizeAct = min(kvBaseTile_, validKvTokens - kvBlockIdx * kvBaseTile_);

                    GemmCoord actualBlockShapeQK{rowNum, kvSTileSizeAct, embed_};
                    uint32_t ubSBufId = kvBlockIdx % UB_S_OTMP_BUF_STAGES;
                    // DN: S^T is [kvTile, qPerSubCore] row-major per sub-block.
                    auto ubSLayoutTla =
                        tla::MakeLayout<ElementS, LayoutS>(RoundUp(kvSTileSizeAct, 16U), RoundUp(qBaseTile_ / 2U, 16U));
                    auto ubSTensorTla = tla::MakeTensor(ubSTensor[ubSBufId], ubSLayoutTla, Arch::PositionUB{});
                    uint32_t Mm1ToSmFlagId = ubSBufId;
                    Arch::CrossCoreFlag mm1ToSmFlag(Mm1ToSmFlagId);

#ifdef __DAV_CUBE__
                    const uint32_t sparseFlag = KV_OFFSET_FLAG_BASE + kvBlockIdx % KV_OFFSET_STAGES;
                    auto gmKLayoutTla = tla::MakeLayout<ElementK, LayoutK>(strideKVRow, kvBaseTile_);

                    // DN: K is the mmad A operand, laid out [kvTile, embed];
                    // keep the ND layout's shape types so the tensor type
                    // (and the shared call site below) stays unchanged.
                    gmKLayoutTla = tla::MakeLayout<ElementK, LayoutK>(static_cast<uint64_t>(kvBaseTile_),
                                                                      static_cast<uint32_t>(strideKVRow));
                    auto gmKTensorTla = tla::MakeTensor(gK[gmOffsetKv], gmKLayoutTla, Arch::PositionGM{});

                    uint64_t prefixSumL0AStages = CalcCrossMm1Mm2PrefixSumL0ABStages(
                        kvBlockIdx, mm1L0ATotalStages_, lastMm1L0AStages, mm2L0ATotalStages_, kvSLoopNum, true);
                    uint64_t prefixSumL0BStages = CalcCrossMm1Mm2PrefixSumL0ABStages(
                        kvBlockIdx, mm1L0BTotalStages_, lastMm1L0BStages, mm2L0BTotalStages_, kvSLoopNum, true);
                    blockMmadQK(gmKTensorTla, ubSTensorTla, gTaskSparseBlockIdx, actualBlockShapeQK, kvBlockIdx,
                                kvBaseTile_, blockShapeY_, prefixSumL0AStages, prefixSumL0BStages, mm1ToSmFlag,
                                strideKVRow, gBatchBlockTable, blockSize_, kStride0_, localExplicitBlockCount, 0U,
                                gKvOffsetInfo[static_cast<uint64_t>(kvBlockIdx % KV_OFFSET_STAGES) * kvOffsetTileWords],
                                kvOffsetGroupWords, sparseFlag);
                    if (kvBlockIdx == kvSLoopNum - 1)
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
#endif
                    uint32_t l1PBufId = kvBlockIdx % pL1BufNum_;
                    uint32_t smToMm2FlagId = l1PBufId + UB_S_OTMP_BUF_STAGES;
                    Arch::CrossCoreFlag smToMm2Flag(smToMm2FlagId);
                    // SoftmaxUnion CopyP writes nZ (same as interface LayoutPDummy / PV L1A).
                    auto l1PLayoutTla = tla::MakeLayout<ElementP, NpuArch::layout::nZ>(pvPhysicalRows, kvSTileSizeAct);
                    auto l1PTensorTla = tla::MakeTensor(l1PTensor[l1PBufId], l1PLayoutTla, Arch::PositionL1{});

#ifdef __DAV_VEC__
                    uint64_t exactMaskTileOffset = 0;

                    // exactMask layout [kvHeads, G, U * blockShapeY_]; this
                    // tile covers columns [kvBlockIdx * kvBaseTile, ...).
                    exactMaskTileOffset = (static_cast<uint64_t>(kvHeadIdx) * qBlockNum_ + globalQBlock) *
                                              static_cast<uint64_t>(topK_ * blockShapeY_) +
                                          static_cast<uint64_t>(kvBlockIdx) * kvBaseTile_;
                    uint64_t exactMaskNextTileOffset = 0;
                    uint32_t exactMaskNextCols = 0;
                    if (kvBlockIdx + 1U < kvSLoopNum) {
                        exactMaskNextTileOffset = exactMaskTileOffset + kvBaseTile_;
                        exactMaskNextCols = min(kvBaseTile_, validKvTokens - (kvBlockIdx + 1U) * kvBaseTile_);
                    }
                    epilogueOnlineSoftmax(l1PTensorTla, actualBlockShapeQK, (kvBlockIdx == 0), ubSBufId, l1PBufId,
                                          mm1ToSmFlag, smToMm2Flag, exactMaskTileOffset, exactMaskNextTileOffset,
                                          exactMaskNextCols);

                    if (kvBlockIdx == kvSLoopNum - 1) {
                        epilogueOnlineSoftmax.FixEmptyRows(static_cast<uint16_t>(rowNum));
                    }
#endif
                }
                if (kvBlockIdx >= PRE_LAUNCH) {
                    uint32_t kvBlockIdxDe = kvBlockIdx - PRE_LAUNCH;
                    uint32_t kvSTileSizeAct = min(kvBaseTile_, validKvTokens - kvBlockIdxDe * kvBaseTile_);

                    GemmCoord actualBlockShapePV{rowNum, embed_, kvSTileSizeAct};
                    uint32_t ubOTmpBufId = kvBlockIdxDe % uBufTileHelper_.loUbBufNum;
                    uint32_t Mm2ToReFlagId = ubOTmpBufId + UB_S_OTMP_BUF_STAGES + pL1BufNum_;

#ifdef __DAV_CUBE__
                    uint32_t l1PBufId = kvBlockIdxDe % pL1BufNum_;
                    GemmCoord pvShape = actualBlockShapePV;
                    uint32_t oTmpRows = rowNumRound;
                    if constexpr (isFp8) {
                        pvShape = GemmCoord{pvPhysicalRows, embed_, kvSTileSizeAct};
                        oTmpRows = pvPhysicalRows;
                    }
                    auto ubOTmpLayoutTla = tla::MakeLayout<ElementOTmp, LayoutOTmp>(oTmpRows, embedRound);
                    auto ubOTmpTensorTla =
                        tla::MakeTensor(ubOTmpTensor[ubOTmpBufId], ubOTmpLayoutTla, Arch::PositionUB{});
                    uint32_t smToMm2FlagId = l1PBufId + UB_S_OTMP_BUF_STAGES;
                    Arch::CrossCoreFlag smToMm2Flag(smToMm2FlagId);
                    Arch::CrossCoreFlag mm2ToReFlag(Mm2ToReFlagId);

                    auto gmVLayoutTla = tla::MakeLayout<ElementV, LayoutV>(kvBaseTile_, strideKVRow);
                    auto gmVTensorTla = tla::MakeTensor(gV[gmOffsetKv], gmVLayoutTla, Arch::PositionGM{});

                    uint64_t prefixSumL0AStages = CalcCrossMm1Mm2PrefixSumL0ABStages(
                        kvBlockIdxDe, mm1L0ATotalStages_, lastMm1L0AStages, mm2L0ATotalStages_, kvSLoopNum, false);
                    uint64_t prefixSumL0BStages = CalcCrossMm1Mm2PrefixSumL0ABStages(
                        kvBlockIdxDe, mm1L0BTotalStages_, lastMm1L0BStages, mm2L0BTotalStages_, kvSLoopNum, false);
                    // The same local tile index controls sparse addressing and P/L0 rotation.

                    blockMmadPV(
                        gmVTensorTla, ubOTmpTensorTla, gTaskSparseBlockIdx, pvShape, kvBlockIdxDe, kvBaseTile_,
                        blockShapeY_, prefixSumL0AStages, prefixSumL0BStages, smToMm2Flag, mm2ToReFlag, strideKVRow,
                        gBatchBlockTable, blockSize_, vStride0_, localExplicitBlockCount, 0U,
                        gKvOffsetInfo[static_cast<uint64_t>(kvBlockIdxDe % KV_OFFSET_STAGES) * kvOffsetTileWords],
                        kvOffsetGroupWords, mm1L1TileN_);

#endif
#ifdef __DAV_VEC__
                    Arch::CrossCoreFlag mm2ToReFlag(Mm2ToReFlagId);
                    uint32_t curTileMod = kvBlockIdxDe % (PRE_LAUNCH + 1);

                    epilogueRescaleO(gmOTensorTla, gmLseTensorTla, actualBlockShapePV, curTileMod, kvBlockIdxDe,
                                     (kvBlockIdxDe == 0), (kvBlockIdxDe == kvSLoopNum - 1), mm2ToReFlag);
                    if (!contiguousIo && kvBlockIdxDe == kvSLoopNum - 1) {
                        ScatterPackedOLSE(gO, gLse, gPackO, gPackLse, qSeq, qTokenInBatch, qHeadStart, rowsInUnit,
                                          groupSize, oPackRowStride);
                    }

#endif
                }
            }
        }
        ReleaseSyncFlags<4, 4, 4>();
    }

private:
#ifdef __DAV_VEC__
    // Zero O (and LSE when present) for a non-contiguous multi-token GQA pack by
    // writing each token's groupSize heads through GetTensorOffset / GetLseOffset.
    __aicore__ inline void WriteEmptyScattered(AscendC::GlobalTensor<ElementO> gO,
                                               AscendC::GlobalTensor<ElementLse> gLse, const BatchSeqInfo& qSeq,
                                               uint32_t qTokenInBatch, uint32_t qHeadStart, uint32_t rowsInUnit,
                                               uint32_t groupSize)
    {
        const uint32_t taskRows = rowsInUnit * groupSize;
        const uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        // Must match Union RescaleO SPLIT32 row ownership; a half-split lets AIV1
        // scatter pack rows that only AIV0 wrote (no cross-AIV sync after Rescale).
        uint32_t rowNumSplit = EpilogueRescaleO::RowNumSplit(taskRows);
        rowNumSplit = taskRows < rowNumSplit ? taskRows : rowNumSplit;
        const uint32_t rowNum = subBlockIdx == 0 ? rowNumSplit : taskRows - rowNumSplit;
        const uint32_t rowOffset = rowNumSplit * subBlockIdx;
        if (rowNum == 0U) {
            return;
        }
        const uint32_t colStride = RoundUp(embed_, 16U);
        auto ubO = resource.ubBuf.template GetBufferByByte<ElementO>(uBufTileHelper_.goStartOffset);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
        AscendC::Duplicate(ubO, static_cast<ElementO>(0), rowNum * colStride);
        AscendC::LocalTensor<ElementLse> ubLse;
        if constexpr (EpilogueRescaleO::DispatchPolicy::LSE_MODE == Epilogue::LseMode::OUT_ONLY) {
            ubLse = resource.ubBuf.template GetBufferByByte<ElementLse>(uBufTileHelper_.lseStartOffset);
            AscendC::Duplicate(ubLse, static_cast<ElementLse>(-3.402823466e+38F), 8U);
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        for (uint32_t r = 0; r < rowNum; ++r) {
            const uint32_t globalRow = rowOffset + r;
            const uint32_t token = globalRow / groupSize;
            const uint32_t head = qHeadStart + globalRow % groupSize;
            const uint64_t dstO = UnionTensorOffset(qSeq, qTokenInBatch + token, head, qHeads_, embed_);
            AscendC::DataCopyPad(gO[dstO], ubO[r * colStride],
                                 AscendC::DataCopyExtParams(1, embed_ * sizeof(ElementO), 0, 0, 0));
            if constexpr (EpilogueRescaleO::DispatchPolicy::LSE_MODE == Epilogue::LseMode::OUT_ONLY) {
                const uint64_t dstLse = UnionLseOffset(qSeq, qTokenInBatch + token, head, qHeads_);
                AscendC::DataCopyPad(gLse[dstLse], ubLse,
                                     AscendC::DataCopyExtParams(1, static_cast<uint32_t>(sizeof(ElementLse)), 0, 0, 0));
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
    }

    // Move packed Rescale O/LSE rows back to layout-correct GM addresses.
    __aicore__ inline void ScatterPackedOLSE(AscendC::GlobalTensor<ElementO> gO, AscendC::GlobalTensor<ElementLse> gLse,
                                             AscendC::GlobalTensor<ElementO> gPackO,
                                             AscendC::GlobalTensor<ElementLse> gPackLse, const BatchSeqInfo& qSeq,
                                             uint32_t qTokenInBatch, uint32_t qHeadStart, uint32_t rowsInUnit,
                                             uint32_t groupSize, uint32_t oPackRowStride)
    {
        const uint32_t taskRows = rowsInUnit * groupSize;
        const uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        // Same SPLIT32 ownership as Union RescaleO (see WriteEmptyScattered).
        uint32_t rowNumSplit = EpilogueRescaleO::RowNumSplit(taskRows);
        rowNumSplit = taskRows < rowNumSplit ? taskRows : rowNumSplit;
        const uint32_t rowNum = subBlockIdx == 0 ? rowNumSplit : taskRows - rowNumSplit;
        const uint32_t rowOffset = rowNumSplit * subBlockIdx;
        if (rowNum == 0U) {
            return;
        }
        auto ubO = resource.ubBuf.template GetBufferByByte<ElementO>(uBufTileHelper_.goStartOffset);
        AscendC::LocalTensor<ElementLse> ubLse;
        if constexpr (EpilogueRescaleO::DispatchPolicy::LSE_MODE == Epilogue::LseMode::OUT_ONLY) {
            ubLse = resource.ubBuf.template GetBufferByByte<ElementLse>(uBufTileHelper_.lseStartOffset);
        }
        for (uint32_t r = 0; r < rowNum; ++r) {
            const uint32_t globalRow = rowOffset + r;
            const uint32_t token = globalRow / groupSize;
            const uint32_t head = qHeadStart + globalRow % groupSize;
            const uint64_t srcO = static_cast<uint64_t>(globalRow) * oPackRowStride;
            const uint64_t dstO = UnionTensorOffset(qSeq, qTokenInBatch + token, head, qHeads_, embed_);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
            // Rescale still reads GO/LSE UB on MTE3. A wait on PIPE_V
            // alone cannot hold back MTE2, whose V_MTE2 token is pre-armed.
            // Order the scratch overwrite directly after those GM stores.
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::DataCopy(ubO, gPackO[srcO], oPackRowStride);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::DataCopyPad(gO[dstO], ubO, AscendC::DataCopyExtParams(1, embed_ * sizeof(ElementO), 0, 0, 0));
            if constexpr (EpilogueRescaleO::DispatchPolicy::LSE_MODE == Epilogue::LseMode::OUT_ONLY) {
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
                // LSE rows are individual FP32 scalars at possibly unaligned
                // GM addresses. Ordinary DataCopy truncates a sub-32B count.
                AscendC::DataCopyPad(ubLse, gPackLse[globalRow],
                                     AscendC::DataCopyExtParams(1, sizeof(ElementLse), 0, 0, 0),
                                     AscendC::DataCopyPadExtParams<ElementLse>{false, 0, 0, 0});
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);
                const uint64_t dstLse = UnionLseOffset(qSeq, qTokenInBatch + token, head, qHeads_);
                AscendC::DataCopyPad(gLse[dstLse], ubLse,
                                     AscendC::DataCopyExtParams(1, static_cast<uint32_t>(sizeof(ElementLse)), 0, 0, 0));
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        }
    }
#endif

    template <class CuTensor, class UsedTensor>
    __aicore__ inline void ReadUnionBatch(CuTensor& cu, UsedTensor& used, bool hasUsed, uint32_t batch,
                                          uint32_t capacity, BatchSeqInfo& seq) const
    {
        if (qFormat_ == Format::TND) {
            ReadBatchSeqInfo<Format::TND>(cu, used, hasUsed, batch, seq);
        } else {
            // Dense Q and KV share the layout, and both batched lengths are required.
            ReadBatchSeqInfo<Format::BSND>(cu, used, hasUsed, batch, seq, capacity);
        }
    }

    __aicore__ inline uint64_t UnionTensorOffset(const BatchSeqInfo& seq, uint32_t token, uint32_t head, uint32_t heads,
                                                 uint32_t dim) const
    {
        return qFormat_ == Format::BNSD ? GetTensorOffset<Format::BNSD>(seq, token, head, heads, dim) :
                                          GetTensorOffset<Format::TND>(seq, token, head, heads, dim);
    }

    __aicore__ inline uint64_t UnionLseOffset(const BatchSeqInfo& seq, uint32_t token, uint32_t head,
                                              uint32_t heads) const
    {
        return qFormat_ == Format::TND ? GetLseOffset<Format::TND>(seq, token, head, heads) :
                                         GetLseOffset<Format::BNSD>(seq, token, head, heads);
    }

    // One unsplit task per Q group/KV head; reject FD schedules.
    __aicore__ inline bool ValidateMetadata(
        __gm__ const GbsaMetadata::Metadata* meta,
        const GenericBlockSparseAttn::GenericBlockSparseAttentionTilingData* tiling) const
    {
        const uint32_t config[] = {tiling->batch,
                                   tiling->numHeads,
                                   tiling->kvHeads,
                                   tiling->embeddingSize,
                                   tiling->blockShapeX,
                                   tiling->blockShapeY,
                                   tiling->topK,
                                   tiling->qBlockNum,
                                   1U,
                                   0U,
                                   0U,
                                   0U};
        uint32_t signature = 2166136261U;
        for (uint32_t value : config) {
            for (uint32_t shift = 0; shift < 32U; shift += 8U) {
                signature = (signature ^ ((value >> shift) & 0xFFU)) * 16777619U;
            }
        }
        return meta != nullptr && meta->magic == GbsaMetadata::METADATA_MAGIC &&
               meta->version == GbsaMetadata::METADATA_VERSION &&
               meta->metadataUsedSize == static_cast<int32_t>(GbsaMetadata::METADATA_USED_SIZE) &&
               meta->configSignature == static_cast<int32_t>(signature) && meta->saTotalTaskNum >= 0 &&
               tiling->maskType == 3U && tiling->residualBlockMode == 0U && tiling->isConsistentTopk == 0U &&
               tiling->fdStaticEnabled == 0U && meta->fdScheduleFlags == GbsaMetadata::FD_ACTUAL_BLOCK_PREFIX &&
               meta->fdActiveCoreNum == 0 && meta->combineTaskNum == 0 && meta->fdPartialTaskNum == 0;
    }

    __aicore__ inline void FetchBaseShapeInfo(
        const GenericBlockSparseAttn::GenericBlockSparseAttentionTilingData* tilingData, GM_ADDR metaData)
    {
        batch_ = tilingData->batch;
        qFormat_ = tilingData->qLayout == 2U ? Format::BNSD : tilingData->qLayout == 1U ? Format::BSND : Format::TND;
        qSeqCapacity_ = tilingData->qSeqCapacity;
        kvSeqCapacity_ = tilingData->kvSeqCapacity;
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
        // DN mode: A tiles KV and B tiles Q, so the per-tile L0 stage counts swap.
        mm1L0ATotalStages_ = CeilDiv(kvBaseTile_, static_cast<uint32_t>(BlockMmadQK::L0_TILE_M)) * qkKLoop;
        mm1L0BTotalStages_ = CeilDiv(qBaseTile_, static_cast<uint32_t>(BlockMmadQK::L0_TILE_N)) * qkKLoop;
        mm2L0ATotalStages_ = mL0LoopPV * pvKLoop;
        mm2L0BTotalStages_ = pvKLoop * CeilDiv(embed_, static_cast<uint32_t>(BlockMmadPV::L0_TILE_N));
    }

    __aicore__ inline uint64_t CalcCrossMm1Mm2PrefixSumL0ABStages(uint32_t kvBlockIdx, uint32_t singleMm1L0Stages,
                                                                  uint32_t lastMm1L0Stages, uint32_t singleMm2L0Stages,
                                                                  uint32_t kvSLoopNum, bool isCurPhaseMm1)
    {
        uint64_t prefixSumStages;
        if (isCurPhaseMm1) {
            const uint32_t launchedMm2Num = kvBlockIdx > PRE_LAUNCH ? kvBlockIdx - PRE_LAUNCH : 0U;
            prefixSumStages = static_cast<uint64_t>(kvBlockIdx) * singleMm1L0Stages +
                              static_cast<uint64_t>(launchedMm2Num) * singleMm2L0Stages;
        } else {
            const uint32_t desiredMm1Num = kvBlockIdx + PRE_LAUNCH + 1U;
            const uint32_t launchedMm1Num = desiredMm1Num < kvSLoopNum ? desiredMm1Num : kvSLoopNum;
            uint64_t launchedMm1Stages = static_cast<uint64_t>(launchedMm1Num) * singleMm1L0Stages;
            if (launchedMm1Num == kvSLoopNum) {
                launchedMm1Stages = static_cast<uint64_t>(kvSLoopNum - 1U) * singleMm1L0Stages + lastMm1L0Stages;
            }
            prefixSumStages = launchedMm1Stages + static_cast<uint64_t>(kvBlockIdx) * singleMm2L0Stages;
        }
        return prefixSumStages;
    }

    __aicore__ inline void CalcUBufTileInfo()
    {
        static_assert(Epilogue::Block::UBufTileHelper::STATS_ROW_NUM == STATS_ROWS);
        // Host tiling checks the complete end address, including mask stages.
        const auto b = MakeUbLayout(qBaseTile_, kvBaseTile_, embed_, mm1L1TileN_, blockShapeY_, isPaged, isFp8);
        uBufTileHelper_ = Epilogue::Block::UBufTileHelper(b.qRows, b.kvRows, b.dimRows, b.s, b.p, b.lo, b.go, b.lm,
                                                          b.gm, b.dm, b.ll, b.gl, b.lse, b.descriptors, b.loBuffers);
        uBufTileHelper_.loUbRowNum = b.qRows;
        uBufTileHelper_.loGmTransit = false;
        uBufTileHelper_.kvOffsetUbBufNum = b.descriptorBuffers;
        uBufTileHelper_.exactMaskStartOffset = b.exactMask;
        uBufTileHelper_.partialMaxStartOffset = b.partialMax;
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
        for (uint32_t i = 0; i < KV_OFFSET_STAGES; ++i) {
            AscendC::CrossCoreSetFlag<4, PIPE_S>(KV_OFFSET_FLAG_BASE + i);
            AscendC::CrossCoreSetFlag<4, PIPE_S>(KV_OFFSET_FLAG_BASE + i + 16U);
        }
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
        // Seed the descriptor scratch buffer's first-use completion event.
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID5);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID6);
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
            AscendC::CrossCoreWaitFlag<MM2_RE_MODE, PIPE_FIX>(5);
            AscendC::CrossCoreWaitFlag<MM2_RE_MODE, PIPE_FIX>(21);
            AscendC::CrossCoreWaitFlag<MM2_RE_MODE, PIPE_FIX>(6);
            AscendC::CrossCoreWaitFlag<MM2_RE_MODE, PIPE_FIX>(22);
        }
#endif
#ifdef __DAV_VEC__
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID5);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID6);
        for (uint32_t i = 0; i < KV_OFFSET_STAGES; ++i) {
            AscendC::CrossCoreWaitFlag<4, PIPE_S>(KV_OFFSET_FLAG_BASE + i);
        }
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

private:
    Arch::Resource<ArchTag> resource;
    // basic shape info
    Format qFormat_;
    uint32_t qSeqCapacity_;
    uint32_t kvSeqCapacity_;
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
    uint32_t mm2L1AddrStart_ = 0;
    Gemm::Block::Mm1L1TileHelper mm1L1TileHelper_;
    Gemm::Block::Mm2L1TileHelper mm2L1TileHelper_;
    Epilogue::Block::UBufTileHelper uBufTileHelper_;
};

} // namespace GbsaKernelArch35
