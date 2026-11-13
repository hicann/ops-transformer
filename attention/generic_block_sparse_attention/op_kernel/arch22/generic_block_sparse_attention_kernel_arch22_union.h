/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GENERIC_BLOCK_SPARSE_ATTENTION_KERNEL_ARCH22_UNION_H
#define GENERIC_BLOCK_SPARSE_ATTENTION_KERNEL_ARCH22_UNION_H

#include "../kernel_common.hpp"
#include "../generic_block_sparse_attention_metadata_kernel.h"
#include "../generic_block_sparse_attention_fd_utils.h"
#include "../gbsa_sparse_kv_geometry.hpp"
#include "kernel_utils.hpp"
#include "gbsa_arch22_layout_utils.hpp"

using namespace NpuArch;

namespace GbsaKernelArch22 {

// maskType=3 (union exactMask): ND GM-workspace pipeline (Approach B).
// One task = one union group of ≤8 Q tokens × groupSize heads.
template <class BlockMmadQK, class EpilogueOnlineSoftmax, class BlockMmadPV, class EpilogueRescaleO,
          bool isPaged = true, bool hasResidualBlock = false>
class GbsaUnionKernelArch22 {
public:
    using ArchTag = typename BlockMmadPV::ArchTag;

    using ElementQ = typename BlockMmadQK::ElementA;
    using ElementK = typename BlockMmadQK::ElementB;
    using ElementS = typename EpilogueOnlineSoftmax::ElementInput;
    using ElementP = typename BlockMmadPV::ElementA;
    using ElementV = typename BlockMmadPV::ElementB;
    using ElementOTmp = typename BlockMmadPV::ElementC;
    using ElementO = typename BlockMmadQK::ElementA;

    using LayoutQ = layout::RowMajor;
    using LayoutK = layout::ColumnMajor;
    using LayoutS = layout::RowMajor;
    using LayoutP = layout::RowMajor;
    using LayoutV = layout::RowMajor;
    using LayoutO = layout::RowMajor;
    using LayoutOTmp = layout::RowMajor;
    using LayoutLse = layout::RowMajor;
    using LayoutUpdate = layout::RowMajor;

    static constexpr uint32_t PRE_LAUNCH = 2;
    static constexpr uint32_t MAX_CROSS_CORE_BUF_STAGES = PRE_LAUNCH + 1;
    static constexpr uint64_t WORKSPACE_BLOCK_SIZE_DB = 131072;
    static constexpr uint32_t QK_READY_ID = 1;
    static constexpr uint32_t SOFTMAX_READY_ID = 2;
    static constexpr uint32_t PV_READY_ID = 3;
    static constexpr uint32_t UNION_SIZE = 8U;

    __aicore__ inline GbsaUnionKernelArch22() {}

    __aicore__ inline void operator()(GbsaKernelParamsArch22 const& params)
    {
        GET_TILING_DATA_WITH_STRUCT(GenericBlockSparseAttn::GenericBlockSparseAttentionTilingData, tilingDataLocal,
                                    params.tiling);
        const GenericBlockSparseAttn::GenericBlockSparseAttentionTilingData* tilingData = &tilingDataLocal;
        FetchTilingData(tilingData, params.metaData);
        const uint32_t qLayout = tilingData->qLayout, kvLayout = tilingData->kvLayout;
        const uint32_t qCapacity = tilingData->qSeqCapacity, kvCapacity = tilingData->kvSeqCapacity;
        const uint64_t qHeadStride = qLayout == 2U ? static_cast<uint64_t>(qCapacity) * embed_ : embed_;
        const uint64_t qTokenStride = qLayout == 2U ? embed_ : static_cast<uint64_t>(qHeads_) * embed_;
        __gm__ const GbsaMetadata::Metadata* meta =
            reinterpret_cast<__gm__ const GbsaMetadata::Metadata*>(params.metaData);
        const uint32_t physicalAicNum = AscendC::GetBlockNum();
        if (!GsaFd::ValidateMetadata(meta, tilingData, physicalAicNum) || tilingData->maskType != 3U ||
            tilingData->residualBlockMode != 0U || tilingData->isConsistentTopk != 0U) {
            return;
        }
        AscendC::GlobalTensor<ElementQ> gQ;
        gQ.SetGlobalBuffer((__gm__ ElementQ*)params.q);
        AscendC::GlobalTensor<ElementK> gK;
        gK.SetGlobalBuffer((__gm__ ElementK*)params.k);
        AscendC::GlobalTensor<ElementV> gV;
        gV.SetGlobalBuffer((__gm__ ElementV*)params.v);
        AscendC::GlobalTensor<int32_t> gSparseBlockIdx;
        gSparseBlockIdx.SetGlobalBuffer((__gm__ int32_t*)params.sparseBlockIdx);
        AscendC::GlobalTensor<int32_t> gBlockTable;
        if constexpr (isPaged) {
            gBlockTable.SetGlobalBuffer((__gm__ int32_t*)params.blockTable);
        }
        StoragePrefix gCuSeqLengthsKv;
        gCuSeqLengthsKv.Init(params.cuSeqLengthsKv, kvCapacity);
        AscendC::GlobalTensor<int32_t> gSparseBlockCount;
        gSparseBlockCount.SetGlobalBuffer((__gm__ int32_t*)params.sparseBlockCount);
        AscendC::GlobalTensor<int32_t> gExactMask;
        gExactMask.SetGlobalBuffer((__gm__ int32_t*)params.attenMask);
        StoragePrefix gCuSeqLengths;
        gCuSeqLengths.Init(params.cuSeqLengths, qCapacity);
        AscendC::GlobalTensor<int32_t> gSequsedQ;
        const bool hasSequsedQ = (params.sequsedQ != nullptr);
        if (hasSequsedQ) {
            gSequsedQ.SetGlobalBuffer((__gm__ int32_t*)params.sequsedQ);
        }
        AscendC::GlobalTensor<int32_t> gSequsedKv;
        if (params.sequsedKv != nullptr) {
            gSequsedKv.SetGlobalBuffer((__gm__ int32_t*)params.sequsedKv);
        }
        AscendC::GlobalTensor<ElementO> gO;
        gO.SetGlobalBuffer((__gm__ ElementO*)params.o);
        AscendC::GlobalTensor<float> gLse;
        gLse.SetGlobalBuffer((__gm__ float*)params.softmaxLse);

        // Workspace: [S][P][OTmp][OUpdate].
        AscendC::GlobalTensor<ElementS> gS;
        gS.SetGlobalBuffer((__gm__ ElementS*)params.workSpace);
        AscendC::GlobalTensor<ElementP> gP;
        gP.SetGlobalBuffer((__gm__ ElementP*)(params.workSpace + mm1OutSize_));
        AscendC::GlobalTensor<ElementOTmp> gOTmp;
        gOTmp.SetGlobalBuffer((__gm__ ElementOTmp*)(params.workSpace + mm1OutSize_ + smOnlineOutSize_));
        AscendC::GlobalTensor<ElementOTmp> gOUpdate;
        gOUpdate.SetGlobalBuffer(
            (__gm__ ElementOTmp*)(params.workSpace + mm1OutSize_ + smOnlineOutSize_ + mm2OutSize_));

        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t coreNum = AscendC::GetBlockNum();

#ifdef __DAV_C220_CUBE__
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
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID5);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID6);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID7);

        uint32_t embedRoundQk = RoundUp(embed_, 16);
        const uint32_t l1KSize = BlockMmadQK::L1TileShape::N * embedRoundQk * sizeof(ElementK);
        const uint32_t l1QkSize = BlockMmadQK::L1TileShape::M * embedRoundQk * sizeof(ElementQ) + 2U * l1KSize;
        // For D=256 and 512 KV columns, place P at K1 and V after P.
        // Q stays resident. The shared K1/P event serializes L1 reuse.
        const bool shareK1WithQk = embed_ == 256U && kvBaseTile_ == 512U;
        const uint32_t l1PvStart = shareK1WithQk ? l1QkSize - l1KSize : l1QkSize;
        BlockMmadQK blockMmadQK(resource, 0, embedRoundQk);
        BlockMmadPV blockMmadPV(resource, l1PvStart, kvBaseTile_, shareK1WithQk);
        constexpr uint32_t copySubTiles =
            GbsaSparsePairCopy::KV_COPY_MAX_TILE_TOKENS / GbsaSparsePairCopy::KV_COPY_SUB_TILE_TOKENS;
        GbsaSparsePairCopy::KvCopySegment kvCopyRecords[MAX_CROSS_CORE_BUF_STAGES *
                                                        GbsaSparsePairCopy::KV_COPY_MAX_TILE_TOKENS /
                                                        GbsaSparsePairCopy::KV_COPY_MIN_BLOCK_Y];
        GbsaSparsePairCopy::KvCopyInfo kvCopyInfos[MAX_CROSS_CORE_BUF_STAGES][copySubTiles];
        const bool saveKvCopyInfo = isPaged && kvBaseTile_ <= GbsaSparsePairCopy::KV_COPY_MAX_TILE_TOKENS &&
                                    blockShapeY_ != 0U &&
                                    (GbsaSparsePairCopy::KV_COPY_SUB_TILE_TOKENS % blockShapeY_) == 0U;
        const uint32_t tileRecordCount = saveKvCopyInfo ? kvBaseTile_ / blockShapeY_ : 0U;
        const uint32_t subTileRecordCount =
            saveKvCopyInfo ? GbsaSparsePairCopy::KV_COPY_SUB_TILE_TOKENS / blockShapeY_ : 0U;
        for (uint32_t slot = 0; slot < MAX_CROSS_CORE_BUF_STAGES; ++slot) {
            for (uint32_t sub = 0; sub < copySubTiles; ++sub) {
                kvCopyInfos[slot][sub].records = kvCopyRecords + slot * tileRecordCount + sub * subTileRecordCount;
            }
        }
#endif

#ifdef __DAV_C220_VEC__
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID4);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID5);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID6);

        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
        EpilogueOnlineSoftmax epilogueOnlineSoftmax(resource, scaleValue_);
        EpilogueRescaleO epilogueRescaleO(resource);
        epilogueRescaleO.SetHeadStrides(qHeadStride, qLayout == 0U ? 1U : qCapacity);
        if (qLayout != 0U || kvHeads_ != 1U) {
            epilogueRescaleO.SetUnionRowStrides(groupSize_, qTokenStride, qLayout == 0U ? qHeads_ : 1U);
        }
        epilogueOnlineSoftmax.SetExactMask(gExactMask, groupSize_);

        coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
#endif

#ifdef __DAV_C220_CUBE__
        coreIdx = AscendC::GetBlockIdx();
#endif

        uint32_t groupSize = groupSize_;
        int64_t strideKVRow = (kvLayout == 2U || kvLayout == 3U) ? embed_ : kvHeads_ * embed_;
        uint32_t embedRound = RoundUp(embed_, 16);

#ifdef __DAV_C220_VEC__
        if (hasSequsedQ) {
            uint64_t paddingTask = 0U;
            for (uint32_t batchIdx = 0U; batchIdx < batch_; ++batchIdx) {
                const int64_t storageStart = gCuSeqLengths.GetValue(batchIdx);
                const int64_t storageEnd = gCuSeqLengths.GetValue(batchIdx + 1U);
                const int64_t actualLenValue = static_cast<int64_t>(gSequsedQ.GetValue(batchIdx));
                if (storageStart < 0 || storageEnd < storageStart || actualLenValue < 0) {
                    continue;
                }
                const int64_t storageLen = storageEnd - storageStart;
                const int64_t actualLen = actualLenValue < storageLen ? actualLenValue : storageLen;
                for (int64_t token = actualLen; token < storageLen; ++token) {
                    for (uint32_t kvHeadIdx = 0U; kvHeadIdx < kvHeads_; ++kvHeadIdx, ++paddingTask) {
                        if (paddingTask % coreNum != coreIdx) {
                            continue;
                        }
                        const uint64_t qStorageToken = static_cast<uint64_t>(storageStart + token);
                        const uint32_t qHeadStart = kvHeadIdx * groupSize;
                        const uint64_t gmOffsetO = TensorOffset(qLayout, batchIdx, token, qStorageToken, qHeadStart,
                                                                qHeads_, embed_, qCapacity);
                        const uint64_t gmOffsetLse =
                            LseOffset(qLayout, batchIdx, token, qStorageToken, qHeadStart, qHeads_, qCapacity);
                        epilogueRescaleO.WriteEmptyOutput(gO[gmOffsetO], gLse[gmOffsetLse], groupSize, embed_);
                    }
                }
            }
        }
#endif

        for (uint32_t taskIdx = coreIdx; taskIdx < totalTaskNum_; taskIdx += coreNum) {
            // One task covers a group of UNION_SIZE Q tokens; qBlockNum_/topK_
            // carry G (total groups) / U (union capacity).
            uint32_t qUnit = taskIdx / kvHeads_;
            uint32_t kvHeadIdx = taskIdx % kvHeads_;
            uint32_t qHeadStart = kvHeadIdx * groupSize;
            uint32_t batchIdx = 0;
            uint32_t unitInBatch = qUnit;
            uint32_t accum = 0;
            for (uint32_t b = 0; b < batch_; ++b) {
                uint32_t storageLen = static_cast<uint32_t>(gCuSeqLengths.GetValue(static_cast<int64_t>(b + 1)) -
                                                            gCuSeqLengths.GetValue(static_cast<int64_t>(b)));
                uint32_t batchLen =
                    hasSequsedQ ? static_cast<uint32_t>(gSequsedQ.GetValue(static_cast<int64_t>(b))) : storageLen;
                const uint32_t unitLen = (batchLen + UNION_SIZE - 1U) / UNION_SIZE;
                if (qUnit < accum + unitLen) {
                    batchIdx = b;
                    unitInBatch = qUnit - accum;
                    break;
                }
                accum += unitLen;
            }

            uint32_t qStorageLen = static_cast<uint32_t>(gCuSeqLengths.GetValue(static_cast<int64_t>(batchIdx + 1)) -
                                                         gCuSeqLengths.GetValue(static_cast<int64_t>(batchIdx)));
            int64_t kvStorageStart = 0;
            uint32_t kvStorageLen = 0;
            uint32_t kvSeqlen = 0;
            if constexpr (isPaged) {
                kvSeqlen = static_cast<uint32_t>(gSequsedKv.GetValue(static_cast<int64_t>(batchIdx)));
            } else {
                kvStorageStart = gCuSeqLengthsKv.GetValue(static_cast<int64_t>(batchIdx));
                kvStorageLen = static_cast<uint32_t>(gCuSeqLengthsKv.GetValue(static_cast<int64_t>(batchIdx + 1U)) -
                                                     kvStorageStart);
                kvSeqlen = params.sequsedKv == nullptr ?
                               kvStorageLen :
                               static_cast<uint32_t>(gSequsedKv.GetValue(static_cast<int64_t>(batchIdx)));
            }
            uint32_t qSeqlen =
                hasSequsedQ ? static_cast<uint32_t>(gSequsedQ.GetValue(static_cast<int64_t>(batchIdx))) : qStorageLen;
            const uint32_t qTokenInBatch = unitInBatch * UNION_SIZE;
            const uint32_t rowsInUnit = min(UNION_SIZE, qSeqlen > qTokenInBatch ? qSeqlen - qTokenInBatch : 0U);
            const uint32_t taskRows = rowsInUnit * groupSize;
            const uint64_t qStorageToken =
                static_cast<uint64_t>(gCuSeqLengths.GetValue(static_cast<int64_t>(batchIdx))) + qTokenInBatch;
            int64_t gmOffsetQ =
                TensorOffset(qLayout, batchIdx, qTokenInBatch, qStorageToken, qHeadStart, qHeads_, embed_, qCapacity);
            int64_t gmOffsetO = gmOffsetQ;
            const int64_t gmOffsetLse =
                LseOffset(qLayout, batchIdx, qTokenInBatch, qStorageToken, qHeadStart, qHeads_, qCapacity);

            if (qSeqlen == 0U || kvSeqlen == 0U || rowsInUnit == 0U) {
#ifdef __DAV_C220_VEC__
                epilogueRescaleO.WriteEmptyOutput(gO[gmOffsetO], gLse[gmOffsetLse],
                                                  taskRows > 0U ? taskRows : groupSize, embed_);
#endif
                continue;
            }

            // Union index is [kvHeads, G, U]; packed task unit index is the global group index.
            const uint32_t globalQBlock = qUnit;
            int64_t sparseIdxBase =
                static_cast<int64_t>(kvHeadIdx) * qBlockNum_ * topK_ + static_cast<int64_t>(globalQBlock) * topK_;
            const int64_t countOffset =
                static_cast<int64_t>(kvHeadIdx) * qBlockNum_ + static_cast<int64_t>(globalQBlock);
            const uint32_t explicitBlockCount = static_cast<uint32_t>(gSparseBlockCount.GetValue(countOffset));
            // Full union list; per-token visibility is applied via exactMask in softmax.
            const uint32_t rawBegin = 0U;
            const uint32_t rawEnd = explicitBlockCount;
            const uint32_t begin = rawBegin < explicitBlockCount ? rawBegin : explicitBlockCount;
            const uint32_t end = rawEnd < explicitBlockCount ? rawEnd : explicitBlockCount;
            const uint32_t selectNum = end > begin ? end - begin : 0U;
            auto gSparseBlockIdxBase = gSparseBlockIdx[sparseIdxBase];
            const uint32_t explicitBegin = begin;
            const uint32_t explicitEnd = end;
            const uint32_t localExplicitBlockCount = explicitEnd - explicitBegin;
            auto gTaskSparseBlockIdx = gSparseBlockIdxBase[explicitBegin];
            // No causal visible-prefix clip; residual is disabled for union.
            // Non-const: BlockMmadQK/PV take uint32_t& for kvSeqlen.
            uint32_t visibleKvEnd = kvSeqlen;
            // Clip any selected block that crosses visibleKvEnd (not only the
            // last-slot index == floor(kvSeqlen/Y) case).
            uint32_t validKvTokens = 0U;
            GbsaSparseGeometry::CalcSparseKvSize(gTaskSparseBlockIdx, localExplicitBlockCount, blockShapeY_,
                                                 visibleKvEnd, validKvTokens);
            uint32_t kvSLoopNum = (validKvTokens + kvBaseTile_ - 1U) / kvBaseTile_;
            uint32_t yBlockNum = (visibleKvEnd + blockShapeY_ - 1U) / blockShapeY_;
            const uint32_t residualBlockId = 0U;
            int64_t gmOffsetKvBase = kvLayout == 3U ? static_cast<int64_t>(kvHeadIdx) * blockSize_ * embed_ :
                                                      TensorOffset(kvLayout, batchIdx, 0U, kvStorageStart, kvHeadIdx,
                                                                   kvHeads_, embed_, kvCapacity);
            if (kvSLoopNum == 0U) {
#ifdef __DAV_C220_VEC__
                epilogueRescaleO.WriteEmptyOutput(gO[gmOffsetO], gLse[gmOffsetLse], taskRows, embed_);
#endif
                continue;
            }
            uint32_t preLaunch = (kvSLoopNum > PRE_LAUNCH) ? PRE_LAUNCH : 0;

            uint32_t rowNum = taskRows;
            uint32_t rowNumRound = RoundUp(rowNum, 16);
            int64_t blockTOffset = static_cast<int64_t>(batchIdx) * maxBlocksPerBatch_;
            // Contiguous TND [tok][head] rows: split rows across subcores.
            uint32_t qSBlockSize = taskRows;
            constexpr uint32_t qNBlockSize = 1U;
            const uint32_t kvTileWidth = kvBaseTile_;
            AscendC::GlobalTensor<int32_t> gBatchBlockTable;
            if constexpr (isPaged) {
                gBatchBlockTable = gBlockTable[blockTOffset];
            }

#ifdef __DAV_C220_CUBE__
            LayoutQ gmQLayout(rowNum, embed_, qHeadStride);
            if (qLayout == 2U || kvHeads_ != 1U) {
                blockMmadQK.LoadUnionQ(gQ[gmOffsetQ], rowsInUnit, groupSize, embed_, qTokenStride, qHeadStride);
            } else {
                uint32_t qLoadGroup = rowNum;
                blockMmadQK.loadQGM(gQ[gmOffsetQ], gmQLayout, rowNum, qLoadGroup, embed_);
            }
#endif

#ifdef __DAV_C220_VEC__
            // Contiguous TND rows: O stride = embed; LSE stride = 1 for TND.
            LayoutO gmOLayout(qSBlockSize, embed_);
            LayoutLse gmLseLayout(qSBlockSize, qLayout == 0U ? 1U : qHeads_);
#endif
            for (uint32_t kvBlockIdx = 0; kvBlockIdx < kvSLoopNum + preLaunch; kvBlockIdx++) {
                if (kvBlockIdx < kvSLoopNum) {
                    const uint32_t tileStart = kvBlockIdx * kvBaseTile_;
                    uint32_t kvSTileSizeAct = min(kvBaseTile_, validKvTokens - tileStart);
                    int64_t gmOffsetK = gmOffsetKvBase;

                    uint32_t stageId = kvBlockIdx % MAX_CROSS_CORE_BUF_STAGES;
                    uint64_t gmOffsetS =
                        static_cast<uint64_t>(coreIdx) * WORKSPACE_BLOCK_SIZE_DB * MAX_CROSS_CORE_BUF_STAGES +
                        static_cast<uint64_t>(stageId) * WORKSPACE_BLOCK_SIZE_DB;

#ifdef __DAV_C220_CUBE__
                    auto* kvCopyInfo = kvCopyInfos[stageId];
                    for (uint32_t pi = 0; pi < copySubTiles; ++pi) {
                        kvCopyInfo[pi].count = GbsaSparsePairCopy::KV_COPY_INVALID_COUNT;
                    }
                    LayoutK gmKLayout(strideKVRow, kvTileWidth);
                    LayoutS ubSLayout(rowNumRound, RoundUp(kvSTileSizeAct, 16));
                    GemmCoord actualBlockShapeQK{rowNum, kvSTileSizeAct, embed_};

                    blockMmadQK(gQ[gmOffsetQ], gK[gmOffsetK], gS[gmOffsetS], gBatchBlockTable, gTaskSparseBlockIdx,
                                gmQLayout, gmKLayout, ubSLayout, actualBlockShapeQK, kvBlockIdx, kvSLoopNum,
                                kvBaseTile_, strideKVRow, blockShapeY_, selectNum, yBlockNum, visibleKvEnd, blockSize_,
                                kStride0_, localExplicitBlockCount, residualBlockId, vStride0_,
                                saveKvCopyInfo ? kvCopyInfo : nullptr);

                    NpuArch::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(qkReady_);
#endif

#ifdef __DAV_C220_VEC__
                    uint64_t gmOffsetP = gmOffsetS;
                    LayoutS ubSLayout(rowNumRound, RoundUp(kvSTileSizeAct, 16));
                    LayoutP ubPLayout(rowNumRound, RoundUp(kvSTileSizeAct, 16));
                    GemmCoord actualBlockShapeQK{rowNum, kvSTileSizeAct, embed_};
                    // exactMask [kvHeads, G, U * blockShapeY]; this tile covers
                    // columns [kvBlockIdx * kvBaseTile, ...).
                    const uint64_t exactMaskTileOffset =
                        (static_cast<uint64_t>(kvHeadIdx) * qBlockNum_ + globalQBlock) *
                            static_cast<uint64_t>(topK_ * blockShapeY_) +
                        static_cast<uint64_t>(kvBlockIdx) * kvBaseTile_;

                    NpuArch::Arch::CrossCoreWaitFlag(qkReady_);

                    epilogueOnlineSoftmax(gP[gmOffsetP], gS[gmOffsetS], ubPLayout, ubSLayout, actualBlockShapeQK,
                                          (kvBlockIdx == 0), 0, qSBlockSize, qNBlockSize, stageId, softmaxReady_,
                                          exactMaskTileOffset);
                    if (kvBlockIdx + 1U == kvSLoopNum) {
                        epilogueOnlineSoftmax.FixEmptyRows(static_cast<uint16_t>(rowNum));
                    }
#endif
                }

                if (kvBlockIdx >= preLaunch) {
                    uint32_t kvBlockIdxDe = kvBlockIdx - preLaunch;
                    const uint32_t tileStart = kvBlockIdxDe * kvBaseTile_;
                    uint32_t kvSTileSizeAct = min(kvBaseTile_, validKvTokens - tileStart);
                    int64_t gmOffsetV = gmOffsetKvBase;

                    uint32_t stageId = kvBlockIdxDe % MAX_CROSS_CORE_BUF_STAGES;
                    uint64_t gmOffsetP =
                        static_cast<uint64_t>(coreIdx) * WORKSPACE_BLOCK_SIZE_DB * MAX_CROSS_CORE_BUF_STAGES +
                        static_cast<uint64_t>(stageId) * WORKSPACE_BLOCK_SIZE_DB;
                    uint64_t gmOffsetOTmp =
                        static_cast<uint64_t>(coreIdx) * WORKSPACE_BLOCK_SIZE_DB * MAX_CROSS_CORE_BUF_STAGES +
                        static_cast<uint64_t>(stageId) * WORKSPACE_BLOCK_SIZE_DB;

#ifdef __DAV_C220_CUBE__
                    LayoutP ubPLayout(rowNumRound, RoundUp(kvSTileSizeAct, 16));
                    LayoutV gmVLayout(kvTileWidth, strideKVRow);
                    LayoutOTmp ubOTmpLayout(rowNumRound, embedRound);
                    GemmCoord actualBlockShapePV{rowNum, embed_, kvSTileSizeAct};

                    blockMmadPV(gP[gmOffsetP], gV[gmOffsetV], gOTmp[gmOffsetOTmp], gBatchBlockTable,
                                gTaskSparseBlockIdx, ubPLayout, gmVLayout, ubOTmpLayout, actualBlockShapePV,
                                kvBlockIdxDe, kvSLoopNum, kvBaseTile_, visibleKvEnd, strideKVRow, 1, softmaxReady_,
                                blockShapeY_, selectNum, yBlockNum, blockSize_, vStride0_, localExplicitBlockCount,
                                residualBlockId, kStride0_, kvBaseTile_ <= 512U ? kvCopyInfos[stageId] : nullptr);
                    NpuArch::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(pvReady_);
#endif

#ifdef __DAV_C220_VEC__
                    uint64_t gmOffsetUpdate = static_cast<uint64_t>(coreIdx) * WORKSPACE_BLOCK_SIZE_DB;
                    LayoutOTmp ubOTmpLayout(rowNumRound, embedRound);
                    LayoutUpdate ubUpdateLayout(rowNumRound, embedRound);
                    GemmCoord actualBlockShapePV{rowNum, embed_, kvSTileSizeAct};

                    NpuArch::Arch::CrossCoreWaitFlag(pvReady_);

                    epilogueRescaleO(gO[gmOffsetO], gOTmp[gmOffsetOTmp], gOUpdate[gmOffsetUpdate], gLse[gmOffsetLse],
                                     gmOLayout, ubOTmpLayout, ubUpdateLayout, gmLseLayout, actualBlockShapePV,
                                     qSBlockSize, qNBlockSize, (kvBlockIdxDe == 0), (kvBlockIdxDe == kvSLoopNum - 1),
                                     stageId);
#endif
                }
            }
        }

#ifdef __DAV_C220_CUBE__
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
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID5);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID6);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID7);
#endif

#ifdef __DAV_C220_VEC__
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID5);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID6);

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
#endif
        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    __aicore__ inline void FetchTilingData(
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
        __gm__ GbsaMetadata::Metadata* meta = reinterpret_cast<__gm__ GbsaMetadata::Metadata*>(metaData);
        totalTaskNum_ = static_cast<uint32_t>(meta->saTotalTaskNum);
        scaleValue_ = tilingData->scaleValue;
        groupSize_ = tilingData->groupSize;
        qBaseTile_ = tilingData->baseTileInfo.qBaseTile;
        kvBaseTile_ = tilingData->baseTileInfo.kvBaseTile;
        mm1OutSize_ = tilingData->mm1OutSize;
        smOnlineOutSize_ = tilingData->smOnlineOutSize;
        mm2OutSize_ = tilingData->mm2OutSize;
        updateSize_ = tilingData->updateSize;
        kStride0_ = tilingData->kStride0;
        vStride0_ = tilingData->vStride0;
        residualBlockMode_ = tilingData->residualBlockMode;
        isConsistentTopk_ = tilingData->isConsistentTopk;
    }

    Arch::Resource<ArchTag> resource;
    Arch::CrossCoreFlag qkReady_{QK_READY_ID};
    Arch::CrossCoreFlag softmaxReady_{SOFTMAX_READY_ID};
    Arch::CrossCoreFlag pvReady_{PV_READY_ID};
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
    uint64_t kStride0_;
    uint64_t vStride0_;
    uint32_t residualBlockMode_;
    uint32_t isConsistentTopk_;
    uint32_t qBaseTile_;
    uint32_t kvBaseTile_;
    uint64_t mm1OutSize_;
    uint64_t smOnlineOutSize_;
    uint64_t mm2OutSize_;
    uint64_t updateSize_;
};

} // namespace GbsaKernelArch22

#endif // GENERIC_BLOCK_SPARSE_ATTENTION_KERNEL_ARCH22_UNION_H
