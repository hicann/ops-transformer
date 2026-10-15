/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BLOCK_SPARSE_ATTENTION_KERNEL_ARCH35_HIF8_FULL_QUANT_H_
#define BLOCK_SPARSE_ATTENTION_KERNEL_ARCH35_HIF8_FULL_QUANT_H_

#include "../arch35/kernel_utils.hpp"

using namespace NpuArch;
using namespace tla;

namespace BsaKernelArch35 {

template <class EpilogueMask2Idx, class BlockMmadQK, class EpilogueOnlineSoftmax, class BlockMmadPV,
          class EpilogueRescaleO, Format qFormat, Format kvFormat, bool transposedMm1, bool zNOnlineSoftmax>
class BsaHif8FullQuantKernelArch35 {
public:
    // quantMode=5: default per-tensor P quantization multiplier when pQuantScale is absent.
    static constexpr float DEFAULT_P_QUANT_SCALE = 16.0f;

    using ArchTag = typename BlockMmadPV::ArchTag;

    using ElementQ = typename BlockMmadQK::ElementQ;               // hifloat8_t
    using ElementK = typename BlockMmadQK::ElementK;               // hifloat8_t
    using ElementS = typename EpilogueOnlineSoftmax::ElementInput; // float
    using ElementP = typename BlockMmadPV::ElementA;               // hifloat8_t
    using ElementV = typename BlockMmadPV::ElementB;               // hifloat8_t
    using ElementOTmp = typename BlockMmadPV::ElementC;            // float
    using ElementO = typename EpilogueRescaleO::ElementO;          // half/bfloat16_t
    using ElementLse = typename EpilogueRescaleO::ElementLse;
    using ElementSparseMask = typename EpilogueMask2Idx::ElementSparseMask;
    using ElementSparseIdx = typename EpilogueMask2Idx::ElementSparseIdx;
    using ElementSparseCount = typename EpilogueMask2Idx::ElementSparseCount;

    using LayoutQ = std::conditional_t<transposedMm1, layout::ColumnMajor, layout::RowMajor>;
    using LayoutK = std::conditional_t<transposedMm1, layout::RowMajor, layout::ColumnMajor>;
    using LayoutS = std::conditional_t<zNOnlineSoftmax, layout::zN, layout::RowMajor>;
    using LayoutP = layout::RowMajor;
    using LayoutV = layout::RowMajor;
    using LayoutO = layout::RowMajor;
    using LayoutLse = layout::RowMajor;
    using LayoutOTmp = layout::RowMajor;
    using LayoutSparseIdx = layout::RowMajor;
    using LayoutSparseCount = layout::RowMajor;

    using LayoutTagL1P = typename BlockMmadPV::LayoutTagL1A;

    static constexpr uint32_t PRE_LAUNCH = 2;
    static constexpr uint32_t MAX_CROSS_CORE_BUF_STAGES = PRE_LAUNCH + 1;
    static constexpr uint32_t UB_S_BUF_STAGES = 2;
    static constexpr uint32_t UB_OTMP_BUF_STAGES = 1;

    // Methods
    __aicore__ inline BsaHif8FullQuantKernelArch35() {}

    __aicore__ inline void operator()(BsaFullQuantKernelParamsArch35 const& params, GM_ADDR pQuantScale)
    {
        __gm__ BlockSparseAttentionTilingData* bsaTilingData =
            reinterpret_cast<__gm__ BlockSparseAttentionTilingData*>(params.tiling);
        FetchBaseShapeInfo(bsaTilingData);
        CalcL1L0BufTileInfo(bsaTilingData);
        CalcUBufTileInfo();
        // global buffers
        AscendC::GlobalTensor<ElementQ> gQ;
        gQ.SetGlobalBuffer((__gm__ ElementQ*)params.query);
        AscendC::GlobalTensor<ElementK> gK;
        gK.SetGlobalBuffer((__gm__ ElementK*)params.key);
        AscendC::GlobalTensor<ElementV> gV;
        gV.SetGlobalBuffer((__gm__ ElementV*)params.value);
        AscendC::GlobalTensor<int64_t> gActualQseqlen;
        gActualQseqlen.SetGlobalBuffer((__gm__ int64_t*)params.actualSeqLengths);
        AscendC::GlobalTensor<int64_t> gActualKvseqlen;
        gActualKvseqlen.SetGlobalBuffer((__gm__ int64_t*)params.actualSeqLengthsKv);
        AscendC::GlobalTensor<uint8_t> gBlockSparseMask;
        gBlockSparseMask.SetGlobalBuffer((__gm__ uint8_t*)params.blockSparseMask);
        qDequantScaleValue_ = *((__gm__ float*)params.qDequantScale);
        kDequantScaleValue_ = *((__gm__ float*)params.kDequantScale);
        vDequantScaleValue_ = *((__gm__ float*)params.vDequantScale);
        AscendC::GlobalTensor<ElementO> gO;
        gO.SetGlobalBuffer((__gm__ ElementO*)params.attentionOut);
        AscendC::GlobalTensor<ElementLse> gLse;
        gLse.SetGlobalBuffer((__gm__ ElementLse*)params.lse);
        AscendC::GlobalTensor<ElementSparseIdx> gSparseIdx;
        gSparseIdx.SetGlobalBuffer((__gm__ ElementSparseIdx*)params.workSpace);
        AscendC::GlobalTensor<ElementSparseCount> gSparseCount;
        gSparseCount.SetGlobalBuffer((__gm__ ElementSparseCount*)(params.workSpace + sparseIdxSize_));

        if (pQuantScale != nullptr) {
            pQuantScale_ = *((__gm__ float*)pQuantScale);
        } else {
            pQuantScale_ = DEFAULT_P_QUANT_SCALE;
        }
        // cross core data move dst buffers
        AscendC::LocalTensor<ElementP> l1PTensor[MAX_CROSS_CORE_BUF_STAGES];
        AscendC::LocalTensor<ElementS> ubSTensor[UB_S_BUF_STAGES];
        AscendC::LocalTensor<ElementOTmp> ubOTmpTensor[UB_OTMP_BUF_STAGES];
        InitCrossCoreDstBuf(l1PTensor, ubSTensor, ubOTmpTensor);
        // core idx
        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t coreNum = AscendC::GetBlockNum();
        // set reverse sync flags
        InitSyncFlags<4, 4, 4>();
#ifdef __DAV_VEC__
        EpilogueMask2Idx epilogueMask2Idx(resource);
        uint32_t totalRowNumBlockMask = batch_ * qHeads_ * xBlockNumAligned_;
        epilogueMask2Idx(gBlockSparseMask, gSparseIdx, gSparseCount, totalRowNumBlockMask, yBlockNumAligned_,
                         avgRowPerSubCore_, preActiveSubCoreNum_);
#endif
        AscendC::SyncAll<false>();
#ifdef __DAV_CUBE__
        coreIdx = AscendC::GetBlockIdx();
        BlockMmadQK blockMmadQK(resource, mm1L1TileHelper_);
        BlockMmadPV blockMmadPV(resource, mm2L1AddrStart_, mm2L1TileHelper_);
#endif
#ifdef __DAV_VEC__
        coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        // MM1 dequantization is applied by the online softmax: the Fixpipe now runs with
        // NO_QUANT so that it can use the hardware dual-destination mode. The multiply order
        // matches the former Fixpipe scalar (qScale * kScale, then * attentionScale).
        EpilogueOnlineSoftmax epilogueOnlineSoftmax(resource, qDequantScaleValue_ * kDequantScaleValue_ * scaleValue_,
                                                    pQuantScale_, uBufTileHelper_);
        // MM2 dequantization is applied by the rescale-O epilogue, after the running sum has
        // been normalized by gl.
        EpilogueRescaleO epilogueRescaleO(resource, uBufTileHelper_, pQuantScale_, vDequantScaleValue_);
#endif
        uint32_t qSTileNumPerFullXBlock = CeilDiv(blockShapeX_, qBaseTile_);
        // Calculate strides based on layout
        // For TND: [T, N, D], stride = N * D
        // For BNSD: [B, N, S, D], strideB = N * S * D, strideN = S * D, strideS = D
        // For BSND: [B, S, N, D], strideB = S * B * D, strideS = N * D, strideN = D
        int64_t strideQO = 0;
        int64_t strideKV = 0;
        int64_t strideLse = 0;
        int64_t strideQOB = 0; // BNSD/BSND batch_ stride for Q
        int64_t strideQON = 0; // BNSD/BSND head stride for Q
        int64_t strideQOS = 0; // BNSD/BSND seq stride for Q
        int64_t strideKVB = 0; // BNSD/BSND batch_ stride for KV
        int64_t strideKVN = 0; // BNSD/BSND head stride for KV
        int64_t strideKVS = 0; // BNSD/BSND seq stride for KV
        int64_t strideLseB = 0;
        int64_t strideLseN = 0;
        int64_t strideLseS = 0;

        if constexpr (qFormat == Format::TND) {
            strideQO = qHeads_ * embed_;
            strideLse = qHeads_;
        } else if constexpr (qFormat == Format::BNSD) {
            strideQOB = qHeads_ * qSeqlenAligned_ * embed_; // batch_ stride
            strideQON = qSeqlenAligned_ * embed_;           // head stride
            strideQOS = embed_;                             // seq stride
            strideLseB = qHeads_ * qSeqlenAligned_;
            strideLseN = qSeqlenAligned_;
            strideLseS = 1;
        } else if constexpr (qFormat == Format::BSND) {
            strideQOB = qSeqlenAligned_ * qHeads_ * embed_; // batch_ stride
            strideQOS = qHeads_ * embed_;                   // seq stride
            strideQON = embed_;                             // head stride
            strideLseB = qSeqlenAligned_ * qHeads_;
            strideLseS = qHeads_;
            strideLseN = 1;
        }

        if constexpr (kvFormat == Format::TND) {
            strideKV = kvHeads_ * embed_;
        } else if constexpr (kvFormat == Format::BNSD) {
            strideKVB = kvHeads_ * kvSeqlenAligned_ * embed_; // batch_ stride
            strideKVN = kvSeqlenAligned_ * embed_;            // head stride
            strideKVS = embed_;                               // seq stride
        } else if constexpr (kvFormat == Format::BSND) {
            strideKVB = kvSeqlenAligned_ * kvHeads_ * embed_; // batch_ stride
            strideKVS = kvHeads_ * embed_;                    // seq stride
            strideKVN = embed_;                               // head stride
        }

        uint32_t embedRound = RoundUp(embed_, 16);
        uint32_t groupSize = qHeads_ / kvHeads_;
        int64_t qBOffset = 0;
        int64_t kBOffset = 0;
        int64_t vBOffset = 0;
        int64_t oBOffset = 0;
        int64_t lseBOffset = 0;
        uint32_t preTotalTaskNum = 0;
        uint32_t curBatch = 0;
        int64_t qSeqlen = actSeqAval_ ? gActualQseqlen.GetValue(curBatch) : qSeqlenAligned_;
        int64_t kvSeqlen = actSeqAval_ ? gActualKvseqlen.GetValue(curBatch) : kvSeqlenAligned_;
        uint32_t curQSTileNum = GetCurQSTileNum(qSeqlen, blockShapeX_, qBaseTile_);
        uint32_t curTotalTaskNum = firstBatchTaskNum_;
        // Go through each task
        for (uint32_t taskIdx = coreIdx; taskIdx < totalTaskNum_; taskIdx += coreNum) {
            while (taskIdx >= curTotalTaskNum) {
                ++curBatch;
                preTotalTaskNum = curTotalTaskNum;
                if constexpr (qFormat == Format::TND) {
                    qBOffset += qSeqlen * strideQO;
                    oBOffset += qSeqlen * strideQO;
                    lseBOffset += qSeqlen * strideLse;
                }
                if constexpr (kvFormat == Format::TND) {
                    kBOffset += kvSeqlen * strideKV;
                    vBOffset += kvSeqlen * strideKV;
                }
                qSeqlen = actSeqAval_ ? gActualQseqlen.GetValue(curBatch) : qSeqlenAligned_;
                kvSeqlen = actSeqAval_ ? gActualKvseqlen.GetValue(curBatch) : kvSeqlenAligned_;
                curQSTileNum = GetCurQSTileNum(qSeqlen, blockShapeX_, qBaseTile_);
                curTotalTaskNum += curQSTileNum * qHeads_;
            }
            uint32_t taskIdxCurBatch = taskIdx - preTotalTaskNum;
            uint32_t qHeadIdx = taskIdxCurBatch / curQSTileNum;
            uint32_t kvHeadIdx = qHeadIdx / groupSize;
            uint32_t qSTileIdx = taskIdxCurBatch - qHeadIdx * curQSTileNum;
            // corresponding xBlock index of cur task
            uint32_t xBlockIdx = qSTileIdx / qSTileNumPerFullXBlock;
            // the q base tile index within the corrsponding xBlock of cur task
            uint32_t qSTileIdxCurXBlock = qSTileIdx - xBlockIdx * qSTileNumPerFullXBlock;
            // corresponding head index of cur task
            // corresponding blockSparseMask gm offset of cur task
            // gmBlockSparseMask has the shape [B, qN, xBlockNumAligned, yBlockNumAligned]
            int64_t sparseMaskBOffset =
                static_cast<int64_t>(curBatch) * qHeads_ * xBlockNumAligned_ * yBlockNumAligned_;
            int64_t sparseMaskNOffset = static_cast<int64_t>(qHeadIdx) * xBlockNumAligned_ * yBlockNumAligned_;
            int64_t sparseMaskXOffset = static_cast<int64_t>(xBlockIdx) * yBlockNumAligned_;
            int64_t gmOffsetSparseMask = sparseMaskBOffset + sparseMaskNOffset + sparseMaskXOffset;
            // corresponding Q/K/V/O gm offset of cur task
            int64_t gmOffsetQ = 0;
            int64_t gmOffsetK = 0;
            int64_t gmOffsetV = 0;
            int64_t gmOffsetO = 0;
            int64_t gmOffsetLse = 0;
            int64_t qSOffset = xBlockIdx * blockShapeX_ + qSTileIdxCurXBlock * qBaseTile_;

            if constexpr (qFormat == Format::TND) {
                gmOffsetQ = qBOffset + qSOffset * strideQO + qHeadIdx * embed_;
                gmOffsetO = oBOffset + qSOffset * strideQO + qHeadIdx * embed_;
                gmOffsetLse = lseBOffset + qSOffset * strideLse + qHeadIdx;
            } else if constexpr (qFormat == Format::BNSD) {
                qBOffset = curBatch * strideQOB;
                oBOffset = curBatch * strideQOB;
                lseBOffset = curBatch * strideLseB;
                gmOffsetQ = qBOffset + qHeadIdx * strideQON + qSOffset * strideQOS;
                gmOffsetO = oBOffset + qHeadIdx * strideQON + qSOffset * strideQOS;
                gmOffsetLse = lseBOffset + qHeadIdx * strideLseN + qSOffset * strideLseS;
            } else if constexpr (qFormat == Format::BSND) {
                qBOffset = curBatch * strideQOB;
                oBOffset = curBatch * strideQOB;
                lseBOffset = curBatch * strideLseB;
                gmOffsetQ = qBOffset + qSOffset * strideQOS + qHeadIdx * strideQON;
                gmOffsetO = oBOffset + qSOffset * strideQOS + qHeadIdx * strideQON;
                gmOffsetLse = lseBOffset + qSOffset * strideLseS + qHeadIdx * strideLseN;
            }

            if constexpr (kvFormat == Format::TND) {
                gmOffsetK = kBOffset + kvHeadIdx * embed_;
                gmOffsetV = vBOffset + kvHeadIdx * embed_;
            } else if constexpr (kvFormat == Format::BNSD) {
                kBOffset = curBatch * strideKVB;
                vBOffset = curBatch * strideKVB;
                gmOffsetK = kBOffset + kvHeadIdx * strideKVN;
                gmOffsetV = vBOffset + kvHeadIdx * strideKVN;
            } else if constexpr (kvFormat == Format::BSND) {
                kBOffset = curBatch * strideKVB;
                vBOffset = curBatch * strideKVB;
                gmOffsetK = kBOffset + kvHeadIdx * strideKVN;
                gmOffsetV = vBOffset + kvHeadIdx * strideKVN;
            }

            // the actual x block num of cur batch_, calc by actual qseqlen
            uint32_t xBlockNumAval = static_cast<uint32_t>(CeilDiv(qSeqlen, static_cast<int64_t>(blockShapeX_)));
            uint32_t xBlockSize =
                (xBlockIdx == xBlockNumAval - 1) ? (qSeqlen - xBlockIdx * blockShapeX_) : blockShapeX_;
            uint32_t qSTileNumCurXBlock = CeilDiv(xBlockSize, qBaseTile_);
            uint32_t qSTileSizeAct = (qSTileIdxCurXBlock == qSTileNumCurXBlock - 1) ?
                                         (xBlockSize - qSTileIdxCurXBlock * qBaseTile_) :
                                         qBaseTile_;
            // calc the gathered kvS from sparse mask
            uint64_t gmOffsetSparseCount = static_cast<uint64_t>(curBatch) * qHeads_ * xBlockNumAligned_ +
                                           static_cast<uint64_t>(qHeadIdx) * xBlockNumAligned_ + xBlockIdx;
            uint32_t yBlockNumRsvd = gSparseCount.GetValue(gmOffsetSparseCount);
            if (yBlockNumRsvd == 0) {
                // Invalid sparse rows retain the caller's O/LSE data; no last index exists.
                continue;
            }

            uint64_t gmOffsetSparseIdx = gmOffsetSparseCount * yBlockNumAligned_;
            uint64_t lastIdxOffset = gmOffsetSparseIdx + yBlockNumRsvd - 1;
            uint32_t lastSparseIdx = gSparseIdx.GetValue(lastIdxOffset);

            uint32_t yBlockNumAval = static_cast<uint32_t>(CeilDiv(kvSeqlen, static_cast<int64_t>(blockShapeY_)));
            uint32_t lastYBlockSize =
                (lastSparseIdx == yBlockNumAval - 1) ? kvSeqlen - lastSparseIdx * blockShapeY_ : blockShapeY_;
            int64_t gatheredKvSeqlen = (yBlockNumRsvd - 1) * blockShapeY_ + lastYBlockSize;
            // the rowNum of cur task
            // no qS*qN combination even in GQA/MQA senario, since each qN has a different sparse pattern
            uint32_t rowNum = qSTileSizeAct;
            uint32_t rowNumRound = RoundUp(rowNum, 16);
            uint32_t kvSLoopNum = static_cast<uint32_t>(CeilDiv(gatheredKvSeqlen, static_cast<int64_t>(kvBaseTile_)));
            uint32_t kvSTileSizeAct = kvBaseTile_;
#ifdef __DAV_CUBE__
            uint32_t qShapeCol = 0;
            uint32_t qShapeRow = 0;
            uint32_t kShapeCol = 0;
            uint32_t kShapeRow = 0;
            uint32_t vShapeCol = 0;
            uint32_t vShapeRow = 0;

            if constexpr (qFormat == Format::TND) {
                qShapeCol = strideQO;
            } else if constexpr (qFormat == Format::BNSD) {
                qShapeCol = strideQOS;
            } else if constexpr (qFormat == Format::BSND) {
                qShapeCol = strideQOS;
            }
            if constexpr (kvFormat == Format::TND) {
                kShapeRow = strideKV;
                vShapeCol = strideKV;
            } else if constexpr (kvFormat == Format::BNSD) {
                kShapeRow = strideKVS;
                vShapeCol = strideKVS;
            } else if constexpr (qFormat == Format::BSND) {
                kShapeRow = strideKVS;
                vShapeCol = strideKVS;
            }
            vShapeRow = kvBaseTile_;
            // K * Q^T views Q as [D, QS] column-major and K as [KVS, D] row-major.
            if constexpr (transposedMm1) {
                qShapeRow = qShapeCol;
                qShapeCol = qBaseTile_;
                kShapeCol = kShapeRow;
                kShapeRow = kvBaseTile_;
            } else {
                qShapeRow = qBaseTile_;
                kShapeCol = kvBaseTile_;
            }

            auto gmQLayoutTla = tla::MakeLayout<ElementQ, LayoutQ>(qShapeRow, qShapeCol);
            auto gmQTensorTla = tla::MakeTensor(gQ[gmOffsetQ], gmQLayoutTla, Arch::PositionGM{});
            GemmCoord actualBlockShapeQ{rowNum, embed_, 0};
            blockMmadQK.loadQGM(gmQTensorTla, actualBlockShapeQ);

            auto gmKLayoutTla = tla::MakeLayout<ElementK, LayoutK>(kShapeRow, kShapeCol);
            auto gmKTensorTla = tla::MakeTensor(gK[gmOffsetK], gmKLayoutTla, Arch::PositionGM{});

            auto gmVLayoutTla = tla::MakeLayout<ElementV, LayoutV>(vShapeRow, vShapeCol);
            auto gmVTensorTla = tla::MakeTensor(gV[gmOffsetV], gmVLayoutTla, Arch::PositionGM{});

            // MM1 and MM2 share two physical L0A/L0B/L0C slots. Advance this
            // counter in real call order and select the slot with stage % 2.
            uint64_t l0Stage = 0;

#endif
#ifdef __DAV_VEC__
            uint32_t oShapeCol = 0;
            uint32_t lseShapeCol = 0;
            if constexpr (qFormat == Format::TND) {
                oShapeCol = strideQO;
                lseShapeCol = strideLse;
            } else if constexpr (qFormat == Format::BNSD) {
                oShapeCol = strideQOS;
                lseShapeCol = strideLseS;
            } else if constexpr (qFormat == Format::BSND) {
                oShapeCol = strideQOS;
                lseShapeCol = strideLseS;
            }

            auto gmOLayoutTla = tla::MakeLayout<ElementO, LayoutO>(qBaseTile_, oShapeCol);
            auto gmOTensorTla = tla::MakeTensor(gO[gmOffsetO], gmOLayoutTla, Arch::PositionGM{});
            auto gmLseLayoutTla = tla::MakeLayout<ElementLse, LayoutLse>(qBaseTile_, lseShapeCol);
            auto gmLseTensorTla = tla::MakeTensor(gLse[gmOffsetLse], gmLseLayoutTla, Arch::PositionGM{});
#endif
            for (uint32_t gatheredKvSTileIdx = 0; gatheredKvSTileIdx < kvSLoopNum + PRE_LAUNCH; gatheredKvSTileIdx++) {
                if (gatheredKvSTileIdx < kvSLoopNum) {
                    if (gatheredKvSTileIdx == kvSLoopNum - 1) {
                        kvSTileSizeAct = gatheredKvSeqlen - gatheredKvSTileIdx * kvBaseTile_;
                    } else {
                        kvSTileSizeAct = kvBaseTile_;
                    }
                    // QK
                    GemmCoord actualBlockShapeQK{rowNum, kvSTileSizeAct, embed_};
                    uint32_t ubSBufId = gatheredKvSTileIdx % UB_S_BUF_STAGES;
                    uint32_t sShapeRow = 0;
                    uint32_t sShapeCol = 0;
                    int64_t sStrideRow = 1;
                    if constexpr (transposedMm1) {
                        // S is [RoundUp(kvSTileSizeAct, 16), RoundUp(qBaseTile_, 32)] rowmajor in UB
                        // with stride [64, 1]. The column extent is the *full* per-block Q width
                        // because that is what the Fixpipe is told as nSize; the 64 is the per-AIV
                        // S buffer pitch that becomes dstStride. The right half of every row is
                        // never materialized: with dualDstCtl=2 the hardware writes columns 0..63
                        // into AIV0 and 64..127 into AIV1 at the same UB offsets, so a single
                        // Fixpipe serves both AIVs. This is why the tile copy must be
                        // SPLIT_N + NO_QUANT; a scalar-dequant Fixpipe cannot use dualDstCtl=2.
                        sShapeRow = RoundUp(kvSTileSizeAct, 16);
                        sShapeCol = RoundUp(qBaseTile_, 32);
                        sStrideRow = 64;
                    } else {
                        sShapeRow = rowNumRound;
                        sShapeCol = RoundUp(kvSTileSizeAct, 16);
                        sStrideRow = static_cast<int64_t>(sShapeCol);
                    }
                    auto ubSLayoutTla = tla::MakeLayout(tla::MakeShape(sShapeRow, sShapeCol),
                                                        tla::MakeStride(sStrideRow, tla::Int<1>{}));
                    auto ubSTensorTla = tla::MakeTensor(ubSTensor[ubSBufId], ubSLayoutTla, Arch::PositionUB{});
                    uint32_t Mm1ToSmFlagId = ubSBufId;
                    Arch::CrossCoreFlag mm1ToSmFlag(Mm1ToSmFlagId);
#ifdef __DAV_CUBE__
                    blockMmadQK(gmKTensorTla, ubSTensorTla, gSparseIdx[gmOffsetSparseIdx], actualBlockShapeQK,
                                gatheredKvSTileIdx, kvSeqlen, kvBaseTile_, blockShapeY_, yBlockNumAval, yBlockNumRsvd,
                                l0Stage, mm1ToSmFlag);
                    if (gatheredKvSTileIdx == kvSLoopNum - 1) {
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
                    }
#endif
                    // SM
                    uint32_t l1PBufId = gatheredKvSTileIdx % pL1BufNum_;
                    uint32_t smToMm2FlagId = l1PBufId + UB_S_BUF_STAGES;
                    Arch::CrossCoreFlag smToMm2Flag(smToMm2FlagId);
                    // HIF8 P keeps a fixed 64-Q-row L1 slot for each AIV even
                    // on a Q tail, matching the QFA HIF8 DN layout.
                    auto l1PLayoutTla = tla::MakeLayout<ElementP, NpuArch::layout::zN>(qBaseTile_, kvSTileSizeAct);
                    auto l1PTensorTla = tla::MakeTensor(l1PTensor[l1PBufId], l1PLayoutTla, Arch::PositionL1{});
#ifdef __DAV_VEC__
                    epilogueOnlineSoftmax(l1PTensorTla, actualBlockShapeQK, (gatheredKvSTileIdx == 0), ubSBufId,
                                          l1PBufId, mm1ToSmFlag, smToMm2Flag);
#endif
                }
                if (gatheredKvSTileIdx >= PRE_LAUNCH) {
                    uint32_t gatheredKvSTileIdxDe = gatheredKvSTileIdx - PRE_LAUNCH;
                    if (gatheredKvSTileIdxDe == kvSLoopNum - 1) {
                        kvSTileSizeAct = gatheredKvSeqlen - gatheredKvSTileIdxDe * kvBaseTile_;
                    } else {
                        kvSTileSizeAct = kvBaseTile_;
                    }
                    // PV
                    GemmCoord actualBlockShapePV{rowNum, embed_, kvSTileSizeAct};
                    // Cube consumes both fixed 64-row P slots.  Invalid tail
                    // lanes are zero-filled by Softmax; RescaleO below still
                    // uses the real row count and writes only valid Q rows.
                    GemmCoord cubeBlockShapePV{qBaseTile_, embed_, kvSTileSizeAct};
                    uint32_t ubOTmpBufId = gatheredKvSTileIdxDe % UB_OTMP_BUF_STAGES;
                    uint32_t Mm2ToReFlagId = ubOTmpBufId + UB_S_BUF_STAGES + pL1BufNum_;
#ifdef __DAV_CUBE__
                    uint32_t l1PBufId = gatheredKvSTileIdxDe % pL1BufNum_;
                    // The row extent is the full Q base tile that MM2 hands to the Fixpipe as mSize
                    // (not rowNumRound, which tracks the real, possibly tail, Q tile). Only the
                    // stride matters for addressing: both AIVs write their half at the same UB
                    // offsets. cubeBlockShapePV already uses qBaseTile_ for the same reason.
                    auto ubOTmpLayoutTla =
                        tla::MakeLayout<ElementOTmp, LayoutOTmp>(RoundUp(qBaseTile_, 32), embedRound);
                    auto ubOTmpTensorTla =
                        tla::MakeTensor(ubOTmpTensor[ubOTmpBufId], ubOTmpLayoutTla, Arch::PositionUB{});
                    uint32_t smToMm2FlagId = l1PBufId + UB_S_BUF_STAGES;

                    Arch::CrossCoreFlag smToMm2Flag(smToMm2FlagId);
                    Arch::CrossCoreFlag mm2ToReFlag(Mm2ToReFlagId);
                    blockMmadPV(gmVTensorTla, ubOTmpTensorTla, gSparseIdx[gmOffsetSparseIdx], cubeBlockShapePV,
                                gatheredKvSTileIdxDe, kvSeqlen, kvBaseTile_, blockShapeY_, yBlockNumAval, yBlockNumRsvd,
                                l0Stage, smToMm2Flag, mm2ToReFlag);
#endif
#ifdef __DAV_VEC__
                    // rescale O
                    Arch::CrossCoreFlag mm2ToReFlag(Mm2ToReFlagId);
                    uint32_t curTileMod = gatheredKvSTileIdxDe % (PRE_LAUNCH + 1);
                    epilogueRescaleO(gmOTensorTla, gmLseTensorTla, actualBlockShapePV, curTileMod, gatheredKvSTileIdxDe,
                                     (gatheredKvSTileIdxDe == 0), (gatheredKvSTileIdxDe == kvSLoopNum - 1),
                                     mm2ToReFlag);
#endif
                }
            }
        }
        // release reverse sync flags
        ReleaseSyncFlags<4, 4, 4>();
    }

    __aicore__ inline void FetchBaseShapeInfo(__gm__ BlockSparseAttentionTilingData* bsaTilingData)
    {
        batch_ = bsaTilingData->batch;
        qHeads_ = bsaTilingData->numHeads;
        kvHeads_ = bsaTilingData->kvHeads;
        embed_ = bsaTilingData->embeddingSize;
        firstBatchTaskNum_ = bsaTilingData->firstBatchTaskNum;
        totalTaskNum_ = bsaTilingData->totalTaskNum;
        blockShapeX_ = bsaTilingData->blockShapeX;
        blockShapeY_ = bsaTilingData->blockShapeY;
        scaleValue_ = bsaTilingData->scaleValue;
        // mask2idx tile info
        xBlockNumAligned_ = bsaTilingData->BsaMask2IdxTileInfo.xBlockNumAligned;
        yBlockNumAligned_ = bsaTilingData->BsaMask2IdxTileInfo.yBlockNumAligned;
        avgRowPerSubCore_ = bsaTilingData->BsaMask2IdxTileInfo.avgRowPerSubCore;
        preActiveSubCoreNum_ = bsaTilingData->BsaMask2IdxTileInfo.preActiveSubCoreNum;
        // base tile info
        qBaseTile_ = bsaTilingData->BsaBaseTileInfo.qBaseTile;
        kvBaseTile_ = bsaTilingData->BsaBaseTileInfo.kvBaseTile;
        // whether actual seqlen is provided
        actSeqAval_ = (!bsaTilingData->useUniformQSeqlen) && (!bsaTilingData->useUniformKvSeqlen);
        sparseIdxSize_ = bsaTilingData->selectIdxSize;
        // aligned seqlen q & kv
        qSeqlenAligned_ = bsaTilingData->maxQSeqlen;
        kvSeqlenAligned_ = bsaTilingData->maxKvSeqlen;
    }

    __aicore__ inline void CalcL1L0BufTileInfo(__gm__ BlockSparseAttentionTilingData* bsaTilingData)
    {
        mm1L1TileM_ = bsaTilingData->BsaMmPhaseL1TileInfo.mm1L1TileM;
        mm1L1TileN_ = bsaTilingData->BsaMmPhaseL1TileInfo.mm1L1TileN;
        mm1L1TileKLeft_ = bsaTilingData->BsaMmPhaseL1TileInfo.mm1L1TileKLeft;
        mm1L1TileKRight_ = bsaTilingData->BsaMmPhaseL1TileInfo.mm1L1TileKRight;
        mm2L1TileM_ = bsaTilingData->BsaMmPhaseL1TileInfo.mm2L1TileM;
        mm2L1TileN_ = bsaTilingData->BsaMmPhaseL1TileInfo.mm2L1TileN;
        mm2L1TileKLeft_ = bsaTilingData->BsaMmPhaseL1TileInfo.mm2L1TileKLeft;
        mm2L1TileKRight_ = bsaTilingData->BsaMmPhaseL1TileInfo.mm2L1TileKRight;
        qL1BufNum_ = bsaTilingData->BsaMmPhaseL1TileInfo.qL1BufNum;
        kL1BufNum_ = bsaTilingData->BsaMmPhaseL1TileInfo.kL1BufNum;
        vL1BufNum_ = bsaTilingData->BsaMmPhaseL1TileInfo.vL1BufNum;
        pL1BufNum_ = bsaTilingData->BsaMmPhaseL1TileInfo.pL1BufNum;
        Gemm::Block::Mm1L1TileHelper mm1L1TileHelper(mm1L1TileM_, mm1L1TileN_, mm1L1TileKLeft_, mm1L1TileKRight_,
                                                     qL1BufNum_, kL1BufNum_);
        mm1L1TileHelper_ = mm1L1TileHelper;
        Gemm::Block::Mm2L1TileHelper mm2L1TileHelper(mm2L1TileM_, mm2L1TileN_, mm2L1TileKLeft_, mm2L1TileKRight_,
                                                     pL1BufNum_, vL1BufNum_);
        mm2L1TileHelper_ = mm2L1TileHelper;
        mm2L1AddrStart_ = mm1L1TileM_ * mm1L1TileKLeft_ * qL1BufNum_ * sizeof(ElementQ) +
                          mm1L1TileKRight_ * mm1L1TileN_ * kL1BufNum_ * sizeof(ElementK);
    }

    __aicore__ inline void CalcUBufTileInfo()
    {
        // Prerequisites: qBaseTile_ <= 128, and kvBaseTile_ is a multiple of 128, guaranteed by tiling strategy.
        uint32_t qBaseTilePerSubCore = 64;
        uint32_t kvBaseTilePerSubCore = RoundUp(kvBaseTile_, 32);
        uint32_t embedPerSubCore = RoundUp(embed_, 32);
        uint32_t pExtraElemNum = 0;
        if constexpr (transposedMm1) {
            pExtraElemNum = Max(qBaseTilePerSubCore, kvBaseTilePerSubCore);
        }
        uint32_t sStartOffset = 0;
        uint32_t pStartOffset =
            sStartOffset + qBaseTilePerSubCore * kvBaseTilePerSubCore * sizeof(ElementS) * UB_S_BUF_STAGES;
        uint32_t loStartOffset = pStartOffset + (qBaseTilePerSubCore * kvBaseTilePerSubCore + pExtraElemNum) *
                                                    sizeof(ElementP) * UB_S_BUF_STAGES;
        uint32_t goStartOffset =
            loStartOffset + qBaseTilePerSubCore * embedPerSubCore * sizeof(ElementOTmp) * UB_OTMP_BUF_STAGES;
        uint32_t lmStartOffset = goStartOffset + qBaseTilePerSubCore * embedPerSubCore * sizeof(ElementOTmp);
        uint32_t gmStartOffset = lmStartOffset + qBaseTilePerSubCore * sizeof(float);
        uint32_t dmStartOffset = gmStartOffset + qBaseTilePerSubCore * sizeof(float);
        uint32_t llStartOffset = dmStartOffset + qBaseTilePerSubCore * (PRE_LAUNCH + 1) * sizeof(float);
        uint32_t glStartOffset = llStartOffset + qBaseTilePerSubCore * sizeof(float);
        uint32_t lseStartOffset = glStartOffset + qBaseTilePerSubCore * sizeof(float);
        uint32_t maskStartOffset = lseStartOffset + qBaseTilePerSubCore * (32 / sizeof(float)) * sizeof(float);

        uBufTileHelper_.qBaseTilePerSubCore = qBaseTilePerSubCore;
        uBufTileHelper_.kvBaseTilePerSubCore = kvBaseTilePerSubCore;
        uBufTileHelper_.embedPerSubCore = embedPerSubCore;
        uBufTileHelper_.sStartOffset = sStartOffset;
        uBufTileHelper_.pStartOffset = pStartOffset;
        uBufTileHelper_.loStartOffset = loStartOffset;
        uBufTileHelper_.goStartOffset = goStartOffset;
        uBufTileHelper_.lmStartOffset = lmStartOffset;
        uBufTileHelper_.gmStartOffset = gmStartOffset;
        uBufTileHelper_.dmStartOffset = dmStartOffset;
        uBufTileHelper_.llStartOffset = llStartOffset;
        uBufTileHelper_.glStartOffset = glStartOffset;
        uBufTileHelper_.lseStartOffset = lseStartOffset;
        uBufTileHelper_.maskStartOffset = maskStartOffset;
    }

    __aicore__ inline void InitCrossCoreDstBuf(AscendC::LocalTensor<ElementP> (&l1PTensor)[MAX_CROSS_CORE_BUF_STAGES],
                                               AscendC::LocalTensor<ElementS> (&ubSTensor)[UB_S_BUF_STAGES],
                                               AscendC::LocalTensor<ElementOTmp> (&ubOTmpTensor)[UB_OTMP_BUF_STAGES])
    {
        for (uint32_t i = 0; i < pL1BufNum_; i++) {
            l1PTensor[i] = resource.l1Buf.template GetBufferByByte<ElementP>(
                mm2L1AddrStart_ + mm2L1TileM_ * mm2L1TileKLeft_ * sizeof(ElementP) * i);
        }
        for (uint32_t i = 0; i < UB_S_BUF_STAGES; i++) {
            ubSTensor[i] = resource.ubBuf.template GetBufferByByte<ElementS>(
                uBufTileHelper_.sStartOffset +
                uBufTileHelper_.qBaseTilePerSubCore * uBufTileHelper_.kvBaseTilePerSubCore * sizeof(ElementS) * i);
        }
        for (uint32_t i = 0; i < UB_OTMP_BUF_STAGES; i++) {
            ubOTmpTensor[i] = resource.ubBuf.template GetBufferByByte<ElementOTmp>(
                uBufTileHelper_.loStartOffset +
                uBufTileHelper_.qBaseTilePerSubCore * uBufTileHelper_.embedPerSubCore * sizeof(ElementOTmp) * i);
        }
    }

    template <uint32_t MM1_SM_MODE, uint32_t MM2_RE_MODE, uint32_t SM_MM2_MODE>
    __aicore__ inline void InitSyncFlags()
    {
#ifdef __DAV_CUBE__
        // same core sync between pipes
        // Query
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
        // Key
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
        // Value
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
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
        // cross core sync
        if constexpr (SM_MM2_MODE == Arch::CROSS_CORE_SYNC_MODE_4) {
            AscendC::CrossCoreSetFlag<SM_MM2_MODE, PIPE_MTE1>(Arch::FLAG_ID2);
            AscendC::CrossCoreSetFlag<SM_MM2_MODE, PIPE_MTE1>(Arch::FLAG_ID18);
            AscendC::CrossCoreSetFlag<SM_MM2_MODE, PIPE_MTE1>(Arch::FLAG_ID3);
            AscendC::CrossCoreSetFlag<SM_MM2_MODE, PIPE_MTE1>(Arch::FLAG_ID19);
            AscendC::CrossCoreSetFlag<SM_MM2_MODE, PIPE_MTE1>(Arch::FLAG_ID4);
            AscendC::CrossCoreSetFlag<SM_MM2_MODE, PIPE_MTE1>(Arch::FLAG_ID20);
        }
#endif
#ifdef __DAV_VEC__
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        // mask2index
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
        // softmax
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID3);
        // rescale
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);

        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID1);
        if constexpr (MM1_SM_MODE == Arch::CROSS_CORE_SYNC_MODE_4) {
            AscendC::CrossCoreSetFlag<MM1_SM_MODE, PIPE_V>(Arch::FLAG_ID0);
            AscendC::CrossCoreSetFlag<MM1_SM_MODE, PIPE_V>(Arch::FLAG_ID1);
        }
        if constexpr (MM2_RE_MODE == Arch::CROSS_CORE_SYNC_MODE_4) {
            AscendC::CrossCoreSetFlag<MM2_RE_MODE, PIPE_V>(Arch::FLAG_ID5);
        }
#endif
    }

    template <uint32_t MM1_SM_MODE, uint32_t MM2_RE_MODE, uint32_t SM_MM2_MODE>
    __aicore__ inline void ReleaseSyncFlags()
    {
#ifdef __DAV_CUBE__
        // same core sync between pipes
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
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
        if constexpr (MM1_SM_MODE == Arch::CROSS_CORE_SYNC_MODE_4) {
            AscendC::CrossCoreWaitFlag<MM1_SM_MODE, PIPE_FIX>(Arch::FLAG_ID0);
            AscendC::CrossCoreWaitFlag<MM1_SM_MODE, PIPE_FIX>(Arch::FLAG_ID1);
            AscendC::CrossCoreWaitFlag<MM1_SM_MODE, PIPE_FIX>(Arch::FLAG_ID16);
            AscendC::CrossCoreWaitFlag<MM1_SM_MODE, PIPE_FIX>(Arch::FLAG_ID17);
        }
        if constexpr (MM2_RE_MODE == Arch::CROSS_CORE_SYNC_MODE_4) {
            AscendC::CrossCoreWaitFlag<MM2_RE_MODE, PIPE_FIX>(Arch::FLAG_ID5);
            AscendC::CrossCoreWaitFlag<MM2_RE_MODE, PIPE_FIX>(Arch::FLAG_ID21);
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
        if constexpr (SM_MM2_MODE == Arch::CROSS_CORE_SYNC_MODE_4) {
            AscendC::CrossCoreWaitFlag<SM_MM2_MODE, PIPE_MTE3>(Arch::FLAG_ID2);
            AscendC::CrossCoreWaitFlag<SM_MM2_MODE, PIPE_MTE3>(Arch::FLAG_ID3);
            AscendC::CrossCoreWaitFlag<SM_MM2_MODE, PIPE_MTE3>(Arch::FLAG_ID4);
        }
#endif
        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    Arch::Resource<ArchTag> resource;
    /*
    tiling info, which are const in each kernel launch
    */
    // basic shape info
    uint32_t batch_;
    uint32_t qHeads_;
    uint32_t kvHeads_;
    uint32_t embed_;
    uint32_t firstBatchTaskNum_;
    uint32_t totalTaskNum_;
    uint32_t blockShapeX_;
    uint32_t blockShapeY_;
    float scaleValue_;
    float qDequantScaleValue_{1.0f};
    float kDequantScaleValue_{1.0f};
    float vDequantScaleValue_{1.0f};
    float pQuantScale_;
    // mask2idx tile info
    uint32_t xBlockNumAligned_;
    uint32_t yBlockNumAligned_;
    uint32_t avgRowPerSubCore_;
    uint32_t preActiveSubCoreNum_;
    // base tile info
    uint32_t qBaseTile_;
    uint32_t kvBaseTile_;
    // whether actual seqlen is provided
    uint32_t actSeqAval_;
    // workspace size
    uint64_t sparseIdxSize_;
    // aligned seqlen q & kv
    int64_t qSeqlenAligned_;
    int64_t kvSeqlenAligned_;
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
    uint32_t mm2L1AddrStart_ = 0;
    Gemm::Block::Mm1L1TileHelper mm1L1TileHelper_;
    Gemm::Block::Mm2L1TileHelper mm2L1TileHelper_;
    Epilogue::Block::UBufTileHelper uBufTileHelper_;
};

} // namespace BsaKernelArch35

#endif // BLOCK_SPARSE_ATTENTION_KERNEL_ARCH35_HIF8_FULL_QUANT_H_
