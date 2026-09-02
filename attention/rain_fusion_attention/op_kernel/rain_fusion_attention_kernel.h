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
 * \file rain_fusion_attention_kernel.h
 * \brief Rain Fusion Attention Kernel Implementation
 */

#ifndef RAIN_FUSION_ATTENTION_KERNEL_H
#define RAIN_FUSION_ATTENTION_KERNEL_H

#include "rain_fusion_attention_kernel_common.hpp"

using namespace NpuArch;
using namespace RfaKenelCommon;

namespace RainFusion {
/**
 * @brief Rain Fusion Attention Inference Kernel
 *
 * This kernel implements rain fusion attention where attention is computed only on
 * selected KV blocks specified by selectIdx. This reduces computation for long sequences
 * by focusing on relevant tokens.
 *
 * @tparam BlockMmadQK Block-level QK matmul module
 * @tparam BlockMmadPV Block-level PV matmul module
 * @tparam EpilogueOnlineSoftmax Online softmax epilogue
 * @tparam EpilogueRescaleO Output rescaling epilogue
 * @tparam PAGED_CACHE_FLAG Whether to use paged KV cache
 * @tparam QUERY_LAYOUT Query tensor layout (0=TND, 1=BNSD)
 * @tparam KV_CACHE_LAYOUT KV cache layout (0=TND, 1=BNSD)
 */
template <class BlockMmadQK, class BlockMmadPV, class EpilogueOnlineSoftmax, class EpilogueRescaleO,
          bool PAGED_CACHE_FLAG, uint32_t QUERY_LAYOUT, uint32_t KV_CACHE_LAYOUT>
class RainFusionAttentionKernel {
public:
    using ArchTag = typename BlockMmadQK::ArchTag;
    using L1TileShape = typename BlockMmadQK::L1TileShape;
    using ElementQ = typename BlockMmadQK::ElementA;
    using LayoutQ = typename BlockMmadQK::LayoutA;
    using ElementK = typename BlockMmadQK::ElementB;
    using LayoutK = typename BlockMmadQK::LayoutB;
    using ElementS = typename BlockMmadQK::ElementC;
    using LayoutS = typename BlockMmadQK::LayoutC;

    using ElementP = typename BlockMmadPV::ElementA;
    using LayoutP = typename BlockMmadPV::LayoutA;

    using ElementV = typename BlockMmadPV::ElementB;
    using LayoutV = typename BlockMmadPV::LayoutB;

    using ElementMask = typename EpilogueOnlineSoftmax::ElementMask;

    using ElementO = typename EpilogueRescaleO::ElementOutput;
    using LayoutO = typename EpilogueRescaleO::LayoutOutput;

    using ElementOTmp = typename EpilogueRescaleO::ElementInput;
    using LayoutOTmp = typename EpilogueRescaleO::LayoutInput;

    using ElementLse = typename EpilogueRescaleO::ElementLse;
    using LayoutLse = typename EpilogueRescaleO::LayoutLse;

    using ElementUpdate = typename EpilogueRescaleO::ElementUpdate;
    using LayoutUpdate = typename EpilogueRescaleO::LayoutUpdate;

    static constexpr Epilogue::LseMode LSE_MODE = EpilogueRescaleO::LSE_MODE;

    // Methods
    __aicore__ inline RainFusionAttentionKernel() {}

    __aicore__ inline void operator()(RainFusionAttentionKernelParams const &params)
    {
        __gm__ RainFusionAttentionTilingData *rainFusionAttentionTilingData =
            reinterpret_cast<__gm__ RainFusionAttentionTilingData *>(params.tiling);
        uint64_t mm1OutSize = rainFusionAttentionTilingData->mm1OutSize;
        uint64_t smOnlineOutSize = rainFusionAttentionTilingData->smOnlineOutSize;
        uint64_t mm2OutSize = rainFusionAttentionTilingData->mm2OutSize;
        uint32_t batch = rainFusionAttentionTilingData->batch;
        uint32_t qHeads = rainFusionAttentionTilingData->numHeads;
        uint32_t kvHeads = rainFusionAttentionTilingData->kvHeads;
        uint32_t embed = rainFusionAttentionTilingData->embeddingSize;
        uint32_t pagedBlockSize = rainFusionAttentionTilingData->blockSize;
        uint32_t maxNumBlocksPerBatch = rainFusionAttentionTilingData->maxNumBlocksPerBatch;
        uint32_t firstBatchTaskNum = rainFusionAttentionTilingData->firstBatchTaskNum;
        uint32_t totalTaskNum = rainFusionAttentionTilingData->totalTaskNum;
        uint32_t maskType = rainFusionAttentionTilingData->maskType;
        ElementS scaleValue = static_cast<ElementS>(rainFusionAttentionTilingData->scaleValue);
        uint32_t totalQBlocks = rainFusionAttentionTilingData->totalQBlocks;
        uint32_t maxKvBlockNum = rainFusionAttentionTilingData->maxKvBlockNum;
        uint32_t qBlockX = rainFusionAttentionTilingData->blockShapeX;
        uint32_t qBlockY = rainFusionAttentionTilingData->blockShapeY;
        uint32_t qBlockNum = totalQBlocks / qBlockX;
        uint32_t qBlockInX = (qBlockX + BASIC_BLOCK_SIZE - 1) / BASIC_BLOCK_SIZE; // CeilDiv
        uint32_t firstQBlockNum = rainFusionAttentionTilingData->firstQBlockNum;
        uint32_t maxQSeqlen = rainFusionAttentionTilingData->maxQSeqlen;
        uint32_t maxKvSeqlen = rainFusionAttentionTilingData->maxKvSeqlen;
        uint32_t useUniformQSeqlen = rainFusionAttentionTilingData->useUniformQSeqlen; // 是否使用统一的qseqlen值
        uint32_t useUniformKvSeqlen = rainFusionAttentionTilingData->useUniformKvSeqlen; // 是否使用统一的kvseqlen值

        // Initialize global tensors
        AscendC::GlobalTensor<ElementQ> gQ;
        gQ.SetGlobalBuffer((__gm__ ElementQ *)params.q);
        AscendC::GlobalTensor<ElementK> gK;
        gK.SetGlobalBuffer((__gm__ ElementK *)params.k);
        AscendC::GlobalTensor<ElementK> gV;
        gV.SetGlobalBuffer((__gm__ ElementK *)params.v);
        AscendC::GlobalTensor<int32_t> gBlockTable;
        gBlockTable.SetGlobalBuffer((__gm__ int32_t *)(params.blockTables));
        AscendC::GlobalTensor<int64_t> gActualQseqlen;
        gActualQseqlen.SetGlobalBuffer((__gm__ int64_t *)params.actualQseqlen);
        AscendC::GlobalTensor<int64_t> gActualKvseqlen;
        gActualKvseqlen.SetGlobalBuffer((__gm__ int64_t *)params.actualKvseqlen);
        AscendC::GlobalTensor<int64_t> gSelectIdx;
        gSelectIdx.SetGlobalBuffer((__gm__ int64_t *)params.selectIdx);
        AscendC::GlobalTensor<int64_t> gSelectNumIdx;
        gSelectNumIdx.SetGlobalBuffer((__gm__ int64_t *)params.selectNumIdx);
        AscendC::GlobalTensor<ElementO> gO;
        gO.SetGlobalBuffer((__gm__ ElementO *)params.o);
        AscendC::GlobalTensor<ElementLse> gLse;
        gLse.SetGlobalBuffer((__gm__ ElementLse *)params.lse);
        AscendC::GlobalTensor<ElementS> gS;
        gS.SetGlobalBuffer((__gm__ ElementS *)params.workspace);
        AscendC::GlobalTensor<ElementP> gP;
        gP.SetGlobalBuffer((__gm__ ElementP *)(params.workspace + mm1OutSize));
        AscendC::GlobalTensor<ElementOTmp> gOTmp;
        gOTmp.SetGlobalBuffer((__gm__ ElementOTmp *)(params.workspace + mm1OutSize + smOnlineOutSize));
        AscendC::GlobalTensor<ElementOTmp> gOUpdate;
        gOUpdate.SetGlobalBuffer((__gm__ ElementOTmp *)(params.workspace + mm1OutSize + smOnlineOutSize + mm2OutSize));

        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t coreNum = AscendC::GetBlockNum();

#ifdef __DAV_C220_CUBE__
        // Initialize hardware events for cube core
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

        static constexpr uint32_t L1_QK_SIZE =
            BlockMmadQK::L1TileShape::M * BlockMmadQK::L1TileShape::K * sizeof(ElementQ) +
            BlockMmadQK::L1TileShape::N * BlockMmadQK::L1TileShape::K * sizeof(ElementK) * 2;
        BlockMmadQK blockMmadQK(resource);
        BlockMmadPV blockMmadPV(resource, L1_QK_SIZE);
#endif

#ifdef __DAV_C220_VEC__
        // Initialize hardware events for vector core
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
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

        EpilogueOnlineSoftmax epilogueOnlineSoftmax(resource, scaleValue);
        EpilogueRescaleO epilogueRescaleO(resource);

        coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
#endif

        // Calculate strides based on layout (compile-time optimization)
        // For TND: [T, N, D], stride = N * D
        // For BNSD: [B, N, S, D], strideB = N * S * D, strideN = S * D, strideS = D
        uint64_t strideQO = 0;
        uint64_t strideKV = 0;
        uint64_t strideQOB = 0; // BNSD batch stride for Q
        uint64_t strideQON = 0; // BNSD head stride for Q
        uint64_t strideQOS = 0; // BNSD seq stride for Q
        uint64_t strideKVB = 0; // BNSD batch stride for KV
        uint64_t strideKVN = 0; // BNSD head stride for KV
        uint64_t strideKVS = 0; // BNSD seq stride for KV

        if constexpr (QUERY_LAYOUT == 1) { // BNSD_Q
            // BNSD: [B, N, S, D]
            // strideB = N * S * D, strideN = S * D, strideS = D
            // maxQSeqlen is the third dimension (S) of query shape, set in tiling
            strideQOB = static_cast<uint64_t>(qHeads) * maxQSeqlen * embed; // batch stride
            strideQON = static_cast<uint64_t>(maxQSeqlen) * embed;          // head stride
            strideQOS = embed;                                              // seq stride
        } else {
            // TND: [T, N, D]
            strideQO = static_cast<uint64_t>(qHeads) * embed;
        }

        if constexpr (KV_CACHE_LAYOUT == 1) { // BNSD
            // BNSD: [B, N, S, D]
            // maxKvSeqlen is the third dimension (S) of value shape, set in tiling
            strideKVB = static_cast<uint64_t>(kvHeads) * maxKvSeqlen * embed; // batch stride
            strideKVN = static_cast<uint64_t>(maxKvSeqlen) * embed;           // head stride
            strideKVS = embed;                                                // seq stride
        } else {
            // TND: [T, N, D]
            strideKV = static_cast<uint64_t>(kvHeads) * embed;
        }

        uint32_t embedRound = AlignUp<uint32_t>(embed, BLOCK_SIZE);
        uint32_t groupSize = qHeads / kvHeads;

        uint64_t qBOffset = 0;
        uint64_t kBOffset = 0;
        uint64_t vBOffset = 0;
        uint64_t oBOffset = 0;
        uint64_t blockBOffset = 0;

        uint32_t preTotalTaskNum = 0;
        uint32_t preTotalQBlockNum = 0;
        uint32_t curBatch = 0;
        // 根据useUniformQSeqlen标志位决定使用actualSeqLengths数组还是maxQSeqlen
        uint32_t qSeqlen = useUniformQSeqlen ?
                               maxQSeqlen :
                               static_cast<uint32_t>(static_cast<int64_t>(gActualQseqlen.GetValue(curBatch)));
        // 根据useUniformKvSeqlen标志位决定使用actualSeqLengthsKv数组还是maxKvSeqlen
        uint32_t kvSeqlen = useUniformKvSeqlen ?
                                maxKvSeqlen :
                                static_cast<uint32_t>(static_cast<int64_t>(gActualKvseqlen.GetValue(curBatch)));
        uint32_t curQNBlockTile = GetQNBlockTile(qSeqlen, groupSize);
        uint32_t qNBlockNumPerGroup =
            curQNBlockTile == 0 ? 1 : (groupSize + curQNBlockTile - 1) / curQNBlockTile; // CeilDiv
        uint32_t curQNBlockNum = qNBlockNumPerGroup * kvHeads;
        uint32_t curQSBlockTile = GetQSBlockTile(kvSeqlen);
        uint32_t curQSBlockNum = GetQBlocks(qSeqlen, qBlockX);
        uint32_t curTotalTaskNum = firstBatchTaskNum;
        uint32_t curQXBlockNum = (qSeqlen + qBlockX - 1) / qBlockX; // CeilDiv
        uint32_t curTotalQBlockNum = firstQBlockNum;

        // Go through each task
        for (uint32_t taskIdx = coreIdx; taskIdx < totalTaskNum; taskIdx += uint32_t(coreNum)) {
            // Get the offset of each core on the GM
            while (taskIdx >= curTotalTaskNum) {
                ++curBatch;
                preTotalTaskNum = curTotalTaskNum;
                preTotalQBlockNum = curTotalQBlockNum;

                // Update offsets based on layout (compile-time optimization)
                if constexpr (QUERY_LAYOUT == 1) { // BNSD_Q
                    // BNSD: [B, N, S, D], offset = batch * strideB
                    qBOffset = static_cast<uint64_t>(curBatch) * strideQOB;
                    oBOffset = static_cast<uint64_t>(curBatch) * strideQOB;
                } else {
                    // TND
                    qBOffset += static_cast<uint64_t>(qSeqlen) * strideQO;
                    oBOffset += static_cast<uint64_t>(qSeqlen) * strideQO;
                }

                if constexpr (!PAGED_CACHE_FLAG) {
                    if constexpr (KV_CACHE_LAYOUT == 1) { // BNSD
                        // BNSD: [B, N, S, D], offset = batch * strideB
                        kBOffset = static_cast<uint64_t>(curBatch) * strideKVB;
                        vBOffset = static_cast<uint64_t>(curBatch) * strideKVB;
                    } else {
                        // TND
                        kBOffset += static_cast<uint64_t>(kvSeqlen) * strideKV;
                        vBOffset += static_cast<uint64_t>(kvSeqlen) * strideKV;
                    }
                } else {
                    blockBOffset += maxNumBlocksPerBatch;
                }

                // 根据useUniformQSeqlen标志位决定使用actualSeqLengths数组还是maxQSeqlen
                qSeqlen = useUniformQSeqlen ?
                              maxQSeqlen :
                              static_cast<uint32_t>(static_cast<int64_t>(gActualQseqlen.GetValue(curBatch)));
                // 根据useUniformKvSeqlen标志位决定使用actualSeqLengthsKv数组还是maxKvSeqlen
                kvSeqlen = useUniformKvSeqlen ?
                               maxKvSeqlen :
                               static_cast<uint32_t>(static_cast<int64_t>(gActualKvseqlen.GetValue(curBatch)));
                curQNBlockTile = GetQNBlockTile(qSeqlen, groupSize);
                qNBlockNumPerGroup = curQNBlockTile == 0 ? 1 : (groupSize + curQNBlockTile - 1) / curQNBlockTile;
                curQNBlockNum = qNBlockNumPerGroup * kvHeads;
                curQSBlockTile = GetQSBlockTile(kvSeqlen);
                curQSBlockNum = GetQBlocks(qSeqlen, qBlockX);
                curTotalTaskNum += curQNBlockNum * curQSBlockNum;
                curQXBlockNum = (qSeqlen + qBlockX - 1) / qBlockX;
                curTotalQBlockNum += qHeads * curQXBlockNum;
            }

            // Q task splitting按照[qNBlockNum, qHead]
            uint32_t taskIdxCurBatch = taskIdx - preTotalTaskNum;
            uint32_t qSBlockIdx = taskIdxCurBatch / curQNBlockNum;
            uint32_t qXIdx = qSBlockIdx / qBlockInX;
            uint32_t qXInnerIdx = qSBlockIdx - qXIdx * qBlockInX;
            uint32_t qNBlockIdx = taskIdxCurBatch - qSBlockIdx * curQNBlockNum;
            uint32_t qNBlockIdxCurGroup = qNBlockIdx % qNBlockNumPerGroup;
            uint32_t xBlockNum = qSeqlen / qBlockX;
            uint32_t xTailNum = qSeqlen - xBlockNum * qBlockX;

            uint32_t kvHeadIdx = qNBlockIdx / qNBlockNumPerGroup;
            uint32_t qHeadIdx = kvHeadIdx * groupSize + qNBlockIdxCurGroup * curQNBlockTile;
            uint32_t curSelectIdx = preTotalQBlockNum + qXIdx * qHeads + qHeadIdx;
            uint32_t curSelectNum = static_cast<uint32_t>(gSelectNumIdx.GetValue(curSelectIdx));
            // In this case, the output should be zero, so we skip this task
            if (curSelectNum == 0) {
                continue;
            }

            uint32_t lastSelectIdx =
                static_cast<int32_t>(gSelectIdx.GetValue(curSelectIdx * maxKvBlockNum + curSelectNum - 1));
            uint32_t kvYBlockNum = (kvSeqlen + qBlockY - 1) / qBlockY; // CeilDiv
            uint32_t curKvSeqLen = (lastSelectIdx == kvYBlockNum - 1 && kvSeqlen % qBlockY != 0) ?
                                       qBlockY * (curSelectNum - 1) + kvSeqlen % qBlockY :
                                       qBlockY * curSelectNum;

            // Calculate offsets based on layout (compile-time optimization)
            uint64_t gmOffsetQ = 0;
            uint64_t gmOffsetK = 0;
            uint64_t gmOffsetV = 0;
            uint64_t gmOffsetO = 0;

            if constexpr (QUERY_LAYOUT == 1) { // BNSD_Q: [B, N, S, D]
                // offset = batch * strideB + head * strideN + seq * strideS
                uint32_t qSeqOffset = qXIdx * qBlockX + qXInnerIdx * BASIC_BLOCK_SIZE;
                gmOffsetQ = qBOffset + qHeadIdx * strideQON + qSeqOffset * strideQOS;
                gmOffsetO = oBOffset + qHeadIdx * strideQON + qSeqOffset * strideQOS;
            } else {
                // TND: [T, N, D]
                uint32_t qSeqOffset = qXIdx * qBlockX + qXInnerIdx * BASIC_BLOCK_SIZE;
                gmOffsetQ = qBOffset + qSeqOffset * strideQO + qHeadIdx * embed;
                gmOffsetO = oBOffset + qSeqOffset * strideQO + qHeadIdx * embed;
            }

            if constexpr (KV_CACHE_LAYOUT == 1) { // BNSD: [B, N, S, D]
                // offset = batch * strideB + head * strideN
                // seq offset will be handled in blockMmadQK/blockMmadPV based on selectIdx
                gmOffsetK = kBOffset + kvHeadIdx * strideKVN;
                gmOffsetV = vBOffset + kvHeadIdx * strideKVN;
            } else {
                // TND: [T, N, D]
                gmOffsetK = kBOffset + kvHeadIdx * embed;
                gmOffsetV = vBOffset + kvHeadIdx * embed;
            }

            uint32_t qSBlockSize =
                (qXIdx == xBlockNum) ?
                    (qXInnerIdx == xTailNum / curQSBlockTile ? xTailNum - qXInnerIdx * curQSBlockTile :
                                                               curQSBlockTile) :
                    ((qXInnerIdx == qBlockInX - 1) ? qBlockX - qXInnerIdx * curQSBlockTile : curQSBlockTile);

            uint32_t qNBlockSize = (qNBlockIdxCurGroup == (qNBlockNumPerGroup - 1)) ?
                                       (groupSize - qNBlockIdxCurGroup * curQNBlockTile) :
                                       curQNBlockTile;
            uint32_t rowNum = qSBlockSize * qNBlockSize;
            uint32_t rowNumRound = AlignUp<uint32_t>(rowNum, BLOCK_SIZE);

            uint32_t noSkipKvS = curKvSeqLen;
            uint32_t kvSLoopNumTotal = (noSkipKvS + pagedBlockSize - 1) / pagedBlockSize; // CeilDiv

            uint32_t blockStackNum = MAX_KV_STACK_LEN / pagedBlockSize;
            uint32_t stackSeqTile;
            uint32_t stackSeqTilePad = blockStackNum * pagedBlockSize;
            uint32_t preKVNum = PRE_LAUNCH * blockStackNum;
            int32_t stackSeqCount = 0;

#ifdef __DAV_C220_CUBE__
            LayoutQ layoutQTemp(rowNum, embed);
            // For BNSD format, use strideKVS; for TND, use strideKV (compile-time)
            uint64_t actualStrideKV = 0;
            if constexpr (KV_CACHE_LAYOUT == 1) {
                actualStrideKV = strideKVS;
            } else {
                actualStrideKV = strideKV;
            }
            LayoutK layoutKTemp(actualStrideKV, blockStackNum * pagedBlockSize);
            LayoutV layoutVTemp(blockStackNum * pagedBlockSize, actualStrideKV);
            // Pass correct Q stride based on data format
            uint64_t qGmStride = 0;
            if constexpr (QUERY_LAYOUT == 1) { // BNSD: [B, N, S, D]
                qGmStride = strideQOS;         // embed
            } else {                           // TND: [T, N, D]
                qGmStride = strideQO;          // qHeads * embed
            }
            blockMmadQK.loadQGM(gQ[gmOffsetQ], layoutQTemp, rowNum, qNBlockSize, qGmStride);
#endif
            // Main computation loop: QK matmul -> Softmax -> PV matmul
            for (uint32_t kvSIdx = 0; kvSIdx < kvSLoopNumTotal + preKVNum; kvSIdx += blockStackNum) {
                // Stage 1: QK matmul (computed on CUBE core)
                if (kvSIdx < kvSLoopNumTotal) {
                    stackSeqTile = noSkipKvS - kvSIdx * pagedBlockSize;
                    if (stackSeqTile >= pagedBlockSize * blockStackNum) {
                        stackSeqTile = pagedBlockSize * blockStackNum;
                    }
                    uint32_t curStackTileMod = stackSeqCount % (PRE_LAUNCH + 1);
                    uint64_t gmOffsetS = coreIdx * WORKSPACE_BLOCK_SIZE_DB * (PRE_LAUNCH + 1) +
                                         curStackTileMod * WORKSPACE_BLOCK_SIZE_DB;
                    GemmCoord actualBlockShapeQK{rowNum, stackSeqTile, embed};
                    LayoutS layOutS(rowNum, stackSeqTile, stackSeqTilePad);
#ifdef __DAV_C220_CUBE__
                    // For BNSD format, pass strideKVS; for TND, pass strideKV (compile-time)
                    uint64_t actualStrideKVForQK = 0;
                    if constexpr (KV_CACHE_LAYOUT == 1) {
                        actualStrideKVForQK = strideKVS;
                    } else {
                        actualStrideKVForQK = strideKV;
                    }
                    blockMmadQK(gQ[gmOffsetQ], gK[gmOffsetK], gS[gmOffsetS], gBlockTable[blockBOffset],
                                gSelectIdx[curSelectIdx * maxKvBlockNum], layoutQTemp, layoutKTemp, layOutS,
                                actualBlockShapeQK, kvSIdx, kvSLoopNumTotal, pagedBlockSize, actualStrideKVForQK,
                                qBlockY, curSelectNum, kvYBlockNum, kvSeqlen);
                    NpuArch::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(qkReady);
#endif
#ifdef __DAV_C220_VEC__
                    // Stage 2: Online softmax (computed on VECTOR core)
                    LayoutP layOutP(rowNum, stackSeqTile, stackSeqTilePad);
                    uint64_t gmOffsetP = gmOffsetS;

                    NpuArch::Arch::CrossCoreWaitFlag(qkReady);
                    // online softmax
                    epilogueOnlineSoftmax(gP[gmOffsetP], gS[gmOffsetS], layOutP, layOutS, actualBlockShapeQK,
                                          (stackSeqCount == 0), 0, qSBlockSize, qNBlockSize, curStackTileMod);
                    NpuArch::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(softmaxReady);
#endif
                }
                // Stage 3: PV matmul and output rescaling
                if (kvSIdx >= preKVNum) {
                    uint32_t nowkvSIdx = kvSIdx - preKVNum;
                    stackSeqTile = noSkipKvS - nowkvSIdx * pagedBlockSize;
                    if (stackSeqTile >= pagedBlockSize * blockStackNum) {
                        stackSeqTile = pagedBlockSize * blockStackNum;
                    }
                    uint32_t curStackTileMod = (stackSeqCount - PRE_LAUNCH) % (PRE_LAUNCH + 1);
                    uint64_t gmOffsetOTmp = coreIdx * WORKSPACE_BLOCK_SIZE_DB * (PRE_LAUNCH + 1) +
                                            curStackTileMod * WORKSPACE_BLOCK_SIZE_DB;
                    GemmCoord actualBlockShapePV{rowNum, embed, stackSeqTile};
                    LayoutOTmp layoutOTmp(rowNum, embed, embedRound);
#ifdef __DAV_C220_CUBE__
                    LayoutP layoutPTemp(rowNum, stackSeqTile, stackSeqTilePad);
                    uint64_t gmOffsetP = coreIdx * WORKSPACE_BLOCK_SIZE_DB * (PRE_LAUNCH + 1) +
                                         curStackTileMod * WORKSPACE_BLOCK_SIZE_DB;
                    // For BNSD format, pass strideKVS; for TND, pass strideKV (compile-time)
                    uint64_t actualStrideKVForPV = 0;
                    if constexpr (KV_CACHE_LAYOUT == 1) {
                        actualStrideKVForPV = strideKVS;
                    } else {
                        actualStrideKVForPV = strideKV;
                    }
                    blockMmadPV(gP[gmOffsetP], gV[gmOffsetV], gOTmp[gmOffsetOTmp], gBlockTable[blockBOffset],
                                gSelectIdx[curSelectIdx * maxKvBlockNum], layoutPTemp, layoutVTemp, layoutOTmp,
                                actualBlockShapePV, nowkvSIdx, kvSLoopNumTotal, pagedBlockSize, kvSeqlen,
                                actualStrideKVForPV, blockStackNum, softmaxReady, qBlockY, curSelectNum, kvYBlockNum);
                    NpuArch::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(pvReady);
#endif
#ifdef __DAV_C220_VEC__
                    // Setup layoutO based on data format
                    LayoutO layoutO;
                    if constexpr (QUERY_LAYOUT == 1) { // BNSD: [B, N, S, D]
                        // BNSD format: stride[0] = embed (strideQOS)
                        layoutO = LayoutO(qSeqlen, embed);
                    } else { // TND: [T, N, D]
                        // TND format: stride[0] = qHeads * embed (strideQO)
                        layoutO = LayoutO(qSeqlen, qHeads * embed);
                    }
                    LayoutUpdate layoutUpdate(rowNum, embed, embedRound);
                    uint64_t gmOffsetUpdate = (uint64_t)(coreIdx * WORKSPACE_BLOCK_SIZE_DB);

                    NpuArch::Arch::CrossCoreWaitFlag(pvReady);
                    // rescale O
                    epilogueRescaleO(gO[gmOffsetO], gOTmp[gmOffsetOTmp], gOUpdate[gmOffsetUpdate], layoutO, layoutOTmp,
                                     layoutUpdate, actualBlockShapePV, qSBlockSize, qNBlockSize,
                                     (stackSeqCount - PRE_LAUNCH == 0), nowkvSIdx + blockStackNum >= kvSLoopNumTotal,
                                     curStackTileMod);
#endif
                }
                stackSeqCount++;
            }
        }
#ifdef __DAV_C220_CUBE__
        // Wait for all CUBE core events
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
        // Wait for all VECTOR core events
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID5);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID6);

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
#endif
        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    NpuArch::Arch::Resource<ArchTag> resource;
    NpuArch::Arch::CrossCoreFlag qkReady{QK_READY_ID};
    NpuArch::Arch::CrossCoreFlag softmaxReady{SOFTMAX_READY_ID};
    NpuArch::Arch::CrossCoreFlag pvReady{PV_READY_ID};
};

} // namespace RainFusion

#endif // RAIN_FUSION_ATTENTION_KERNEL_H
