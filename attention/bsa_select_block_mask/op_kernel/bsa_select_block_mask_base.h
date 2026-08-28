/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file bsa_select_block_mask_base.h
 * \brief
 */
#ifndef BSA_SELECT_BLOCK_MASK_BASE_H
#define BSA_SELECT_BLOCK_MASK_BASE_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "bsa_select_block_mask_common.h"
#include "bsa_select_block_mask_tiling_data.h"
#include "bsa_vector_service.h"
#include "bsa_matmul_service.h"
#include "bsa_radix_topk_service.h"

using namespace matmul;
using AscendC::CacheMode;
using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;

template <typename BSAT>
class BSASelectBlockMaskBase {
public:
    using T = float;
    using IN_T = typename BSAT::inputT;
    using OUT_T = typename BSAT::outputT;
    using MM_OUT_T = T;
    using POOL_OUT_T = half;
    using SFTMAX_OUT_T = half;
    static constexpr BSALayout LAYOUT_Q = BSAT::layoutQ;
    static constexpr BSALayout LAYOUT_KV = BSAT::layoutKV;

    __aicore__ inline BSASelectBlockMaskBase(){};
    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *blockShape,
                                __gm__ uint8_t *postBlockShape, __gm__ uint8_t *actualSeqLensQ,
                                __gm__ uint8_t *actualSeqLensKV, __gm__ uint8_t *actualBlockLenQ,
                                __gm__ uint8_t *actualBlockLenKV, __gm__ uint8_t *maskOut, __gm__ uint8_t *workspace,
                                const optiling::BSASelectBlockMaskTilingData *__restrict tiling, TPipe *tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitConstInfo();
    __aicore__ inline void InitWorkspace(__gm__ uint8_t *workspace);
    __aicore__ inline void CalcMultiCoreOffset(uint32_t &startRow, uint32_t &endRow);
    __aicore__ inline void CalcKPoolingRange(uint32_t &startKBlock, uint32_t &endKBlock);
    // D2: dynamic partition by validYBlocks
    __aicore__ inline void CalcKPoolingRangeValid(uint32_t &startKBlock, uint32_t &endKBlock, uint32_t validYBlocks);
    // D2: dynamic partition by validXBlocks
    __aicore__ inline void CalcMultiCoreOffsetValid(uint32_t &startRow, uint32_t &endRow, uint32_t validXBlocks);
    __aicore__ inline void ProcessPoolingK(uint32_t batchIdx, uint32_t headIdx, uint64_t seqPrefixSumKV,
                                           uint64_t blockPrefixSumKV, uint32_t validYBlocks);
    __aicore__ inline void ProcessPoolingQ(uint32_t batchIdx, uint32_t headIdx, uint32_t tokenStart, uint32_t tokenEnd,
                                           uint64_t seqPrefixSumQ, uint64_t blockPrefixSumQ);
    __aicore__ inline void ProcessMatmulSoftmax(uint32_t batchIdx, uint32_t headIdx, uint32_t qChunkStart,
                                                uint32_t curQChunkSize, uint32_t validYBlocks);
    __aicore__ inline void ProcessSoftmaxSecondPass(uint32_t qChunkStart, uint32_t curQChunkSize, uint32_t batchIdx,
                                                    uint32_t headIdx, uint32_t validYBlocks);
    __aicore__ inline void ProcessPostSoftmaxPooling(uint32_t batchIdx, uint32_t headIdx, uint32_t validXBlocks,
                                                     uint32_t validYBlocks);
    // 标量兜底路径（postBlockShape 超出向量化路径支持上限时使用，现实规格不会命中）
    __aicore__ inline void ProcessPostSoftmaxPoolingScalar(uint32_t batchIdx, uint32_t headIdx, uint32_t validXBlocks,
                                                           uint32_t validYBlocks);

    TPipe *pipe = nullptr;
    const optiling::BSASelectBlockMaskTilingData *__restrict tilingData = nullptr;
    BSAConstInfo constInfo;

    BSAMatmulService<BSAT> matmulService;
    BSAVectorService<BSAT> vectorService;
    BSARadixTopKService<BSAT> radixTopKService;

    GlobalTensor<IN_T> queryGm, keyGm;
    GlobalTensor<int64_t> blockShapeGm;
    GlobalTensor<int64_t> actualSeqLensQGm, actualSeqLensKVGm;
    GlobalTensor<int64_t> actualBlockLenQGm, actualBlockLenKVGm;
    GlobalTensor<uint8_t> maskOutGmU8;

    GlobalTensor<POOL_OUT_T> qCmpGm, kCmpGm;
    GlobalTensor<SFTMAX_OUT_T> attnScoreFp16Gm;
    GlobalTensor<SFTMAX_OUT_T> pooledScoreGm;
    GlobalTensor<T> scoreFp32Gm;
    GlobalTensor<int32_t> topkWorkspaceGm;
};

template <typename BSAT>
__aicore__ inline void BSASelectBlockMaskBase<BSAT>::InitConstInfo()
{
    if ASCEND_IS_AIV {
        constInfo.aivIdx = GetBlockIdx();
        constInfo.aicIdx = constInfo.aivIdx / 2;
        constInfo.subBlockIdx = constInfo.aivIdx % 2;
    } else {
        constInfo.aicIdx = GetBlockIdx();
    }
    constInfo.aivNum = GetBlockNum() * AIC_AIV_RATIO;

    auto &baseInfo = tilingData->baseParams;
    constInfo.batchSize = baseInfo.batchSize;
    constInfo.numHeads = baseInfo.numHeads;
    constInfo.maxQSeqlen = baseInfo.maxQSeqlen;
    constInfo.maxKvSeqlen = baseInfo.maxKvSeqlen;
    constInfo.dSize = baseInfo.dSize;
    constInfo.blockShapeX = baseInfo.blockShapeX;
    constInfo.blockShapeY = baseInfo.blockShapeY;
    constInfo.xBlocks = baseInfo.xBlocks;
    constInfo.yBlocks = baseInfo.yBlocks;
    constInfo.scaleValue = baseInfo.scaleValue;
    constInfo.sparsity = baseInfo.sparsity;
    constInfo.topKValue = baseInfo.topKValue;
    constInfo.sparsityMode = static_cast<BSASparseMode>(baseInfo.sparsityMode);

    auto &mcInfo = tilingData->multiCoreParams;
    constInfo.coreNum = mcInfo.coreNum;
    constInfo.activeCoreNum = mcInfo.activeCoreNum;
    constInfo.rowsPerCore = mcInfo.rowsPerCore;
    constInfo.extraCores = mcInfo.extraCores;
    constInfo.totalRows = mcInfo.totalRows;
    constInfo.yBlocksPerCore = mcInfo.yBlocksPerCore;
    constInfo.extraYCores = mcInfo.extraYCores;
    constInfo.qChunkSize = baseInfo.qChunkSize;
    constInfo.kChunkSize = baseInfo.kChunkSize;
    constInfo.activeYVecCoreNum = mcInfo.activeYVecCoreNum;

    auto &outInfo = tilingData->outputParams;
    constInfo.qCmpOffset = 0;
    constInfo.kCmpOffset = constInfo.qCmpOffset + outInfo.qCmpSize;
    constInfo.attnScoreOffset = constInfo.kCmpOffset + outInfo.kCmpSize;
    constInfo.pooledScoreOffset = constInfo.attnScoreOffset + outInfo.attnScoreSize;
    constInfo.softmaxTmpOffset = constInfo.pooledScoreOffset + outInfo.pooledScoreSize;
    constInfo.topkWorkspaceOffset = constInfo.softmaxTmpOffset + outInfo.softmaxTmpSize;

    constInfo.usePostBlockShape = baseInfo.usePostBlockShape;
    constInfo.postBlockShapeX = baseInfo.postBlockShapeX;
    constInfo.postBlockShapeY = baseInfo.postBlockShapeY;
    constInfo.postXBlocks = baseInfo.postXBlocks;
    constInfo.postYBlocks = baseInfo.postYBlocks;
}

template <typename BSAT>
__aicore__ inline void BSASelectBlockMaskBase<BSAT>::InitWorkspace(__gm__ uint8_t *workspace)
{
    auto &outInfo = tilingData->outputParams;

    qCmpGm.SetGlobalBuffer((__gm__ POOL_OUT_T *)(workspace + constInfo.qCmpOffset),
                           outInfo.qCmpSize / sizeof(POOL_OUT_T));
    kCmpGm.SetGlobalBuffer((__gm__ POOL_OUT_T *)(workspace + constInfo.kCmpOffset),
                           outInfo.kCmpSize / sizeof(POOL_OUT_T));
    attnScoreFp16Gm.SetGlobalBuffer((__gm__ SFTMAX_OUT_T *)(workspace + constInfo.attnScoreOffset),
                                    outInfo.attnScoreSize / sizeof(SFTMAX_OUT_T));
    if (constInfo.usePostBlockShape) {
        pooledScoreGm.SetGlobalBuffer((__gm__ SFTMAX_OUT_T *)(workspace + constInfo.pooledScoreOffset),
                                      outInfo.pooledScoreSize / sizeof(SFTMAX_OUT_T));
    }
    scoreFp32Gm.SetGlobalBuffer((__gm__ T *)(workspace + constInfo.softmaxTmpOffset),
                                outInfo.softmaxTmpSize / sizeof(T));
    topkWorkspaceGm.SetGlobalBuffer((__gm__ int32_t *)(workspace + constInfo.topkWorkspaceOffset),
                                    outInfo.topkWorkspaceSize / sizeof(int32_t));

    SyncAll();
}

template <typename BSAT>
__aicore__ inline void BSASelectBlockMaskBase<BSAT>::CalcMultiCoreOffset(uint32_t &startRow, uint32_t &endRow)
{
    uint32_t coreIdx = constInfo.aicIdx;
    uint32_t extraCores = constInfo.extraCores;
    uint32_t rowsPerCore = constInfo.rowsPerCore; // blocks

    if (coreIdx < extraCores || extraCores == 0) {
        startRow = coreIdx * rowsPerCore;
        endRow = startRow + rowsPerCore;
    } else {
        uint32_t baseRows = rowsPerCore - 1;
        startRow = extraCores * rowsPerCore + (coreIdx - extraCores) * baseRows;
        endRow = startRow + baseRows;
    }
    if (startRow >= constInfo.totalRows) {
        startRow = constInfo.totalRows;
        endRow = constInfo.totalRows;
    }
}

template <typename BSAT>
__aicore__ inline void BSASelectBlockMaskBase<BSAT>::CalcKPoolingRange(uint32_t &startKBlock, uint32_t &endKBlock)
{
    uint32_t coreIdx = constInfo.aivIdx; // vec idx
    uint32_t extraYCores = constInfo.extraYCores;
    uint32_t yBlocksPerCore = constInfo.yBlocksPerCore;

    if (coreIdx < extraYCores || extraYCores == 0) {
        startKBlock = coreIdx * yBlocksPerCore;
        endKBlock = startKBlock + yBlocksPerCore;
    } else {
        uint32_t baseYRows = yBlocksPerCore - 1;
        startKBlock = extraYCores * yBlocksPerCore + (coreIdx - extraYCores) * baseYRows;
        endKBlock = startKBlock + baseYRows;
    }
    if (startKBlock >= constInfo.yBlocks) {
        startKBlock = constInfo.yBlocks;
        endKBlock = constInfo.yBlocks;
    }
}

// D2: dynamic partition by validYBlocks (early-return style)
template <typename BSAT>
__aicore__ inline void BSASelectBlockMaskBase<BSAT>::CalcKPoolingRangeValid(uint32_t &startKBlock, uint32_t &endKBlock,
                                                                            uint32_t validYBlocks)
{
    if (validYBlocks == 0) {
        startKBlock = 0;
        endKBlock = 0;
        return;
    }

    uint32_t coreIdx = constInfo.aivIdx;
    uint32_t activeYVecCoreNum = constInfo.activeYVecCoreNum;
    uint32_t actualActiveCores = BSAMin(validYBlocks, activeYVecCoreNum);

    if (coreIdx >= actualActiveCores) {
        startKBlock = 0;
        endKBlock = 0;
        return;
    }

    uint32_t baseRows = validYBlocks / actualActiveCores;
    uint32_t extraCores = validYBlocks % actualActiveCores;
    uint32_t rowsPerCore = baseRows + (extraCores > 0 ? 1 : 0);

    if (extraCores == 0) {
        startKBlock = coreIdx * rowsPerCore;
        endKBlock = startKBlock + rowsPerCore;
        return;
    }

    if (coreIdx < extraCores) {
        startKBlock = coreIdx * rowsPerCore;
        endKBlock = startKBlock + rowsPerCore;
        return;
    }

    uint32_t baseYRows = rowsPerCore - 1;
    startKBlock = extraCores * rowsPerCore + (coreIdx - extraCores) * baseYRows;
    endKBlock = startKBlock + baseYRows;
}

// D2: dynamic partition by validXBlocks (early-return style)
template <typename BSAT>
__aicore__ inline void BSASelectBlockMaskBase<BSAT>::CalcMultiCoreOffsetValid(uint32_t &startRow, uint32_t &endRow,
                                                                              uint32_t validXBlocks)
{
    if (validXBlocks == 0) {
        startRow = 0;
        endRow = 0;
        return;
    }

    uint32_t coreIdx = constInfo.aicIdx;
    uint32_t activeCoreNum = constInfo.activeCoreNum;
    uint32_t actualActiveCores = BSAMin(validXBlocks, activeCoreNum);

    if (coreIdx >= actualActiveCores) {
        startRow = 0;
        endRow = 0;
        return;
    }

    uint32_t baseRows = validXBlocks / actualActiveCores;
    uint32_t extraCores = validXBlocks % actualActiveCores;
    uint32_t rowsPerCore = baseRows + (extraCores > 0 ? 1 : 0);

    if (extraCores == 0) {
        startRow = coreIdx * rowsPerCore;
        endRow = startRow + rowsPerCore;
        return;
    }

    if (coreIdx < extraCores) {
        startRow = coreIdx * rowsPerCore;
        endRow = startRow + rowsPerCore;
        return;
    }

    uint32_t baseYRows = rowsPerCore - 1;
    startRow = extraCores * rowsPerCore + (coreIdx - extraCores) * baseYRows;
    endRow = startRow + baseYRows;
}

template <typename BSAT>
__aicore__ inline void BSASelectBlockMaskBase<BSAT>::Init(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *blockShape, __gm__ uint8_t *postBlockShape,
    __gm__ uint8_t *actualSeqLensQ, __gm__ uint8_t *actualSeqLensKV, __gm__ uint8_t *actualBlockLenQ,
    __gm__ uint8_t *actualBlockLenKV, __gm__ uint8_t *maskOut, __gm__ uint8_t *workspace,
    const optiling::BSASelectBlockMaskTilingData *__restrict tiling, TPipe *tPipe)
{
    pipe = tPipe;
    tilingData = tiling;
    InitConstInfo();

    queryGm.SetGlobalBuffer((__gm__ IN_T *)query);
    keyGm.SetGlobalBuffer((__gm__ IN_T *)key);
    if (blockShape != nullptr) {
        blockShapeGm.SetGlobalBuffer((__gm__ int64_t *)blockShape, 2);
    }
    if (actualSeqLensQ != nullptr) {
        actualSeqLensQGm.SetGlobalBuffer((__gm__ int64_t *)actualSeqLensQ, constInfo.batchSize);
    } else {
        actualSeqLensQGm.SetGlobalBuffer((__gm__ int64_t *)actualSeqLensQ, 0);
    }
    if (actualSeqLensKV != nullptr) {
        actualSeqLensKVGm.SetGlobalBuffer((__gm__ int64_t *)actualSeqLensKV, constInfo.batchSize);
    } else {
        actualSeqLensKVGm.SetGlobalBuffer((__gm__ int64_t *)actualSeqLensKV, 0);
    }
    if (actualBlockLenQ != nullptr) {
        uint32_t totalQBlocks = constInfo.batchSize * constInfo.xBlocks;
        actualBlockLenQGm.SetGlobalBuffer((__gm__ int64_t *)actualBlockLenQ, totalQBlocks);
    } else {
        actualBlockLenQGm.SetGlobalBuffer((__gm__ int64_t *)actualBlockLenQ, 0);
    }
    if (actualBlockLenKV != nullptr) {
        uint32_t totalKVBlocks = constInfo.batchSize * constInfo.yBlocks;
        actualBlockLenKVGm.SetGlobalBuffer((__gm__ int64_t *)actualBlockLenKV, totalKVBlocks);
    } else {
        actualBlockLenKVGm.SetGlobalBuffer((__gm__ int64_t *)actualBlockLenKV, 0);
    }
    maskOutGmU8.SetGlobalBuffer((__gm__ uint8_t *)maskOut);

    InitWorkspace(workspace);

    if ASCEND_IS_AIV {
        vectorService.InitParams(constInfo, tilingData);
        vectorService.InitBuffers(pipe);
        vectorService.InitGM(qCmpGm, kCmpGm, attnScoreFp16Gm, scoreFp32Gm, queryGm, keyGm, actualBlockLenQGm,
                             actualBlockLenKVGm, actualSeqLensQGm, actualSeqLensKVGm);
        if (constInfo.usePostBlockShape) {
            vectorService.InitPostPoolGM(attnScoreFp16Gm, pooledScoreGm);
        }

        radixTopKService.InitParams(constInfo, tilingData);
        radixTopKService.InitBuffers(pipe);
        // post 启用时 TopK 在粗粒度 pooledScore 上选择并直接输出粗粒度 mask [B,N,postX,postY]
        if (constInfo.usePostBlockShape) {
            radixTopKService.InitGM(pooledScoreGm, topkWorkspaceGm, maskOutGmU8);
        } else {
            radixTopKService.InitGM(attnScoreFp16Gm, topkWorkspaceGm, maskOutGmU8);
        }
    } else if ASCEND_IS_AIC {
        matmulService.InitParams(constInfo);
        matmulService.InitBuffers(pipe);
        matmulService.InitGM(qCmpGm, kCmpGm, scoreFp32Gm);
    }
}

template <typename BSAT>
__aicore__ inline void BSASelectBlockMaskBase<BSAT>::Process()
{
    if ASCEND_IS_AIV {
        vectorService.AllocEventID();
        radixTopKService.AllocEventID();
    } else {
        matmulService.AllocEventID();
    }

    // 1行等于1 block
    // startRow: q_comp 的start行数
    // endRow: q_comp 的end行数
    // D2: startRow/endRow moved to batch loop (per-batch dynamic partition)

    // TND 布局下各 batch 的 token 在 GM 中变长拼接，需用前缀和计算 batch 偏移。
    // 在 batch 循环外初始化为 0，每个 batch 结束后累加该 batch 的实际序列长度，
    // 避免在每次 PoolingSingleQ/KBlock 调用中重复从 GM 读取 actualSeqLens 做 O(B²) 前缀和。
    // BNSD 布局下 seqPrefixSum 不被使用（pool 的 BNSD 分支忽略），保持为 0 不影响行为。
    uint64_t seqPrefixSumQ = 0;
    uint64_t seqPrefixSumKV = 0;
    uint64_t blockPrefixSumQ = 0;
    uint64_t blockPrefixSumKV = 0;

    for (uint32_t batchIdx = 0; batchIdx < constInfo.batchSize; batchIdx++) {
        // D1: compute valid block counts for this batch
        uint32_t curBatchSq, curBatchSkv;
        if (tilingData->baseParams.useActualSeqLenQ) {
            curBatchSq = static_cast<uint32_t>(actualSeqLensQGm.GetValue(batchIdx));
        } else {
            curBatchSq = constInfo.maxQSeqlen;
        }
        if (tilingData->baseParams.useActualSeqLenK) {
            curBatchSkv = static_cast<uint32_t>(actualSeqLensKVGm.GetValue(batchIdx));
        } else {
            curBatchSkv = constInfo.maxKvSeqlen;
        }
        uint32_t validXBlocks = BSACeilDiv(curBatchSq, static_cast<uint32_t>(constInfo.blockShapeX));
        uint32_t validYBlocks = BSACeilDiv(curBatchSkv, static_cast<uint32_t>(constInfo.blockShapeY));

        if ASCEND_IS_AIV {
            radixTopKService.UpdateRuntimeParams(validXBlocks, validYBlocks);
        }

        // D2: dynamic partition by validXBlocks
        uint32_t startRow, endRow;
        CalcMultiCoreOffsetValid(startRow, endRow, validXBlocks);

        for (uint32_t headIdx = 0; headIdx < constInfo.numHeads; headIdx++) {
            ProcessPoolingK(batchIdx, headIdx, seqPrefixSumKV, blockPrefixSumKV, validYBlocks);
            AscendC::PipeBarrier<PIPE_ALL>();
            SyncAll<false>();

            // D2: Q chunk loop with dynamic partition (no truncation needed)
            for (uint32_t qChunkStart = startRow; qChunkStart < endRow; qChunkStart += constInfo.qChunkSize) {
                uint32_t qChunkEnd = BSAMin(qChunkStart + constInfo.qChunkSize, endRow);
                uint32_t curQChunkSize = qChunkEnd - qChunkStart;

                uint32_t tokenStart = qChunkStart * constInfo.blockShapeX;
                uint32_t tokenEnd = BSAMin(qChunkEnd * constInfo.blockShapeX, curBatchSq);
                ProcessPoolingQ(batchIdx, headIdx, tokenStart, tokenEnd, seqPrefixSumQ, blockPrefixSumQ);

                if ASCEND_IS_AIV {
                    CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(SYNC_V1_TO_C1_FLAG[qChunkStart & 1]);
                }
                if ASCEND_IS_AIC {
                    CrossCoreWaitFlag(SYNC_V1_TO_C1_FLAG[qChunkStart & 1]);
                }

                ProcessMatmulSoftmax(batchIdx, headIdx, qChunkStart, curQChunkSize, validYBlocks);

                ProcessSoftmaxSecondPass(qChunkStart, curQChunkSize, batchIdx, headIdx, validYBlocks);
            }

            AscendC::PipeBarrier<PIPE_ALL>();
            SyncAll<false>();

            if (constInfo.usePostBlockShape) {
                ProcessPostSoftmaxPooling(batchIdx, headIdx, validXBlocks, validYBlocks);
                AscendC::PipeBarrier<PIPE_ALL>();
                SyncAll<false>();
            }

            if ASCEND_IS_AIV {
                radixTopKService.ProcessRadixTopKAndWriteMask(batchIdx, headIdx);
            }
            SyncAll();
        }

        // 当前 batch 处理完毕，累加其序列长度为下一个 batch 准备前缀和。
        // 仅 TND 布局需要；BNSD 分支由 constexpr 消除，不会访问 actualSeqLens GM（BNSD 下可能为空）。
        if (batchIdx + 1 < constInfo.batchSize) {
            if constexpr (LAYOUT_Q == BSALayout::TND) {
                seqPrefixSumQ += static_cast<uint64_t>(actualSeqLensQGm.GetValue(batchIdx));
                blockPrefixSumQ += validXBlocks;
            }
            if constexpr (LAYOUT_KV == BSALayout::TND) {
                seqPrefixSumKV += static_cast<uint64_t>(actualSeqLensKVGm.GetValue(batchIdx));
                blockPrefixSumKV += validYBlocks;
            }
        }
    }

    if ASCEND_IS_AIV {
        vectorService.FreeEventID();
        radixTopKService.FreeEventID();
    } else {
        matmulService.FreeEventID();
    }
}

template <typename BSAT>
__aicore__ inline void BSASelectBlockMaskBase<BSAT>::ProcessPoolingK(uint32_t batchIdx, uint32_t headIdx,
                                                                     uint64_t seqPrefixSumKV, uint64_t blockPrefixSumKV,
                                                                     uint32_t validYBlocks)
{
    if ASCEND_IS_AIV {
        if (constInfo.aivIdx >= constInfo.activeYVecCoreNum) {
            return;
        }
        uint32_t startKBlock, endKBlock;
        CalcKPoolingRangeValid(startKBlock, endKBlock, validYBlocks);

        for (uint32_t kBlockIdx = startKBlock; kBlockIdx < endKBlock; kBlockIdx++) {
            vectorService.PoolingSingleKBlock(batchIdx, headIdx, kBlockIdx, keyGm, actualBlockLenKVGm, kCmpGm,
                                              seqPrefixSumKV, blockPrefixSumKV);
        }
    }
}

template <typename BSAT>
__aicore__ inline void BSASelectBlockMaskBase<BSAT>::ProcessPoolingQ(uint32_t batchIdx, uint32_t headIdx,
                                                                     uint32_t tokenStart, uint32_t tokenEnd,
                                                                     uint64_t seqPrefixSumQ, uint64_t blockPrefixSumQ)
{
    if ASCEND_IS_AIV {
        uint32_t coreIdx = constInfo.aivIdx; // vec idx
        uint32_t subVecIdx = constInfo.aivIdx % AIC_AIV_RATIO;
        uint32_t qRows = tokenEnd - tokenStart;
        uint32_t qBlocks = (qRows + constInfo.blockShapeX - 1) / constInfo.blockShapeX; // blockShapeX行为1个block
        uint32_t excuteBlock = 0;
        uint32_t startQBlockIdx = 0;
        uint32_t endQBlockIdx = 0;

        // vec 0 and 1 进行分核
        if (subVecIdx == 0) {
            excuteBlock = qBlocks / AIC_AIV_RATIO + qBlocks % AIC_AIV_RATIO;
            startQBlockIdx = tokenStart / constInfo.blockShapeX;
            endQBlockIdx = startQBlockIdx + excuteBlock;
        } else {
            excuteBlock = qBlocks / AIC_AIV_RATIO;
            if (excuteBlock == 0) {
                // sub vec 1 没有分到数据
                return;
            }
            uint32_t subvec0ExcuteBlocks = qBlocks / AIC_AIV_RATIO + qBlocks % AIC_AIV_RATIO;
            startQBlockIdx = tokenStart / constInfo.blockShapeX + subvec0ExcuteBlocks;
            endQBlockIdx = startQBlockIdx + excuteBlock;
        }

        for (uint32_t qBlockIdx = startQBlockIdx; qBlockIdx < endQBlockIdx; qBlockIdx++) {
            vectorService.PoolingSingleQBlock(batchIdx, headIdx, qBlockIdx, queryGm, actualBlockLenQGm, qCmpGm,
                                              seqPrefixSumQ, blockPrefixSumQ);
        }
    }
}

template <typename BSAT>
__aicore__ inline void BSASelectBlockMaskBase<BSAT>::ProcessMatmulSoftmax(uint32_t batchIdx, uint32_t headIdx,
                                                                          uint32_t qChunkStart, uint32_t curQChunkSize,
                                                                          uint32_t validYBlocks)
{
    uint32_t loopChunkSize = CV_EXEC_RATIO * constInfo.kChunkSize;
    uint32_t flagId = 0;
    // D1: K loop truncated to validYBlocks
    for (uint32_t kChunkStart = 0; kChunkStart < validYBlocks; kChunkStart += loopChunkSize) {
        uint32_t kChunkEnd = BSAMin(kChunkStart + loopChunkSize, validYBlocks);
        uint32_t curKChunkSize = kChunkEnd - kChunkStart; // 实际执行k的blocks, 即k_comp的行数

        if ASCEND_IS_AIC {
            matmulService.ComputeMatmulChunk(qChunkStart, curQChunkSize, kChunkStart, curKChunkSize, batchIdx, headIdx);
            CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SYNC_C1_TO_V1_FLAG[flagId % SYNC_C1_TO_V1_FLAG_NUMS]);
        }

        if ASCEND_IS_AIV {
            CrossCoreWaitFlag(SYNC_C1_TO_V1_FLAG[flagId % SYNC_C1_TO_V1_FLAG_NUMS]);
            vectorService.OnlineSoftmaxFirstPassChunk(qChunkStart, curQChunkSize, kChunkStart, curKChunkSize,
                                                      validYBlocks);
        }
        flagId++;
    }
}

template <typename BSAT>
__aicore__ inline void BSASelectBlockMaskBase<BSAT>::ProcessSoftmaxSecondPass(uint32_t qChunkStart,
                                                                              uint32_t curQChunkSize, uint32_t batchIdx,
                                                                              uint32_t headIdx, uint32_t validYBlocks)
{
    if ASCEND_IS_AIV {
        AscendC::PipeBarrier<PIPE_ALL>();
        uint32_t loopChunkSize = CV_EXEC_RATIO * constInfo.kChunkSize;
        // D1: K loop truncated to validYBlocks
        for (uint32_t kChunkStart = 0; kChunkStart < validYBlocks; kChunkStart += loopChunkSize) {
            uint32_t kChunkEnd = BSAMin(kChunkStart + loopChunkSize, validYBlocks);
            uint32_t curKChunkSize = kChunkEnd - kChunkStart;

            vectorService.SoftmaxSecondPassAndCast(qChunkStart, curQChunkSize, kChunkStart, curKChunkSize, batchIdx,
                                                   headIdx, validYBlocks);
        }
    }
}

template <typename BSAT>
__aicore__ inline void BSASelectBlockMaskBase<BSAT>::ProcessPostSoftmaxPooling(uint32_t batchIdx, uint32_t headIdx,
                                                                               uint32_t validXBlocks,
                                                                               uint32_t validYBlocks)
{
    if ASCEND_IS_AIV {
        if (constInfo.postBlockShapeY > BSA_POST_POOL_COL_CAP) {
            // 超大 post 列宽超出向量化路径支持上限，走标量兜底
            ProcessPostSoftmaxPoolingScalar(batchIdx, headIdx, validXBlocks, validYBlocks);
            return;
        }
        vectorService.PostPoolRange(batchIdx, headIdx, validXBlocks, validYBlocks);
    }
}

template <typename BSAT>
__aicore__ inline void BSASelectBlockMaskBase<BSAT>::ProcessPostSoftmaxPoolingScalar(uint32_t batchIdx,
                                                                                     uint32_t headIdx,
                                                                                     uint32_t validXBlocks,
                                                                                     uint32_t validYBlocks)
{
    if ASCEND_IS_AIV {
        if (constInfo.aivIdx != 0) {
            return;
        }
        // attnScore workspace 是单 head 复用的 scratch buffer（每个 head 的 softmax 重写同一区域），
        // 读取当前 head 的 attn_score 偏移恒为 0。
        // pooledScore 逐 head 布局：head 间距为最大粗粒度网格 postXBlocks*postYBlocks，
        // head 内部为紧凑布局 [validPostX × validPostY]（与 TopK 线性读取一致）。
        uint64_t pooledHeadOffset = static_cast<uint64_t>(batchIdx) * constInfo.numHeads;
        pooledHeadOffset = (pooledHeadOffset + headIdx) * constInfo.postXBlocks * constInfo.postYBlocks;

        uint32_t postBlockShapeX = constInfo.postBlockShapeX;
        uint32_t postBlockShapeY = constInfo.postBlockShapeY;
        uint32_t validPostXBlocks = BSACeilDiv(validXBlocks, postBlockShapeX);
        uint32_t validPostYBlocks = BSACeilDiv(validYBlocks, postBlockShapeY);
        uint32_t validCount = validPostXBlocks * validPostYBlocks;

        // UB 布局（借用 postPoolOP 的 readUb scratch，与向量化路径同相位）: [readUb: 256 half = 512B | pooledUb: 1536
        // half = 3KB]
        constexpr uint32_t READ_CHUNK = 256;
        constexpr uint32_t POOLED_CHUNK = 1536;
        LocalTensor<half> readUb = vectorService.GetPostPoolScalarScratch();
        LocalTensor<half> pooledUb = readUb[READ_CHUNK];

        // pooledScore 在 head 区域内按紧凑布局 [validPostX × validPostY] 存储，
        // 与 TopK 的线性读取（sortLen = validPostX*validPostY，跨距 validPostY）一致。
        // head 间距仍为最大网格 postXBlocks*postYBlocks。
        for (uint32_t base = 0; base < validCount; base += POOLED_CHUNK) {
            uint32_t end = BSAMin(base + POOLED_CHUNK, validCount);
            for (uint32_t idx = base; idx < end; idx++) {
                uint32_t px = idx / validPostYBlocks;
                uint32_t py = idx % validPostYBlocks;
                uint32_t dstIdx = idx - base;
                uint32_t numFineRows = BSAMin(postBlockShapeX, validXBlocks - px * postBlockShapeX);
                uint32_t numFineCols = BSAMin(postBlockShapeY, validYBlocks - py * postBlockShapeY);
                if (numFineCols == 0 || numFineRows == 0) {
                    pooledUb.SetValue(dstIdx, static_cast<half>(0.0f));
                    continue;
                }
                float totalSum = 0.0f;
                for (uint32_t x = 0; x < numFineRows; x++) {
                    uint32_t fineX = px * postBlockShapeX + x;
                    // attnScore scratch 为紧凑布局 [validX × validY]（softmax 按 validYBlocks 跨距写出）
                    uint64_t gmOffset = static_cast<uint64_t>(fineX) * validYBlocks + py * postBlockShapeY;
                    uint32_t remain = numFineCols;
                    uint64_t cur = gmOffset;
                    while (remain > 0) {
                        uint32_t curLen = BSAMin(remain, READ_CHUNK);
                        DataCopyPad(readUb, attnScoreFp16Gm[cur], DataCopyExtParams(1, curLen * sizeof(half), 0, 0, 0),
                                    DataCopyPadExtParams<half>{true, 0, 0, 0});
                        // 标量消费前必须用 PipeBarrier 等 MTE2 完成
                        // （WaitFlag<MTE2_V> 在标量上下文不阻塞标量流水，存在竞争）
                        AscendC::PipeBarrier<PIPE_ALL>();
                        for (uint32_t c = 0; c < curLen; c++) {
                            totalSum += static_cast<float>(readUb.GetValue(c));
                        }
                        remain -= curLen;
                        cur += curLen;
                    }
                }
                float count = static_cast<float>(static_cast<int32_t>(numFineRows * numFineCols));
                float meanVal = totalSum / count;
                pooledUb.SetValue(dstIdx, static_cast<half>(meanVal));
            }
            AscendC::PipeBarrier<PIPE_ALL>();
            // 写出长度对齐到 32B（向上取整；尾部溢出部分落入下一 head 区域，会被其后续写出覆盖，
            // 最后一个 head 的溢出落入 workspace 之后的 softmaxTmp 区域，不会影响任何读取方，安全）
            uint32_t writeBytes = (end - base) * sizeof(half);
            uint32_t alignedBytes = (writeBytes + UB_BLOCK_SIZE - 1) / UB_BLOCK_SIZE * UB_BLOCK_SIZE;
            DataCopyPad(pooledScoreGm[pooledHeadOffset + base], pooledUb, DataCopyExtParams(1, alignedBytes, 0, 0, 0));
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }
}

#undef BSA_TILING_VERIFY_PRINTF

#endif // BSA_SELECT_BLOCK_MASK_BASE_H
