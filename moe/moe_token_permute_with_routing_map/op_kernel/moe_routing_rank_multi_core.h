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
 * \file moe_routing_rank_multi_core.h
 * \brief 非对齐路径：按 routingMap 专家维前缀和构建 sortPosition 矩阵。
 */
#ifndef MOE_ROUTING_RANK_MULTI_CORE_H
#define MOE_ROUTING_RANK_MULTI_CORE_H

#include "kernel_tiling/kernel_tiling.h"
#include "moe_token_permute_with_routing_map_common.h"
#include "moe_sort_base_with_routing_map.h"

namespace MoeTokenPermute {
using namespace AscendC;

// rank 与 masked select 的 UB 布局不同，不可直接复用 formertileLength。
// 单 tile 约占用 27×Align(L,8) 字节（mask + 5×int32 + float + half 缓冲）。
// L=4096 时约 110592B，低于 192KB UB
constexpr int64_t RANK_MAX_TOKEN_TILE_LENGTH = 4096;

class MoeRoutingRankMultiCore {
public:
    __aicore__ inline MoeRoutingRankMultiCore() {}

    __aicore__ inline void Init(GM_ADDR routingMap, GM_ADDR workspace,
                                const MaskedSelectRMTilingData *maskedSelectTilingData, int64_t numExperts,
                                int64_t topKIn, TPipe *tPipeIn);

    __aicore__ inline void Process();

private:
    __aicore__ inline int64_t GetExpertStart() const;
    __aicore__ inline int64_t GetExpertCount() const;
    __aicore__ inline void GetTokenTileParams(int64_t tileIdx, int64_t &tokenStart, int64_t &tileLen) const;
    __aicore__ inline void CastMaskRowToInt32(const LocalTensor<int32_t> &maskInt32,
                                              const LocalTensor<uint8_t> &rowMask, int64_t tileLen);
    __aicore__ inline void CopyInMaskRow(int64_t gmOffset, int64_t tileLen);
    __aicore__ inline void CopyInPartial(int64_t coreId, int64_t tokenStart, int64_t tileLen);
    __aicore__ inline void ComputeAccumulateMask(int64_t tileLen);
    __aicore__ inline void ComputePartialOut(int64_t tileLen);
    __aicore__ inline void CopyOutPartialGm(int64_t gmOffset, int64_t tileLen);
    __aicore__ inline void ComputeAccumulatePartialPrefix(int64_t tileLen);
    __aicore__ inline void ComputeSortRow(int64_t tileLen);
    __aicore__ inline void CopyOutSortRow(int64_t gmOffset, int64_t tileLen);

    __aicore__ inline void AccumulatePartialCount();
    __aicore__ inline void BuildSortPosition();

private:
    TPipe *pipe;
    TQue<QuePosition::VECIN, 1> inQueueMask;
    TQue<QuePosition::VECIN, 1> inQueueInt32;
    TQue<QuePosition::VECOUT, 1> outQueueInt32;
    TBuf<TPosition::VECCALC> partialBuf;
    TBuf<TPosition::VECCALC> maskWorkBuf;
    TBuf<TPosition::VECCALC> maskFloatBuf;
    TBuf<TPosition::VECCALC> maskCastHalfBuf;
    TBuf<TPosition::VECCALC> tokenBaseBuf;

    GlobalTensor<uint8_t> maskGlobal;
    GlobalTensor<int32_t> partialGm;
    GlobalTensor<int32_t> sortPositionGm;

    int64_t blockIdx;
    int64_t numExperts;
    int64_t numTokens;
    int64_t topK;
    int64_t needCoreNum;
    int64_t formerNum;
    int64_t tokenTileLength;
    int64_t tokenTileNum;
    int64_t lastTokenTileLength;
    int64_t calcLength;
    int64_t partialOffset;
    int64_t sortPositionOffset;
};

__aicore__ inline int64_t MoeRoutingRankMultiCore::GetExpertStart() const
{
    int64_t expertsPerFormer = (numExperts + needCoreNum - 1) / needCoreNum;
    if (blockIdx < formerNum) {
        return blockIdx * expertsPerFormer;
    }
    int64_t formerExpertCount = formerNum * expertsPerFormer;
    int64_t tailExperts = numExperts - formerExpertCount;
    int64_t tailCores = needCoreNum - formerNum;
    if (tailCores <= 0 || blockIdx >= needCoreNum) {
        return numExperts;
    }
    int64_t expertsPerTail = tailExperts / tailCores;
    return formerExpertCount + (blockIdx - formerNum) * expertsPerTail;
}

__aicore__ inline int64_t MoeRoutingRankMultiCore::GetExpertCount() const
{
    int64_t expertsPerFormer = (numExperts + needCoreNum - 1) / needCoreNum;
    if (blockIdx < formerNum) {
        return expertsPerFormer;
    }
    int64_t formerExpertCount = formerNum * expertsPerFormer;
    int64_t tailExperts = numExperts - formerExpertCount;
    int64_t tailCores = needCoreNum - formerNum;
    if (tailCores <= 0 || blockIdx >= needCoreNum) {
        return 0;
    }
    int64_t expertsPerTail = tailExperts / tailCores;
    int64_t tailIdx = blockIdx - formerNum;
    if (tailIdx == tailCores - 1) {
        return tailExperts - expertsPerTail * (tailCores - 1);
    }
    return expertsPerTail;
}

__aicore__ inline void MoeRoutingRankMultiCore::GetTokenTileParams(int64_t tileIdx, int64_t &tokenStart,
                                                                   int64_t &tileLen) const
{
    tokenStart = tileIdx * tokenTileLength;
    tileLen = (tileIdx == tokenTileNum - 1) ? lastTokenTileLength : tokenTileLength;
}

__aicore__ inline void MoeRoutingRankMultiCore::CastMaskRowToInt32(const LocalTensor<int32_t> &maskInt32,
                                                                   const LocalTensor<uint8_t> &rowMask, int64_t tileLen)
{
    LocalTensor<half> maskHalf = maskCastHalfBuf.Get<half>();
    LocalTensor<float> maskFloat = maskFloatBuf.Get<float>();
    Cast(maskHalf, rowMask, RoundMode::CAST_NONE, tileLen);
    PipeBarrier<PIPE_V>();
    Cast(maskFloat, maskHalf, RoundMode::CAST_NONE, tileLen);
    PipeBarrier<PIPE_V>();
    Cast(maskInt32, maskFloat, RoundMode::CAST_ROUND, tileLen);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void MoeRoutingRankMultiCore::CopyInMaskRow(int64_t gmOffset, int64_t tileLen)
{
    LocalTensor<uint8_t> maskLocal = inQueueMask.AllocTensor<uint8_t>();
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(tileLen), 0, 0, 0};
    DataCopyPadExtParams<uint8_t> padParams{false, 0, 0, 0};
    DataCopyPad(maskLocal, maskGlobal[gmOffset], copyParams, padParams);
    inQueueMask.EnQue(maskLocal);
}

__aicore__ inline void MoeRoutingRankMultiCore::CopyInPartial(int64_t coreId, int64_t tokenStart, int64_t tileLen)
{
    LocalTensor<int32_t> partialLocal = inQueueInt32.AllocTensor<int32_t>();
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(tileLen * sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
    DataCopyPad(partialLocal, partialGm[coreId * numTokens + tokenStart], copyParams, padParams);
    inQueueInt32.EnQue(partialLocal);
}

__aicore__ inline void MoeRoutingRankMultiCore::ComputeAccumulateMask(int64_t tileLen)
{
    LocalTensor<uint8_t> maskLocal = inQueueMask.DeQue<uint8_t>();

    LocalTensor<int32_t> maskInt32 = maskWorkBuf.Get<int32_t>();
    CastMaskRowToInt32(maskInt32, maskLocal, tileLen);
    inQueueMask.FreeTensor(maskLocal);

    LocalTensor<int32_t> partialTile = partialBuf.Get<int32_t>();
    Add(partialTile, partialTile, maskInt32, tileLen);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void MoeRoutingRankMultiCore::ComputePartialOut(int64_t tileLen)
{
    LocalTensor<int32_t> partialTile = partialBuf.Get<int32_t>();
    LocalTensor<int32_t> outLocal = outQueueInt32.AllocTensor<int32_t>();
    Duplicate(outLocal, static_cast<int32_t>(0), tileLen);
    PipeBarrier<PIPE_V>();
    Add(outLocal, outLocal, partialTile, tileLen);
    PipeBarrier<PIPE_V>();
    outQueueInt32.EnQue(outLocal);
}

__aicore__ inline void MoeRoutingRankMultiCore::CopyOutPartialGm(int64_t gmOffset, int64_t tileLen)
{
    LocalTensor<int32_t> outLocal = outQueueInt32.DeQue<int32_t>();
    DataCopyExtParams copyOutParams{1, static_cast<uint32_t>(tileLen * sizeof(int32_t)), 0, 0, 0};
    DataCopyPad(partialGm[gmOffset], outLocal, copyOutParams);
    outQueueInt32.FreeTensor(outLocal);
}

__aicore__ inline void MoeRoutingRankMultiCore::ComputeAccumulatePartialPrefix(int64_t tileLen)
{
    LocalTensor<int32_t> partialLocal = inQueueInt32.DeQue<int32_t>();

    LocalTensor<int32_t> runningTile = partialBuf.Get<int32_t>();
    Add(runningTile, runningTile, partialLocal, tileLen);
    PipeBarrier<PIPE_V>();

    inQueueInt32.FreeTensor(partialLocal);
}

__aicore__ inline void MoeRoutingRankMultiCore::ComputeSortRow(int64_t tileLen)
{
    LocalTensor<uint8_t> maskLocal = inQueueMask.DeQue<uint8_t>();

    LocalTensor<int32_t> maskInt32 = maskWorkBuf.Get<int32_t>();
    CastMaskRowToInt32(maskInt32, maskLocal, tileLen);
    inQueueMask.FreeTensor(maskLocal);

    LocalTensor<int32_t> runningTile = partialBuf.Get<int32_t>();
    LocalTensor<int32_t> tokenBase = tokenBaseBuf.Get<int32_t>();
    LocalTensor<int32_t> sortRow = outQueueInt32.AllocTensor<int32_t>();
    Add(sortRow, runningTile, tokenBase, tileLen);
    PipeBarrier<PIPE_V>();
    outQueueInt32.EnQue(sortRow);

    Add(runningTile, runningTile, maskInt32, tileLen);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void MoeRoutingRankMultiCore::CopyOutSortRow(int64_t gmOffset, int64_t tileLen)
{
    LocalTensor<int32_t> sortRow = outQueueInt32.DeQue<int32_t>();
    DataCopyExtParams copyOutParams{1, static_cast<uint32_t>(tileLen * sizeof(int32_t)), 0, 0, 0};
    DataCopyPad(sortPositionGm[gmOffset], sortRow, copyOutParams);
    outQueueInt32.FreeTensor(sortRow);
}

__aicore__ inline void MoeRoutingRankMultiCore::Init(GM_ADDR routingMap, GM_ADDR workspace,
                                                     const MaskedSelectRMTilingData *maskedSelectTilingData,
                                                     int64_t numExpertsIn, int64_t topKIn, TPipe *tPipeIn)
{
    pipe = tPipeIn;
    blockIdx = GetBlockIdx();
    numExperts = numExpertsIn;
    topK = topKIn;

    needCoreNum = static_cast<int64_t>(maskedSelectTilingData->needCoreNum);
    formerNum = static_cast<int64_t>(maskedSelectTilingData->formerNum);
    numTokens = static_cast<int64_t>(maskedSelectTilingData->tokenNum);

    int64_t msTileLength = static_cast<int64_t>(maskedSelectTilingData->formertileLength);
    tokenTileLength = msTileLength;
    if (tokenTileLength <= 0 || tokenTileLength > RANK_MAX_TOKEN_TILE_LENGTH) {
        tokenTileLength = RANK_MAX_TOKEN_TILE_LENGTH;
    }
    if (tokenTileLength > numTokens) {
        tokenTileLength = numTokens;
    }
    if (tokenTileLength <= 0) {
        tokenTileLength = 1;
    }
    tokenTileNum = Ceil(numTokens, tokenTileLength);
    lastTokenTileLength = numTokens % tokenTileLength;
    if (lastTokenTileLength == 0) {
        lastTokenTileLength = tokenTileLength;
    }
    calcLength = Align(tokenTileLength, INT32_ONE_BLOCK_NUM);

    partialOffset = Align(needCoreNum, sizeof(int32_t));
    sortPositionOffset = partialOffset + needCoreNum * numTokens;

    maskGlobal.SetGlobalBuffer((__gm__ uint8_t *)routingMap, numExperts * numTokens);
    partialGm.SetGlobalBuffer((__gm__ int32_t *)workspace + partialOffset, needCoreNum * numTokens);
    sortPositionGm.SetGlobalBuffer((__gm__ int32_t *)workspace + sortPositionOffset, numExperts * numTokens);

    pipe->InitBuffer(inQueueMask, 1, calcLength);
    pipe->InitBuffer(inQueueInt32, 1, calcLength * sizeof(int32_t));
    pipe->InitBuffer(outQueueInt32, 1, calcLength * sizeof(int32_t));
    pipe->InitBuffer(partialBuf, calcLength * sizeof(int32_t));
    pipe->InitBuffer(maskWorkBuf, calcLength * sizeof(int32_t));
    pipe->InitBuffer(maskFloatBuf, calcLength * sizeof(float));
    pipe->InitBuffer(maskCastHalfBuf, calcLength * static_cast<int64_t>(sizeof(half)));
    pipe->InitBuffer(tokenBaseBuf, calcLength * sizeof(int32_t));
}

__aicore__ inline void MoeRoutingRankMultiCore::AccumulatePartialCount()
{
    int64_t expertStart = GetExpertStart();
    int64_t expertCount = GetExpertCount();

    for (int64_t tileIdx = 0; tileIdx < tokenTileNum; ++tileIdx) {
        int64_t tokenStart = 0;
        int64_t tileLen = 0;
        GetTokenTileParams(tileIdx, tokenStart, tileLen);

        LocalTensor<int32_t> partialTile = partialBuf.Get<int32_t>();
        Duplicate(partialTile, static_cast<int32_t>(0), tileLen);
        PipeBarrier<PIPE_V>();

        for (int64_t e = 0; e < expertCount; ++e) {
            int64_t rowOffset = (expertStart + e) * numTokens + tokenStart;
            CopyInMaskRow(rowOffset, tileLen);
            ComputeAccumulateMask(tileLen);
        }

        ComputePartialOut(tileLen);
        CopyOutPartialGm(blockIdx * numTokens + tokenStart, tileLen);
    }
}

__aicore__ inline void MoeRoutingRankMultiCore::BuildSortPosition()
{
    int64_t expertStart = GetExpertStart();
    int64_t expertCount = GetExpertCount();

    for (int64_t tileIdx = 0; tileIdx < tokenTileNum; ++tileIdx) {
        int64_t tokenStart = 0;
        int64_t tileLen = 0;
        GetTokenTileParams(tileIdx, tokenStart, tileLen);

        LocalTensor<int32_t> runningTile = partialBuf.Get<int32_t>();
        Duplicate(runningTile, static_cast<int32_t>(0), tileLen);
        PipeBarrier<PIPE_V>();

        for (int64_t c = 0; c < blockIdx; ++c) {
            CopyInPartial(c, tokenStart, tileLen);
            ComputeAccumulatePartialPrefix(tileLen);
        }

        LocalTensor<int32_t> tokenBase = tokenBaseBuf.Get<int32_t>();
        ArithProgressionSupportInt32<int32_t>(tokenBase, static_cast<int32_t>(tokenStart * topK),
                                              static_cast<int32_t>(topK), static_cast<int32_t>(tileLen));
        PipeBarrier<PIPE_V>();

        for (int64_t e = 0; e < expertCount; ++e) {
            int64_t expert = expertStart + e;
            int64_t rowOffset = expert * numTokens + tokenStart;

            CopyInMaskRow(rowOffset, tileLen);
            ComputeSortRow(tileLen);
            CopyOutSortRow(rowOffset, tileLen);
        }
    }
}

__aicore__ inline void MoeRoutingRankMultiCore::Process()
{
    AscendC::SyncAll();
    if (blockIdx < needCoreNum) {
        AccumulatePartialCount();
    }
    AscendC::SyncAll();

    if (blockIdx < needCoreNum) {
        BuildSortPosition();
    }
    AscendC::SyncAll();
}

} // namespace MoeTokenPermute

#endif // MOE_ROUTING_RANK_MULTI_CORE_H
