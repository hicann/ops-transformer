/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file moe_v2_scatter_out_batch_row.h
 * \brief 按输出位置批量Scatter输出模板，用于token数多、hiddenSize、k小的场景优化
 *        利用dst_to_src索引（输出位置连续），批量写出，减少搬出次数，提升性能
 */
#ifndef MOE_V2_SCATTER_OUT_BATCH_ROW_H
#define MOE_V2_SCATTER_OUT_BATCH_ROW_H

#include "moe_init_routing_v2_arch35_common.h"
#include "kernel_operator.h"

constexpr int64_t BUFFER_NUM_BATCH = 2;

namespace MoeInitRoutingV2 {
using namespace AscendC;

template <typename T>
class MoeV2ScatterOutBatchRow {
public:
    __aicore__ inline MoeV2ScatterOutBatchRow(){};
    __aicore__ inline void Init(GM_ADDR inputX, GM_ADDR expandedRowIdx, GM_ADDR expandedX, GM_ADDR workspace,
                                const MoeInitRoutingV2TilingData *tilingData, TPipe *tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyInDstToSrcIndices(int64_t progress, int64_t loopRows);
    __aicore__ inline void ScatterOutOneBlock(int64_t outPosStartIdx, int64_t outPosRowsNum, int64_t indexOffsetVal,
                                              LocalTensor<int32_t> &dstToSrcLocal);
    __aicore__ inline void ScatterOutCompute(int64_t progress, int64_t loopRows);

private:
    TPipe *pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM_BATCH> dstToSrcIndexQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM_BATCH> outputRowsQueue;

    GlobalTensor<T> inputXGm;
    GlobalTensor<T> expandedXGm;
    GlobalTensor<int32_t> dstToSrcIndexGm;

    const MoeV2GatherOutComputeTilingData *gatherOutTilingData;

    int64_t needCoreNum;
    int64_t blockIdx;
    int64_t cols;
    int64_t n;
    int64_t k;
    int64_t expertNum;
    int64_t totalLength;
    int64_t activateRows;

    int64_t ubRowsPerLoop;

    int64_t currentLoopRows;
    int64_t coreRows;
    int64_t perLoopRows;
    int64_t lastLoopRows;
    int64_t rowLoops;
};

template <typename T>
__aicore__ inline void MoeV2ScatterOutBatchRow<T>::CopyInDstToSrcIndices(int64_t progress, int64_t loopRows)
{
    LocalTensor<int32_t> indicesLocal = dstToSrcIndexQueue.AllocTensor<int32_t>();
    DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(loopRows * sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams<int32_t> dataCopyPadParams{false, 0, 0, 0};
    DataCopyPad(indicesLocal, dstToSrcIndexGm[progress * this->perLoopRows], dataCopyParams, dataCopyPadParams);
    dstToSrcIndexQueue.EnQue<int32_t>(indicesLocal);
}

template <typename T>
__aicore__ inline void MoeV2ScatterOutBatchRow<T>::ScatterOutOneBlock(int64_t outPosStartIdx, int64_t outPosRowsNum,
                                                                      int64_t indexOffsetVal,
                                                                      LocalTensor<int32_t> &dstToSrcLocal)
{
    int64_t cols = this->cols;
    int64_t k = this->k;
    int64_t n = this->n;
    int64_t totalLength = this->totalLength;
    int64_t activateRows = this->activateRows;

    LocalTensor<T> outRowsLocal = outputRowsQueue.AllocTensor<T>();

    SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);

    int64_t validOutPosRowsNum = Min(outPosRowsNum, activateRows - outPosStartIdx);

    DataCopyExtParams copyInParams{1, static_cast<uint32_t>(cols * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> copyInPadParams{false, 0, 0, 0};

    for (int64_t outPosIndex = 0; outPosIndex < outPosRowsNum; outPosIndex++) {
        if (outPosIndex >= validOutPosRowsNum) {
            break;
        }

        int32_t expandedTokenIdx = dstToSrcLocal.GetValue(indexOffsetVal + outPosIndex);

        if (expandedTokenIdx < 0 || expandedTokenIdx >= totalLength) {
            continue;
        }

        int64_t srcRowIdx = expandedTokenIdx / k;

        if (srcRowIdx < 0 || srcRowIdx >= n) {
            continue;
        }

        DataCopyPad(outRowsLocal[outPosIndex * cols], inputXGm[srcRowIdx * cols], copyInParams, copyInPadParams);
    }

    SetWaitFlag<HardEvent::MTE2_MTE3>(HardEvent::MTE2_MTE3);

    if (validOutPosRowsNum > 0 && outPosStartIdx < activateRows) {
        DataCopyExtParams copyOutParams{static_cast<uint16_t>(validOutPosRowsNum),
                                        static_cast<uint32_t>(cols * sizeof(T)), 0, 0, 0};
        DataCopyPad(expandedXGm[outPosStartIdx * cols], outRowsLocal, copyOutParams);
    }

    outputRowsQueue.FreeTensor(outRowsLocal);
}

template <typename T>
__aicore__ inline void MoeV2ScatterOutBatchRow<T>::ScatterOutCompute(int64_t progress, int64_t loopRows)
{
    LocalTensor<int32_t> dstToSrcLocal = dstToSrcIndexQueue.DeQue<int32_t>();

    int64_t outPosStartIdx = this->gatherOutTilingData->perCoreRows * this->blockIdx + this->perLoopRows * progress;

    if (outPosStartIdx >= this->activateRows) {
        dstToSrcIndexQueue.FreeTensor(dstToSrcLocal);
        return;
    }

    int64_t totalOutPosRowsNum = loopRows;
    int64_t ubRowsLoops = Ceil(totalOutPosRowsNum, this->ubRowsPerLoop);

    for (int64_t ubBlockIdx = 0; ubBlockIdx < ubRowsLoops; ubBlockIdx++) {
        int64_t currentUbRows = this->ubRowsPerLoop;
        int64_t remainingRows = totalOutPosRowsNum - ubBlockIdx * this->ubRowsPerLoop;
        if (remainingRows < this->ubRowsPerLoop) {
            currentUbRows = remainingRows;
        }

        int64_t blockOutPosStartIdx = outPosStartIdx + ubBlockIdx * this->ubRowsPerLoop;
        int64_t blockIndexOffsetVal = ubBlockIdx * this->ubRowsPerLoop;
        ScatterOutOneBlock(blockOutPosStartIdx, currentUbRows, blockIndexOffsetVal, dstToSrcLocal);
    }

    dstToSrcIndexQueue.FreeTensor(dstToSrcLocal);
}

template <typename T>
__aicore__ inline void MoeV2ScatterOutBatchRow<T>::Init(GM_ADDR inputX, GM_ADDR expandedRowIdx, GM_ADDR expandedX,
                                                        GM_ADDR workspace, const MoeInitRoutingV2TilingData *tilingData,
                                                        TPipe *tPipe)
{
    this->pipe = tPipe;
    this->blockIdx = GetBlockIdx();
    this->gatherOutTilingData = &(tilingData->gatherOutComputeParamsOp);

    this->needCoreNum = this->gatherOutTilingData->needCoreNum;
    this->activateRows = this->gatherOutTilingData->activateRows;
    this->cols = tilingData->cols;
    this->n = tilingData->n;
    this->k = tilingData->k;
    this->expertNum = tilingData->expertNum;
    this->totalLength = tilingData->n * tilingData->k;

    if (this->blockIdx == this->gatherOutTilingData->needCoreNum - 1) {
        this->coreRows = this->gatherOutTilingData->lastCoreRows;
        this->perLoopRows = this->gatherOutTilingData->perCorePerLoopRows;
        this->lastLoopRows = this->gatherOutTilingData->lastCoreLastLoopRows;
        this->rowLoops = this->gatherOutTilingData->lastCoreLoops;
    } else {
        this->coreRows = this->gatherOutTilingData->perCoreRows;
        this->perLoopRows = this->gatherOutTilingData->perCorePerLoopRows;
        this->lastLoopRows = this->gatherOutTilingData->perCoreLastLoopRows;
        this->rowLoops = this->gatherOutTilingData->perCoreLoops;
    }

    this->ubRowsPerLoop = this->gatherOutTilingData->perLoopCols;

    inputXGm.SetGlobalBuffer((__gm__ T *)inputX, this->n * this->cols);
    expandedXGm.SetGlobalBuffer((__gm__ T *)expandedX, this->activateRows * this->cols);

    dstToSrcIndexGm.SetGlobalBuffer((__gm__ int32_t *)workspace + Align(this->totalLength, sizeof(int32_t)) +
                                        this->blockIdx * this->gatherOutTilingData->perCoreRows,
                                    Align(this->coreRows, sizeof(int32_t)));

    pipe->InitBuffer(outputRowsQueue, BUFFER_NUM_BATCH, AlignBytes(this->ubRowsPerLoop * this->cols, sizeof(T)));
    pipe->InitBuffer(dstToSrcIndexQueue, BUFFER_NUM_BATCH,
                     AlignBytes(this->gatherOutTilingData->perCorePerLoopRows, sizeof(int32_t)));
}

template <typename T>
__aicore__ inline void MoeV2ScatterOutBatchRow<T>::Process()
{
    if (this->blockIdx >= this->needCoreNum) {
        return;
    }

    if (this->rowLoops == 0) {
        return;
    }

    CopyInDstToSrcIndices(0, this->perLoopRows);

    for (int64_t loop = 0; loop < this->rowLoops; loop++) {
        int64_t currentRows = (loop == this->rowLoops - 1) ? this->lastLoopRows : this->perLoopRows;

        ScatterOutCompute(loop, currentRows);

        if (loop < this->rowLoops - 1) {
            int64_t nextRows = (loop + 1 == this->rowLoops - 1) ? this->lastLoopRows : this->perLoopRows;
            CopyInDstToSrcIndices(loop + 1, nextRows);
        }
    }
}

} // namespace MoeInitRoutingV2
#endif // MOE_V2_SCATTER_OUT_BATCH_ROW_H
