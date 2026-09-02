/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file moe_v2_gather_out_for_simt.h
 * \brief
 */
#ifndef MOE_V2_GATHER_OUT_SIMT_H
#define MOE_V2_GATHER_OUT_SIMT_H

#include "moe_init_routing_v2_arch35_common.h"
#include "kernel_operator.h"

constexpr int64_t BUFFER_NUM = 2;

namespace MoeInitRoutingV2 {
using namespace AscendC;

template <typename T>
class MoeV2GatherOutSimt {
public:
    __aicore__ inline MoeV2GatherOutSimt(){};
    __aicore__ inline void Init(GM_ADDR inputX, GM_ADDR expandedRowIdx, GM_ADDR expandedX, GM_ADDR workspace,
                                const MoeInitRoutingV2TilingData *tilingData, TPipe *tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyInIndices(int64_t progress);
    __aicore__ inline void CopyOut(int64_t progress);
    __aicore__ inline void CopyInZeroIndices(int64_t progress);
    __aicore__ inline void CopyOutZero(int64_t progress);

private:
    TPipe *pipe;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, BUFFER_NUM> inputActivationsCopyInQueue;
    TQue<QuePosition::VECIN, BUFFER_NUM> expandedRowIdxIndexCopyInQueue;

    GlobalTensor<T> inputXGm;
    GlobalTensor<T> expandedXGm;
    GlobalTensor<int32_t> expandedRowIdxGm;
    GlobalTensor<int32_t> expandedRowIdxIndexGm_;

    const MoeV2GatherOutComputeTilingData *gatherOutTilingData;

    int64_t needCoreNum;
    int64_t blockIdx;
    int64_t cols;
    int64_t n;
    int64_t k;
    int64_t expertNum_;
    int64_t totalLength_;
    int64_t activateRows;
    int64_t currentLoopRows;
    int64_t coreRows;
    int64_t perLoopRows;
    int64_t lastLoopRows;
    int64_t rowLoops;
    int64_t colsTileLength;
    int64_t perLoopCols;
    int64_t lastLoopCols;
    int64_t colLoops;
    int64_t dropPadMode;

    int64_t indicesOffset;
    int64_t inputOffset;
    int64_t outOffset;
};

template <typename T>
__aicore__ inline void MoeV2GatherOutSimt<T>::CopyInIndices(int64_t progress)
{
    this->indicesOffset = progress * this->perLoopRows;
    LocalTensor<int32_t> indicesLocal = expandedRowIdxIndexCopyInQueue.AllocTensor<int32_t>();
    DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(this->currentLoopRows * sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams<int32_t> dataCopyPadParams{false, 0, 0, 0};
    DataCopyPad(indicesLocal, expandedRowIdxGm[indicesOffset], dataCopyParams, dataCopyPadParams);
    expandedRowIdxIndexCopyInQueue.EnQue<int32_t>(indicesLocal);
}

template <typename T>
__aicore__ inline void MoeV2GatherOutSimt<T>::CopyInZeroIndices(int64_t progress)
{
    this->indicesOffset = progress * this->perLoopRows;
    LocalTensor<int32_t> expandedRowIdxIndexLocal = expandedRowIdxIndexCopyInQueue.AllocTensor<int32_t>();
    DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>((this->currentLoopRows + 1) * sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams<int32_t> dataCopyPadParams{false, 0, 0, 0};
    DataCopyPad(expandedRowIdxIndexLocal, expandedRowIdxIndexGm_[indicesOffset], dataCopyParams, dataCopyPadParams);
    expandedRowIdxIndexCopyInQueue.EnQue<int32_t>(expandedRowIdxIndexLocal);
}

template <typename T>
__aicore__ inline void MoeV2GatherOutSimt<T>::CopyOutZero(int64_t progress)
{
    LocalTensor<int32_t> indicesLocal = expandedRowIdxIndexCopyInQueue.DeQue<int32_t>();
    LocalTensor<T> zeroTensor = inputActivationsCopyInQueue.AllocTensor<T>();
    if (this->blockIdx == 0 && progress == 0) {
        int32_t curIndex = 0;
        int32_t nextIndex = indicesLocal.GetValue(0);
        int32_t count = nextIndex - curIndex;
        if (count > 0) {
            InitGmValue(expandedXGm[curIndex * cols], zeroTensor, count * cols, static_cast<T>(0));
        }
    }
    for (int i = 0; i < currentLoopRows; i++) {
        int32_t curIndex = indicesLocal.GetValue(i) + 1;
        if (activateRows <= curIndex) {
            break;
        }
        int32_t nextIndex;
        if (this->blockIdx == this->gatherOutTilingData->needCoreNum - 1 && progress == this->rowLoops - 1 &&
            i == currentLoopRows - 1) {
            nextIndex = activateRows;
        } else {
            nextIndex = indicesLocal.GetValue(i + 1);
        }
        int32_t count = nextIndex - curIndex;
        if (count > 0) {
            InitGmValue(expandedXGm[curIndex * cols], zeroTensor, count * cols, static_cast<T>(0));
        }
    }
    SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
    inputActivationsCopyInQueue.FreeTensor(zeroTensor);
    expandedRowIdxIndexCopyInQueue.FreeTensor(indicesLocal);
}

template <typename T>
__aicore__ inline void MoeV2GatherOutSimt<T>::CopyOut(int64_t progress)
{
    LocalTensor<int32_t> indicesLocal = expandedRowIdxIndexCopyInQueue.DeQue<int32_t>();
    SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
    colsTileLength = this->perLoopCols;
    for (int64_t colsLoop = 0; colsLoop < this->colLoops; colsLoop++) {
        int64_t initialRow = this->gatherOutTilingData->perCoreRows * this->blockIdx + this->perLoopRows * progress;
        int64_t curLoopRow = 0;
        if (colsLoop == this->colLoops - 1) {
            colsTileLength = this->lastLoopCols;
        }
        int64_t currentLoopStartRow = initialRow / this->k;
        int64_t currentLoopLastRow = (initialRow + this->currentLoopRows - 1) / this->k;
        for (int64_t row = currentLoopStartRow; row <= currentLoopLastRow; row++) {
            LocalTensor<T> inLocal = inputActivationsCopyInQueue.AllocTensor<T>();
            // input row position
            inputOffset = row * this->cols + colsLoop * this->perLoopCols;
            DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(this->colsTileLength * sizeof(T)), 0, 0, 0};
            DataCopyPadExtParams<T> dataCopyPadParams{false, 0, 0, 0};
            DataCopyPad(inLocal, inputXGm[inputOffset], dataCopyParams, dataCopyPadParams);
            inputActivationsCopyInQueue.EnQue<T>(inLocal);
            inLocal = inputActivationsCopyInQueue.DeQue<T>();
            DataCopyExtParams intriParams{1, static_cast<uint32_t>(this->colsTileLength * sizeof(T)), 0, 0, 0};
            while (curLoopRow < this->currentLoopRows && initialRow / this->k == row) {
                int32_t outIndex = indicesLocal.GetValue(curLoopRow);
                curLoopRow++;
                initialRow++;
                if (outIndex == -1 || (this->dropPadMode == DROPLESS_MODE && outIndex >= this->activateRows)) {
                    continue;
                }
                outOffset = outIndex * cols + colsLoop * this->perLoopCols;
                DataCopyPad(expandedXGm[outOffset], inLocal, intriParams);
            }
            inputActivationsCopyInQueue.FreeTensor(inLocal);
        }
    }
    expandedRowIdxIndexCopyInQueue.FreeTensor(indicesLocal);
}

template <typename T>
__aicore__ inline void MoeV2GatherOutSimt<T>::Init(GM_ADDR inputX, GM_ADDR expandedRowIdx, GM_ADDR expandedX,
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
    this->expertNum_ = tilingData->expertNum;
    this->dropPadMode = tilingData->dropPadMode;
    this->totalLength_ = tilingData->n * tilingData->k;

    if (this->dropPadMode == 1) {
        this->activateRows = tilingData->expertCapacity * tilingData->expertNum;
    }

    if (this->blockIdx == this->gatherOutTilingData->needCoreNum - 1) {
        this->coreRows = this->gatherOutTilingData->lastCoreRows;
        this->perLoopRows = this->gatherOutTilingData->lastCorePerLoopRows;
        this->lastLoopRows = this->gatherOutTilingData->lastCoreLastLoopRows;
        this->rowLoops = this->gatherOutTilingData->perCoreLoops;
    } else {
        this->coreRows = this->gatherOutTilingData->perCoreRows;
        this->perLoopRows = this->gatherOutTilingData->perCorePerLoopRows;
        this->lastLoopRows = this->gatherOutTilingData->perCoreLastLoopRows;
        this->rowLoops = this->gatherOutTilingData->lastCoreLoops;
    }
    this->colLoops = this->gatherOutTilingData->colLoops;

    inputXGm.SetGlobalBuffer((__gm__ T *)inputX, this->coreRows * this->cols);
    expandedXGm.SetGlobalBuffer((__gm__ T *)expandedX, tilingData->n * tilingData->k * this->cols);
    expandedRowIdxGm.SetGlobalBuffer(
        (__gm__ int32_t *)expandedRowIdx + this->blockIdx * this->gatherOutTilingData->perCoreRows,
        Align(this->coreRows, sizeof(int32_t)));

    expandedRowIdxIndexGm_.SetGlobalBuffer((__gm__ int32_t *)workspace +
                                               Align(this->totalLength_, sizeof(int32_t)) * 2 + this->expertNum_ +
                                               this->blockIdx * this->gatherOutTilingData->perCoreRows,
                                           0);
    if (this->gatherOutTilingData->perLoopCols > this->cols) {
        this->perLoopCols = this->cols;
        this->lastLoopCols = this->cols;
        pipe->InitBuffer(inputActivationsCopyInQueue, CONSTANT_FOUR, AlignBytes(this->perLoopCols, sizeof(T)));
    } else {
        this->perLoopCols = this->gatherOutTilingData->perLoopCols;
        this->lastLoopCols = this->gatherOutTilingData->lastLoopCols;
        pipe->InitBuffer(inputActivationsCopyInQueue, BUFFER_NUM, AlignBytes(this->perLoopCols, sizeof(T)));
    }
    pipe->InitBuffer(expandedRowIdxIndexCopyInQueue, BUFFER_NUM, AlignBytes(this->perLoopRows, sizeof(int32_t)));
}

template <typename T>
__aicore__ inline void MoeV2GatherOutSimt<T>::Process()
{
    if (this->blockIdx < this->needCoreNum) {
        currentLoopRows = perLoopRows;
        for (int64_t loop = 0; loop < this->rowLoops; loop++) {
            if (loop == this->rowLoops - 1) {
                currentLoopRows = lastLoopRows;
            }
            if (this->dropPadMode == DROP_PAD_MODE) {
                CopyInZeroIndices(loop);
                CopyOutZero(loop);
            }
            CopyInIndices(loop);
            CopyOut(loop);
        }
    }
}
} // namespace MoeInitRoutingV2
#endif // MOE_V2_GATHER_OUT_SIMT_H
