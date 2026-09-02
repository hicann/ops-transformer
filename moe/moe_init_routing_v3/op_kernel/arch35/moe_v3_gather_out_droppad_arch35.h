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
 * \file moe_v3_gather_out_droppad_arch35.h
 * \brief
 */
#ifndef MOE_V3_GATHER_OUT_DROPPAD_H_REGBASE_ARCH35
#define MOE_V3_GATHER_OUT_DROPPAD_H_REGBASE_ARCH35

#include "moe_v3_common.h"
#include "kernel_operator.h"

namespace MoeInitRoutingV3 {
using namespace AscendC;

template <typename T>
class MoeV3GatherOutDropPad {
public:
    __aicore__ inline MoeV3GatherOutDropPad(){};
    __aicore__ inline void InitBaseData(GM_ADDR workspace, const MoeInitRoutingV3Arch35TilingData *tilingData,
                                        TPipe *tPipe);
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR scale, GM_ADDR workspace, GM_ADDR expandedRowIdx, GM_ADDR expandedX,
                                GM_ADDR expandedScale, const MoeInitRoutingV3Arch35TilingData *tilingData,
                                TPipe *tPipe);
    __aicore__ inline void Process();
    __aicore__ inline void CopyExpertIn(int64_t curExpertLoopOffset, int64_t curLoopElements);
    __aicore__ inline void CopyXIn(int64_t xSrcOffset, int64_t scaleSrcOffset, int64_t curLoopCols);
    __aicore__ inline void CopyXOut(int64_t xDstOffset, int64_t scaleDstOffset, int64_t curLoopCols);
    __aicore__ inline void CopyScaleIn(int64_t scaleSrcOffset);
    __aicore__ inline void CopyScaleOut(int64_t scaleDstOffset);
    __aicore__ inline void ProcessCompactOutputRows();
    __aicore__ inline void CopySourceRowsIn(int64_t outputRowOffset, int64_t curLoopRows);

private:
    TPipe *pipe_;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, GATHER_OUT_BUFFER_NUM> xCopyInQueue_;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, GATHER_OUT_BUFFER_NUM> scaleCopyInQueue_;
    TQue<QuePosition::VECIN, GATHER_OUT_BUFFER_NUM> expandedRowIdxCopyInQueue_;

    GlobalTensor<T> xGm_;
    GlobalTensor<uint8_t> xUint8tGm_;
    GlobalTensor<float> xGscaleGm_;
    GlobalTensor<int32_t> sortedExpertIdxGm_;
    GlobalTensor<int32_t> expandDstToSrcRowGm_;
    GlobalTensor<int32_t> outputToSrcRowGm_;
    GlobalTensor<T> expandedXGm_;
    GlobalTensor<int32_t> expandedRowIdxGm_;
    GlobalTensor<float> expandedScaleGm_;

    int64_t blockIdx_;
    int64_t cols_;
    int64_t n_;
    int64_t k_;

    int64_t colsLoops_;
    int64_t perLoopCols_;
    int64_t lastLoopCols_;

    int64_t indicesLoops_;

    int64_t perCoreIndicesElements_;
    int64_t lastCoreIndicesElements_;
    int64_t perCorePerLoopIndicesElements_;
    int64_t lastCorePerLoopIndicesElements_;
    int64_t curCorePerLoopIndicesElements_;
    int64_t curCoreLastLoopIndicesElements_;
    int64_t needCoreNum_;
    int64_t curCoreIndicesElements_;

    int64_t actualExpertNum_;
    int64_t outputRows_ = 0;

    int64_t rowIdxType_ = 0;
    int64_t isInputScale_ = 0;
    int64_t dropPadMode_ = DROP_PAD_MODE;
    int64_t activeNum_ = 0;
    int64_t useCompactOutputRows_ = 0;
};

template <typename T>
__aicore__ inline void MoeV3GatherOutDropPad<T>::InitBaseData(GM_ADDR workspace,
                                                              const MoeInitRoutingV3Arch35TilingData *tilingData,
                                                              TPipe *tPipe)
{
    pipe_ = tPipe;
    blockIdx_ = GetBlockIdx();

    cols_ = tilingData->cols;
    n_ = tilingData->n;
    k_ = tilingData->k;

    isInputScale_ = tilingData->isInputScale;
    rowIdxType_ = tilingData->rowIdxType;
    dropPadMode_ = tilingData->dropPadMode;
    activeNum_ = tilingData->activeNum;

    colsLoops_ = tilingData->gatherOutComputeParamsOp.colsLoops;
    perLoopCols_ = tilingData->gatherOutComputeParamsOp.perLoopCols;
    lastLoopCols_ = tilingData->gatherOutComputeParamsOp.lastLoopCols;

    actualExpertNum_ = tilingData->actualExpertNum;
    outputRows_ = tilingData->expertNum * tilingData->expertCapacity;
    useCompactOutputRows_ = tilingData->gatherOutComputeParamsOp.useCompactGatherOutDropPad;

    // Tiling split matches Process(): compact scans outputRows, fallback scans the n*k expandedRowIdx space.
    needCoreNum_ = tilingData->gatherOutComputeParamsOp.needCoreNum;
    perCoreIndicesElements_ = tilingData->gatherOutComputeParamsOp.perCoreIndicesElements;
    lastCoreIndicesElements_ = tilingData->gatherOutComputeParamsOp.lastCoreIndicesElements;
    perCorePerLoopIndicesElements_ = tilingData->gatherOutComputeParamsOp.perCorePerLoopIndicesElements;
    lastCorePerLoopIndicesElements_ = tilingData->gatherOutComputeParamsOp.lastCorePerLoopIndicesElements;

    // 从gatherOutComputeParamsOp读取循环参数
    if (blockIdx_ == needCoreNum_ - 1) {
        curCoreIndicesElements_ = lastCoreIndicesElements_;
        indicesLoops_ = tilingData->gatherOutComputeParamsOp.lastCoreIndicesLoops;
        curCorePerLoopIndicesElements_ = tilingData->gatherOutComputeParamsOp.lastCorePerLoopIndicesElements;
        curCoreLastLoopIndicesElements_ = tilingData->gatherOutComputeParamsOp.lastCoreLastLoopIndicesElements;
    } else {
        curCoreIndicesElements_ = perCoreIndicesElements_;
        indicesLoops_ = tilingData->gatherOutComputeParamsOp.perCoreIndicesLoops;
        curCorePerLoopIndicesElements_ = tilingData->gatherOutComputeParamsOp.perCorePerLoopIndicesElements;
        curCoreLastLoopIndicesElements_ = tilingData->gatherOutComputeParamsOp.perCoreLastLoopIndicesElements;
    }
}

template <typename T>
__aicore__ inline void MoeV3GatherOutDropPad<T>::Init(GM_ADDR x, GM_ADDR scale, GM_ADDR workspace,
                                                      GM_ADDR expandedRowIdx, GM_ADDR expandedX, GM_ADDR expandedScale,
                                                      const MoeInitRoutingV3Arch35TilingData *tilingData, TPipe *tPipe)
{
    InitBaseData(workspace, tilingData, tPipe);

    if (isInputScale_ == 1) {
        xGscaleGm_.SetGlobalBuffer((__gm__ float *)scale, n_);
    }
    if constexpr (IsSameType<T, hifloat8_t>::value) {
        xUint8tGm_.SetGlobalBuffer((__gm__ uint8_t *)x, n_ * cols_);
    } else {
        xGm_.SetGlobalBuffer((__gm__ T *)x, n_ * cols_);
    }
    // DropPad模式：输出expandedX为3D[expertNum, expertCapacity, cols]，所有核共享整个输出区域
    // 写入位置由expandedRowIdx决定，分布在整个输出范围
    expandedXGm_.SetGlobalBuffer((__gm__ T *)expandedX, tilingData->expertNum * tilingData->expertCapacity * cols_);
    if (isInputScale_ == 1) {
        expandedScaleGm_.SetGlobalBuffer((__gm__ float *)expandedScale,
                                         tilingData->expertNum * tilingData->expertCapacity);
    }

    pipe_->InitBuffer(expandedRowIdxCopyInQueue_, GATHER_OUT_BUFFER_NUM,
                      AlignBytes(curCorePerLoopIndicesElements_, sizeof(int32_t)));
    if (isInputScale_ == 1) {
        pipe_->InitBuffer(scaleCopyInQueue_, GATHER_OUT_BUFFER_NUM, AlignBytes(1, sizeof(float)));
    }

    // DropPad compact路径按输出行搬运 X，保留单行 queue 即可。
    int64_t xCopyInQueueBufferNum =
        max(tilingData->gatherOutComputeParamsOp.xCopyInQueueBufferNum, GATHER_OUT_BUFFER_NUM);
    pipe_->InitBuffer(xCopyInQueue_, xCopyInQueueBufferNum, AlignBytes(perLoopCols_, sizeof(T)));

    sortedExpertIdxGm_.SetGlobalBuffer((__gm__ int32_t *)workspace + blockIdx_ * perCoreIndicesElements_,
                                       Align(curCoreIndicesElements_, sizeof(int32_t)));

    int64_t length = Align(n_ * k_, sizeof(int32_t));
    expandDstToSrcRowGm_.SetGlobalBuffer((__gm__ int32_t *)workspace + length + blockIdx_ * perCoreIndicesElements_,
                                         Align(curCoreIndicesElements_, sizeof(int32_t)));
    int64_t actualExpertNumOffset = Align(tilingData->actualExpertNum, sizeof(int32_t));
    if (useCompactOutputRows_ == 1) {
        outputToSrcRowGm_.SetGlobalBuffer(
            (__gm__ int32_t *)workspace + length * 2 + actualExpertNumOffset + tilingData->coreNum * 2, outputRows_);
    }

    if (rowIdxType_ == SCATTER) {
        expandedRowIdxGm_.SetGlobalBuffer((__gm__ int32_t *)expandedRowIdx + blockIdx_ * perCoreIndicesElements_,
                                          Align(curCoreIndicesElements_, sizeof(int32_t)));
    } else {
        // GATHER模式下，expandedRowIdx以全局排序位置为索引，row_idx_gather_droppad用全局索引写入
        expandedRowIdxGm_.SetGlobalBuffer((__gm__ int32_t *)expandedRowIdx, Align(n_ * k_, sizeof(int32_t)));
    }
}

template <typename T>
__aicore__ inline void MoeV3GatherOutDropPad<T>::CopyExpertIn(int64_t curExpertLoopOffset, int64_t curLoopElements)
{
    LocalTensor<int32_t> subRowIdxLocal = expandedRowIdxCopyInQueue_.AllocTensor<int32_t>();
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(curLoopElements * sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
    DataCopyPad(subRowIdxLocal, expandedRowIdxGm_[curExpertLoopOffset], copyParams, padParams);
    expandedRowIdxCopyInQueue_.EnQue(subRowIdxLocal);
}

template <typename T>
__aicore__ inline void MoeV3GatherOutDropPad<T>::CopyXIn(int64_t xSrcOffset, int64_t scaleSrcOffset,
                                                         int64_t curLoopCols)
{
    if constexpr (IsSameType<T, hifloat8_t>::value) {
        LocalTensor<uint8_t> xLocal = xCopyInQueue_.AllocTensor<uint8_t>();
        DataCopyExtParams copyParams0{static_cast<uint16_t>(1), static_cast<uint32_t>(curLoopCols * sizeof(uint8_t)), 0,
                                      0, 0};
        DataCopyPadExtParams<uint8_t> padParams0{false, 0, 0, 0};
        DataCopyPad(xLocal, xUint8tGm_[xSrcOffset], copyParams0, padParams0);
        xCopyInQueue_.EnQue(xLocal);
    } else {
        LocalTensor<T> xLocal = xCopyInQueue_.AllocTensor<T>();
        DataCopyExtParams copyParams0{static_cast<uint16_t>(1), static_cast<uint32_t>(curLoopCols * sizeof(T)), 0, 0,
                                      0};
        DataCopyPadExtParams<T> padParams0{false, 0, 0, 0};
        DataCopyPad(xLocal, xGm_[xSrcOffset], copyParams0, padParams0);
        xCopyInQueue_.EnQue(xLocal);
    }
}

template <typename T>
__aicore__ inline void MoeV3GatherOutDropPad<T>::CopyXOut(int64_t xDstOffset, int64_t scaleDstOffset,
                                                          int64_t curLoopCols)
{
    LocalTensor<T> xLocal = xCopyInQueue_.DeQue<T>();
    DataCopyExtParams copyParams2{1, static_cast<uint32_t>(curLoopCols * sizeof(T)), 0, 0, 0};
    DataCopyPad(expandedXGm_[xDstOffset], xLocal, copyParams2);
    xCopyInQueue_.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void MoeV3GatherOutDropPad<T>::CopyScaleIn(int64_t scaleSrcOffset)
{
    LocalTensor<float> scaleLocal = scaleCopyInQueue_.AllocTensor<float>();
    DataCopyExtParams copyParams1{static_cast<uint16_t>(1), static_cast<uint32_t>(1 * sizeof(float)), 0, 0, 0};
    DataCopyPadExtParams<float> padParams1{false, 0, 0, 0};
    DataCopyPad(scaleLocal, xGscaleGm_[scaleSrcOffset], copyParams1, padParams1);
    scaleCopyInQueue_.EnQue(scaleLocal);
}

template <typename T>
__aicore__ inline void MoeV3GatherOutDropPad<T>::CopyScaleOut(int64_t scaleDstOffset)
{
    LocalTensor<float> scaleLocal = scaleCopyInQueue_.DeQue<float>();
    DataCopyExtParams copyParams3{1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
    DataCopyPad(expandedScaleGm_[scaleDstOffset], scaleLocal, copyParams3);
    scaleCopyInQueue_.FreeTensor(scaleLocal);
}

template <typename T>
__aicore__ inline void MoeV3GatherOutDropPad<T>::CopySourceRowsIn(int64_t outputRowOffset, int64_t curLoopRows)
{
    LocalTensor<int32_t> sourceRowsLocal = expandedRowIdxCopyInQueue_.AllocTensor<int32_t>();
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(curLoopRows * sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
    DataCopyPad(sourceRowsLocal, outputToSrcRowGm_[outputRowOffset], copyParams, padParams);
    expandedRowIdxCopyInQueue_.EnQue(sourceRowsLocal);
}

template <typename T>
__aicore__ inline void MoeV3GatherOutDropPad<T>::ProcessCompactOutputRows()
{
    int64_t perCoreOutputRows = Ceil(outputRows_, needCoreNum_);
    int64_t startOutputRow = blockIdx_ * perCoreOutputRows;
    int64_t endOutputRow = Min(startOutputRow + perCoreOutputRows, outputRows_);
    if (startOutputRow >= endOutputRow) {
        return;
    }

    int64_t tileRows = Max(curCorePerLoopIndicesElements_, static_cast<int64_t>(1));
    for (int64_t outputRowOffset = startOutputRow; outputRowOffset < endOutputRow; outputRowOffset += tileRows) {
        int64_t curLoopRows = Min(tileRows, endOutputRow - outputRowOffset);

        CopySourceRowsIn(outputRowOffset, curLoopRows);
        LocalTensor<int32_t> sourceRowsLocal = expandedRowIdxCopyInQueue_.DeQue<int32_t>();

        // 按列分块拷贝：当cols很大时，一行数据需要分多次拷贝
        for (int64_t colsLoop = 0; colsLoop < colsLoops_; colsLoop++) {
            int64_t curLoopCols = (colsLoop == colsLoops_ - 1) ? lastLoopCols_ : perLoopCols_;
            int64_t colsLoopOffset = colsLoop * perLoopCols_;
            DataCopyExtParams copyXParams{1, static_cast<uint32_t>(curLoopCols * sizeof(T)), 0, 0, 0};
            DataCopyExtParams copyScaleParams{1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0};

            for (int64_t idx = 0; idx < curLoopRows; idx++) {
                int64_t srcRow = sourceRowsLocal.GetValue(idx);
                if (srcRow < 0 || srcRow >= n_) {
                    continue;
                }

                SetWaitFlag<HardEvent::S_MTE2>(HardEvent::S_MTE2);
                CopyXIn(srcRow * cols_ + colsLoopOffset, srcRow, curLoopCols);
                LocalTensor<T> xLocal = xCopyInQueue_.DeQue<T>();

                LocalTensor<float> scaleLocal;
                if (isInputScale_ == 1 && colsLoop == 0) {
                    CopyScaleIn(srcRow);
                    scaleLocal = scaleCopyInQueue_.DeQue<float>();
                }

                int64_t outputRow = outputRowOffset + idx;
                DataCopyPad(expandedXGm_[outputRow * cols_ + colsLoopOffset], xLocal, copyXParams);
                if (isInputScale_ == 1 && colsLoop == 0) {
                    DataCopyPad(expandedScaleGm_[outputRow], scaleLocal, copyScaleParams);
                }

                if (isInputScale_ == 1 && colsLoop == 0) {
                    scaleCopyInQueue_.FreeTensor(scaleLocal);
                }
                xCopyInQueue_.FreeTensor(xLocal);
            }
        }
        expandedRowIdxCopyInQueue_.FreeTensor(sourceRowsLocal);
    }
}

template <typename T>
__aicore__ inline void MoeV3GatherOutDropPad<T>::Process()
{
    if (blockIdx_ < needCoreNum_) {
        if (useCompactOutputRows_ == 1) {
            ProcessCompactOutputRows();
            return;
        }

        int64_t curLoopElements = curCorePerLoopIndicesElements_;
        int64_t globalOffsetBase = blockIdx_ * perCoreIndicesElements_;
        for (int64_t indicesLoop = 0; indicesLoop < indicesLoops_; indicesLoop++) {
            if (indicesLoop == indicesLoops_ - 1) {
                curLoopElements = curCoreLastLoopIndicesElements_;
            }
            int64_t curExpertLoopOffset = indicesLoop * curCorePerLoopIndicesElements_;
            int64_t globalReadOffset = globalOffsetBase + curExpertLoopOffset;
            SetWaitFlag<HardEvent::S_MTE2>(HardEvent::S_MTE2);
            CopyExpertIn(globalReadOffset, curLoopElements);

            LocalTensor<int32_t> subRowIdxLocal = expandedRowIdxCopyInQueue_.DeQue<int32_t>();
            SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);

            for (int64_t colsLoop = 0; colsLoop < colsLoops_; colsLoop++) {
                int64_t curLoopCols = (colsLoop == colsLoops_ - 1) ? lastLoopCols_ : perLoopCols_;
                int64_t colsLoopOffset = colsLoop * perLoopCols_;
                int64_t curIndex = 0;
                int64_t globalSortIdx = globalReadOffset;
                int64_t startRow = globalSortIdx / k_;
                int64_t endRow = (globalSortIdx + curLoopElements - 1) / k_;

                for (int64_t row = startRow; row <= endRow; row++) {
                    SetWaitFlag<HardEvent::S_MTE2>(HardEvent::S_MTE2);
                    CopyXIn(row * cols_ + colsLoopOffset, row, curLoopCols);
                    LocalTensor<T> xLocal = xCopyInQueue_.DeQue<T>();

                    LocalTensor<float> scaleLocal;
                    if (isInputScale_ == 1 && colsLoop == 0) {
                        CopyScaleIn(row);
                        scaleLocal = scaleCopyInQueue_.DeQue<float>();
                    }

                    DataCopyExtParams copyXParams{1, static_cast<uint32_t>(curLoopCols * sizeof(T)), 0, 0, 0};
                    DataCopyExtParams copyScaleParams{1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
                    while (curIndex < curLoopElements && globalSortIdx / k_ == row) {
                        int64_t outPosition = subRowIdxLocal.GetValue(curIndex);
                        curIndex++;
                        globalSortIdx++;
                        if (outPosition < 0 || outPosition >= outputRows_) {
                            continue;
                        }
                        DataCopyPad(expandedXGm_[outPosition * cols_ + colsLoopOffset], xLocal, copyXParams);
                        if (isInputScale_ == 1 && colsLoop == 0) {
                            DataCopyPad(expandedScaleGm_[outPosition], scaleLocal, copyScaleParams);
                        }
                    }

                    if (isInputScale_ == 1 && colsLoop == 0) {
                        scaleCopyInQueue_.FreeTensor(scaleLocal);
                    }
                    xCopyInQueue_.FreeTensor(xLocal);
                }
            }
            expandedRowIdxCopyInQueue_.FreeTensor(subRowIdxLocal);
        }
    }
}
} // namespace MoeInitRoutingV3
#endif // MOE_V3_GATHER_OUT_DROPPAD_H_REGBASE_ARCH35
