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
 * \file moe_v3_gather_out_mxfp8.h
 * \brief
 */
#ifndef MOE_V3_GATHER_OUT_MXFP8_H_REGBASE
#define MOE_V3_GATHER_OUT_MXFP8_H_REGBASE

#include "moe_v3_common.h"
#include "kernel_operator.h"

namespace MoeInitRoutingV3 {
using namespace AscendC;

template <typename T>
class MoeV3GatherOutMxfp8 {
public:
    __aicore__ inline MoeV3GatherOutMxfp8(){};
    __aicore__ inline void Init(GM_ADDR xAddr, GM_ADDR unused_ScaleAddr, GM_ADDR workspace, GM_ADDR expandedRowIdxAddr,
                                GM_ADDR expandedXAddr, GM_ADDR expandedScaleAddr,
                                const MoeInitRoutingV3Arch35TilingData *tilingData, TPipe *tPipe);
    __aicore__ inline void Process();
    __aicore__ inline void CopyInExpandedExpertIdx(int64_t progress);
    __aicore__ inline void CopyIn(int64_t srcIdx, int64_t colIdx, int64_t loopCols);
    __aicore__ inline void CopyScaleIn(int64_t srcIdx, int64_t colIdx, int64_t loopCols);

private:
    __aicore__ inline void InitKernelTiling(GM_ADDR workspace, const MoeInitRoutingV3Arch35TilingData *tilingData);
    __aicore__ inline void ScatterCopyExpandedXandMXQuant(int64_t progress);
    __aicore__ inline void GatherCopyExpandedXandMXQuant(int64_t progress);
    TPipe *pipe_;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, GATHER_OUT_BUFFER_NUM> xInQueue_;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, GATHER_OUT_BUFFER_NUM> scaleCopyInQueue_;
    TQue<QuePosition::VECIN, GATHER_OUT_BUFFER_NUM> sortedRowIdxInQueue_;

    GlobalTensor<uint8_t> xInGm_;
    GlobalTensor<uint8_t> xGscaleGm_;
    GlobalTensor<uint8_t> expandedXOutGm_;
    GlobalTensor<int32_t> sortedRowIdxGm_;
    GlobalTensor<uint8_t> expandedScaleOutGm_;
    GlobalTensor<int32_t> expertTotalCountGm_;

    const MoeV3Arch35GatherOutComputeTilingData *gatherOutTilingData_;

    int64_t needCoreNum_;
    int64_t blockIdx_;
    int64_t cols_;
    int64_t scaleCols_; // 一个token的scale有多少个元素（列）
    int64_t n_;
    int64_t k_;
    int64_t perCoreRow_;
    int64_t currentLoopRows_;
    int64_t coreRows_;
    int64_t perLoopRows_;
    int64_t lastLoopRows_;
    int64_t rowLoops_;
    int64_t perLoopCols_;
    int64_t lastLoopCols_;
    int64_t colLoops_;
    int64_t perLoopScaleCols_;
    int64_t lastLoopScaleCols_;
    int64_t indicesOffset_;
    int64_t rowIdxType_ = 0;
    int64_t useGatherCopy_ = 0;
    int64_t isInputScale_ = 0;
    int64_t xCopyInQueueBufferNum_ = 2;
};

template <typename T>
__aicore__ inline void MoeV3GatherOutMxfp8<T>::InitKernelTiling(GM_ADDR workspace,
                                                                const MoeInitRoutingV3Arch35TilingData *tilingData)
{
    gatherOutTilingData_ = &(tilingData->gatherOutComputeParamsOp);
    cols_ = tilingData->cols;
    scaleCols_ = Ops::Base::CeilAlign<int64_t>(Ops::Base::CeilDiv<int64_t>(cols_, MX_BLOCK_SIZE), 2);
    n_ = tilingData->n;
    k_ = tilingData->k;
    rowIdxType_ = tilingData->rowIdxType;
    useGatherCopy_ = tilingData->useGatherCopy;
    isInputScale_ = tilingData->isInputScale;

    // core split
    int64_t actualExpertNum_ = tilingData->actualExpertNum;
    int64_t scanRowCount = n_ * k_;
    if (!useGatherCopy_) {
        expertTotalCountGm_.SetGlobalBuffer((__gm__ int32_t *)workspace + Align(n_ * k_, sizeof(int32_t)) * 2 +
                                                Align(actualExpertNum_, sizeof(int32_t)),
                                            1);
        DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(expertTotalCountGm_);
        scanRowCount = expertTotalCountGm_.GetValue(0);
    }
    perCoreRow_ = Ceil(scanRowCount, tilingData->coreNum);
    needCoreNum_ = Ceil(scanRowCount, perCoreRow_);
    int64_t lastCoreIndicesElements = scanRowCount - (needCoreNum_ - 1) * perCoreRow_;
    xCopyInQueueBufferNum_ = gatherOutTilingData_->xCopyInQueueBufferNum;

    // inner core split
    coreRows_ = perCoreRow_;
    int64_t originPerLoopElements = gatherOutTilingData_->perCorePerLoopIndicesElements;
    if (blockIdx_ == needCoreNum_ - 1) {
        coreRows_ = lastCoreIndicesElements;
        originPerLoopElements = gatherOutTilingData_->lastCorePerLoopIndicesElements;
    }
    perLoopRows_ = Min(coreRows_, originPerLoopElements);
    rowLoops_ = Ceil(coreRows_, perLoopRows_);
    lastLoopRows_ = coreRows_ - (rowLoops_ - 1) * perLoopRows_;

    // cols split
    perLoopCols_ = gatherOutTilingData_->perLoopCols;
    lastLoopCols_ = gatherOutTilingData_->lastLoopCols;
    colLoops_ = gatherOutTilingData_->colsLoops;
    perLoopScaleCols_ = perLoopCols_ / MX_BLOCK_SIZE; // perLoopCols_在tiling侧计算，已经对齐到32的整数倍了
    lastLoopScaleCols_ = scaleCols_ - (colLoops_ - 1) * perLoopScaleCols_;
}

template <typename T>
__aicore__ inline void MoeV3GatherOutMxfp8<T>::Init(GM_ADDR xAddr, GM_ADDR unused_ScaleAddr, GM_ADDR workspace,
                                                    GM_ADDR expandedRowIdxAddr, GM_ADDR expandedXAddr,
                                                    GM_ADDR expandedScaleAddr,
                                                    const MoeInitRoutingV3Arch35TilingData *tilingData, TPipe *tPipe)
{
#if (__NPU_ARCH__ == 3510)
    SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(0);
#endif

    pipe_ = tPipe;
    blockIdx_ = GetBlockIdx();
    InitKernelTiling(workspace, tilingData);

    xInGm_.SetGlobalBuffer((__gm__ uint8_t *)xAddr);
    xGscaleGm_.SetGlobalBuffer((__gm__ uint8_t *)unused_ScaleAddr);
    expandedXOutGm_.SetGlobalBuffer((__gm__ uint8_t *)expandedXAddr);
    if (useGatherCopy_) {
        if (rowIdxType_ == SCATTER) {
            sortedRowIdxGm_.SetGlobalBuffer(
                (__gm__ int32_t *)workspace + Align(n_ * k_, sizeof(int32_t)) + blockIdx_ * perCoreRow_,
                Align(perCoreRow_, sizeof(int32_t)));
        } else {
            sortedRowIdxGm_.SetGlobalBuffer((__gm__ int32_t *)expandedRowIdxAddr + blockIdx_ * perCoreRow_,
                                            Align(perCoreRow_, sizeof(int32_t)));
        }
    } else {
        if (rowIdxType_ == SCATTER) {
            sortedRowIdxGm_.SetGlobalBuffer((__gm__ int32_t *)expandedRowIdxAddr + blockIdx_ * perCoreRow_,
                                            Align(perCoreRow_, sizeof(int32_t)));
        } else {
            sortedRowIdxGm_.SetGlobalBuffer(
                (__gm__ int32_t *)workspace + Align(n_ * k_, sizeof(int32_t)) + blockIdx_ * perCoreRow_,
                Align(perCoreRow_, sizeof(int32_t)));
        }
    }

    expandedScaleOutGm_.SetGlobalBuffer((__gm__ uint8_t *)expandedScaleAddr);

    // perrows * 2 * 2 * 4 expandRowIdx + sortedExpertId
    pipe_->InitBuffer(sortedRowIdxInQueue_, GATHER_OUT_BUFFER_NUM, AlignBytes(perLoopRows_, sizeof(int32_t)));
    pipe_->InitBuffer(xInQueue_, xCopyInQueueBufferNum_, AlignBytes(perLoopCols_, sizeof(int8_t)));
    pipe_->InitBuffer(scaleCopyInQueue_, xCopyInQueueBufferNum_, AlignBytes(perLoopScaleCols_, sizeof(int8_t)));
}

template <typename T>
__aicore__ inline void MoeV3GatherOutMxfp8<T>::CopyInExpandedExpertIdx(int64_t progress)
{
    indicesOffset_ = progress * perLoopRows_;
    LocalTensor<int32_t> indicesLocal = sortedRowIdxInQueue_.AllocTensor<int32_t>();
    DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(currentLoopRows_ * sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams<int32_t> dataCopyPadParams{false, 0, 0, 0};
    DataCopyPad(indicesLocal, sortedRowIdxGm_[indicesOffset_], dataCopyParams, dataCopyPadParams);
    sortedRowIdxInQueue_.EnQue<int32_t>(indicesLocal);
}

template <typename T>
__aicore__ inline void MoeV3GatherOutMxfp8<T>::CopyIn(int64_t srcIdx, int64_t colIdx, int64_t loopCols)
{
    LocalTensor<uint8_t> inLocal = xInQueue_.AllocTensor<uint8_t>();
    DataCopyExtParams copyInParam = {1, static_cast<uint32_t>(loopCols * sizeof(uint8_t)), 0, 0, 0};
    DataCopyPadExtParams<uint8_t> padParams = {false, 0, 0, 0};
    DataCopyPad(inLocal, xInGm_[srcIdx * cols_ + colIdx * perLoopCols_], copyInParam, padParams);
    xInQueue_.EnQue(inLocal);
}

template <typename T>
__aicore__ inline void MoeV3GatherOutMxfp8<T>::CopyScaleIn(int64_t srcIdx, int64_t colIdx, int64_t loopCols)
{
    LocalTensor<uint8_t> scaleLocal = scaleCopyInQueue_.AllocTensor<uint8_t>();
    DataCopyExtParams copyParams1{static_cast<uint16_t>(1), static_cast<uint32_t>(loopCols * sizeof(uint8_t)), 0, 0, 0};
    DataCopyPadExtParams<uint8_t> padParams1{false, 0, 0, 0};
    DataCopyPad(scaleLocal, xGscaleGm_[srcIdx * scaleCols_ + colIdx * perLoopScaleCols_], copyParams1, padParams1);
    scaleCopyInQueue_.EnQue(scaleLocal);
}

template <typename T>
__aicore__ inline void MoeV3GatherOutMxfp8<T>::Process()
{
    if (blockIdx_ < needCoreNum_) {
        currentLoopRows_ = perLoopRows_;
        for (int64_t loop = 0; loop < rowLoops_ - 1; loop++) {
            CopyInExpandedExpertIdx(loop);
            if (!useGatherCopy_) {
                ScatterCopyExpandedXandMXQuant(loop);
            } else {
                GatherCopyExpandedXandMXQuant(loop);
            }
        }

        currentLoopRows_ = lastLoopRows_;
        CopyInExpandedExpertIdx(rowLoops_ - 1);
        if (!useGatherCopy_) {
            ScatterCopyExpandedXandMXQuant(rowLoops_ - 1);
        } else {
            GatherCopyExpandedXandMXQuant(rowLoops_ - 1);
        }
    }
}

template <typename T>
__aicore__ inline void MoeV3GatherOutMxfp8<T>::ScatterCopyExpandedXandMXQuant(int64_t progress)
{
    LocalTensor<int32_t> indicesLocal = sortedRowIdxInQueue_.DeQue<int32_t>();
    SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
    for (int64_t index = 0; index < currentLoopRows_; index++) {
        int32_t srcIdx = indicesLocal.GetValue(index);
        int64_t dstIdx = perCoreRow_ * blockIdx_ + perLoopRows_ * progress + index;
        for (int64_t j = 0; j < colLoops_; j++) {
            int64_t loopCols = (j == colLoops_ - 1) ? lastLoopCols_ : perLoopCols_;
            uint32_t loopScaleCols = (j == colLoops_ - 1) ? lastLoopScaleCols_ : perLoopScaleCols_;

            SetWaitFlag<HardEvent::S_MTE2>(HardEvent::S_MTE2);
            CopyIn(srcIdx / k_, j, loopCols);
            if (isInputScale_ == 1) {
                CopyScaleIn(srcIdx / k_, j, loopScaleCols);
            }

            LocalTensor<uint8_t> outLocal = xInQueue_.DeQue<uint8_t>();
            DataCopyExtParams copyOutParams = {1, static_cast<uint32_t>(loopCols * sizeof(uint8_t)), 0, 0, 0};
            int64_t outOffset = dstIdx * cols_ + j * perLoopCols_;
            DataCopyPad<uint8_t>(expandedXOutGm_[outOffset], outLocal, copyOutParams);
            xInQueue_.FreeTensor(outLocal);

            if (isInputScale_ == 1) {
                LocalTensor<uint8_t> mxScaleLocal = scaleCopyInQueue_.DeQue<uint8_t>();
                DataCopyExtParams copyScaleParams = {1, static_cast<uint32_t>(loopScaleCols * sizeof(uint8_t)), 0, 0,
                                                     0};
                int64_t outScaleOffset = dstIdx * scaleCols_ + j * perLoopScaleCols_;
                DataCopyPad<uint8_t>(expandedScaleOutGm_[outScaleOffset], mxScaleLocal, copyScaleParams);
                scaleCopyInQueue_.FreeTensor(mxScaleLocal);
            }
        }
    }
    sortedRowIdxInQueue_.FreeTensor(indicesLocal);
}

template <typename T>
__aicore__ inline void MoeV3GatherOutMxfp8<T>::GatherCopyExpandedXandMXQuant(int64_t progress)
{
    LocalTensor<int32_t> indicesLocal = sortedRowIdxInQueue_.DeQue<int32_t>();
    SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);

    for (int64_t j = 0; j < colLoops_; j++) {
        int64_t loopCols = (j == colLoops_ - 1) ? lastLoopCols_ : perLoopCols_;
        uint32_t loopScaleCols = (j == colLoops_ - 1) ? lastLoopScaleCols_ : perLoopScaleCols_;

        int64_t globalSortIdx = perCoreRow_ * blockIdx_ + perLoopRows_ * progress;
        int64_t curLoopRow = 0;
        int64_t currentLoopStartRow = globalSortIdx / k_;
        int64_t currentLoopLastRow = (globalSortIdx + currentLoopRows_ - 1) / k_;

        for (int64_t row = currentLoopStartRow; row <= currentLoopLastRow; row++) {
            bool hasValidOut = false;
            while (curLoopRow < currentLoopRows_ && globalSortIdx / k_ == row) {
                if (indicesLocal.GetValue(curLoopRow) >= 0) {
                    hasValidOut = true;
                    break;
                }
                curLoopRow++;
                globalSortIdx++;
            }
            if (!hasValidOut) {
                continue;
            }

            SetWaitFlag<HardEvent::S_MTE2>(HardEvent::S_MTE2);
            CopyIn(row, j, loopCols);
            if (isInputScale_ == 1) {
                CopyScaleIn(row, j, loopScaleCols);
            }

            LocalTensor<uint8_t> outLocal = xInQueue_.DeQue<uint8_t>();
            LocalTensor<uint8_t> mxScaleLocal;
            if (isInputScale_ == 1) {
                mxScaleLocal = scaleCopyInQueue_.DeQue<uint8_t>();
            }

            DataCopyExtParams copyOutParams = {1, static_cast<uint32_t>(loopCols * sizeof(uint8_t)), 0, 0, 0};
            DataCopyExtParams copyScaleParams = {1, static_cast<uint32_t>(loopScaleCols * sizeof(uint8_t)), 0, 0, 0};
            while (curLoopRow < currentLoopRows_ && globalSortIdx / k_ == row) {
                int32_t outIndex = indicesLocal.GetValue(curLoopRow);
                curLoopRow++;
                globalSortIdx++;
                if (outIndex < 0) {
                    continue;
                }
                int64_t outOffset = static_cast<int64_t>(outIndex) * cols_ + j * perLoopCols_;
                DataCopyPad<uint8_t>(expandedXOutGm_[outOffset], outLocal, copyOutParams);

                if (isInputScale_ == 1) {
                    int64_t outScaleOffset = static_cast<int64_t>(outIndex) * scaleCols_ + j * perLoopScaleCols_;
                    DataCopyPad<uint8_t>(expandedScaleOutGm_[outScaleOffset], mxScaleLocal, copyScaleParams);
                }
            }

            xInQueue_.FreeTensor(outLocal);
            if (isInputScale_ == 1) {
                scaleCopyInQueue_.FreeTensor(mxScaleLocal);
            }
        }
    }
    sortedRowIdxInQueue_.FreeTensor(indicesLocal);
}

} // namespace MoeInitRoutingV3
#endif // MOE_V3_GATHER_OUT_MXFP8_H_REGBASE
