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
 * \file moe_v2_gather_quant_simt.h
 * \brief
 */
#ifndef MOE_V2_QUANT_GATHER_QUANT_H
#define MOE_V2_QUANT_GATHER_QUANT_H

#include "moe_init_routing_quant_v2_arch35_common.h"
#include "kernel_operator.h"

namespace MoeInitRoutingQuantV2 {
using namespace AscendC;
using namespace MoeInitRoutingQuantV2;

constexpr int64_t BUFFER_NUM = 2;

template <typename T>
class MoeV2GatherQuant {
public:
    __aicore__ inline MoeV2GatherQuant(){};
    __aicore__ inline void Init(GM_ADDR inputX, GM_ADDR scale, GM_ADDR offset, GM_ADDR expandedRowIdx,
                                GM_ADDR expandedX, GM_ADDR workspace, const MoeInitRoutingQuantV2TilingData *tilingData,
                                TPipe *tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyInIndices(int64_t progress);
    __aicore__ inline void Compute();
    __aicore__ inline void CopyOut(int64_t progress);
    __aicore__ inline void CopyInZeroIndices(int64_t progress);
    __aicore__ inline void CopyOutZero(int64_t progress);

private:
    TPipe *pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputXCopyInQueue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> expandRowIdxCopyInQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> inputXCopyOutQueue_;
    TQue<QuePosition::VECOUT, 1> floatQueue_;
    TQue<QuePosition::VECOUT, 1> halfQueue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> expandedRowIdxIndexCopyInQueue_;

    GlobalTensor<T> inputXGm_;
    GlobalTensor<int8_t> expandedXGm_;
    GlobalTensor<int32_t> expandedRowIdxGm_;
    GlobalTensor<float> scaleGm_;
    GlobalTensor<float> offsetGm_;
    GlobalTensor<int32_t> expandedRowIdxIndexGm_;

    const InnerMoeV2GatherOutComputeTilingData *gatherOutTilingData_;

    int64_t needCoreNum_;
    int64_t blockIdx_;
    int64_t cols_;
    int64_t n_;
    int64_t k_;
    int64_t activateRows_;
    int64_t currentLoopRows_;
    int64_t coreRows_;
    int64_t perLoopRows_;
    int64_t lastLoopRows_;
    int64_t rowLoops_;
    int64_t colsTileLength_;
    int64_t perLoopCols_;
    int64_t lastLoopCols_;
    int64_t colLoops_;
    int64_t dropPadMode_;
    float scale_;
    float offset_;
    int64_t expertNum_;
    int64_t totalLength_;

    int64_t indicesOffset_;
    int64_t inputOffset_;
    int64_t outOffset_;
};

template <typename T>
__aicore__ inline void MoeV2GatherQuant<T>::CopyInIndices(int64_t progress)
{
    indicesOffset_ = progress * perLoopRows_;
    LocalTensor<int32_t> indicesLocal = expandRowIdxCopyInQueue_.AllocTensor<int32_t>();
    DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(currentLoopRows_ * sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams<int32_t> dataCopyPadParams{false, 0, 0, 0};
    DataCopyPad(indicesLocal, expandedRowIdxGm_[indicesOffset_], dataCopyParams, dataCopyPadParams);
    expandRowIdxCopyInQueue_.EnQue<int32_t>(indicesLocal);
}

template <typename T>
__aicore__ inline void MoeV2GatherQuant<T>::CopyInZeroIndices(int64_t progress)
{
    indicesOffset_ = progress * perLoopRows_;
    LocalTensor<int32_t> expandedRowIdxIndexLocal = expandedRowIdxIndexCopyInQueue_.AllocTensor<int32_t>();
    DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>((currentLoopRows_ + 1) * sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams<int32_t> dataCopyPadParams{false, 0, 0, 0};
    DataCopyPad(expandedRowIdxIndexLocal, expandedRowIdxIndexGm_[indicesOffset_], dataCopyParams, dataCopyPadParams);
    expandedRowIdxIndexCopyInQueue_.EnQue<int32_t>(expandedRowIdxIndexLocal);
}

template <typename T>
__aicore__ inline void MoeV2GatherQuant<T>::CopyOutZero(int64_t progress)
{
    LocalTensor<int32_t> indicesLocal = expandedRowIdxIndexCopyInQueue_.DeQue<int32_t>();
    if (blockIdx_ == 0) {
        int32_t curIndex = 0;
        int32_t nextIndex = indicesLocal.GetValue(0);
        int32_t count = nextIndex - curIndex;
        if (count > 0) {
            InitOutput(expandedXGm_[curIndex * cols_], count * cols_, static_cast<int8_t>(0));
        }
    }
    for (int i = 0; i < currentLoopRows_; i++) {
        int32_t curIndex = indicesLocal.GetValue(i) + 1;
        if (activateRows_ <= curIndex) {
            break;
        }
        int32_t nextIndex;
        if (blockIdx_ == gatherOutTilingData_->needCoreNum - 1 && progress == rowLoops_ - 1 &&
            i == currentLoopRows_ - 1) {
            nextIndex = activateRows_;
        } else {
            nextIndex = indicesLocal.GetValue(i + 1);
        }
        int32_t count = nextIndex - curIndex;
        if (count > 0) {
            InitOutput(expandedXGm_[curIndex * cols_], count * cols_, static_cast<int8_t>(0));
        }
    }
    expandedRowIdxIndexCopyInQueue_.FreeTensor(indicesLocal);
}

template <typename T>
__aicore__ inline void MoeV2GatherQuant<T>::Compute()
{
    LocalTensor<T> inLocal = inputXCopyInQueue_.DeQue<T>();
    LocalTensor<int8_t> outLocal = inputXCopyOutQueue_.AllocTensor<int8_t>();
    LocalTensor<float> floatLocal = floatQueue_.AllocTensor<float>();
    LocalTensor<half> halfLocal = halfQueue_.AllocTensor<half>();
    uint32_t elements = Align(colsTileLength_, sizeof(T));
    if constexpr (IsSameType<T, bfloat16_t>::value) {
        Cast(floatLocal, inLocal, RoundMode::CAST_NONE, colsTileLength_);
        TQueSync<PIPE_MTE3, PIPE_V> sync;
        sync.SetFlag(0);
        sync.WaitFlag(0);
        Cast(halfLocal, floatLocal, RoundMode::CAST_NONE, elements);
        Muls(halfLocal, halfLocal, static_cast<half>(scale_), elements);
        Adds(halfLocal, halfLocal, static_cast<half>(offset_), elements);
        LocalTensor<int32_t> intLocal = floatLocal.ReinterpretCast<int32_t>();
        Cast(intLocal, halfLocal, RoundMode::CAST_RINT, elements);
        SetDeqScale((half)1.000000e+00f);
        Cast(halfLocal, intLocal, RoundMode::CAST_RINT, elements);
        Cast(outLocal, halfLocal, RoundMode::CAST_RINT, elements);
    } else if constexpr (IsSameType<T, float>::value) {
        Cast(halfLocal, inLocal, RoundMode::CAST_NONE, elements);
        Muls(halfLocal, halfLocal, static_cast<half>(scale_), elements);
        Adds(halfLocal, halfLocal, static_cast<half>(offset_), elements);
        Cast(outLocal, halfLocal, RoundMode::CAST_RINT, elements);
    } else {
        Muls(inLocal, inLocal, static_cast<T>(scale_), elements);
        Adds(inLocal, inLocal, static_cast<T>(offset_), elements);
        Cast(outLocal, inLocal, RoundMode::CAST_RINT, elements);
    }
    inputXCopyOutQueue_.EnQue(outLocal);
    floatQueue_.FreeTensor(floatLocal);
    halfQueue_.FreeTensor(halfLocal);
}

template <typename T>
__aicore__ inline void MoeV2GatherQuant<T>::CopyOut(int64_t progress)
{
    LocalTensor<int32_t> indicesLocal = expandRowIdxCopyInQueue_.DeQue<int32_t>();
    SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
    colsTileLength_ = perLoopCols_;
    for (int64_t colsLoop = 0; colsLoop < colLoops_; colsLoop++) {
        int64_t initialRow = gatherOutTilingData_->perCoreRows * blockIdx_ + perLoopRows_ * progress;
        int64_t curLoopRow = 0;
        if (colsLoop == colLoops_ - 1) {
            colsTileLength_ = lastLoopCols_;
        }
        int64_t currentLoopStartRow = initialRow / k_;
        int64_t currentLoopLastRow = (initialRow + currentLoopRows_ - 1) / k_;
        for (int64_t row = currentLoopStartRow; row <= currentLoopLastRow; row++) {
            LocalTensor<T> inLocal = inputXCopyInQueue_.AllocTensor<T>();
            // input row position
            inputOffset_ = row * cols_ + colsLoop * perLoopCols_;
            DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(colsTileLength_ * sizeof(T)), 0, 0, 0};
            DataCopyPadExtParams<T> dataCopyPadParams{false, 0, 0, 0};
            DataCopyPad(inLocal, inputXGm_[inputOffset_], dataCopyParams, dataCopyPadParams);
            inputXCopyInQueue_.EnQue<T>(inLocal);
            Compute();
            LocalTensor<int8_t> outLocal = inputXCopyOutQueue_.DeQue<int8_t>();
            DataCopyExtParams intriParams{1, static_cast<uint32_t>(colsTileLength_ * sizeof(int8_t)), 0, 0, 0};
            while (curLoopRow < currentLoopRows_ && initialRow / k_ == row) {
                int32_t outIndex = indicesLocal.GetValue(curLoopRow);
                curLoopRow++;
                initialRow++;
                if (outIndex == -1 || (dropPadMode_ == DROPLESS_MODE && outIndex >= activateRows_)) {
                    continue;
                }
                outOffset_ = outIndex * cols_ + colsLoop * perLoopCols_;
                DataCopyPad(expandedXGm_[outOffset_], outLocal, intriParams);
            }
            inputXCopyInQueue_.FreeTensor(inLocal);
            inputXCopyOutQueue_.FreeTensor(outLocal);
        }
    }
    expandRowIdxCopyInQueue_.FreeTensor(indicesLocal);
}

template <typename T>
__aicore__ inline void MoeV2GatherQuant<T>::Init(GM_ADDR inputX, GM_ADDR scale, GM_ADDR offset, GM_ADDR expandedRowIdx,
                                                 GM_ADDR expandedX, GM_ADDR workspace,
                                                 const MoeInitRoutingQuantV2TilingData *tilingData, TPipe *tPipe)
{
    pipe_ = tPipe;
    blockIdx_ = GetBlockIdx();
    gatherOutTilingData_ = &(tilingData->gatherOutComputeParamsOp);

    needCoreNum_ = gatherOutTilingData_->needCoreNum;
    activateRows_ = gatherOutTilingData_->activateRows;
    cols_ = tilingData->cols;
    n_ = tilingData->n;
    k_ = tilingData->k;
    dropPadMode_ = tilingData->dropPadMode;
    expertNum_ = tilingData->expertNum;
    totalLength_ = tilingData->n * tilingData->k;

    if (dropPadMode_ == 1) {
        activateRows_ = tilingData->expertCapacity * tilingData->expertNum;
    }

    if (blockIdx_ == gatherOutTilingData_->needCoreNum - 1) {
        coreRows_ = gatherOutTilingData_->lastCoreRows;
        perLoopRows_ = gatherOutTilingData_->lastCorePerLoopRows;
        lastLoopRows_ = gatherOutTilingData_->lastCoreLastLoopRows;
        rowLoops_ = gatherOutTilingData_->lastCoreLoops;
    } else {
        coreRows_ = gatherOutTilingData_->perCoreRows;
        perLoopRows_ = gatherOutTilingData_->perCorePerLoopRows;
        lastLoopRows_ = gatherOutTilingData_->perCoreLastLoopRows;
        rowLoops_ = gatherOutTilingData_->perCoreLoops;
    }
    perLoopCols_ = gatherOutTilingData_->perLoopCols;
    lastLoopCols_ = gatherOutTilingData_->lastLoopCols;
    colLoops_ = gatherOutTilingData_->colLoops;

    inputXGm_.SetGlobalBuffer((__gm__ T *)inputX);
    expandedXGm_.SetGlobalBuffer((__gm__ int8_t *)expandedX);
    expandedRowIdxGm_.SetGlobalBuffer((__gm__ int32_t *)expandedRowIdx + blockIdx_ * gatherOutTilingData_->perCoreRows,
                                      Align(coreRows_, sizeof(int32_t)));
    scaleGm_.SetGlobalBuffer((__gm__ float *)scale, 1);
    offsetGm_.SetGlobalBuffer((__gm__ float *)offset, 1);
    expandedRowIdxIndexGm_.SetGlobalBuffer((__gm__ int32_t *)workspace + Align(totalLength_, sizeof(int32_t)) * 2 +
                                               expertNum_ + blockIdx_ * gatherOutTilingData_->perCoreRows,
                                           0);
    scale_ = scaleGm_.GetValue(0);
    offset_ = offsetGm_.GetValue(0);

    pipe_->InitBuffer(inputXCopyInQueue_, BUFFER_NUM, AlignBytes(perLoopCols_, sizeof(T)));
    pipe_->InitBuffer(inputXCopyOutQueue_, BUFFER_NUM, AlignBytes(perLoopCols_, sizeof(int8_t)));
    pipe_->InitBuffer(expandRowIdxCopyInQueue_, BUFFER_NUM, AlignBytes(perLoopRows_, sizeof(int32_t)));
    pipe_->InitBuffer(floatQueue_, 1, AlignBytes(Align(perLoopCols_, sizeof(T)), sizeof(float)));
    pipe_->InitBuffer(halfQueue_, 1, AlignBytes(perLoopCols_, sizeof(half)));
    pipe_->InitBuffer(expandedRowIdxIndexCopyInQueue_, BUFFER_NUM, AlignBytes(perLoopRows_ + 1, sizeof(int32_t)));
}

template <typename T>
__aicore__ inline void MoeV2GatherQuant<T>::Process()
{
    if (blockIdx_ < needCoreNum_) {
        currentLoopRows_ = perLoopRows_;
        for (int64_t loop = 0; loop < rowLoops_; loop++) {
            if (loop == rowLoops_ - 1) {
                currentLoopRows_ = lastLoopRows_;
            }

            if (dropPadMode_ == DROP_PAD_MODE) {
                CopyInZeroIndices(loop);
                CopyOutZero(loop);
            }
            CopyInIndices(loop);
            CopyOut(loop);
        }
    }
}
} // namespace MoeInitRoutingQuantV2
#endif // MOE_V2_QUANT_GATHER_QUANT_H
