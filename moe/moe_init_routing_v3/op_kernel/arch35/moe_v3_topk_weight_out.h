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
 * \file moe_v3_topk_weight_out.h
 * \brief TopkWeight rearrangement for non-FullLoad path (arch35/regbase)
 */
#ifndef MOE_V3_TOPK_WEIGHT_OUT_H
#define MOE_V3_TOPK_WEIGHT_OUT_H

#include "moe_v3_common.h"

namespace MoeInitRoutingV3 {
using namespace AscendC;

constexpr int32_t ROW_IDX_QUEUE_DEPTH = 2;
constexpr int32_t ROW_IDX_BUFFER_NUM = 2;

class MoeV3TopkWeightOut {
public:
    __aicore__ inline MoeV3TopkWeightOut() {}
    __aicore__ inline void Init(GM_ADDR topkWeight, GM_ADDR expandedRowIdx, GM_ADDR expandedTopkWeight,
                                GM_ADDR workspace, const MoeInitRoutingV3Arch35TilingData *tilingData, TPipe *tPipe);
    __aicore__ inline void Process();

private:
    TPipe *pipe_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> topkWeightCopyInQueue_;
    TQue<QuePosition::VECIN, ROW_IDX_QUEUE_DEPTH> rowIdxCopyInQueue_;

    GlobalTensor<float> topkWeightGm_;
    GlobalTensor<float> expandedTopkWeightGm_;
    GlobalTensor<int32_t> expandDstToSrcRowGm_;
    GlobalTensor<int32_t> expandedRowIdxGm_;

    int64_t blockIdx_;
    int64_t n_;
    int64_t k_;
    int64_t totalLength_;
    int64_t activeNum_;
    int64_t expertTotalCount_;
    int64_t rowIdxType_;
    int64_t dropPadMode_;
    int64_t expertNum_;
    int64_t expertCapacity_;
    int64_t outputRows_;
    int64_t needCoreNum_;
    int64_t perCoreElements_;
    int64_t coreElements_;
    int64_t coreNum_;
    int64_t perCorePerLoopElements_;
    int64_t indicesLoops_;

    __aicore__ inline void CopyRowIdxIn(int64_t offset, int64_t curLoopElements);
    __aicore__ inline void ProcessDropPadGather(int64_t startIdx);
    __aicore__ inline void ProcessDroplessScatter(int64_t startIdx);
};

__aicore__ inline void MoeV3TopkWeightOut::Init(GM_ADDR topkWeight, GM_ADDR expandedRowIdx,
                                                GM_ADDR expandedTopkWeight, GM_ADDR workspace,
                                                const MoeInitRoutingV3Arch35TilingData *tilingData, TPipe *tPipe)
{
    pipe_ = tPipe;
    blockIdx_ = GetBlockIdx();
    coreNum_ = tilingData->coreNum;
    n_ = tilingData->n;
    k_ = tilingData->k;
    totalLength_ = n_ * k_;
    activeNum_ = tilingData->activeNum;
    rowIdxType_ = tilingData->rowIdxType;
    dropPadMode_ = tilingData->dropPadMode;
    expertNum_ = tilingData->expertNum;
    expertCapacity_ = tilingData->expertCapacity;

    outputRows_ = (dropPadMode_ == DROP_PAD_MODE) ? expertNum_ * expertCapacity_ : activeNum_;

    topkWeightGm_.SetGlobalBuffer((__gm__ float *)topkWeight, totalLength_);
    expandedTopkWeightGm_.SetGlobalBuffer((__gm__ float *)expandedTopkWeight, outputRows_);

    if (dropPadMode_ != DROP_PAD_MODE) {
        GlobalTensor<int32_t> expertTotalCountGm;
        expertTotalCountGm.SetGlobalBuffer((__gm__ int32_t *)workspace +
            Align(totalLength_, sizeof(int32_t)) * 2 + Align(tilingData->actualExpertNum, sizeof(int32_t)), 1);
        AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
            AscendC::DcciDst::CACHELINE_OUT>(expertTotalCountGm);
        expertTotalCount_ = expertTotalCountGm.GetValue(0);

        if (rowIdxType_ == SCATTER) {
            expandDstToSrcRowGm_.SetGlobalBuffer((__gm__ int32_t *)expandedRowIdx, totalLength_);
        } else {
            expandDstToSrcRowGm_.SetGlobalBuffer((__gm__ int32_t *)workspace +
                Align(totalLength_, sizeof(int32_t)), Align(totalLength_, sizeof(int32_t)));
        }
    } else {
        expandedRowIdxGm_.SetGlobalBuffer((__gm__ int32_t *)expandedRowIdx, totalLength_);
    }

    int64_t splitBase;
    if (dropPadMode_ == DROP_PAD_MODE) {
        splitBase = totalLength_;
    } else {
        splitBase = expertTotalCount_;
    }
    perCoreElements_ = Ceil(splitBase, coreNum_);
    needCoreNum_ = Ceil(splitBase, perCoreElements_);
    if (blockIdx_ == needCoreNum_ - 1) {
        coreElements_ = splitBase - (needCoreNum_ - 1) * perCoreElements_;
    } else {
        coreElements_ = perCoreElements_;
    }

    perCorePerLoopElements_ = tilingData->topkWeightOutPerCorePerLoopElements;
    indicesLoops_ = Ceil(coreElements_, perCorePerLoopElements_);

    pipe_->InitBuffer(topkWeightCopyInQueue_, 1, AlignBytes(1, sizeof(float)));
    pipe_->InitBuffer(rowIdxCopyInQueue_, ROW_IDX_BUFFER_NUM, AlignBytes(perCorePerLoopElements_, sizeof(int32_t)));
}

__aicore__ inline void MoeV3TopkWeightOut::CopyRowIdxIn(int64_t offset, int64_t curLoopElements)
{
    LocalTensor<int32_t> rowIdxLocal = rowIdxCopyInQueue_.AllocTensor<int32_t>();
    DataCopyExtParams idxCopyParams{1, static_cast<uint32_t>(curLoopElements * sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams<int32_t> idxPadParams{false, 0, 0, 0};
    if (dropPadMode_ == DROP_PAD_MODE) {
        DataCopyPad(rowIdxLocal, expandedRowIdxGm_[offset], idxCopyParams, idxPadParams);
    } else {
        DataCopyPad(rowIdxLocal, expandDstToSrcRowGm_[offset], idxCopyParams, idxPadParams);
    }
    rowIdxCopyInQueue_.EnQue(rowIdxLocal);
}

__aicore__ inline void MoeV3TopkWeightOut::ProcessDropPadGather(int64_t startIdx)
{
    DataCopyExtParams copyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
    DataCopyPadExtParams<float> padParams{false, 0, 0, 0};

    for (int64_t loop = 0; loop < indicesLoops_; loop++) {
        int64_t curLoopElements = (loop == indicesLoops_ - 1)
                                      ? coreElements_ - loop * perCorePerLoopElements_
                                      : perCorePerLoopElements_;
        int64_t offset = startIdx + loop * perCorePerLoopElements_;

        CopyRowIdxIn(offset, curLoopElements);
        LocalTensor<int32_t> rowIdxLocal = rowIdxCopyInQueue_.DeQue<int32_t>();
        SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);

        for (int64_t i = 0; i < curLoopElements; i++) {
            int64_t dstIdx = rowIdxLocal.GetValue(i);
            if (dstIdx >= 0 && dstIdx < outputRows_) {
                int64_t globalSortIdx = offset + i;
                LocalTensor<float> topkWeightLocal = topkWeightCopyInQueue_.AllocTensor<float>();
                DataCopyPad(topkWeightLocal, topkWeightGm_[globalSortIdx], copyParams, padParams);
                topkWeightCopyInQueue_.EnQue(topkWeightLocal);
                topkWeightLocal = topkWeightCopyInQueue_.DeQue<float>();
                DataCopyPad(expandedTopkWeightGm_[dstIdx], topkWeightLocal, copyParams);
                topkWeightCopyInQueue_.FreeTensor(topkWeightLocal);
            }
        }
        rowIdxCopyInQueue_.FreeTensor(rowIdxLocal);
    }
}

__aicore__ inline void MoeV3TopkWeightOut::ProcessDroplessScatter(int64_t startIdx)
{
    DataCopyExtParams copyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
    DataCopyPadExtParams<float> padParams{false, 0, 0, 0};

    for (int64_t loop = 0; loop < indicesLoops_; loop++) {
        int64_t curLoopElements = (loop == indicesLoops_ - 1)
                                    ? coreElements_ - loop * perCorePerLoopElements_
                                    : perCorePerLoopElements_;
        int64_t offset = startIdx + loop * perCorePerLoopElements_;

        CopyRowIdxIn(offset, curLoopElements);
        LocalTensor<int32_t> rowIdxLocal = rowIdxCopyInQueue_.DeQue<int32_t>();
        SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);

        for (int64_t i = 0; i < curLoopElements; i++) {
            int64_t originalPos = rowIdxLocal.GetValue(i);
            if (originalPos >= 0 && originalPos < totalLength_) {
                int64_t curIndex = offset + i;
                LocalTensor<float> topkWeightLocal = topkWeightCopyInQueue_.AllocTensor<float>();
                DataCopyPad(topkWeightLocal, topkWeightGm_[originalPos], copyParams, padParams);
                topkWeightCopyInQueue_.EnQue(topkWeightLocal);
                topkWeightLocal = topkWeightCopyInQueue_.DeQue<float>();
                DataCopyPad(expandedTopkWeightGm_[curIndex], topkWeightLocal, copyParams);
                topkWeightCopyInQueue_.FreeTensor(topkWeightLocal);
            }
        }
        rowIdxCopyInQueue_.FreeTensor(rowIdxLocal);
    }
}

__aicore__ inline void MoeV3TopkWeightOut::Process()
{
    int64_t startIdx = blockIdx_ * perCoreElements_;

    if (dropPadMode_ == DROP_PAD_MODE) {
        int64_t perCoreZeroRows = Ceil(outputRows_, coreNum_);
        int64_t startRow = blockIdx_ * perCoreZeroRows;
        int64_t endRow = Min(startRow + perCoreZeroRows, outputRows_);
        if (startRow < endRow) {
            GlobalTensor<float> zeroGm;
            zeroGm.SetGlobalBuffer(
                (__gm__ float *)expandedTopkWeightGm_.GetPhyAddr() + startRow,
                endRow - startRow);
            SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
            InitGlobalMemory(zeroGm, endRow - startRow, 0.0f);
            SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
        }
#ifndef __CCE_KT_TEST__
        AscendC::SyncAll();
#endif

        if (blockIdx_ < needCoreNum_) {
            ProcessDropPadGather(startIdx);
        }
        return;
    }

    if (blockIdx_ < needCoreNum_) {
        ProcessDroplessScatter(startIdx);
    }
}

} // namespace MoeInitRoutingV3
#endif // MOE_V3_TOPK_WEIGHT_OUT_H
