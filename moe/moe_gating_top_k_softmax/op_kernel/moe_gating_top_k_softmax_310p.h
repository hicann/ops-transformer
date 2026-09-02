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
 * \file moe_gating_top_k_softmax_310p.h
 * \brief
 */
#include "kernel_operator.h"

namespace MoeGatingTopKSoftmax {
using namespace AscendC;

namespace {
constexpr int64_t ALIGN_FACTOR = 16;
const constexpr uint32_t FP16_PER_REPEAT = 16;
const constexpr uint32_t MAX_UB_SIZE = 245760;
const constexpr uint32_t BLOCK_SIZE = 8;
const constexpr uint32_t BLOCK_BYTES = 32;
} // namespace

template <typename T1, typename T2> // T1 T1 : gating, y (half float16)  // T2 : expertIdx (int32)
class MoeGatingTopKSoftmax310P {
public:
    __aicore__ inline MoeGatingTopKSoftmax310P<T1, T1, T2>(){};

    __aicore__ inline int32_t GetBlock()
    {
        return counter;
    }

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR expertIdx, GM_ADDR rowIdx, GM_ADDR workspace,
                                const MoeGatingTopKSoftmax310PTilingData &tilingData, AscendC::TPipe *pipeIn,
                                int32_t count)
    {
        counter = count;
        ParesTiling(&tilingData);
        if (GetBlock() >= activateCore_) {
            return;
        }
        int32_t rowWork_ = (GetBlock() < activateCore_ - 1) ? oneCoreRow_ : tailRow_;
        int32_t WorkspaceSize = 256 * sizeof(int32_t);
        pipe = pipeIn;
        gmXOffset_ = oneCoreRow_ * numCol_; // input data offset
        gmYOffset_ = oneCoreRow_ * k_;      //* numCol_;
        gmExpertIdxOffset_ = oneCoreRow_ * k_;
        gmX_.SetGlobalBuffer((__gm__ T1 *)x + GetBlock() * gmXOffset_, gmXOffset_);
        gmY_.SetGlobalBuffer((__gm__ T1 *)y + GetBlock() * gmYOffset_, gmYOffset_);
        gmExpertIdx_.SetGlobalBuffer((__gm__ T2 *)expertIdx + GetBlock() * gmExpertIdxOffset_, gmExpertIdxOffset_);
        gmRowIdx_.SetGlobalBuffer((__gm__ T2 *)rowIdx + GetBlock() * gmExpertIdxOffset_, gmExpertIdxOffset_);

        gmSync_.SetGlobalBuffer((__gm__ int32_t *)workspace, WorkspaceSize);
        gmYTmp_.SetGlobalBuffer(
            (__gm__ T1 *)workspace + WorkspaceSize / sizeof(T1) + GetBlock() * oneCoreRow_ * kAlign_,
            rowWork_ * kAlign_);
        gmExpertTmp_.SetGlobalBuffer((__gm__ T2 *)workspace + WorkspaceSize / sizeof(T2) +
                                         numRow_ * kAlign_ * sizeof(T1) / sizeof(T2) +
                                         GetBlock() * oneCoreRow_ * kAlign_,
                                     rowWork_ * kAlign_);
        gmRowTmp_.SetGlobalBuffer((__gm__ T2 *)workspace + WorkspaceSize / sizeof(T2) +
                                      numRow_ * kAlign_ * sizeof(T1) / sizeof(T2) + GetBlock() * oneCoreRow_ * kAlign_ +
                                      rowWork_ * kAlign_,
                                  rowWork_ * kAlign_);
        offset0 = 0;                                            // input x
        offset1 = offset0 + oneCoreRow_ * numCol_ * sizeof(T1); // output Y
        offset2 = offset1 + oneCoreRow_ * kAlign_ * sizeof(T1); // output ExpertIdx
        offset3 = offset2 + oneCoreRow_ * kAlign_ * sizeof(T2);
        offset4 = offset3 + oneCoreRow_ * numCol_ * sizeof(T1);
        offset5 = offset4 + oneCoreRow_ * kAlign_ * sizeof(T2);
        offset6 = offset5 + 0; // to the end

        pipe->InitBuffer(bufUb_, MAX_UB_SIZE - WorkspaceSize);
        pipe->InitBuffer(syncIn, WorkspaceSize);
        xLT = bufUb_.GetWithOffset<T1>(1, offset0);
        yLT = bufUb_.GetWithOffset<T1>(1, offset1);
        expertIdxLT = bufUb_.GetWithOffset<T2>(1, offset2);
        rowIdxLT = bufUb_.GetWithOffset<T2>(1, offset5);
        yTmpLT = bufUb_.GetWithOffset<T1>(1, offset3);
        tmpLT = bufUb_.GetWithOffset<T1>(1, offset4);
        syncLT = syncIn.Get<int32_t>();
    }

    __aicore__ inline void ParesTiling(const MoeGatingTopKSoftmax310PTilingData *__restrict tilingData)
    {
        tilingKey = tilingData->tilingKey;
        numRow_ = tilingData->row;
        numCol_ = tilingData->col;
        k_ = tilingData->k;
        kAlign_ = tilingData->kAlign;
        numCore_ = tilingData->blockNum;

        oneCoreRow_ = tilingData->oneCoreRow;
        activateCore_ = tilingData->activateCore;
        tailRow_ = tilingData->tailRow;

        hasFinished_ = tilingData->hasFinished;

        workspaceSize_ = tilingData->workspaceSize;
        FormerSoftmaxTilingData = tilingData->FormerSoftmaxTilingData;
        TailSoftmaxTilingData = tilingData->TailSoftmaxTilingData;

        FormerTopkTilingData = tilingData->FormerTopkTilingData;
        TailTopkTilingData = tilingData->TailTopkTilingData;

        tmp_minsize = tilingData->FormerTmpMinsize;
    }

    __aicore__ inline void Process()
    {
        if (GetBlock() >= activateCore_)
            return;
        CopyIn();
        AscendC::SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        Compute();
        AscendC::SetFlag<HardEvent::V_MTE3>(EVENT_ID1);
        AscendC::WaitFlag<HardEvent::V_MTE3>(EVENT_ID1);
        CopyOut();
    }

    __aicore__ inline int64_t Align16(const int64_t elementNum)
    {
        return (elementNum + ALIGN_FACTOR - 1) / ALIGN_FACTOR * ALIGN_FACTOR;
    }

    __aicore__ inline void CopyIn()
    {
        int32_t rowWork_ = (GetBlock() < activateCore_ - 1) ? oneCoreRow_ : tailRow_;
        DataCopy<T1>(xLT, gmX_, rowWork_ * numCol_);
    }

    __aicore__ inline void Compute()
    {
        int32_t rowWork_ = (GetBlock() < activateCore_ - 1) ? oneCoreRow_ : tailRow_;
        AscendC::SoftMaxShapeInfo softmaxShapeInfoData;
        softmaxShapeInfoData.srcK = Align16(numCol_);
        softmaxShapeInfoData.srcM = rowWork_;
        softmaxShapeInfoData.oriSrcK = numCol_;
        softmaxShapeInfoData.oriSrcM = rowWork_;
        SoftMaxTiling *softmaxTilingData =
            (GetBlock() < activateCore_ - 1) ? &FormerSoftmaxTilingData : &TailSoftmaxTilingData;
        TopKInfo topkInfo;
        topkInfo.outter = rowWork_;
        topkInfo.inner = numCol_;
        topkInfo.n = numCol_;
        TopkTiling *TopkTilingData = (GetBlock() < activateCore_ - 1) ? &FormerTopkTilingData : &TailTopkTilingData;
        AscendC::LocalTensor<T1> sumTmpLT = tmpLT;
        AscendC::LocalTensor<T1> maxTmpLT = sumTmpLT[oneCoreRow_ * FP16_PER_REPEAT];
        AscendC::LocalTensor<T1> softmaxTmpLT = maxTmpLT[oneCoreRow_ * FP16_PER_REPEAT];
        AscendC::LocalTensor<uint8_t> softmaxTmpUint8LT = softmaxTmpLT.template ReinterpretCast<uint8_t>();
        SoftMax<T1, false, false>(yTmpLT, sumTmpLT, maxTmpLT, xLT, softmaxTmpUint8LT, *softmaxTilingData,
                                  softmaxShapeInfoData);
        AscendC::PipeBarrier<PIPE_V>();
        // Init TopK
        AscendC::LocalTensor<T2> srcIndexLocal = tmpLT.template ReinterpretCast<T2>();
        AscendC::LocalTensor<bool> finishedLocal;
        AscendC::LocalTensor<uint8_t> tmpUint8LT = srcIndexLocal[numCol_].template ReinterpretCast<uint8_t>();
        AscendC::ArithProgression<T2>(srcIndexLocal, static_cast<T2>(0), static_cast<T2>(1), numCol_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::TopK<T1, true, false, false, AscendC::TopKMode::TOPK_NORMAL>(yLT, expertIdxLT, yTmpLT, srcIndexLocal,
                                                                              finishedLocal, tmpUint8LT, kAlign_,
                                                                              *TopkTilingData, topkInfo, true);
        for (int32_t i = 0; i < rowWork_; i++) {
            ArithProgression<T2>(rowIdxLT[i * kAlign_], (i + GetBlock() * oneCoreRow_) * k_, 1, k_);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void CopyOut()
    {
        int32_t rowWork_ = (GetBlock() < activateCore_ - 1) ? oneCoreRow_ : tailRow_;
        for (int32_t i = 0; i < rowWork_; i++) {
            DataCopy(gmYTmp_[i * k_], yLT[i * kAlign_], kAlign_);
            DataCopy(gmExpertTmp_[i * k_], expertIdxLT[i * kAlign_], kAlign_);
            DataCopy(gmRowTmp_[i * k_], rowIdxLT[i * kAlign_], kAlign_);
            PipeBarrier<PIPE_ALL>();
        }
        DataCopy(yLT, gmYTmp_, Align16(rowWork_ * k_));
        DataCopy(expertIdxLT, gmExpertTmp_, Align16(rowWork_ * k_));
        DataCopy(rowIdxLT, gmRowTmp_, Align16(rowWork_ * k_));
        AscendC::SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        PipeBarrier<PIPE_ALL>();
        DataCopy(gmY_, yLT, Align16(rowWork_ * k_));
        DataCopy(gmExpertIdx_, expertIdxLT, Align16(rowWork_ * k_));
        DataCopy(gmRowIdx_, rowIdxLT, Align16(rowWork_ * k_));
        PipeBarrier<PIPE_ALL>();
    }

private:
    TPipe *pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufUb_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> syncIn;
    AscendC::GlobalTensor<T1> gmX_;
    AscendC::GlobalTensor<T1> gmY_;
    AscendC::GlobalTensor<T2> gmExpertIdx_;
    AscendC::GlobalTensor<T2> gmRowIdx_;
    AscendC::GlobalTensor<int32_t> gmSync_;
    AscendC::GlobalTensor<T1> gmYTmp_;
    AscendC::GlobalTensor<T2> gmExpertTmp_;
    AscendC::GlobalTensor<T2> gmRowTmp_;
    int32_t numCore_{0}; // 一共激活多少AICORE
    int32_t numCol_{0};  // 输入的列数
    int32_t numRow_{0};
    int32_t tmp_minsize;
    int64_t gmXOffset_{0}; // GM数据起始位置偏移量
    int64_t gmYOffset_{0};
    int64_t gmExpertIdxOffset_{0};
    int32_t k_{0};
    int32_t kAlign_{0};
    int32_t num_experts_{0};
    uint32_t tilingKey{0};
    int32_t oneCoreRow_{0};
    int32_t activateCore_{0};
    int32_t tailRow_{0};
    int32_t counter{0};
    uint32_t workspaceSize_;
    bool hasFinished_;
    int32_t offset0;
    int32_t offset1;
    int32_t offset2;
    int32_t offset3;
    int32_t offset4;
    int32_t offset5;
    int32_t offset6;
    SoftMaxTiling FormerSoftmaxTilingData;
    SoftMaxTiling TailSoftmaxTilingData;
    TopkTiling FormerTopkTilingData;
    TopkTiling TailTopkTilingData;
    AscendC::LocalTensor<T1> xLT;
    AscendC::LocalTensor<T1> yLT;
    AscendC::LocalTensor<T2> expertIdxLT;
    AscendC::LocalTensor<T2> rowIdxLT;
    AscendC::LocalTensor<T1> yTmpLT;
    AscendC::LocalTensor<T1> tmpLT;
    AscendC::LocalTensor<int32_t> syncLT;
};
} // namespace MoeGatingTopKSoftmax
