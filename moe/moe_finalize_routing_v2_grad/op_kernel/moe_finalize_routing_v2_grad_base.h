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
 * \file moe_finalize_routing_v2_grad_base.h
 * \brief
 */
#ifndef MOE_FINALIZE_ROUTING_V2_GRAD_BASE_H
#define MOE_FINALIZE_ROUTING_V2_GRAD_BASE_H

#include "kernel_operator.h"

namespace MoeFinalizeRoutingV2Grad {
using namespace AscendC;

constexpr uint32_t BYTE_BLOCK = 32;
constexpr uint32_t MAX_REPEAT = 255;
constexpr uint32_t MASK_FLOAT_ONE_REPEAT = 64;
constexpr uint32_t NUM_FLOAT_ONE_BLOCK = 8;
constexpr uint32_t STRIDE = 8;
constexpr uint32_t BLOCK_NUM_ONE_RPT = 8;

template <typename T1, typename T2>
class MoeFinalizeRoutingV2GradBase {
public:
    __aicore__ inline MoeFinalizeRoutingV2GradBase(){};

protected:
    __aicore__ inline void SubInit(GM_ADDR gradY, GM_ADDR expandedRowIdx, GM_ADDR gradExpandedX,
                                   const MoeFinalizeRoutingV2GradTilingData *tilingData, TPipe *pipe);
    __aicore__ inline void GetInitOutBatches();
    __aicore__ inline void GetSubProcessBatches();
    __aicore__ inline void InitOutCutH();
    __aicore__ inline void InitOutNotCutH();
    __aicore__ inline void Mte3ToMte2();
    __aicore__ inline void WaitMte2ToVec();
    __aicore__ inline void ReduceCalc(LocalTensor<float> &inputUb, LocalTensor<float> &tempUb,
                                      int64_t innerLoopCopyNum);
    __aicore__ inline void CopyInWithBias(int64_t batchIdx, int64_t innerOffset, int64_t innerLoopCopyNum);
    __aicore__ inline void CopyInWithOutBias(int64_t batchIdx, int64_t innerOffset, int64_t innerLoopCopyNum);
    __aicore__ inline void ComputeGradExpandedX(LocalTensor<T1> &gradYUb, LocalTensor<T1> &tempUb,
                                                LocalTensor<T1> &gradExpandedXUb, int64_t innerLoopCopyNum);
    __aicore__ inline void DoComputeOnlyWithBias(LocalTensor<float> &gradYUb, LocalTensor<float> &biasUb,
                                                 int64_t innerLoopCopyNum);
    __aicore__ inline void ComputeWithOutBias(int64_t innerOffset, int64_t innerLoopCopyNum);
    __aicore__ inline void ComputeWithBias(int64_t innerOffset, int64_t innerLoopCopyNum);
    __aicore__ inline void CopyOutGradScales(int64_t batchIdx);
    __aicore__ inline void CopyOutGradExpandedX(int64_t innerOffset, int64_t innerLoopCopyNum);

protected:
    GlobalTensor<T1> gradYGm_;
    GlobalTensor<T2> expandedRowIdxGm_;
    GlobalTensor<T1> expandedXGm_;
    GlobalTensor<T1> scalesGm_;
    GlobalTensor<T2> expertIdxGm_;
    GlobalTensor<T1> biasGm_;
    GlobalTensor<T1> gradExpandedXGm_;
    GlobalTensor<T1> gradExpandedXInitGm_;
    GlobalTensor<T1> gradScalesGm_;

    const MoeFinalizeRoutingV2GradTilingData *tilingData_;

    TPipe *pipe_;
    TQue<QuePosition::VECIN, 1> gradYInQueue_;
    TQue<QuePosition::VECIN, 1> expandedRowIdxInQueue_;
    TQue<QuePosition::VECIN, 1> expandedXInQueue_;
    TQue<QuePosition::VECIN, 1> scalesInQueue_;
    TQue<QuePosition::VECIN, 1> expertIdxInQueue_;
    TQue<QuePosition::VECIN, 1> biasInQueue_;
    TQue<QuePosition::VECOUT, 1> gradExpandedXOutQueue_;
    TQue<QuePosition::VECOUT, 1> gradScalesOutQueue_;

    int64_t blockIdx_ = 0;
    int64_t startBatchIdx_ = 0;
    int64_t endBatchIdx_ = 0;
    int64_t srcOffset_ = 0;
    int64_t dstOffset_ = 0;
    int64_t orgF16OBf16UbOffset_ = 0;
    T2 expandedRowIdx_ = 0;
    T2 expertIdx_ = 0;
    T1 scale_ = 0.0;
    float gradScalesSum_ = 0.0;
    bool hasBias = false;
};

__aicore__ inline void MTE3ToVSync()
{
    event_t eventIDMTE3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(eventIDMTE3ToV);
    WaitFlag<HardEvent::MTE3_V>(eventIDMTE3ToV);
}

__aicore__ inline void VToMTE3Sync()
{
    event_t eventIDVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
    WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
}

template <typename T1, typename T2>
__aicore__ inline void MoeFinalizeRoutingV2GradBase<T1, T2>::SubInit(
    GM_ADDR gradY, GM_ADDR expandedRowIdx, GM_ADDR gradExpandedX, const MoeFinalizeRoutingV2GradTilingData *tilingData,
    TPipe *pipe)
{
    gradYGm_.SetGlobalBuffer((__gm__ T1 *)gradY);
    expandedRowIdxGm_.SetGlobalBuffer((__gm__ T2 *)expandedRowIdx);
    gradExpandedXGm_.SetGlobalBuffer((__gm__ T1 *)gradExpandedX);
    tilingData_ = tilingData;
    pipe_ = pipe;
    blockIdx_ = GetBlockIdx();
}

template <typename T1, typename T2>
__aicore__ inline void MoeFinalizeRoutingV2GradBase<T1, T2>::GetInitOutBatches()
{
    if (blockIdx_ < tilingData_->initOutModCoreNum) {
        startBatchIdx_ = blockIdx_ * (tilingData_->initOutEachCoreBatchNum + 1);
        endBatchIdx_ = startBatchIdx_ + tilingData_->initOutEachCoreBatchNum + 1;
    } else if (blockIdx_ < tilingData_->initOutNeedCoreNum) {
        startBatchIdx_ = blockIdx_ * tilingData_->initOutEachCoreBatchNum + tilingData_->initOutModCoreNum;
        endBatchIdx_ = startBatchIdx_ + tilingData_->initOutEachCoreBatchNum;
    } else {
        startBatchIdx_ = 0;
        endBatchIdx_ = 0;
    }
}

template <typename T1, typename T2>
__aicore__ inline void MoeFinalizeRoutingV2GradBase<T1, T2>::GetSubProcessBatches()
{
    if (blockIdx_ < tilingData_->computeModCoreNum) {
        startBatchIdx_ = blockIdx_ * (tilingData_->computeEachCoreBatchNum + 1);
        endBatchIdx_ = startBatchIdx_ + tilingData_->computeEachCoreBatchNum + 1;
    } else if (blockIdx_ < tilingData_->computeNeedCoreNum) {
        startBatchIdx_ = blockIdx_ * tilingData_->computeEachCoreBatchNum + tilingData_->computeModCoreNum;
        endBatchIdx_ = startBatchIdx_ + tilingData_->computeEachCoreBatchNum;
    } else {
        startBatchIdx_ = 0;
        endBatchIdx_ = 0;
    }
}

template <typename T1, typename T2>
__aicore__ inline void MoeFinalizeRoutingV2GradBase<T1, T2>::InitOutCutH()
{
    GetInitOutBatches();
    for (int64_t batchIdx = startBatchIdx_; batchIdx < endBatchIdx_; batchIdx++) {
        for (int64_t innerLoopIdx = 0; innerLoopIdx < tilingData_->hiddenInnerLoops; innerLoopIdx++) {
            dstOffset_ = batchIdx * tilingData_->hidden + innerLoopIdx * tilingData_->hiddenPrePart;
            gradExpandedXInitGm_ = gradExpandedXGm_[dstOffset_];
            InitOutput<T1>(gradExpandedXInitGm_, tilingData_->hiddenPrePart, 0);
        }
        if (tilingData_->hiddenLastPart != 0) {
            dstOffset_ = batchIdx * tilingData_->hidden + tilingData_->hiddenInnerLoops * tilingData_->hiddenPrePart;
            gradExpandedXInitGm_ = gradExpandedXGm_[dstOffset_];
            InitOutput<T1>(gradExpandedXInitGm_, tilingData_->hiddenLastPart, 0);
        }
    }
}

template <typename T1, typename T2>
__aicore__ inline void MoeFinalizeRoutingV2GradBase<T1, T2>::InitOutNotCutH()
{
    GetInitOutBatches();
    for (int64_t batchIdx = startBatchIdx_; batchIdx < endBatchIdx_; batchIdx++) {
        dstOffset_ = batchIdx * tilingData_->hidden;
        gradExpandedXInitGm_ = gradExpandedXGm_[dstOffset_];
        InitOutput<T1>(gradExpandedXInitGm_, tilingData_->hidden, 0);
        MTE3ToVSync();
    }
}

template <typename T1, typename T2>
__aicore__ inline void MoeFinalizeRoutingV2GradBase<T1, T2>::Mte3ToMte2()
{
    event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
}

template <typename T1, typename T2>
__aicore__ inline void MoeFinalizeRoutingV2GradBase<T1, T2>::WaitMte2ToVec()
{
    event_t eventIdMte2ToVec = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToVec);
    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToVec);
}

template <typename T1, typename T2>
__aicore__ inline void MoeFinalizeRoutingV2GradBase<T1, T2>::CopyInWithBias(int64_t batchIdx, int64_t innerOffset,
                                                                            int64_t innerLoopCopyNum)
{
    LocalTensor<T1> gradYUb = gradYInQueue_.template AllocTensor<T1>();
    LocalTensor<T2> expandedRowIdxUb = expandedRowIdxInQueue_.template AllocTensor<T2>();
    LocalTensor<T1> scalesUb = scalesInQueue_.template AllocTensor<T1>();
    LocalTensor<T2> expertIdxUb = expertIdxInQueue_.template AllocTensor<T2>();
    LocalTensor<T1> biasUb = biasInQueue_.template AllocTensor<T1>();

    DataCopyExtParams copyExtParams{1, 1, 0, 0, 0};
    DataCopyPadExtParams<T1> copyPadExtparamsT1{false, 0, 0, 0};
    DataCopyPadExtParams<T2> copyPadExtparamsT2{false, 0, 0, 0};

    srcOffset_ = batchIdx / tilingData_->topK * tilingData_->hidden + innerOffset;
    copyExtParams.blockLen = innerLoopCopyNum * sizeof(T1);
    DataCopyPad(gradYUb[orgF16OBf16UbOffset_], gradYGm_[srcOffset_], copyExtParams, copyPadExtparamsT1);

    srcOffset_ = batchIdx;
    copyExtParams.blockLen = sizeof(T2);
    DataCopyPad(expandedRowIdxUb, expandedRowIdxGm_[srcOffset_], copyExtParams, copyPadExtparamsT2);
    DataCopyPad(expertIdxUb, expertIdxGm_[srcOffset_], copyExtParams, copyPadExtparamsT2);

    copyExtParams.blockLen = sizeof(T1);
    DataCopyPad(scalesUb, scalesGm_[srcOffset_], copyExtParams, copyPadExtparamsT1);

    expertIdx_ = expertIdxUb.GetValue(0);
    expandedRowIdx_ = expandedRowIdxUb.GetValue(0);
    scale_ = scalesUb.GetValue(0);

    srcOffset_ = expertIdx_ * tilingData_->hidden + innerOffset;

    copyExtParams.blockLen = innerLoopCopyNum * sizeof(T1);
    DataCopyPad(biasUb[orgF16OBf16UbOffset_], biasGm_[srcOffset_], copyExtParams, copyPadExtparamsT1);

    gradYInQueue_.template EnQue(gradYUb);
    biasInQueue_.template EnQue(biasUb);
    expandedRowIdxInQueue_.FreeTensor(expandedRowIdxUb);
    scalesInQueue_.FreeTensor(scalesUb);
    expertIdxInQueue_.FreeTensor(expertIdxUb);
}

template <typename T1, typename T2>
__aicore__ inline void MoeFinalizeRoutingV2GradBase<T1, T2>::CopyInWithOutBias(int64_t batchIdx, int64_t innerOffset,
                                                                               int64_t innerLoopCopyNum)
{
    LocalTensor<T1> gradYUb = gradYInQueue_.template AllocTensor<T1>();
    LocalTensor<T2> expandedRowIdxUb = expandedRowIdxInQueue_.template AllocTensor<T2>();
    LocalTensor<T1> scalesUb = scalesInQueue_.template AllocTensor<T1>();

    DataCopyExtParams copyExtParams{1, 1, 0, 0, 0};
    DataCopyPadExtParams<T2> copyPadExtparamsT2{false, 0, 0, 0};
    DataCopyPadExtParams<T1> copyPadExtparamsT1{false, 0, 0, 0};

    srcOffset_ = batchIdx / tilingData_->topK * tilingData_->hidden + innerOffset;
    copyExtParams.blockLen = innerLoopCopyNum * sizeof(T1);
    DataCopyPad(gradYUb[orgF16OBf16UbOffset_], gradYGm_[srcOffset_], copyExtParams, copyPadExtparamsT1);

    srcOffset_ = batchIdx;
    copyExtParams.blockLen = sizeof(T2);
    DataCopyPad(expandedRowIdxUb, expandedRowIdxGm_[srcOffset_], copyExtParams, copyPadExtparamsT2);

    copyExtParams.blockLen = sizeof(T1);
    DataCopyPad(scalesUb, scalesGm_[srcOffset_], copyExtParams, copyPadExtparamsT1);

    expandedRowIdx_ = expandedRowIdxUb.GetValue(0);
    scale_ = scalesUb.GetValue(0);

    gradYInQueue_.template EnQue(gradYUb);
    expandedRowIdxInQueue_.FreeTensor(expandedRowIdxUb);
    scalesInQueue_.FreeTensor(scalesUb);
}

template <typename T1, typename T2>
__aicore__ inline void MoeFinalizeRoutingV2GradBase<T1, T2>::ComputeGradExpandedX(LocalTensor<T1> &gradYUb,
                                                                                  LocalTensor<T1> &tempUb,
                                                                                  LocalTensor<T1> &gradExpandedXUb,
                                                                                  int64_t innerLoopCopyNum)
{
    auto gradYUbF32 = gradYUb.template ReinterpretCast<float>();
    auto tempUbF32 = tempUb.template ReinterpretCast<float>();

    if constexpr (IsSameType<T1, half>::value || IsSameType<T1, bfloat16_t>::value) {
        Cast(gradYUbF32, gradYUb[orgF16OBf16UbOffset_], RoundMode::CAST_NONE, innerLoopCopyNum);
        if constexpr (IsSameType<T1, half>::value) {
            Muls(tempUbF32, gradYUbF32, (float)scale_, innerLoopCopyNum);
        } else {
            Muls(tempUbF32, gradYUbF32, ToFloat(scale_), innerLoopCopyNum);
        }
        Cast(gradExpandedXUb, tempUbF32, RoundMode::CAST_RINT, innerLoopCopyNum);
    } else {
        Muls(gradExpandedXUb, gradYUb, (float)scale_, innerLoopCopyNum);
    }

    gradExpandedXOutQueue_.template EnQue(gradExpandedXUb);
}

template <typename T1, typename T2>
__aicore__ inline void MoeFinalizeRoutingV2GradBase<T1, T2>::ReduceCalc(LocalTensor<float> &inputUb,
                                                                        LocalTensor<float> &tempUb,
                                                                        int64_t innerLoopCopyNum)
{
    int32_t repeat = innerLoopCopyNum / MASK_FLOAT_ONE_REPEAT;
    uint32_t modNum = innerLoopCopyNum % MASK_FLOAT_ONE_REPEAT;
    uint32_t maxRptNum = repeat / MAX_REPEAT;
    uint32_t modRptNum = repeat % MAX_REPEAT;
    uint32_t offset = 0;

    if (maxRptNum != 0) {
        for (uint32_t rptIdx = 0; rptIdx < maxRptNum; rptIdx++) {
            offset = rptIdx * MAX_REPEAT * MASK_FLOAT_ONE_REPEAT;
            BlockReduceSum<float>(tempUb[rptIdx * MAX_REPEAT * BLOCK_NUM_ONE_RPT], inputUb[offset], MAX_REPEAT,
                                  MASK_FLOAT_ONE_REPEAT, 1, 1, STRIDE);
        }
    }

    if (modRptNum != 0) {
        offset = maxRptNum * MAX_REPEAT * MASK_FLOAT_ONE_REPEAT;
        BlockReduceSum<float>(tempUb[maxRptNum * MAX_REPEAT * BLOCK_NUM_ONE_RPT], inputUb[offset], modRptNum,
                              MASK_FLOAT_ONE_REPEAT, 1, 1, STRIDE);
    }

    if (modNum != 0) {
        offset = repeat * MASK_FLOAT_ONE_REPEAT;
        BlockReduceSum<float>(tempUb[repeat * BLOCK_NUM_ONE_RPT], inputUb[offset], 1, modNum, 1, 1, STRIDE);
    }

    ReduceSum(inputUb, tempUb, inputUb, (innerLoopCopyNum + NUM_FLOAT_ONE_BLOCK - 1) / NUM_FLOAT_ONE_BLOCK);
    gradScalesSum_ += inputUb.GetValue(0);
}

template <typename T1, typename T2>
__aicore__ inline void MoeFinalizeRoutingV2GradBase<T1, T2>::ComputeWithOutBias(int64_t innerOffset,
                                                                                int64_t innerLoopCopyNum)
{
    LocalTensor<T1> gradYUb = gradYInQueue_.template DeQue<T1>();
    LocalTensor<T1> expandedXUb = expandedXInQueue_.template AllocTensor<T1>();

    DataCopyExtParams copyExtParams{1, static_cast<uint32_t>(innerLoopCopyNum * sizeof(T1)), 0, 0, 0};
    DataCopyPadExtParams<T1> copyPadExtParams{false, 0, 0, 0};

    if (expandedRowIdx_ >= 0 && expandedRowIdx_ < tilingData_->expandedXDim0) {
        LocalTensor<T1> gradExpandedXUb = gradExpandedXOutQueue_.template AllocTensor<T1>();
        ComputeGradExpandedX(gradYUb, expandedXUb, gradExpandedXUb, innerLoopCopyNum);
        CopyOutGradExpandedX(innerOffset, innerLoopCopyNum);
        srcOffset_ = expandedRowIdx_ * tilingData_->hidden + innerOffset;
        DataCopyPad(expandedXUb[orgF16OBf16UbOffset_], expandedXGm_[srcOffset_], copyExtParams, copyPadExtParams);
        WaitMte2ToVec();

        if constexpr (IsSameType<T1, half>::value || IsSameType<T1, bfloat16_t>::value) {
            auto expandedXUbF32 = expandedXUb.template ReinterpretCast<float>();
            auto gradYUbF32 = gradYUb.template ReinterpretCast<float>();

            Cast(expandedXUbF32, expandedXUb[orgF16OBf16UbOffset_], RoundMode::CAST_NONE, innerLoopCopyNum);
            Mul(gradYUbF32, expandedXUbF32, gradYUbF32, innerLoopCopyNum);
            ReduceCalc(gradYUbF32, expandedXUbF32, innerLoopCopyNum);
        } else {
            Mul(expandedXUb, expandedXUb, gradYUb, innerLoopCopyNum);
            ReduceCalc(expandedXUb, gradYUb, innerLoopCopyNum);
        }
    }

    gradYInQueue_.FreeTensor(gradYUb);
    expandedXInQueue_.FreeTensor(expandedXUb);
}

template <typename T1, typename T2>
__aicore__ inline void MoeFinalizeRoutingV2GradBase<T1, T2>::DoComputeOnlyWithBias(LocalTensor<float> &gradYUb,
                                                                                   LocalTensor<float> &biasUb,
                                                                                   int64_t innerLoopCopyNum)
{
    Mul(biasUb, biasUb, gradYUb, innerLoopCopyNum);
    ReduceCalc(biasUb, gradYUb, innerLoopCopyNum);
}

template <typename T1, typename T2>
__aicore__ inline void MoeFinalizeRoutingV2GradBase<T1, T2>::ComputeWithBias(int64_t innerOffset,
                                                                             int64_t innerLoopCopyNum)
{
    LocalTensor<T1> gradYUb = gradYInQueue_.template DeQue<T1>();
    LocalTensor<T1> expandedXUb = expandedXInQueue_.template AllocTensor<T1>();
    LocalTensor<T1> biasUb = biasInQueue_.template DeQue<T1>();

    if (expandedRowIdx_ >= 0 && expandedRowIdx_ < tilingData_->expandedXDim0) {
        LocalTensor<T1> gradExpandedXUb = gradExpandedXOutQueue_.template AllocTensor<T1>();
        ComputeGradExpandedX(gradYUb, expandedXUb, gradExpandedXUb, innerLoopCopyNum);
        CopyOutGradExpandedX(innerOffset, innerLoopCopyNum);

        DataCopyExtParams copyExtParams{1, static_cast<uint32_t>(innerLoopCopyNum * sizeof(T1)), 0, 0, 0};
        DataCopyPadExtParams<T1> copyPadExtParams{false, 0, 0, 0};
        srcOffset_ = expandedRowIdx_ * tilingData_->hidden + innerOffset;
        DataCopyPad(expandedXUb[orgF16OBf16UbOffset_], expandedXGm_[srcOffset_], copyExtParams, copyPadExtParams);
        WaitMte2ToVec();

        if constexpr (IsSameType<T1, half>::value || IsSameType<T1, bfloat16_t>::value) {
            auto expandedXUbF32 = expandedXUb.template ReinterpretCast<float>();
            auto gradYUbF32 = gradYUb.template ReinterpretCast<float>();
            auto biasUbF32 = biasUb.template ReinterpretCast<float>();
            Cast(expandedXUbF32, expandedXUb[orgF16OBf16UbOffset_], RoundMode::CAST_NONE, innerLoopCopyNum);
            Cast(biasUbF32, biasUb[orgF16OBf16UbOffset_], RoundMode::CAST_NONE, innerLoopCopyNum);
            Add(biasUbF32, biasUbF32, expandedXUbF32, innerLoopCopyNum);
            DoComputeOnlyWithBias(gradYUbF32, biasUbF32, innerLoopCopyNum);
        } else {
            Add(biasUb, biasUb, expandedXUb, innerLoopCopyNum);
            DoComputeOnlyWithBias(gradYUb, biasUb, innerLoopCopyNum);
        }
    } else {
        if constexpr (IsSameType<T1, half>::value || IsSameType<T1, bfloat16_t>::value) {
            auto gradYUbF32 = gradYUb.template ReinterpretCast<float>();
            auto biasUbF32 = biasUb.template ReinterpretCast<float>();
            Cast(gradYUbF32, gradYUb[orgF16OBf16UbOffset_], RoundMode::CAST_NONE, innerLoopCopyNum);
            Cast(biasUbF32, biasUb[orgF16OBf16UbOffset_], RoundMode::CAST_NONE, innerLoopCopyNum);
            DoComputeOnlyWithBias(gradYUbF32, biasUbF32, innerLoopCopyNum);
        } else {
            DoComputeOnlyWithBias(gradYUb, biasUb, innerLoopCopyNum);
        }
    }

    gradYInQueue_.FreeTensor(gradYUb);
    expandedXInQueue_.FreeTensor(expandedXUb);
    biasInQueue_.FreeTensor(biasUb);
}

template <typename T1, typename T2>
__aicore__ inline void MoeFinalizeRoutingV2GradBase<T1, T2>::CopyOutGradScales(int64_t batchIdx)
{
    LocalTensor<T1> gradScalesUb = gradScalesOutQueue_.template AllocTensor<T1>();

    dstOffset_ = batchIdx;
    DataCopyExtParams copyExtParams{1, 1, 0, 0, 0};
    copyExtParams.blockLen = sizeof(T1);
    if constexpr (IsSameType<T1, bfloat16_t>::value || IsSameType<T1, half>::value) {
        auto tempUb1 = gradScalesUb.template ReinterpretCast<float>();
        tempUb1.SetValue(0, gradScalesSum_);
        Cast(gradScalesUb, tempUb1, RoundMode::CAST_RINT, 1);
    } else {
        gradScalesUb.SetValue(0, gradScalesSum_);
    }
    VToMTE3Sync();
    gradScalesSum_ = 0;
    DataCopyPad(gradScalesGm_[dstOffset_], gradScalesUb, copyExtParams);
    Mte3ToMte2();

    gradScalesOutQueue_.FreeTensor(gradScalesUb);
}

template <typename T1, typename T2>
__aicore__ inline void MoeFinalizeRoutingV2GradBase<T1, T2>::CopyOutGradExpandedX(int64_t innerOffset,
                                                                                  int64_t innerLoopCopyNum)
{
    LocalTensor<T1> gradExpandedXUb = gradExpandedXOutQueue_.template DeQue<T1>();

    dstOffset_ = expandedRowIdx_ * tilingData_->hidden + innerOffset;
    DataCopyExtParams copyExtParams{1, 1, 0, 0, 0};
    copyExtParams.blockLen = innerLoopCopyNum * sizeof(T1);
    DataCopyPad(gradExpandedXGm_[dstOffset_], gradExpandedXUb, copyExtParams);
    Mte3ToMte2();
    gradExpandedXOutQueue_.FreeTensor(gradExpandedXUb);
}
} // namespace MoeFinalizeRoutingV2Grad
#endif
// MOE_FINALIZE_ROUTING_V2_GRAD_BASE_H
