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
 * \file moe_finalize_routing_v2_grad_without_scale_cut_h.h
 * \brief
 */
#ifndef MOE_FINALIZE_ROUTING_V2_GRAD_WITHOUT_SCALE_CUT_H_H
#define MOE_FINALIZE_ROUTING_V2_GRAD_WITHOUT_SCALE_CUT_H_H

#include "moe_finalize_routing_v2_grad_base.h"

namespace MoeFinalizeRoutingV2Grad {
using namespace AscendC;

template <typename T1, typename T2>
class MoeFinalizeRoutingV2GradWithoutScaleCutH : public MoeFinalizeRoutingV2GradBase<T1, T2> {
public:
    __aicore__ inline MoeFinalizeRoutingV2GradWithoutScaleCutH() {}
    __aicore__ inline void Init(GM_ADDR gradY, GM_ADDR expandedRowIdx, GM_ADDR gradExpandedX, GM_ADDR workspace,
                                const MoeFinalizeRoutingV2GradTilingData *tilingData, TPipe *pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void SubProcess();
    __aicore__ inline void CopyIn(int64_t batchIdx, int64_t innerLoopIdx, int64_t innerLoopCopyNum);
    __aicore__ inline void CopyOut(int64_t innerLoopIdx, int64_t innerLoopCopyNum);
};

template <typename T1, typename T2>
__aicore__ inline void MoeFinalizeRoutingV2GradWithoutScaleCutH<T1, T2>::Init(
    GM_ADDR gradY, GM_ADDR expandedRowIdx, GM_ADDR gradExpandedX, GM_ADDR workspace,
    const MoeFinalizeRoutingV2GradTilingData *tilingData, TPipe *pipe)
{
    this->SubInit(gradY, expandedRowIdx, gradExpandedX, tilingData, pipe);
    this->pipe_->InitBuffer(this->gradYInQueue_, 1, this->tilingData_->hiddenPrePart * sizeof(T1));
    this->pipe_->InitBuffer(this->expandedRowIdxInQueue_, 1, BYTE_BLOCK);
}

template <typename T1, typename T2>
__aicore__ inline void MoeFinalizeRoutingV2GradWithoutScaleCutH<T1, T2>::Process()
{
    if (this->tilingData_->dropPadMode == 1) {
        this->InitOutCutH();
        SyncAll();
    }
    SubProcess();
}

template <typename T1, typename T2>
__aicore__ inline void MoeFinalizeRoutingV2GradWithoutScaleCutH<T1, T2>::SubProcess()
{
    this->GetSubProcessBatches();
    for (int64_t batchIdx = this->startBatchIdx_; batchIdx < this->endBatchIdx_; batchIdx++) {
        for (int64_t innerLoopIdx = 0; innerLoopIdx < this->tilingData_->hiddenInnerLoops; innerLoopIdx++) {
            CopyIn(batchIdx, innerLoopIdx, this->tilingData_->hiddenPrePart);
            CopyOut(innerLoopIdx, this->tilingData_->hiddenPrePart);
        }
        if (this->tilingData_->hiddenLastPart != 0) {
            CopyIn(batchIdx, this->tilingData_->hiddenInnerLoops, this->tilingData_->hiddenLastPart);
            CopyOut(this->tilingData_->hiddenInnerLoops, this->tilingData_->hiddenLastPart);
        }
    }
}

template <typename T1, typename T2>
__aicore__ inline void MoeFinalizeRoutingV2GradWithoutScaleCutH<T1, T2>::CopyIn(int64_t batchIdx, int64_t innerLoopIdx,
                                                                                int64_t innerLoopCopyNum)
{
    LocalTensor<T1> gradYUb = this->gradYInQueue_.template AllocTensor<T1>();
    LocalTensor<T2> expandedRowIdxUb = this->expandedRowIdxInQueue_.template AllocTensor<T2>();

    DataCopyExtParams copyExtParams{1, 1, 0, 0, 0};
    DataCopyPadExtParams<T1> copyPadExtParamsT1{false, 0, 0, 0};
    DataCopyPadExtParams<T2> copyPadExtParamsT2{false, 0, 0, 0};

    this->srcOffset_ = batchIdx * this->tilingData_->hidden + innerLoopIdx * this->tilingData_->hiddenPrePart;
    copyExtParams.blockLen = innerLoopCopyNum * sizeof(T1);
    DataCopyPad(gradYUb, this->gradYGm_[this->srcOffset_], copyExtParams, copyPadExtParamsT1);

    this->srcOffset_ = batchIdx;
    copyExtParams.blockLen = sizeof(T2);
    DataCopyPad(expandedRowIdxUb, this->expandedRowIdxGm_[this->srcOffset_], copyExtParams, copyPadExtParamsT2);

    this->gradYInQueue_.template EnQue(gradYUb);
    this->expandedRowIdxInQueue_.template EnQue(expandedRowIdxUb);
}

template <typename T1, typename T2>
__aicore__ inline void MoeFinalizeRoutingV2GradWithoutScaleCutH<T1, T2>::CopyOut(int64_t innerLoopIdx,
                                                                                 int64_t innerLoopCopyNum)
{
    LocalTensor<T1> gradYUb = this->gradYInQueue_.template DeQue<T1>();
    LocalTensor<T2> expandedRowIdxUb = this->expandedRowIdxInQueue_.template DeQue<T2>();

    this->expandedRowIdx_ = expandedRowIdxUb.GetValue(0);
    if (this->expandedRowIdx_ >= 0 && this->expandedRowIdx_ < this->tilingData_->expandedXDim0) {
        this->dstOffset_ =
            this->expandedRowIdx_ * this->tilingData_->hidden + innerLoopIdx * this->tilingData_->hiddenPrePart;
        DataCopyExtParams copyExtParams{1, 1, 0, 0, 0};
        copyExtParams.blockLen = innerLoopCopyNum * sizeof(T1);
        DataCopyPad(this->gradExpandedXGm_[this->dstOffset_], gradYUb, copyExtParams);
        this->Mte3ToMte2();
    }

    this->gradYInQueue_.FreeTensor(gradYUb);
    this->expandedRowIdxInQueue_.FreeTensor(expandedRowIdxUb);
}
} // namespace MoeFinalizeRoutingV2Grad

#endif // MOE_FINALIZE_ROUTING_V2_GRAD_WITHOUT_SCALE_CUT_H_H
