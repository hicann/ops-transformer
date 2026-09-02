/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file ffn_wb_sort_one_core.h
 * \brief
 */
#ifndef OP_KERNEL_FFN_WB_SORT_ONE_CORE_H
#define OP_KERNEL_FFN_WB_SORT_ONE_CORE_H

#include "ffn_wb_sort_base.h"
#include "ffn_wb_common.h"

namespace FfnWbBatching {
using namespace AscendC;

class KernelSortMaskOneCore : public SortMaskBase {
public:
    __aicore__ inline KernelSortMaskOneCore(){};
    __aicore__ inline void Init(GM_ADDR expert_ids, GM_ADDR workspace, const SortCustomTilingDataKernel *tilingData,
                                const ScheduleContextInfo *contextInfo, TPipe *tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn();
    __aicore__ inline void SortCompute();
    __aicore__ inline void ExpertCountCompute();
    __aicore__ inline void CopyOut();

private:
    int64_t sortNum = 0;
    int32_t validCnt = 0;
    int32_t needCoreNum_ = 0;
    const ScheduleContextInfo *contextInfo_ = nullptr;
};

__aicore__ inline void KernelSortMaskOneCore::CopyIn()
{
    LocalTensor<int32_t> inLocal = sortDataCopyInQueue.AllocTensor<int32_t>();
    DataCopyExtParams dataCopyParams{static_cast<uint16_t>(1),
                                     static_cast<uint32_t>(this->totalLength * sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams dataCopyPadParams{false, 0, 0, 0};
    DataCopyPad(inLocal[0], expertIdsGm, dataCopyParams, dataCopyPadParams);

    LocalTensor<int32_t> rowIdsLocal = inLocal[this->sortNum];
    ArithProgression<int32_t>(rowIdsLocal, 0, 1, this->totalLength);
    sortDataCopyInQueue.EnQue(inLocal);
}

__aicore__ inline void KernelSortMaskOneCore::SortCompute()
{
    LocalTensor<int32_t> inLocal = sortDataCopyInQueue.DeQue<int32_t>();
    LocalTensor<float> expertIdsFp32 = inLocal.ReinterpretCast<float>();
    Cast(expertIdsFp32, inLocal, RoundMode::CAST_ROUND, this->totalLength);
    PipeBarrier<PIPE_V>();

    LocalTensor<uint32_t> maskLocalUInt32 = sortedBuffer.Get<uint32_t>();
    uint64_t rsvdCnt = 0;

    Muls(expertIdsFp32, expertIdsFp32, (float)-1, this->totalLength);
    PipeBarrier<PIPE_V>();

    LocalTensor<uint8_t> maskLocalTensorUInt8 = maskLocalUInt32.ReinterpretCast<uint8_t>();
    AscendC::CompareScalar(
        maskLocalTensorUInt8, expertIdsFp32, static_cast<float>(-expertStart_), AscendC::CMPMODE::GT,
        (this->totalLength + ONE_REPEAT_COMPARE_NUM - 1) / ONE_REPEAT_COMPARE_NUM * ONE_REPEAT_COMPARE_NUM);
    PipeBarrier<PIPE_V>();

    GatherMaskParams gatherMaskParams;
    gatherMaskParams.repeatTimes = 1;
    gatherMaskParams.src0BlockStride = 1;
    gatherMaskParams.src0RepeatStride = 8; // 8 blocks
    gatherMaskParams.src1RepeatStride = 0;
    GatherMask(expertIdsFp32, expertIdsFp32, maskLocalUInt32, true, this->totalLength, gatherMaskParams, rsvdCnt);
    PipeBarrier<PIPE_V>();
    this->validCnt = rsvdCnt;
    if (rsvdCnt == 0) {
        sortDataCopyInQueue.FreeTensor(inLocal);
        return;
    }
    int64_t duplicateNum = rsvdCnt % ONE_REPEAT_SORT_NUM;
    if (duplicateNum > 0) {
        int duplicateIndex = rsvdCnt - duplicateNum;
        uint64_t mask0 = UINT64_MAX;
        mask0 = mask0 << duplicateNum;
        mask0 = mask0 & (UINT64_MAX >> ONE_REPEAT_SORT_NUM);
        uint64_t mask[2] = {mask0, 0};
        Duplicate(expertIdsFp32[duplicateIndex], MIN_FP32, mask, 1, DST_BLK_STRIDE, DST_REP_STRIDE);
        PipeBarrier<PIPE_V>();
    }
    int32_t selectedCnt = (rsvdCnt + ONE_REPEAT_SORT_NUM - 1) / ONE_REPEAT_SORT_NUM * ONE_REPEAT_SORT_NUM;

    LocalTensor<float> concatLocal = expertIdsFp32;
    LocalTensor<float> tempTensor = tempBuffer.Get<float>(GetSortLen<float>(selectedCnt));
    Concat(concatLocal, expertIdsFp32, tempTensor, selectedCnt / ONE_REPEAT_SORT_NUM);
    PipeBarrier<PIPE_V>();

    LocalTensor<uint32_t> rowIdsLocal = inLocal[this->sortNum].ReinterpretCast<uint32_t>();
    GatherMask(rowIdsLocal, rowIdsLocal, maskLocalUInt32, true, this->totalLength, gatherMaskParams, rsvdCnt);
    PipeBarrier<PIPE_V>();
    LocalTensor<float> sortedLocal = sortedBuffer.Get<float>(GetSortLen<float>(selectedCnt));
    Sort<float, true>(sortedLocal, concatLocal, rowIdsLocal, tempTensor, selectedCnt / ONE_REPEAT_SORT_NUM);
    PipeBarrier<PIPE_V>();

    LocalTensor<float> outLocal = sortDataCopyOutQueue.AllocTensor<float>();
    LocalTensor<float> sortedExpertIdsLocal = outLocal[0];
    LocalTensor<uint32_t> sortedRowIdsLocal = outLocal[this->sortNum].ReinterpretCast<uint32_t>();
    Extract(sortedExpertIdsLocal, sortedRowIdsLocal, sortedLocal, selectedCnt / ONE_REPEAT_SORT_NUM);
    PipeBarrier<PIPE_V>();

    Muls(sortedExpertIdsLocal, sortedExpertIdsLocal, (float)-1, rsvdCnt);
    PipeBarrier<PIPE_V>();
    LocalTensor<int32_t> sortedExpertIdsLocalInt32 = sortedExpertIdsLocal.ReinterpretCast<int32_t>();
    Cast(sortedExpertIdsLocalInt32, sortedExpertIdsLocal, RoundMode::CAST_ROUND, rsvdCnt);
    PipeBarrier<PIPE_V>();
    sortDataCopyOutQueue.EnQue<float>(outLocal);
    sortDataCopyInQueue.FreeTensor(inLocal);
}

__aicore__ inline void KernelSortMaskOneCore::CopyOut()
{
    if (this->validCnt != 0) {
        LocalTensor<int32_t> outLocal = sortDataCopyOutQueue.DeQue<int32_t>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(this->validCnt * sizeof(int32_t)), 0, 0, 0};
        DataCopyPad(sortedexpertIdsGm, outLocal[0], copyParams);
        DataCopyPad(sortedRowIdsGm, outLocal[this->sortNum], copyParams);
        sortDataCopyOutQueue.FreeTensor(outLocal);
    }

    rsvdCntGm.SetValue(0, this->validCnt);
    DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_ALL>(
        rsvdCntGm);
}

__aicore__ inline void KernelSortMaskOneCore::Init(GM_ADDR expert_ids, GM_ADDR workspace,
                                                   const SortCustomTilingDataKernel *tilingData,
                                                   const ScheduleContextInfo *contextInfo, TPipe *tPipe)
{
    this->pipe = tPipe;
    contextInfo_ = contextInfo;

    needCoreNum_ = tilingData->needCoreNum;
    this->totalLength = tilingData->totalLength;
    this->sortNum = Ceil(this->totalLength, ONE_REPEAT_SORT_NUM) * ONE_REPEAT_SORT_NUM;

    expertIdsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(expert_ids), this->totalLength);
    rsvdCntGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(workspace), OFFSET_SORTED_EXPERT_IDS);
    sortedexpertIdsGm.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t *>(workspace) + OFFSET_SORTED_EXPERT_IDS + contextInfo_->sortNumWorkSpace,
        this->totalLength);
    sortedRowIdsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(workspace) + OFFSET_SORTED_EXPERT_IDS +
                                       contextInfo_->sortNumWorkSpace + this->totalLength,
                                   this->totalLength);

    if (GetBlockIdx() == 0) {
        GM_ADDR targetAddr = workspace + OFFSET_SORTED_EXPERT_IDS * sizeof(int32_t) +
                             contextInfo_->sortNumWorkSpace * sizeof(int32_t) * (NUM_TWO * NUM_FOUR + 1);
        groupListTmpGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(targetAddr), contextInfo_->expertNum);
        InitGlobalMemory(groupListTmpGm, contextInfo_->expertNum, 0);
        SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);

        if (needCoreNum_ == 0) {
            rsvdCntGm.SetValue(0, 0);
            DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_ALL>(
                rsvdCntGm);
        }
    }

    int64_t kvFactor = 2;
    int64_t buffSize = this->sortNum * sizeof(int32_t) * kvFactor;
    pipe->InitBuffer(sortDataCopyInQueue, bufferNum, buffSize);
    pipe->InitBuffer(sortDataCopyOutQueue, bufferNum, buffSize);
    pipe->InitBuffer(tempBuffer, buffSize);
    pipe->InitBuffer(sortedBuffer, buffSize);
}

__aicore__ inline void KernelSortMaskOneCore::Process()
{
    if (GetBlockIdx() < needCoreNum_) {
        CopyIn();
        SortCompute();
        CopyOut();
    }
    SyncAll();
}
} // namespace FfnWbBatching
#endif // OP_KERNEL_FFN_WB_SORT_ONE_CORE_H
