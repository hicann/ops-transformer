/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
 * \file ffn_wb_scan_sort_one_core.h
 * \brief
 */
#ifndef OP_KERNEL_FFN_WB_SCAN_SORT_ONE_CORE_H
#define OP_KERNEL_FFN_WB_SCAN_SORT_ONE_CORE_H

#include "ffn_wb_sort_base.h"
#include "ffn_wb_scan_token_info.h"

namespace FfnWbBatching {
using namespace AscendC;

class KernelScanSortMaskOneCore : public SortMaskBase {
public:
    __aicore__ inline KernelScanSortMaskOneCore(){};
    __aicore__ inline void Init(GM_ADDR tokenInfoGm, GM_ADDR workspace, SortCustomTilingDataKernel *tilingData,
                                const ScheduleContextInfo *contextInfo, TPipe *tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyInAndClear();
    __aicore__ inline void SortCompute();
    __aicore__ inline void ExpertCountCompute();
    __aicore__ inline void CopyOut();

private:
    int64_t sortNum = 0;
    int32_t validCnt = 0;
    int32_t needCoreNum_ = 0;
    int64_t F_ = 0;
    const ScheduleContextInfo *contextInfo_ = nullptr;
};

__aicore__ inline void KernelScanSortMaskOneCore::CopyInAndClear()
{
    LocalTensor<int32_t> inLocal = sortDataCopyInQueue.AllocTensor<int32_t>();
    DataCopyExtParams dataCopyParams{static_cast<uint16_t>(contextInfo_->A),
                                     static_cast<uint32_t>(contextInfo_->BS * contextInfo_->K * INT32_SIZE),
                                     static_cast<uint32_t>(((contextInfo_->M - 1) * F_ + NUM_TWO) * INT32_SIZE), 0, 0};
    DataCopyPadExtParams dataCopyPadParams{true, 0, static_cast<uint8_t>(contextInfo_->BsKPaddingCount), INT_MAX};
    DataCopyPad(inLocal, expertIdsGm[contextInfo_->curMicroBatchID * F_ + NUM_TWO], dataCopyParams, dataCopyPadParams);

    LocalTensor<int32_t> rowIdsLocal = inLocal[this->sortNum];
    ArithProgression<int32_t>(rowIdsLocal, 0, 1, this->totalLength);
    sortDataCopyInQueue.EnQue(inLocal);

    // clear expertIds and flag in token_info_buf
    LocalTensor<int32_t> clearLocal = sortDataCopyOutQueue.AllocTensor<int32_t>();
    Duplicate<int32_t>(clearLocal, INT_MAX, this->totalLength);
    Duplicate<int32_t>(clearLocal[this->sortNum], 0, contextInfo_->A * 8); // 8: block num
    sortDataCopyOutQueue.EnQue(clearLocal);

    clearLocal = sortDataCopyOutQueue.DeQue<int32_t>();
    DataCopyExtParams copyoutParams{
        static_cast<uint16_t>(contextInfo_->A),
        static_cast<uint32_t>(contextInfo_->BS * contextInfo_->K * sizeof(int32_t)), 0,
        static_cast<uint32_t>((contextInfo_->M * F_ - contextInfo_->BS * contextInfo_->K) * sizeof(int32_t)), 0};
    DataCopyExtParams copyOutParamsF{static_cast<uint16_t>(contextInfo_->A), static_cast<uint32_t>(sizeof(int32_t)), 0,
                                     static_cast<uint32_t>((contextInfo_->M * F_ - 1) * sizeof(int32_t)), 0};

    SetWaitFlag<HardEvent::MTE2_MTE3>(HardEvent::MTE2_MTE3);
    DataCopyPad(expertIdsGm[contextInfo_->curMicroBatchID * F_ + NUM_TWO], clearLocal, copyoutParams);
    DataCopyPad(expertIdsGm[contextInfo_->curMicroBatchID * F_], clearLocal[this->sortNum], copyOutParamsF);
    sortDataCopyOutQueue.FreeTensor(clearLocal);
}

__aicore__ inline void KernelScanSortMaskOneCore::SortCompute()
{
    // step1: 取数据并转为float32类型，乘以-1
    LocalTensor<int32_t> inLocal = sortDataCopyInQueue.DeQue<int32_t>();
    LocalTensor<float> expertIdsFp32 = inLocal.ReinterpretCast<float>();
    Cast(expertIdsFp32, inLocal, RoundMode::CAST_ROUND, this->totalLength);
    PipeBarrier<PIPE_V>();

    // gatherMask start
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
        uint64_t mask[NUM_TWO] = {mask0, 0};
        Duplicate(expertIdsFp32[duplicateIndex], MIN_FP32, mask, 1, DST_BLK_STRIDE, DST_REP_STRIDE);
        PipeBarrier<PIPE_V>();
    }
    int32_t selectedCnt = (rsvdCnt + ONE_REPEAT_SORT_NUM - 1) / ONE_REPEAT_SORT_NUM * ONE_REPEAT_SORT_NUM;

    // step2: 做sort操作
    LocalTensor<float> concatLocal = expertIdsFp32;
    LocalTensor<float> tempTensor = tempBuffer.Get<float>(GetSortLen<float>(selectedCnt));
    Concat(concatLocal, expertIdsFp32, tempTensor, selectedCnt / ONE_REPEAT_SORT_NUM);
    PipeBarrier<PIPE_V>();

    LocalTensor<uint32_t> rowIdsLocal = inLocal[this->sortNum].ReinterpretCast<uint32_t>();
    // gathermask index
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

    // step3: 与step1相反，先乘以-1并转为int32
    Muls(sortedExpertIdsLocal, sortedExpertIdsLocal, (float)-1, rsvdCnt);
    PipeBarrier<PIPE_V>();
    LocalTensor<int32_t> sortedExpertIdsLocalInt32 = sortedExpertIdsLocal.ReinterpretCast<int32_t>();
    Cast(sortedExpertIdsLocalInt32, sortedExpertIdsLocal, RoundMode::CAST_ROUND, rsvdCnt);
    PipeBarrier<PIPE_V>();
    sortDataCopyOutQueue.EnQue<float>(outLocal);
    sortDataCopyInQueue.FreeTensor(inLocal);
}

__aicore__ inline void KernelScanSortMaskOneCore::CopyOut()
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

__aicore__ inline void KernelScanSortMaskOneCore::Init(GM_ADDR tokenInfoGm, GM_ADDR workspace,
                                                       SortCustomTilingDataKernel *tilingData,
                                                       const ScheduleContextInfo *contextInfo, TPipe *tPipe)
{
    this->pipe = tPipe;
    contextInfo_ = contextInfo;
    F_ = contextInfo_->BS * contextInfo_->K + 1 + 1;

    needCoreNum_ = tilingData->needCoreNum;
    // 单核排序时 lastCorePerLoopElements = totalLengthWithPad, 是针对BS*K补了pad后的长度
    this->totalLength = tilingData->totalLengthWithPad;
    this->sortNum = Ceil(this->totalLength, ONE_REPEAT_SORT_NUM) * ONE_REPEAT_SORT_NUM;

    expertIdsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(tokenInfoGm));
    rsvdCntGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(workspace), OFFSET_SORTED_EXPERT_IDS);
    sortedexpertIdsGm.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t *>(workspace) + OFFSET_SORTED_EXPERT_IDS + contextInfo_->sortNumWorkSpace,
        this->totalLength);
    sortedRowIdsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(workspace) + OFFSET_SORTED_EXPERT_IDS +
                                       contextInfo_->sortNumWorkSpace + this->totalLength,
                                   this->totalLength);

    if (needCoreNum_ == 0 && GetBlockIdx() == 0) {
        rsvdCntGm.SetValue(0, 0);
        DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_ALL>(
            rsvdCntGm);
    }

    // key and value
    int64_t buffSize = this->sortNum * sizeof(int32_t) * NUM_TWO;
    pipe->InitBuffer(sortDataCopyInQueue, bufferNum, buffSize);
    pipe->InitBuffer(sortDataCopyOutQueue, bufferNum, buffSize);
    pipe->InitBuffer(tempBuffer, buffSize);
    pipe->InitBuffer(sortedBuffer, buffSize);
}

__aicore__ inline void KernelScanSortMaskOneCore::Process()
{
    if (GetBlockIdx() < needCoreNum_) {
        CopyInAndClear();
        SortCompute();
        CopyOut();
    }
    SyncAll();
}
} // namespace FfnWbBatching
#endif // OP_KERNEL_FFN_WB_SCAN_SORT_ONE_CORE_H
