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
 * \file ffn_wb_sort_multi_core.h
 * \brief
 */
#ifndef OP_KERNEL_FFN_WB_SORT_MULTI_CORE_H
#define OP_KERNEL_FFN_WB_SORT_MULTI_CORE_H

#include "ffn_wb_sort_base.h"
#include "ffn_wb_common.h"
#include "ffn_wb_sort_mrgsort.h"
#include "ffn_wb_sort_mrgsort_out.h"

namespace FfnWbBatching {
using namespace AscendC;

class KernelSortMaskMultiCore : public SortMaskBase {
public:
    __aicore__ inline KernelSortMaskMultiCore(){};
    __aicore__ inline void Init(GM_ADDR expert_ids, GM_ADDR workspace, SortCustomTilingDataKernel *tilingData,
                                const ScheduleContextInfo *contextInfo, TPipe *tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void VBSProcess();
    __aicore__ inline void UBSortProcess(int64_t progress, int64_t size, int64_t sortNum);
    __aicore__ inline void OneCoreVMSProcess(int64_t listNum, int64_t perListElements, int64_t lastListElements);
    __aicore__ inline void VMSProcess();
    __aicore__ inline void SortOutProcess();
    __aicore__ inline void VBSCopyIn(int64_t progress, int64_t size, int64_t sortNum);
    __aicore__ inline void UBSortCompute(int64_t progress, int64_t size, int64_t sortNum);
    __aicore__ inline void VBSCopyOut(int64_t progress, int64_t size, int64_t sortNum);
    __aicore__ inline void InitSortMaskMrgSort(SortCustomMrgsort *sorter, int64_t listNum, int64_t coreOffset,
                                               int64_t sortNumCoreOffset, int64_t loopOffset, int64_t loopIdxOffset);
    __aicore__ inline void InitSortMaskMrgSortOut(SortCustomMrgsortOut *sorter, int64_t listNum, int64_t coreOffset);
    __aicore__ inline void CopyOutValidCount();

private:
    GlobalTensor<float> workspaceGms[2];
    GlobalTensor<int32_t> workspaceSortNumGm_;

    SortCustomTilingDataKernel *tilingData_ = nullptr;
    const ScheduleContextInfo *contextInfo_ = nullptr;

    int32_t totalValidCnt_ = 0;
    int32_t curValidCnt_ = 0;

    SortCustomMrgsort mrgsorter;
    SortCustomMrgsortParam mrgsortParam;

    int64_t blockIdx_ = 0;
    int64_t srcWsIndex = 0;

    int64_t listNum;
    int64_t perListElements;
    int64_t lastListElements;
    int64_t vmsSortNumStride_ = 0;

    int64_t sortTotalLength;
    int64_t sortCoreLoops;
    int64_t sortCoreLoopElements;
    int64_t sortCoreLastLoopElements;

    int64_t perCoreExpert;
    int64_t needInitExpertCore;
    int64_t currentCoreExpert;

    static constexpr int64_t MAX_MRGSORT_LIST = 4;
};

__aicore__ inline void KernelSortMaskMultiCore::VBSCopyIn(int64_t progress, int64_t size, int64_t sortNum)
{
    LocalTensor<int32_t> inLocal = sortDataCopyInQueue.AllocTensor<int32_t>();
    int64_t inOffset = progress * sortCoreLoopElements;
    DataCopyExtParams dataCopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(size * sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams<int32_t> dataCopyPadParams{false, 0, 0, 0};
    DataCopyPad(inLocal[0], expertIdsGm[inOffset], dataCopyParams, dataCopyPadParams);

    LocalTensor<int32_t> rowIdsLocal = inLocal[sortNum];
    int64_t startValue = blockIdx_ * tilingData_->perCoreElements + inOffset;
    ArithProgression<int32_t>(rowIdsLocal, startValue, 1, size);
    sortDataCopyInQueue.EnQue(inLocal);
}

__aicore__ inline void KernelSortMaskMultiCore::UBSortCompute(int64_t progress, int64_t size, int64_t sortNum)
{
    LocalTensor<int32_t> inLocal = sortDataCopyInQueue.DeQue<int32_t>();
    LocalTensor<int32_t> expertIdsLocal = inLocal[0];
    LocalTensor<float> expertIdsLocalFp32;

    expertIdsLocalFp32 = expertIdsLocal.ReinterpretCast<float>();
    Cast(expertIdsLocalFp32, expertIdsLocal, RoundMode::CAST_ROUND, size);

    LocalTensor<uint32_t> maskLocalTensor = sortedBuffer.Get<uint32_t>();
    uint64_t rsvdCnt = 0;

    Muls(expertIdsLocalFp32, expertIdsLocalFp32, (float)-1, size);

    LocalTensor<uint8_t> maskLocalTensorUInt8 = maskLocalTensor.ReinterpretCast<uint8_t>();
    AscendC::CompareScalar(maskLocalTensorUInt8, expertIdsLocalFp32, static_cast<float>(-expertStart_),
                           AscendC::CMPMODE::GT,
                           (size + ONE_REPEAT_COMPARE_NUM - 1) / ONE_REPEAT_COMPARE_NUM * ONE_REPEAT_COMPARE_NUM);

    GatherMaskParams gatherMaskParams;
    gatherMaskParams.repeatTimes = 1;
    gatherMaskParams.src0BlockStride = 1;
    gatherMaskParams.src0RepeatStride = 8; // 8 blocks
    gatherMaskParams.src1RepeatStride = 0;
    GatherMask(expertIdsLocalFp32, expertIdsLocalFp32, maskLocalTensor, true, size, gatherMaskParams, rsvdCnt);
    curValidCnt_ = rsvdCnt;
    if (rsvdCnt == 0) {
        sortDataCopyInQueue.FreeTensor(inLocal);
        return;
    }
    this->totalValidCnt_ += rsvdCnt;
    int64_t duplicateNum = rsvdCnt % ONE_REPEAT_SORT_NUM;
    if (duplicateNum > 0) {
        int duplicateIndex = rsvdCnt - duplicateNum;
        uint64_t mask0 = UINT64_MAX;
        mask0 = mask0 << duplicateNum;
        mask0 = mask0 & (UINT64_MAX >> ONE_REPEAT_SORT_NUM);
        uint64_t mask[2] = {mask0, 0};
        Duplicate(expertIdsLocalFp32[duplicateIndex], MIN_FP32, mask, 1, DST_BLK_STRIDE, DST_REP_STRIDE);
    }
    int32_t selectedCnt = (rsvdCnt + ONE_REPEAT_SORT_NUM - 1) / ONE_REPEAT_SORT_NUM * ONE_REPEAT_SORT_NUM;

    LocalTensor<uint32_t> rowIdsLocal = inLocal[sortNum].ReinterpretCast<uint32_t>();
    GatherMask(rowIdsLocal, rowIdsLocal, maskLocalTensor, true, size, gatherMaskParams, rsvdCnt);
    LocalTensor<float> concatLocal = expertIdsLocalFp32;
    LocalTensor<float> sortedLocal = sortedBuffer.Get<float>(GetSortLen<float>(sortNum));
    LocalTensor<float> outLocal = sortDataCopyOutQueue.AllocTensor<float>();
    Sort<float, true>(outLocal, concatLocal, rowIdsLocal, sortedLocal, selectedCnt / ONE_REPEAT_SORT_NUM);

    sortDataCopyOutQueue.EnQue<float>(outLocal);
    sortDataCopyInQueue.FreeTensor(inLocal);
}

__aicore__ inline void KernelSortMaskMultiCore::VBSCopyOut(int64_t progress, int64_t size, int64_t sortNum)
{
    if (curValidCnt_ > 0) {
        LocalTensor<float> outLocal = sortDataCopyOutQueue.DeQue<float>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(GetSortLen<float>(curValidCnt_) * sizeof(int32_t)), 0, 0,
                                     0};
        int64_t wkOffset = blockIdx_ * GetSortLen<float>(tilingData_->perCoreElements) +
                           GetSortLen<float>(progress * sortCoreLoopElements);
        DataCopyPad(workspaceGms[0][wkOffset], outLocal, copyParams);
        sortDataCopyOutQueue.FreeTensor(outLocal);
    }

    LocalTensor<int32_t> tempTensor = tempBuffer.Get<int32_t>(BLOCK_BYTES / sizeof(int32_t));
    tempTensor.SetValue(0, curValidCnt_);
    SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
    DataCopyExtParams copyParams1{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
    DataCopyPad(workspaceSortNumGm_[blockIdx_ * tilingData_->sortNumWorkSpacePerCore + progress], tempTensor,
                copyParams1);
}

__aicore__ inline void KernelSortMaskMultiCore::InitSortMaskMrgSort(SortCustomMrgsort *sorter, int64_t listNum,
                                                                    int64_t coreOffset, int64_t sortNumCoreOffset,
                                                                    int64_t loopOffset, int64_t loopIdxOffset)
{
    GlobalTensor<float> srcWsGm = workspaceGms[srcWsIndex][blockIdx_ * coreOffset + loopOffset];
    GlobalTensor<int32_t> srcSortNumGm = workspaceSortNumGm_[blockIdx_ * sortNumCoreOffset + loopIdxOffset];
    LocalTensor<float> inLocal = sortDataCopyInQueue.AllocTensor<float>();
    LocalTensor<float> outLocal = sortDataCopyOutQueue.AllocTensor<float>();
    for (int64_t i = 0; i < listNum; i++) {
        LocalTensor<float> inLocalT = inLocal[GetSortLen<float>(tilingData_->oneLoopMaxElementsMrg) * i];
        sorter->SetInput(srcWsGm, srcSortNumGm, inLocalT);
    }
    GlobalTensor<float> dstWsGm = workspaceGms[1 - srcWsIndex][blockIdx_ * coreOffset + loopOffset];
    LocalTensor<int32_t> outSortNumLocal = tempBuffer.Get<int32_t>(BLOCK_BYTES / sizeof(int32_t));
    sorter->SetOutput(dstWsGm, outLocal, outSortNumLocal);
    sortDataCopyInQueue.FreeTensor(inLocal);
    sortDataCopyOutQueue.FreeTensor(outLocal);
    tempBuffer.FreeTensor(outSortNumLocal);
}

__aicore__ inline void KernelSortMaskMultiCore::InitSortMaskMrgSortOut(SortCustomMrgsortOut *sorter, int64_t listNum,
                                                                       int64_t coreOffset)
{
    GlobalTensor<float> srcWsGm = workspaceGms[srcWsIndex];
    GlobalTensor<int32_t> srcSortNumGm = workspaceSortNumGm_;
    LocalTensor<float> inLocal = sortDataCopyInQueue.AllocTensor<float>();
    LocalTensor<float> outLocal = sortDataCopyOutQueue.AllocTensor<float>();

    for (int64_t i = 0; i < listNum; i++) {
        LocalTensor<float> inLocalT = inLocal[GetSortLen<float>(tilingData_->oneLoopMaxElementsMrg) * i];
        sorter->SetInput(srcWsGm, srcSortNumGm, inLocalT);
    }

    LocalTensor<float> outLocalV = outLocal[tilingData_->oneLoopMaxElementsMrg * MAX_MRGSORT_LIST];
    sorter->SetOutput(this->sortedexpertIdsGm, this->sortedRowIdsGm, outLocal, outLocalV);

    LocalTensor<float> tempBuffer =
        sortedBuffer.Get<float>(GetSortLen<float>(tilingData_->oneLoopMaxElementsMrg) * MAX_MRGSORT_LIST);
    sorter->SetBuffer(tempBuffer);
    sortDataCopyInQueue.FreeTensor(inLocal);
    sortDataCopyOutQueue.FreeTensor(outLocal);
}

__aicore__ inline void KernelSortMaskMultiCore::OneCoreVMSProcess(int64_t listNum, int64_t perListElements,
                                                                  int64_t lastListElements)
{
    int64_t coreOffset = GetSortLen<float>(tilingData_->perCoreElements);
    int64_t sortNumCoreOffset = tilingData_->sortNumWorkSpacePerCore;
    mrgsortParam.oneLoopMaxElements = tilingData_->oneLoopMaxElementsMrg;

    int64_t curSortNumStride = 1;
    for (int64_t i = 0; listNum >= 1; i++) {
        int64_t loops = (listNum + MAX_MRGSORT_LIST - 1) / MAX_MRGSORT_LIST;
        int64_t remainListNum = listNum - (loops - 1) * MAX_MRGSORT_LIST;

        mrgsortParam.perListElements = perListElements;
        mrgsortParam.sortNumStride = curSortNumStride;

        int64_t loopOffset = GetSortLen<float>(mrgsortParam.perListElements * MAX_MRGSORT_LIST);
        int64_t loopIdxOffset = mrgsortParam.sortNumStride * MAX_MRGSORT_LIST;
        for (int64_t loop = 0; loop < loops - 1; loop++) {
            InitSortMaskMrgSort(&mrgsorter, MAX_MRGSORT_LIST, coreOffset, sortNumCoreOffset, loop * loopOffset,
                                loop * loopIdxOffset);
            mrgsorter.Init(&mrgsortParam);
            mrgsorter.Process();
        }

        InitSortMaskMrgSort(&mrgsorter, remainListNum, coreOffset, sortNumCoreOffset, (loops - 1) * loopOffset,
                            (loops - 1) * loopIdxOffset);
        mrgsorter.Init(&mrgsortParam);
        mrgsorter.Process();

        listNum = loops;
        perListElements = perListElements * MAX_MRGSORT_LIST;
        curSortNumStride = curSortNumStride * MAX_MRGSORT_LIST;
        srcWsIndex = (srcWsIndex + 1) % WORK_GM_NUM;
        if (loops == 1) {
            break;
        }
    }
}

__aicore__ inline void KernelSortMaskMultiCore::UBSortProcess(int64_t progress, int64_t size, int64_t sortNum)
{
    VBSCopyIn(progress, size, sortNum);
    UBSortCompute(progress, size, sortNum);
    VBSCopyOut(progress, size, sortNum);
}

__aicore__ inline void KernelSortMaskMultiCore::VBSProcess()
{
    if (blockIdx_ < tilingData_->needCoreNum) {
        int64_t sortNum = Ceil(sortCoreLoopElements, ONE_REPEAT_SORT_NUM) * ONE_REPEAT_SORT_NUM;
        for (int64_t loop = 0; loop < sortCoreLoops - 1; loop++) {
            UBSortProcess(loop, sortCoreLoopElements, sortNum);
        }

        sortNum = Ceil(sortCoreLastLoopElements, ONE_REPEAT_SORT_NUM) * ONE_REPEAT_SORT_NUM;
        UBSortProcess(sortCoreLoops - 1, sortCoreLastLoopElements, sortNum);

        CopyOutValidCount();

        if (sortCoreLoops > 1) {
            OneCoreVMSProcess(sortCoreLoops, sortCoreLoopElements, sortCoreLastLoopElements);
        }
    }
    SyncAll();
}

__aicore__ inline void KernelSortMaskMultiCore::VMSProcess()
{
    int64_t currentStageNeedCoreNum = tilingData_->needCoreNumMrg;
    perListElements = tilingData_->perCoreElements;
    listNum = tilingData_->needCoreNum;
    vmsSortNumStride_ = tilingData_->sortNumWorkSpacePerCore;

    for (; listNum > MAX_MRGSORT_LIST;) {
        currentStageNeedCoreNum = Ceil(listNum, MAX_MRGSORT_LIST);
        int64_t coreOffset = GetSortLen<float>(perListElements * MAX_MRGSORT_LIST);
        int64_t sortNumCoreOffset = vmsSortNumStride_ * MAX_MRGSORT_LIST;
        int64_t remainListNum = listNum - (currentStageNeedCoreNum - 1) * MAX_MRGSORT_LIST;

        mrgsortParam.perListElements = perListElements;
        mrgsortParam.sortNumStride = vmsSortNumStride_;
        mrgsortParam.oneLoopMaxElements = tilingData_->oneLoopMaxElementsMrg;

        if (blockIdx_ < currentStageNeedCoreNum - 1) {
            InitSortMaskMrgSort(&mrgsorter, MAX_MRGSORT_LIST, coreOffset, sortNumCoreOffset, 0, 0);
            mrgsorter.Init(&mrgsortParam);
            mrgsorter.Process();
        } else if (blockIdx_ == currentStageNeedCoreNum - 1) {
            InitSortMaskMrgSort(&mrgsorter, remainListNum, coreOffset, sortNumCoreOffset, 0, 0);
            mrgsorter.Init(&mrgsortParam);
            mrgsorter.Process();
        }
        listNum = currentStageNeedCoreNum;
        srcWsIndex = (srcWsIndex + 1) % WORK_GM_NUM;

        perListElements = perListElements * MAX_MRGSORT_LIST;
        vmsSortNumStride_ = vmsSortNumStride_ * MAX_MRGSORT_LIST;

        SyncAll();
    }
}

__aicore__ inline void KernelSortMaskMultiCore::SortOutProcess()
{
    if (blockIdx_ < 1) {
        mrgsortParam.perListElements = perListElements;
        mrgsortParam.sortNumStride = vmsSortNumStride_;
        mrgsortParam.oneLoopMaxElements = tilingData_->oneLoopMaxElementsMrg;

        SortCustomMrgsortOut sorter;
        InitSortMaskMrgSortOut(&sorter, listNum, GetSortLen<float>(perListElements));
        sorter.Init(&mrgsortParam, pipe);
        sorter.Process();
    }
    SyncAll();
}

__aicore__ inline void KernelSortMaskMultiCore::CopyOutValidCount()
{
    LocalTensor<int32_t> outLocal = sortDataCopyOutQueue.AllocTensor<int32_t>();
    outLocal.SetValue(0, this->totalValidCnt_);
    DataCopyParams params;
    params.blockCount = 1;
    params.blockLen = 1 * sizeof(int32_t);
    SetAtomicAdd<int32_t>();
    DataCopyPad(rsvdCntGm[0], outLocal[0], params);
    SetAtomicNone();
    sortDataCopyOutQueue.FreeTensor(outLocal);
}

__aicore__ inline void KernelSortMaskMultiCore::Init(GM_ADDR expert_ids, GM_ADDR workspace,
                                                     SortCustomTilingDataKernel *tilingData,
                                                     const ScheduleContextInfo *contextInfo, TPipe *tPipe)
{
    this->pipe = tPipe;
    contextInfo_ = contextInfo;
    tilingData_ = tilingData;

    this->totalLength = tilingData->totalLength;

    blockIdx_ = GetBlockIdx();
    if (blockIdx_ == tilingData_->needCoreNum - 1) {
        sortCoreLoops = tilingData_->lastCoreLoops;
        sortCoreLoopElements = tilingData_->lastCorePerLoopElements;
        sortCoreLastLoopElements = tilingData_->lastCoreLastLoopElements;
        this->sortTotalLength = tilingData_->lastCoreElements;
    } else {
        sortCoreLoops = tilingData_->perCoreLoops;
        sortCoreLoopElements = tilingData_->perCorePerLoopElements;
        sortCoreLastLoopElements = tilingData_->perCoreLastLoopElements;
        this->sortTotalLength = tilingData_->perCoreElements;
    }

    expertIdsGm.SetGlobalBuffer((__gm__ int32_t *)expert_ids + blockIdx_ * tilingData_->perCoreElements,
                                this->sortTotalLength);
    rsvdCntGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(workspace), OFFSET_SORTED_EXPERT_IDS);
    workspaceSortNumGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(workspace) + OFFSET_SORTED_EXPERT_IDS,
                                        contextInfo_->sortNumWorkSpace);

    if (blockIdx_ == 0) {
        InitGlobalMemory(rsvdCntGm, OFFSET_SORTED_EXPERT_IDS, 0);
        GM_ADDR targetAddr = workspace + OFFSET_SORTED_EXPERT_IDS * sizeof(int32_t) +
                             contextInfo_->sortNumWorkSpace * sizeof(int32_t) * (NUM_TWO * NUM_FOUR + 1);
        groupListTmpGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(targetAddr), contextInfo_->expertNum);
        InitGlobalMemory(groupListTmpGm, contextInfo_->expertNum, 0);
        SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
    }
    SyncAll();

    sortedexpertIdsGm.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t *>(workspace) + OFFSET_SORTED_EXPERT_IDS + contextInfo_->sortNumWorkSpace,
        this->totalLength);
    sortedRowIdsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(workspace) + OFFSET_SORTED_EXPERT_IDS +
                                       contextInfo_->sortNumWorkSpace + this->totalLength,
                                   this->totalLength);

    int64_t kvFactor = 2;
    workspaceGms[0].SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace) + OFFSET_SORTED_EXPERT_IDS +
                                        contextInfo_->sortNumWorkSpace + this->totalLength * kvFactor,
                                    this->totalLength * kvFactor);
    workspaceGms[1].SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace) + OFFSET_SORTED_EXPERT_IDS +
                                        contextInfo_->sortNumWorkSpace + this->totalLength * (kvFactor + kvFactor),
                                    this->totalLength * kvFactor);

    int64_t bufferSize =
        Ceil(Max(tilingData_->oneLoopMaxElementsMrg * MAX_MRGSORT_LIST, sortCoreLoopElements), ONE_REPEAT_SORT_NUM) *
        ONE_REPEAT_SORT_NUM * sizeof(int32_t) * kvFactor;
    pipe->InitBuffer(sortDataCopyInQueue, bufferNum, bufferSize);
    pipe->InitBuffer(sortDataCopyOutQueue, bufferNum, bufferSize);
    pipe->InitBuffer(sortedBuffer, bufferSize);
    pipe->InitBuffer(tempBuffer, bufferSize);
}

__aicore__ inline void KernelSortMaskMultiCore::Process()
{
    VBSProcess();
    VMSProcess();
    SortOutProcess();
}
} // namespace FfnWbBatching
#endif // OP_KERNEL_FFN_WB_SORT_MULTI_CORE_H
