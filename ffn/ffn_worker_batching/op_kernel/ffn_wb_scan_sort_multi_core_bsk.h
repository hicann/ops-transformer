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
 * \file ffn_wb_scan_sort_multi_core_bsk.h
 * \brief
 */
#ifndef OP_KERNEL_FFN_WB_SCAN_SORT_MULTI_CORE_BSK_H
#define OP_KERNEL_FFN_WB_SCAN_SORT_MULTI_CORE_BSK_H

#include "ffn_wb_sort_base.h"
#include "ffn_wb_sort_mrgsort.h"
#include "ffn_wb_sort_mrgsort_out.h"

namespace FfnWbBatching {
using namespace AscendC;

class KernelScanSortMaskMultiCoreBsK : public SortMaskBase {
public:
    __aicore__ inline KernelScanSortMaskMultiCoreBsK(){};
    __aicore__ inline void Init(GM_ADDR tokenInfoGm, GM_ADDR workspace, SortCustomTilingDataKernel *tilingData,
                                const ScheduleContextInfo *contextInfo, TPipe *tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void VBSProcess();
    __aicore__ inline void UBSortProcess(int64_t progress, int64_t size, int64_t sortNum, int64_t loopSessionCnt);
    __aicore__ inline void OneCoreVMSProcess(int64_t listNum, int64_t perListElements, int64_t lastListElements);
    __aicore__ inline void VMSProcess();
    __aicore__ inline void SortOutProcess();
    __aicore__ inline void VBSCopyInAndClear(int64_t progress, int64_t size, int64_t sortNum, int64_t loopSessionCnt);
    __aicore__ inline void UBSortCompute(int64_t progress, int64_t size, int64_t sortNum);
    __aicore__ inline void VBSCopyOut(int64_t progress, int64_t size, int64_t sortNum);
    __aicore__ inline void InitSortMaskMrgSort(SortCustomMrgsort *sorter, int64_t listNum, int64_t coreOffset,
                                               int64_t sortNumCoreOffset, int64_t loopOffset, int64_t loopIdxOffset);
    __aicore__ inline void InitSortMaskMrgSortOut(SortCustomMrgsortOut *sorter, int64_t listNum, int64_t coreOffset);
    __aicore__ inline void CopyOutValidCount();
    __aicore__ inline void ClearTokenInfoFlag();

private:
    GlobalTensor<float> workspaceGms[NUM_TWO];
    GlobalTensor<int32_t> workspaceSortNumGm_;
    GlobalTensor<int32_t> expertIdsGmFStart_;

    SortCustomTilingDataKernel *tilingData_ = nullptr;
    const ScheduleContextInfo *contextInfo_ = nullptr;

    int32_t totalValidCnt_ = 0;
    int32_t curValidCnt_ = 0;

    int64_t F = 0;
    int64_t BsKLenWithPading = 0; // BS*K_plus_1按block对齐后的个数

    int64_t perCoreSortNum = 0; // perCoreSessionNum * BsKLenWithPading 主核 每个核排序的元素总个数; 不保证32个数对齐
    int64_t lastCoreSortNum = 0; // lastCoreSessionNum * BsKLenWithPading;
    int64_t sessionLoops = 0;    // 当前核 ub循环次数

    int64_t perLoopElements = 0;
    int64_t lastLoopElement = 0;

    // for MoeMrgsort
    SortCustomMrgsort mrgsorter;
    SortCustomMrgsortParam mrgsortParam;

    int64_t blockIdx_ = 0;
    int64_t srcWsIndex = 0;
    int64_t bufferSize_ = 0;

    int64_t listNum = 0;
    int64_t perListElements = 0;
    int64_t lastListElements = 0;
    int64_t vmsSortNumStride_ = 0; // 核间

    int64_t perCorePad_ = 0;  // 记录pad
    int64_t lastCorePad_ = 0; // 记录pad
    int64_t splitNum_ = 1;    // 原始块被切分后的块数
    int64_t recordCnt_ = 0;   // 记录1个原始块的有效专家数

    static constexpr int64_t MAX_MRGSORT_LIST = 4;
};

__aicore__ inline void KernelScanSortMaskMultiCoreBsK::VBSCopyInAndClear(int64_t progress, int64_t size,
                                                                         int64_t sortNum, int64_t loopSessionCnt)
{
    int64_t curCorePad = 0;

    // 根据当前块位置计算参数
    if ((progress % splitNum_) != (splitNum_ - 1)) {
        curCorePad = perCorePad_;
    } else {
        curCorePad = lastCorePad_;
    }

    LocalTensor<int32_t> inLocal = sortDataCopyInQueue.AllocTensor<int32_t>();
    int64_t inOffset = 0;

    // 偏移量应该是上一次实际参与排序的元素个数（不包含pad）
    inOffset = (progress % splitNum_) * (perLoopElements - perCorePad_) +
               (progress / splitNum_) * (contextInfo_->M * this->F); // 单循环处理元素个数

    DataCopyExtParams dataCopyParams{static_cast<uint16_t>(loopSessionCnt),                        // 搬运块数 = 1
                                     static_cast<uint32_t>((size - curCorePad) * sizeof(int32_t)), // 实际元素个数
                                     0, 0, 0};
    DataCopyPadExtParams<int32_t> dataCopyPadParams{true, 0, static_cast<uint8_t>(curCorePad),
                                                    INT_MAX}; // 当前pad长度=perCorePad_

    DataCopyPad(inLocal[0], expertIdsGm[inOffset], dataCopyParams, dataCopyPadParams);

    LocalTensor<int32_t> rowIdsLocal = inLocal[sortNum];
    // 索引生成只有(A, BsKPad)

    int64_t startValue = blockIdx_ * tilingData_->perCoreSessionNum * BsKLenWithPading +
                         (progress % splitNum_) * perLoopElements +
                         (progress / splitNum_) * BsKLenWithPading; // 起始值为上一次参与排序的元素个数（包括pad）

    // size是向上32位补齐后的元素个数
    ArithProgression<int32_t>(rowIdsLocal, startValue, 1, size); // size包括: BsKLenWithPading

    sortDataCopyInQueue.EnQue(inLocal);

    // clear expertIds
    LocalTensor<int32_t> clearLocal = sortDataCopyOutQueue.AllocTensor<int32_t>();
    Duplicate<int32_t>(clearLocal, INT_MAX, size);
    // 初始化最大元素值张量
    sortDataCopyOutQueue.EnQue(clearLocal);

    clearLocal = sortDataCopyOutQueue.DeQue<int32_t>();

    DataCopyExtParams copyoutParams{static_cast<uint16_t>(loopSessionCnt),
                                    static_cast<uint32_t>((size - curCorePad) * sizeof(int32_t)), // 实际元素个数
                                    0, 0, 0};
    SetWaitFlag<HardEvent::MTE2_MTE3>(HardEvent::MTE2_MTE3);
    DataCopyPad(expertIdsGm[inOffset], clearLocal, copyoutParams);

    sortDataCopyOutQueue.FreeTensor(clearLocal);
}

__aicore__ inline void KernelScanSortMaskMultiCoreBsK::UBSortCompute(int64_t progress, int64_t size, int64_t sortNum)
{
    LocalTensor<int32_t> inLocal = sortDataCopyInQueue.DeQue<int32_t>();
    LocalTensor<int32_t> expertIdsLocal = inLocal[0];
    LocalTensor<float> expertIdsLocalFp32;

    expertIdsLocalFp32 = expertIdsLocal.ReinterpretCast<float>();
    Cast(expertIdsLocalFp32, expertIdsLocal, RoundMode::CAST_ROUND, size);

    // gathermask start
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
        uint64_t mask[NUM_TWO] = {mask0, 0};
        Duplicate(expertIdsLocalFp32[duplicateIndex], MIN_FP32, mask, 1, DST_BLK_STRIDE, DST_REP_STRIDE);
    }
    int32_t selectedCnt = (rsvdCnt + ONE_REPEAT_SORT_NUM - 1) / ONE_REPEAT_SORT_NUM * ONE_REPEAT_SORT_NUM;

    // step2: sort
    LocalTensor<uint32_t> rowIdsLocal = inLocal[sortNum].ReinterpretCast<uint32_t>();

    GatherMask(rowIdsLocal, rowIdsLocal, maskLocalTensor, true, size, gatherMaskParams, rsvdCnt);
    LocalTensor<float> concatLocal = expertIdsLocalFp32;
    LocalTensor<float> sortedLocal = sortedBuffer.Get<float>(GetSortLen<float>(sortNum));
    LocalTensor<float> outLocal = sortDataCopyOutQueue.AllocTensor<float>();

    Sort<float, true>(outLocal, concatLocal, rowIdsLocal, sortedLocal, selectedCnt / ONE_REPEAT_SORT_NUM);

    sortDataCopyOutQueue.EnQue<float>(outLocal);
    sortDataCopyInQueue.FreeTensor(inLocal);
}

__aicore__ inline void KernelScanSortMaskMultiCoreBsK::VBSCopyOut(int64_t progress, int64_t size, int64_t sortNum)
{
    if (curValidCnt_ > 0) {
        LocalTensor<float> outLocal = sortDataCopyOutQueue.DeQue<float>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(GetSortLen<float>(curValidCnt_) * sizeof(int32_t)), 0, 0,
                                     0};
        int64_t wkOffset =
            blockIdx_ * GetSortLen<float>(perCoreSortNum) + GetSortLen<float>(progress * (perLoopElements));

        DataCopyPad(workspaceGms[0][wkOffset], outLocal, copyParams);

        sortDataCopyOutQueue.FreeTensor(outLocal);
    }

    // 个数为0，也要设置到gm上，防止脏数据
    LocalTensor<int32_t> tempTensor = tempBuffer.Get<int32_t>(BLOCK_BYTES / sizeof(int32_t));

    tempTensor.SetValue(0, curValidCnt_);
    SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
    DataCopyExtParams copyParams1{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
    DataCopyPad(workspaceSortNumGm_[blockIdx_ * tilingData_->sortNumWorkSpacePerCore + progress], tempTensor,
                copyParams1);
}

__aicore__ inline void KernelScanSortMaskMultiCoreBsK::InitSortMaskMrgSort(SortCustomMrgsort *sorter, int64_t listNum,
                                                                           int64_t coreOffset,
                                                                           int64_t sortNumCoreOffset,
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

__aicore__ inline void KernelScanSortMaskMultiCoreBsK::InitSortMaskMrgSortOut(SortCustomMrgsortOut *sorter,
                                                                              int64_t listNum, int64_t coreOffset)
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

    LocalTensor<float> useTempBuffer =
        sortedBuffer.Get<float>(GetSortLen<float>(tilingData_->oneLoopMaxElementsMrg) * MAX_MRGSORT_LIST);
    sorter->SetBuffer(useTempBuffer);
    sortDataCopyInQueue.FreeTensor(inLocal);
    sortDataCopyOutQueue.FreeTensor(outLocal);
}

__aicore__ inline void KernelScanSortMaskMultiCoreBsK::OneCoreVMSProcess(int64_t listNum, int64_t perListElements,
                                                                         int64_t lastListElements)
{
    int64_t coreOffset = GetSortLen<float>(perCoreSortNum);
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

    if (blockIdx_ == tilingData_->needCoreNum - 1) {
        int64_t perCoreGmIdx = CeilDiv(tilingData_->perCoreSessionNum * splitNum_, NUM_FOUR) % NUM_TWO == 1 ? 1 : 0;
        int64_t lastCoreGmIdx = CeilDiv(tilingData_->lastCoreSessionNum * splitNum_, NUM_FOUR) % NUM_TWO == 1 ? 1 : 0;
        if (lastCoreGmIdx != perCoreGmIdx) {
            GlobalTensor<float> srcWsGm = workspaceGms[lastCoreGmIdx][blockIdx_ * coreOffset];
            GlobalTensor<float> dstWsGm = workspaceGms[perCoreGmIdx][blockIdx_ * coreOffset];
            LocalTensor<float> inLocal = sortDataCopyInQueue.AllocTensor<float>();

            // 搬入
            SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);

            DataCopyParams intriParamsIn;
            intriParamsIn.blockCount = 1;
            intriParamsIn.blockLen = GetSortLen<float>(this->totalValidCnt_ * NUM_TWO) * sizeof(float);
            DataCopyPadParams intriPadParamsIn{true, 0, 0, 0};
            DataCopyPad(inLocal, srcWsGm, intriParamsIn, intriPadParamsIn);

            // 搬出
            SetWaitFlag<HardEvent::MTE2_MTE3>(HardEvent::MTE2_MTE3);

            DataCopyParams intriParamsOut;
            intriParamsOut.blockCount = 1;
            intriParamsOut.blockLen = GetSortLen<float>(this->totalValidCnt_ * NUM_TWO) * sizeof(float);
            DataCopyPad(dstWsGm, inLocal, intriParamsOut);

            sortDataCopyInQueue.FreeTensor(inLocal);
        }
    }
}

__aicore__ inline void KernelScanSortMaskMultiCoreBsK::UBSortProcess(int64_t progress, int64_t size, int64_t sortNum,
                                                                     int64_t loopSessionCnt)
{
    VBSCopyInAndClear(progress, size, sortNum, loopSessionCnt);
    UBSortCompute(progress, size, sortNum);
    VBSCopyOut(progress, size, sortNum);
}

__aicore__ inline void KernelScanSortMaskMultiCoreBsK::VBSProcess()
{
    if (blockIdx_ < tilingData_->needCoreNum) {
        int64_t sortNum = Ceil(perLoopElements, ONE_REPEAT_SORT_NUM) * ONE_REPEAT_SORT_NUM;
        int64_t cureLoopElement = 0;
        for (int64_t loop = 0; loop < sessionLoops; loop++) {
            // 处理中间的尾块
            if ((loop % splitNum_) == (splitNum_ - 1)) {
                sortNum = Ceil(lastLoopElement, ONE_REPEAT_SORT_NUM) * ONE_REPEAT_SORT_NUM;
                cureLoopElement = lastLoopElement;
            } else {
                sortNum = Ceil(perLoopElements, ONE_REPEAT_SORT_NUM) * ONE_REPEAT_SORT_NUM;
                cureLoopElement = perLoopElements;
            }
            UBSortProcess(loop, cureLoopElement, sortNum, 1);
        }

        CopyOutValidCount();

        if (sessionLoops > 1) {
            OneCoreVMSProcess(sessionLoops, perLoopElements, lastLoopElement);
        }
    }
    SyncAll();
}

__aicore__ inline void KernelScanSortMaskMultiCoreBsK::VMSProcess()
{
    int64_t currentStageNeedCoreNum = tilingData_->needCoreNumMrg;
    perListElements = perCoreSortNum;
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

__aicore__ inline void KernelScanSortMaskMultiCoreBsK::SortOutProcess()
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

__aicore__ inline void KernelScanSortMaskMultiCoreBsK::CopyOutValidCount()
{
    LocalTensor<int32_t> outLocal = sortDataCopyOutQueue.AllocTensor<int32_t>();

    outLocal.SetValue(0, totalValidCnt_);
    DataCopyParams params;
    params.blockCount = 1;
    params.blockLen = 1 * sizeof(int32_t);

    SetAtomicAdd<int32_t>();
    DataCopyPad(rsvdCntGm[0], outLocal[0], params);
    SetAtomicNone();

    sortDataCopyOutQueue.FreeTensor(outLocal);
}

__aicore__ inline void KernelScanSortMaskMultiCoreBsK::Init(GM_ADDR tokenInfoGm, GM_ADDR workspace,
                                                            SortCustomTilingDataKernel *tilingData,
                                                            const ScheduleContextInfo *contextInfo, TPipe *tPipe)
{
    this->pipe = tPipe;
    tilingData_ = tilingData;
    contextInfo_ = contextInfo;

    F = contextInfo_->BS * contextInfo_->K + 1 + 1;
    this->totalLength = tilingData_->totalLengthWithPad;

    BsKLenWithPading = contextInfo_->BS * contextInfo_->K + contextInfo_->BsKPaddingCount;

    blockIdx_ = GetBlockIdx();

    int64_t curCoreSessionNum = 0;
    if (blockIdx_ == tilingData_->needCoreNum - 1) {
        curCoreSessionNum = tilingData_->lastCoreSessionNum;
    } else {
        curCoreSessionNum = tilingData_->perCoreSessionNum;
    }

    int64_t perCoreBsKLen = contextInfo_->sortLoopMaxElement;
    // 尾块也单独占一个UB
    perCoreSortNum =
        tilingData_->perCoreSessionNum * CeilDiv(contextInfo_->BS * contextInfo_->K, perCoreBsKLen) * perCoreBsKLen;
    lastCoreSortNum =
        tilingData_->lastCoreSessionNum * CeilDiv(contextInfo_->BS * contextInfo_->K, perCoreBsKLen) * perCoreBsKLen;

    // 切BS*K，计算每个循环可容纳BS*K大小
    // 非尾块元素个数均为UB最大元素个数
    perLoopElements = Align(perCoreBsKLen, sizeof(int32_t));

    // 切分后可能需要的pad
    perCorePad_ = perLoopElements - perCoreBsKLen;

    // 尾块的元素个数是原始块总元素个数 对 UB对齐元素个数 取余
    int64_t temp = (contextInfo_->BS * contextInfo_->K) % perCoreBsKLen;
    int64_t lastCoreBsKLen = temp == 0 ? perCoreBsKLen : temp;
    lastLoopElement = Align(lastCoreBsKLen, sizeof(int32_t));

    // 切分后尾块可能需要的pad
    lastCorePad_ = lastLoopElement - lastCoreBsKLen;

    // 原始块被切分后的块数
    splitNum_ = CeilDiv(contextInfo_->BS * contextInfo_->K, perCoreBsKLen);

    // 重新计算单核排序所需变量
    // 需要循环多少次，ub总数 = 原始session数 * 原始块被切分后的块数
    sessionLoops = curCoreSessionNum * splitNum_;

    int64_t expertIdStartPos = contextInfo_->curMicroBatchID * F + 1 + 1;
    expertIdsGmFStart_.SetGlobalBuffer((__gm__ int32_t *)tokenInfoGm + contextInfo_->curMicroBatchID * F);
    expertIdsGm.SetGlobalBuffer((__gm__ int32_t *)tokenInfoGm + expertIdStartPos +
                                blockIdx_ * tilingData_->perCoreSessionNum * contextInfo_->M * F);
    // rsvdCntGm 在scan阶段已经清零
    rsvdCntGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(workspace), SCAN_BATCHID_GM_OFFSET);
    workspaceSortNumGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(workspace) + OFFSET_SORTED_EXPERT_IDS,
                                        contextInfo_->sortNumWorkSpace);

    sortedexpertIdsGm.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t *>(workspace) + OFFSET_SORTED_EXPERT_IDS + contextInfo_->sortNumWorkSpace,
        this->totalLength);
    sortedRowIdsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(workspace) + OFFSET_SORTED_EXPERT_IDS +
                                       contextInfo_->sortNumWorkSpace + this->totalLength,
                                   this->totalLength);

    workspaceGms[0].SetGlobalBuffer((__gm__ float *)workspace + OFFSET_SORTED_EXPERT_IDS +
                                        contextInfo_->sortNumWorkSpace + this->totalLength * NUM_TWO,
                                    this->totalLength * NUM_TWO);
    workspaceGms[1].SetGlobalBuffer((__gm__ float *)workspace + OFFSET_SORTED_EXPERT_IDS +
                                        contextInfo_->sortNumWorkSpace + this->totalLength * (NUM_TWO + NUM_TWO),
                                    this->totalLength * NUM_TWO);

    bufferSize_ = Ceil(Max(tilingData_->oneLoopMaxElementsMrg * MAX_MRGSORT_LIST, perCoreBsKLen), ONE_REPEAT_SORT_NUM) *
                  ONE_REPEAT_SORT_NUM * sizeof(int32_t) * NUM_TWO;
    pipe->InitBuffer(sortDataCopyInQueue, bufferNum, bufferSize_);
    pipe->InitBuffer(sortDataCopyOutQueue, bufferNum, bufferSize_);
    pipe->InitBuffer(sortedBuffer, bufferSize_);
    pipe->InitBuffer(tempBuffer, bufferSize_);
}

__aicore__ inline void KernelScanSortMaskMultiCoreBsK::ClearTokenInfoFlag()
{
    // 用最后一个核清理flag. 一个block一个有效数(0, int32_t). 总共需要A个block.
    if (blockIdx_ == contextInfo_->coreNum - 1) {
        int64_t perLoopElement = bufferSize_ / BLOCK_SIZE; // buffer总共可以支持的block个数
        int64_t loops = Ceil(contextInfo_->A, perLoopElement);
        int64_t lastLoopElement = contextInfo_->A - (loops - 1) * perLoopElement;
        int64_t duplicateNum = Min(static_cast<int64_t>(contextInfo_->A), perLoopElement) * 8; // 8: block num

        LocalTensor<int32_t> clearLocal = tempBuffer.Get<int32_t>();
        Duplicate<int32_t>(clearLocal, 0, duplicateNum);

        SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);

        int64_t curElementA = perLoopElement;
        for (int64_t idx = 0; idx < loops; idx++) {
            if (idx == loops - 1) {
                curElementA = lastLoopElement;
            }

            DataCopyExtParams copyOutParams{static_cast<uint16_t>(curElementA), static_cast<uint32_t>(sizeof(int32_t)),
                                            0, static_cast<uint32_t>((contextInfo_->M * F - 1) * sizeof(int32_t)), 0};
            DataCopyPad(expertIdsGmFStart_[idx * perLoopElement * contextInfo_->M * F], clearLocal, copyOutParams);
        }
    }
}

__aicore__ inline void KernelScanSortMaskMultiCoreBsK::Process()
{
    VBSProcess();
    ClearTokenInfoFlag();
    VMSProcess();
    SortOutProcess();
}
} // namespace FfnWbBatching
#endif // OP_KERNEL_FFN_WB_SCAN_SORT_MULTI_CORE_BSK_H
