/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_sort_multi_core_last_dim.h
 * \brief
 */
#ifndef MOE_TOKEN_PREMUTE_MOE_SORT_MULTI_CORE_LAST_H
#define MOE_TOKEN_PREMUTE_MOE_SORT_MULTI_CORE_LAST_H

#include "moe_sort_base_with_routing_map.h"
#include "moe_mrgsort_with_routing_map.h"
#include "moe_mrgsort_out_with_routing_map.h"

namespace MoeTokenPermute {
using namespace AscendC;
template <typename T>
class MoeSortMultiLastDimCore : public MoeSortBase {
public:
    __aicore__ inline MoeSortMultiLastDimCore(){};
    __aicore__ inline void Init(GM_ADDR expertForSourceRow, GM_ADDR sortedExpertForSourceRow, GM_ADDR workspace,
                                const MoeTokenPermuteWithRoutingMapTilingData *tilingData, TPipe *tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitTilingConfig(const MoeTokenPermuteWithRoutingMapTilingData *tilingData);
    __aicore__ inline void VBSProcess();
    __aicore__ inline void UBSortProcess(int64_t progress, int64_t size, int64_t sortNum);
    __aicore__ inline void OneCoreVMSProcess(int64_t listNum, int64_t perListElements, int64_t lastListElements);
    __aicore__ inline void VMSProcess();
    __aicore__ inline void SortOutProcess();
    __aicore__ inline void VBSCopyIn(int64_t progress, int64_t size, int64_t sortNum);
    __aicore__ inline void UBSortCompute(int64_t progress, int64_t size, int64_t sortNum);
    __aicore__ inline void VBSCopyOut(int64_t progress, int64_t size, int64_t sortNum);
    __aicore__ inline void InitMoeMrgSort(MoeMrgsort *sorter, int64_t listNum, int64_t coreOffset, int64_t loopOffset);
    __aicore__ inline void InitMoeMrgSortOut(MoeMrgsortOut<int32_t, int32_t> *sorter, int64_t listNum,
                                             int64_t coreOffset);

private:
    GlobalTensor<float> workspaceGms[2];

    const PermuteVBSComputeRMTilingData *vbsTilingData;
    const PermuteVMSMiddleComputeRMTilingData *vmsTilingData;
    const PermuteSortOutComputeRMTilingData *sortOutTilingData;

    // for MoeMrgsort
    MoeMrgsort mrgsorter;
    MoeMrgsortParam mrgsortParam;
    TBuf<TPosition::VECCALC> indexBuffer;
    LocalTensor<int32_t> indexLocal;
    LocalTensor<float> concatLocal;

    TBuf<TPosition::VECCALC> concatBuffer;
    TBuf<TPosition::VECCALC> indexConcatBuffer;

    GlobalTensor<T> expertForSourceRowGm;
    GlobalTensor<int32_t> expertForSourceRowGmB32;

    GlobalTensor<int32_t> sortedExpertForSourceRowGm;
    GlobalTensor<float> debugGm;
    GlobalTensor<float> debugGm1;
    GlobalTensor<float> debugGm2;
    LocalTensor<float> expertForSourceRowLocalFp32;
    GM_ADDR expertForSourceRow_;
    GM_ADDR sortedExpertForSourceRow_;
    GM_ADDR workspace_;
    int64_t coreTaskNum;
    int64_t coreTaskNumFront;
    int64_t tailcoreTask;
    int64_t coreNum;
    int64_t taskId = 0;
    int64_t blockIdx;
    int64_t srcWsIndex = 0;
    int64_t blockFactor;
    int64_t listNum;
    int64_t perListElements;
    int64_t lastListElements;
    int64_t capacity;
    int64_t sortTotalLength;
    int64_t sortCoreLoops;
    int64_t sortCoreLoopElements;
    int64_t sortCoreLastLoopElements;
    static constexpr int32_t BLOCK_DATA_NUM = ONE_BLK_SIZE / sizeof(T);
    static constexpr int64_t MAX_MRGSORT_LIST = 4;
};

template <typename T>
__aicore__ inline void MoeSortMultiLastDimCore<T>::VBSCopyIn(int64_t progress, int64_t size, int64_t sortNum)
{
    LocalTensor<T> inLocal = sortDataCopyInQueue.AllocTensor<T>();
    int64_t inOffset = progress * sortCoreLoopElements;
    DataCopyExtParams dataCopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(size * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams DataCopyPadCustomParams{false, 0, 0, (T)0};

    if constexpr (IsSameType<T, int64_t>::value) {
        DataCopyB64(inLocal, expertForSourceRowGm[inOffset], dataCopyParams, DataCopyPadCustomParams);
    } else {
        DataCopyPadCustom(inLocal, expertForSourceRowGm[inOffset], dataCopyParams, DataCopyPadCustomParams);
    }

    PipeBarrier<PIPE_V>();

    sortDataCopyInQueue.EnQue(inLocal);
}

template <typename T>
__aicore__ inline void MoeSortMultiLastDimCore<T>::UBSortCompute(int64_t progress, int64_t size, int64_t sortNum)
{
    LocalTensor<T> sortDataLocal = sortDataCopyInQueue.DeQue<T>();
    expertForSourceRowLocalFp32 = concatLocal;
    LocalTensor<int32_t> expertForSourceRowLocalInt32 = expertForSourceRowLocalFp32.template ReinterpretCast<int32_t>();
    LocalTensor<half> expertForSourceRowLocalHalf = expertForSourceRowLocalFp32.template ReinterpretCast<half>();
    auto castOffset = Align(size, sizeof(half));
    if constexpr (IsSameType<T, uint8_t>::value) {
        PipeBarrier<PIPE_V>();
        Cast(expertForSourceRowLocalHalf[castOffset], sortDataLocal, RoundMode::CAST_NONE, size);
        PipeBarrier<PIPE_V>();
        Cast(expertForSourceRowLocalFp32, expertForSourceRowLocalHalf[castOffset], RoundMode::CAST_NONE, size);
    } else {
        Cast(expertForSourceRowLocalFp32, sortDataLocal, RoundMode::CAST_ROUND, size);
    }

    sortDataCopyInQueue.FreeTensor(sortDataLocal);
    PipeBarrier<PIPE_V>();

    int64_t duplicateNum = size % ONE_REPEAT_SORT_NUM;
    if (duplicateNum > 0) {
        int duplicateIndex = size - duplicateNum;
        uint64_t mask0 = UINT64_MAX;
        mask0 = mask0 << duplicateNum;
        mask0 = mask0 & (UINT64_MAX >> ONE_REPEAT_SORT_NUM);
        uint64_t mask[2] = {mask0, 0};
        Duplicate(expertForSourceRowLocalFp32[duplicateIndex], MIN_FP32, mask, 1, DST_BLK_STRIDE, DST_REP_STRIDE);
    }
    PipeBarrier<PIPE_V>();

    LocalTensor<float> concatLocal;

    LocalTensor<float> sortedLocal = sortedBuffer.Get<float>(GetSortLen<float>(sortNum));
    LocalTensor<float> outLocal = sortDataCopyOutQueue.AllocTensor<float>();
    LocalTensor<uint32_t> sourceRowLocal = indexLocal.ReinterpretCast<uint32_t>();
    PipeBarrier<PIPE_V>();

    Sort<float, true>(outLocal, expertForSourceRowLocalFp32, sourceRowLocal, sortedLocal,
                      sortNum / ONE_REPEAT_SORT_NUM);

    sortDataCopyOutQueue.EnQue<float>(outLocal);
}

template <typename T>
__aicore__ inline void MoeSortMultiLastDimCore<T>::VBSCopyOut(int64_t progress, int64_t size, int64_t sortNum)
{
    LocalTensor<float> outLocal = sortDataCopyOutQueue.DeQue<float>();

    DataCopyExtParams dataCopyParams{static_cast<uint16_t>(1),
                                     static_cast<uint32_t>(GetSortLen<float>(size) * sizeof(float)), 0, 0, 0};

    DataCopyPadCustom(
        workspaceGms[srcWsIndex][this->blockIdx * GetSortLen<float>(this->vbsTilingData->perCoreElements) +
                                 GetSortLen<float>(progress * sortCoreLoopElements)],
        outLocal, dataCopyParams);

    sortDataCopyOutQueue.FreeTensor(outLocal);
}

template <typename T>
__aicore__ inline void MoeSortMultiLastDimCore<T>::InitMoeMrgSort(MoeMrgsort *sorter, int64_t listNum,
                                                                  int64_t coreOffset, int64_t loopOffset)
{
    GlobalTensor<float> srcWsGm = workspaceGms[srcWsIndex][blockIdx * coreOffset + loopOffset];
    LocalTensor<float> inLocal = sortedBuffer.Get<float>();
    LocalTensor<float> outLocal = sortDataCopyOutQueue.AllocTensor<float>();
    for (int64_t i = 0; i < listNum; i++) {
        LocalTensor<float> inLocalT = inLocal[GetSortLen<float>(this->sortOutTilingData->oneLoopMaxElements) * i];
        sorter->SetInput(srcWsGm, inLocalT);
    }
    GlobalTensor<float> dstWsGm = workspaceGms[1 - srcWsIndex][blockIdx * coreOffset + loopOffset];

    sorter->SetOutput(dstWsGm, outLocal);
    sortDataCopyOutQueue.FreeTensor(outLocal);
}

template <typename T>
__aicore__ inline void MoeSortMultiLastDimCore<T>::InitMoeMrgSortOut(MoeMrgsortOut<int32_t, int32_t> *sorter,
                                                                     int64_t listNum, int64_t coreOffset)
{
    GlobalTensor<float> srcWsGm = workspaceGms[srcWsIndex][blockIdx * coreOffset];
    LocalTensor<float> inLocal = indexConcatBuffer.Get<float>();
    LocalTensor<float> outLocal = sortDataCopyOutQueue.AllocTensor<float>();

    for (int64_t i = 0; i < listNum; i++) {
        LocalTensor<float> inLocalT = inLocal[GetSortLen<float>(this->sortOutTilingData->oneLoopMaxElements) * i];
        sorter->SetInput(srcWsGm, inLocalT);
    }

    LocalTensor<float> outLocalV = outLocal[this->sortOutTilingData->oneLoopMaxElements * MAX_MRGSORT_LIST];
    sorter->SetOutput(this->sortedExpertForSourceRowGm, this->sortedExpertForSourceRowGm, outLocal, outLocalV);

    LocalTensor<float> tempBuffer =
        sortedBuffer.Get<float>(GetSortLen<float>(this->sortOutTilingData->oneLoopMaxElements) * MAX_MRGSORT_LIST);
    sorter->SetBuffer(tempBuffer);
    sortDataCopyOutQueue.FreeTensor(outLocal);
}

template <typename T>
__aicore__ inline void MoeSortMultiLastDimCore<T>::OneCoreVMSProcess(int64_t listNum, int64_t perListElements,
                                                                     int64_t lastListElements)
{
    int64_t coreOffset = GetSortLen<float>(this->vbsTilingData->perCoreElements);
    mrgsortParam.oneLoopMaxElements = this->sortOutTilingData->oneLoopMaxElements;

    for (int64_t i = 0; listNum > 4; i++) {
        int64_t loops = (listNum + MAX_MRGSORT_LIST - 1) / MAX_MRGSORT_LIST;
        int64_t remainListNum = listNum - (loops - 1) * MAX_MRGSORT_LIST;

        mrgsortParam.perListElements = perListElements;
        mrgsortParam.lastListElements = perListElements;

        int64_t loopOffset = GetSortLen<float>(mrgsortParam.perListElements * MAX_MRGSORT_LIST);
        for (int64_t loop = 0; loop < loops - 1; loop++) {
            InitMoeMrgSort(&mrgsorter, MAX_MRGSORT_LIST, coreOffset, loop * loopOffset);
            mrgsorter.Init(&mrgsortParam);
            mrgsorter.Process();
        }

        mrgsortParam.perListElements = perListElements;
        mrgsortParam.lastListElements = lastListElements;
        InitMoeMrgSort(&mrgsorter, remainListNum, coreOffset, (loops - 1) * loopOffset);
        mrgsorter.Init(&mrgsortParam);
        mrgsorter.Process();

        listNum = loops;
        lastListElements = perListElements * (remainListNum - 1) + lastListElements;
        perListElements = perListElements * MAX_MRGSORT_LIST;
        srcWsIndex = (srcWsIndex + 1) % WORK_GM_NUM;
    }
    mrgsortParam.perListElements = perListElements;
    mrgsortParam.lastListElements = lastListElements;
    mrgsortParam.oneLoopMaxElements = this->sortOutTilingData->oneLoopMaxElements;

    MoeMrgsortOut<int32_t, int32_t> sorter;
    InitMoeMrgSortOut(&sorter, listNum, coreOffset);
    sorter.Init(&mrgsortParam, pipe);
    sorter.Process(capacity);
}

template <typename T>
__aicore__ inline void MoeSortMultiLastDimCore<T>::UBSortProcess(int64_t progress, int64_t size, int64_t sortNum)
{
    VBSCopyIn(progress, size, sortNum);
    UBSortCompute(progress, size, sortNum);
    VBSCopyOut(progress, size, sortNum);
}

template <typename T>
__aicore__ inline void MoeSortMultiLastDimCore<T>::VBSProcess()
{
    if (this->blockIdx < this->vbsTilingData->needCoreNum) {
        int64_t sortNum = Ceil(sortCoreLoopElements, ONE_REPEAT_SORT_NUM) * ONE_REPEAT_SORT_NUM;
        ArithProgressionSupportInt32<int32_t>(indexLocal, static_cast<int32_t>(0), static_cast<int32_t>(1),
                                              sortCoreLoopElements);
        for (int64_t loop = 0; loop < sortCoreLoops - 1; loop++) {
            UBSortProcess(loop, sortCoreLoopElements, sortNum);
            PipeBarrier<PIPE_V>();

            Adds(indexLocal, indexLocal, (int32_t)sortCoreLoopElements, sortCoreLoopElements);
        }

        sortNum = Ceil(sortCoreLastLoopElements, ONE_REPEAT_SORT_NUM) * ONE_REPEAT_SORT_NUM;
        UBSortProcess(sortCoreLoops - 1, sortCoreLastLoopElements, sortNum);

        OneCoreVMSProcess(sortCoreLoops, sortCoreLoopElements, sortCoreLastLoopElements);
    }
}

template <typename T>
__aicore__ inline void MoeSortMultiLastDimCore<T>::VMSProcess()
{
    int64_t currentStageNeedCoreNum = this->vmsTilingData->needCoreNum;
    perListElements = this->vbsTilingData->perCoreElements;
    lastListElements = this->vbsTilingData->lastCoreElements;
    listNum = this->vbsTilingData->needCoreNum;

    for (; listNum > MAX_MRGSORT_LIST;) {
        currentStageNeedCoreNum = Ceil(listNum, MAX_MRGSORT_LIST);
        int64_t coreOffset = GetSortLen<float>(perListElements * MAX_MRGSORT_LIST);
        int64_t remainListNum = listNum - (currentStageNeedCoreNum - 1) * MAX_MRGSORT_LIST;

        if (this->blockIdx < currentStageNeedCoreNum - 1) {
            mrgsortParam.perListElements = perListElements;
            mrgsortParam.lastListElements = perListElements;
            mrgsortParam.oneLoopMaxElements = this->sortOutTilingData->oneLoopMaxElements;
            InitMoeMrgSort(&mrgsorter, MAX_MRGSORT_LIST, coreOffset, 0);
            mrgsorter.Init(&mrgsortParam);
            mrgsorter.Process();
        } else if (this->blockIdx == currentStageNeedCoreNum - 1) {
            mrgsortParam.perListElements = perListElements;
            mrgsortParam.lastListElements = lastListElements;
            mrgsortParam.oneLoopMaxElements = this->sortOutTilingData->oneLoopMaxElements;
            InitMoeMrgSort(&mrgsorter, remainListNum, coreOffset, 0);
            mrgsorter.Init(&mrgsortParam);
            mrgsorter.Process();
        }
        listNum = currentStageNeedCoreNum;
        currentStageNeedCoreNum = Ceil(listNum, MAX_MRGSORT_LIST);
        srcWsIndex = (srcWsIndex + 1) % WORK_GM_NUM;

        lastListElements = perListElements * (remainListNum - 1) + lastListElements;
        perListElements = perListElements * MAX_MRGSORT_LIST;
#ifndef __CCE_KT_TEST__
        AscendC::SyncAll();
#endif
    }
}

template <typename T>
__aicore__ inline void MoeSortMultiLastDimCore<T>::SortOutProcess()
{
    if (this->blockIdx < 1) {
        mrgsortParam.perListElements = perListElements;
        mrgsortParam.lastListElements = lastListElements;
        mrgsortParam.oneLoopMaxElements = this->sortOutTilingData->oneLoopMaxElements;

        MoeMrgsortOut<int32_t, int32_t> sorter;
        InitMoeMrgSortOut(&sorter, listNum, GetSortLen<float>(perListElements));
        sorter.Init(&mrgsortParam, pipe);
        sorter.Process();
    }
#ifndef __CCE_KT_TEST__
    AscendC::SyncAll();
#endif
}

template <typename T>
__aicore__ inline void MoeSortMultiLastDimCore<T>::InitTilingConfig(
    const MoeTokenPermuteWithRoutingMapTilingData *tilingData)
{
    this->totalLength = tilingData->n;
    this->coreNum = tilingData->coreNum;
    this->capacity = tilingData->capacity;

    this->vbsTilingData = &(tilingData->vbsComputeParamsOp);
    this->vmsTilingData = &(tilingData->vmsMiddleComputeParamsOp);
    this->sortOutTilingData = &(tilingData->sortOutComputeParamsOp);
    tailcoreTask = this->vbsTilingData->tailcoreTask;
    this->blockIdx = GetBlockIdx();
    if (this->blockIdx > tilingData->vbsComputeParamsOp.frontCoreNum - 1) {
        coreTaskNumFront = tilingData->vbsComputeParamsOp.frontCoreNum;
        this->coreTaskNum = this->vbsTilingData->tailcoreTask;
    } else {
        coreTaskNumFront = this->blockIdx * (this->vbsTilingData->frontcoreTask - this->vbsTilingData->tailcoreTask);
        this->coreTaskNum = this->vbsTilingData->frontcoreTask;
    }

    this->tileLength = this->vbsTilingData->perCorePerLoopElements;
    this->sortTotalLength = this->vbsTilingData->perCoreElements;
}

template <typename T>
__aicore__ inline void MoeSortMultiLastDimCore<T>::Init(GM_ADDR expertForSourceRow, GM_ADDR sortedExpertForSourceRow,
                                                        GM_ADDR workspace,
                                                        const MoeTokenPermuteWithRoutingMapTilingData *tilingData,
                                                        TPipe *tPipe)
{
    expertForSourceRow_ = expertForSourceRow;
    sortedExpertForSourceRow_ = sortedExpertForSourceRow;
    workspace_ = workspace;
    InitTilingConfig(tilingData);

    sortCoreLoops = this->vbsTilingData->perCoreLoops;
    sortCoreLoopElements = this->vbsTilingData->perCorePerLoopElements;
    sortCoreLastLoopElements = this->vbsTilingData->perCoreLastLoopElements;

    this->pipe = tPipe;
    int64_t coreNum = GetBlockNum();
    blockFactor = tilingData->vbsComputeParamsOp.perCoreElements;
    expertForSourceRowGm.SetGlobalBuffer(
        (__gm__ T *)expertForSourceRow + this->blockIdx * tilingData->vbsComputeParamsOp.perCoreElements,
        this->sortTotalLength);
    sortedExpertForSourceRowGm.SetGlobalBuffer((__gm__ int32_t *)sortedExpertForSourceRow, this->totalLength);
    // for sort: expandDstToSrcRowGm.SetGlobalBuffer((__gm__ int32_t*)workspace, Align(this->totalLength,
    // sizeof(int32_t)));

    int64_t kvFactor = 2;

    workspaceGms[0].SetGlobalBuffer(
        (__gm__ float *)workspace,
        Align(this->totalLength * tilingData->vbsComputeParamsOp.needCoreNum, sizeof(int32_t)) * kvFactor);
    workspaceGms[1].SetGlobalBuffer(
        (__gm__ float *)workspace +
            Align(this->totalLength * tilingData->vbsComputeParamsOp.needCoreNum, sizeof(int32_t)) * kvFactor,
        Align(this->totalLength * tilingData->vbsComputeParamsOp.needCoreNum, sizeof(int32_t)) * kvFactor);

    int64_t indexNum = Ceil(Max(this->sortOutTilingData->oneLoopMaxElements * MAX_MRGSORT_LIST, sortCoreLoopElements),
                            ONE_REPEAT_SORT_NUM) *
                       ONE_REPEAT_SORT_NUM;
    int64_t indexBufferSize = indexNum * sizeof(int32_t);
    int64_t sortDataBufferSize = indexNum * sizeof(int64_t);
    int64_t bufferSize = indexBufferSize * kvFactor;
    pipe->InitBuffer(sortDataCopyInQueue, bufferNum, sortDataBufferSize);
    pipe->InitBuffer(sortDataCopyOutQueue, bufferNum, bufferSize);
    pipe->InitBuffer(indexConcatBuffer, bufferSize);
    pipe->InitBuffer(sortedBuffer, bufferSize);
    auto indexConcatLocal = indexConcatBuffer.Get<int32_t>();
    indexLocal = indexConcatLocal;
    concatLocal = indexConcatLocal.template ReinterpretCast<float>()[indexNum];
}

template <typename T>
__aicore__ inline void MoeSortMultiLastDimCore<T>::Process()
{
    uint64_t processoffset = (this->blockIdx * tailcoreTask + coreTaskNumFront) * blockFactor;
    uint64_t processOutOffset = (this->blockIdx * tailcoreTask + coreTaskNumFront) * capacity;

    while (taskId < coreTaskNum) {
        expertForSourceRowGm.SetGlobalBuffer((__gm__ T *)expertForSourceRow_ + processoffset + taskId * blockFactor);
        sortedExpertForSourceRowGm.SetGlobalBuffer((__gm__ int32_t *)sortedExpertForSourceRow_ + processOutOffset +
                                                   taskId * capacity);

        VBSProcess();
        taskId += 1;
    }
}
} // namespace MoeTokenPermute
#endif // MOE_TOKEN_PREMUTE_MOE_SORT_MULTI_CORE_LAST_H
