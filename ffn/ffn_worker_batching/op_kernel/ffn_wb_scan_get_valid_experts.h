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
 * \file ffn_wb_scan_get_valid_experts.h
 * \brief
 */
#ifndef OP_KERNEL_FFN_WB_SCAN_GET_VALID_EXPERTS_H
#define OP_KERNEL_FFN_WB_SCAN_GET_VALID_EXPERTS_H

#include "ffn_wb_sort_base.h"
#include "kernel_operator.h"

namespace FfnWbBatching {
using namespace AscendC;

class KernelScanGetValidExperts : public SortMaskBase {
public:
    __aicore__ inline KernelScanGetValidExperts(){};
    __aicore__ inline void Init(GM_ADDR tokenInfoGm, GM_ADDR workspace, GM_ADDR groupList,
                                const ScheduleContextInfo *contextInfo, TPipe *tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void VBSProcess();
    __aicore__ inline void UBSortProcess(int64_t progress, int64_t size, int64_t loopSessionCnt);

    __aicore__ inline void GatherOutProcess();
    __aicore__ inline void VBSCopyInAndClear(int64_t progress, int64_t size, int64_t loopSessionCnt);
    __aicore__ inline void UBSortCompute(int64_t progress, int64_t size);
    __aicore__ inline void VBSCopyOut(int64_t progress, int64_t size);

    __aicore__ inline void CopyOutValidCount();
    __aicore__ inline void ClearTokenInfoFlag();

private:
    GlobalTensor<int32_t> workspaceGms[NUM_TWO];
    GlobalTensor<int32_t> workspaceSortNumGm_;
    GlobalTensor<int32_t> expertIdsGmFStart_;

    GlobalTensor<int64_t> groupListGm_;

    const ScheduleContextInfo *contextInfo_ = nullptr;

    int32_t totalValidCnt_ = 0;
    int32_t curValidCnt_ = 0;

    int64_t F_ = 0;
    int64_t BsKLenWithPading_ = 0; // BS*K_plus_1按block对齐后的个数

    int64_t perCoreSortNum_ = 0; // perCoreSessionNum * BsKLenWithPading_ 主核 每个核排序的元素总个数; 不保证32个数对齐
    int64_t sessionLoops_ = 0; // 当前核 ub循环次数

    int64_t perLoopSessionNum_ = 0;  // 当前核 一次ub处理A中的几个
    int64_t lastLoopSessionNum_ = 0; // 当前核 尾ub处理A中的几个
    int64_t perLoopElements_ = 0;    // perLoopSessionNum_ * BsKLenWithPading_;
    int64_t lastLoopElement_ = 0;    // lastLoopSessionNum_ * BsKLenWithPading_;

    int64_t blockIdx_ = 0;
    int64_t bufferSize_ = 0;

    int64_t needCoreNum_ = 0;
    int64_t sortNumWorkSpacePerCore_ = 0;
    int64_t perCoreSessionNum_ = 0;
    int64_t curCoreSessionNum_ = 0;
};

__aicore__ inline void KernelScanGetValidExperts::VBSCopyInAndClear(int64_t progress, int64_t size,
                                                                    int64_t loopSessionCnt)
{
    LocalTensor<int32_t> inLocal = sortDataCopyInQueue.AllocTensor<int32_t>();
    int64_t inOffset = progress * perLoopSessionNum_ * contextInfo_->M * F_;
    DataCopyExtParams dataCopyParams{
        static_cast<uint16_t>(loopSessionCnt),
        static_cast<uint32_t>(contextInfo_->BS * contextInfo_->K * sizeof(int32_t)),
        static_cast<uint32_t>((contextInfo_->M * F_ - contextInfo_->BS * contextInfo_->K) * sizeof(int32_t)), 0, 0};
    DataCopyPadExtParams<int32_t> dataCopyPadParams{true, 0, static_cast<uint8_t>(contextInfo_->BsKPaddingCount),
                                                    INT_MAX};
    DataCopyPad(inLocal[0], expertIdsGm[inOffset], dataCopyParams, dataCopyPadParams);

    int64_t interVal = bufferSize_ / NUM_TWO / sizeof(int32_t);
    LocalTensor<int32_t> rowIdsLocal = inLocal[interVal];
    // 索引生成只有(A, BsKPad)
    int64_t startValue =
        blockIdx_ * perCoreSessionNum_ * BsKLenWithPading_ + progress * perLoopSessionNum_ * BsKLenWithPading_;
    ArithProgression<int32_t>(rowIdsLocal, startValue, 1, size); // size包括: BsKLenWithPading_
    sortDataCopyInQueue.EnQue(inLocal);

    // clear expertIds
    LocalTensor<int32_t> clearLocal = sortDataCopyOutQueue.AllocTensor<int32_t>();
    Duplicate<int32_t>(clearLocal, INT_MAX, size);
    sortDataCopyOutQueue.EnQue(clearLocal);

    clearLocal = sortDataCopyOutQueue.DeQue<int32_t>();
    DataCopyExtParams copyoutParams{
        static_cast<uint16_t>(loopSessionCnt),
        static_cast<uint32_t>(contextInfo_->BS * contextInfo_->K * sizeof(int32_t)), 0,
        static_cast<uint32_t>((contextInfo_->M * F_ - contextInfo_->BS * contextInfo_->K) * sizeof(int32_t)), 0};
    SetWaitFlag<HardEvent::MTE2_MTE3>(HardEvent::MTE2_MTE3);
    DataCopyPad(expertIdsGm[inOffset], clearLocal, copyoutParams);
    sortDataCopyOutQueue.FreeTensor(clearLocal);
}

__aicore__ inline void KernelScanGetValidExperts::UBSortCompute(int64_t progress, int64_t size)
{
    LocalTensor<int32_t> inLocal = sortDataCopyInQueue.DeQue<int32_t>();

    LocalTensor<float> expertIdsLocalFp32 = inLocal.ReinterpretCast<float>();
    Cast(expertIdsLocalFp32, inLocal, RoundMode::CAST_ROUND, size);
    PipeBarrier<PIPE_V>();

    // gathermask start
    LocalTensor<uint32_t> maskLocalTensor = sortedBuffer.Get<uint32_t>();
    uint64_t rsvdCnt = 0;

    Muls(expertIdsLocalFp32, expertIdsLocalFp32, (float)-1, size);
    PipeBarrier<PIPE_V>();

    LocalTensor<uint8_t> maskLocalTensorUInt8 = maskLocalTensor.ReinterpretCast<uint8_t>();
    AscendC::CompareScalar(maskLocalTensorUInt8, expertIdsLocalFp32, static_cast<float>(-expertStart_),
                           AscendC::CMPMODE::GT,
                           (size + ONE_REPEAT_COMPARE_NUM - 1) / ONE_REPEAT_COMPARE_NUM * ONE_REPEAT_COMPARE_NUM);
    PipeBarrier<PIPE_V>();

    GatherMaskParams gatherMaskParams;
    gatherMaskParams.repeatTimes = 1;
    gatherMaskParams.src0BlockStride = 1;
    gatherMaskParams.src0RepeatStride = 8; // 8 blocks
    gatherMaskParams.src1RepeatStride = 0;
    GatherMask(expertIdsLocalFp32, expertIdsLocalFp32, maskLocalTensor, true, size, gatherMaskParams, rsvdCnt);
    PipeBarrier<PIPE_V>();
    curValidCnt_ = rsvdCnt;
    if (rsvdCnt == 0) {
        sortDataCopyInQueue.FreeTensor(inLocal);
        return;
    }
    totalValidCnt_ += rsvdCnt;
    LocalTensor<int32_t> outLocal = sortDataCopyOutQueue.AllocTensor<int32_t>();
    Muls(expertIdsLocalFp32, expertIdsLocalFp32, (float)-1, rsvdCnt);
    PipeBarrier<PIPE_V>();
    Cast(outLocal, expertIdsLocalFp32, RoundMode::CAST_ROUND, rsvdCnt);

    // step2: sort
    uint64_t rsvdCnt1 = 0;
    int64_t interVal = bufferSize_ / NUM_TWO / sizeof(int32_t);
    LocalTensor<int32_t> rowIdsLocal = inLocal[interVal];
    LocalTensor<int32_t> expertIdx = outLocal[interVal];
    GatherMask(expertIdx, rowIdsLocal, maskLocalTensor, true, size, gatherMaskParams, rsvdCnt1);

    sortDataCopyOutQueue.EnQue<int32_t>(outLocal);
    sortDataCopyInQueue.FreeTensor(inLocal);
}

__aicore__ inline void KernelScanGetValidExperts::VBSCopyOut(int64_t progress, int64_t size)
{
    if (curValidCnt_ > 0) {
        LocalTensor<int32_t> outLocal = sortDataCopyOutQueue.DeQue<int32_t>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(curValidCnt_ * sizeof(int32_t)), 0, 0, 0};
        int64_t wkOffset = blockIdx_ * GetSortLen<float>(perCoreSortNum_) + (totalValidCnt_ - curValidCnt_);
        int64_t interVal = bufferSize_ / NUM_TWO / sizeof(int32_t);
        DataCopyPad(workspaceGms[0][wkOffset], outLocal, copyParams);

        DataCopyPad(workspaceGms[1][wkOffset], outLocal[interVal], copyParams);
        sortDataCopyOutQueue.FreeTensor(outLocal);
    }
}

__aicore__ inline void KernelScanGetValidExperts::UBSortProcess(int64_t progress, int64_t size, int64_t loopSessionCnt)
{
    VBSCopyInAndClear(progress, size, loopSessionCnt);
    UBSortCompute(progress, size);
    VBSCopyOut(progress, size);
}

__aicore__ inline void KernelScanGetValidExperts::VBSProcess()
{
    if (blockIdx_ < needCoreNum_) {
        int64_t curLoopSessionNum = perLoopSessionNum_;
        int64_t size = perLoopElements_;
        for (int64_t loop = 0; loop < sessionLoops_; loop++) {
            if (loop == sessionLoops_ - 1) {
                curLoopSessionNum = lastLoopSessionNum_;
                size = lastLoopElement_;
            }
            UBSortProcess(loop, size, curLoopSessionNum);
        }

        CopyOutValidCount();
    }
    SyncAll();
}

__aicore__ inline void KernelScanGetValidExperts::GatherOutProcess()
{
    if (blockIdx_ == 0) {
        LocalTensor<int32_t> inLocal = sortDataCopyInQueue.AllocTensor<int32_t>();
        int64_t interVal = bufferSize_ / NUM_TWO / sizeof(int32_t);
        uint64_t rsvdCnt = 0;
        int64_t inOffset = 0;
        for (int64_t i = 0; i < needCoreNum_; i++) {
            GlobalTensor<int32_t> srcSortNumGm = workspaceSortNumGm_[i * sortNumWorkSpacePerCore_];
            DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_ALL>(
                srcSortNumGm[0]);
            int32_t vNum = srcSortNumGm.GetValue(0);
            if (vNum > 0) {
                DataCopyExtParams dataCopyParams{static_cast<uint16_t>(1),
                                                 static_cast<uint32_t>(vNum * sizeof(int32_t)), 0, 0, 0};
                int32_t padCnt = Align(vNum, sizeof(int32_t)) - vNum;
                // 太大，不支持，走排序流程
                ASSERT_MSG((inOffset + vNum + padCnt) <= interVal, "gather valid num too big");
                DataCopyPadExtParams<int32_t> dataCopyPadParams{true, 0, static_cast<uint8_t>(padCnt), INT_MAX};

                DataCopyPad(inLocal[inOffset], workspaceGms[0][i * GetSortLen<float>(perCoreSortNum_)], dataCopyParams,
                            dataCopyPadParams);
                DataCopyPad(inLocal[interVal + inOffset], workspaceGms[1][i * GetSortLen<float>(perCoreSortNum_)],
                            dataCopyParams, dataCopyPadParams);

                inOffset += vNum + padCnt;
            }
        }

        sortDataCopyInQueue.EnQue(inLocal);
        inLocal = sortDataCopyInQueue.DeQue<int32_t>();

        LocalTensor<float> expertIdsLocalFp32 = inLocal.ReinterpretCast<float>();
        Cast(expertIdsLocalFp32, inLocal, RoundMode::CAST_ROUND, inOffset);
        PipeBarrier<PIPE_V>();

        LocalTensor<uint32_t> maskLocalUInt32 = tempBuffer.Get<uint32_t>();
        LocalTensor<uint8_t> maskLocalTensorUInt8 = maskLocalUInt32.ReinterpretCast<uint8_t>();
        LocalTensor<float> tmpIds = sortedBuffer.Get<float>();
        AscendC::CompareScalar(
            maskLocalTensorUInt8, expertIdsLocalFp32, static_cast<float>(expertStart_), AscendC::CMPMODE::LE,
            (inOffset + ONE_REPEAT_COMPARE_NUM - 1) / ONE_REPEAT_COMPARE_NUM * ONE_REPEAT_COMPARE_NUM);
        PipeBarrier<PIPE_V>();

        GatherMaskParams gatherMaskParams;
        gatherMaskParams.repeatTimes = 1;
        gatherMaskParams.src0BlockStride = 1;
        gatherMaskParams.src0RepeatStride = 8; // 8 blocks
        gatherMaskParams.src1RepeatStride = 0;
        GatherMask(expertIdsLocalFp32, expertIdsLocalFp32, maskLocalUInt32, true, inOffset, gatherMaskParams, rsvdCnt);
        GatherMask(inLocal[interVal], inLocal[interVal], maskLocalUInt32, true, inOffset, gatherMaskParams, rsvdCnt);
        PipeBarrier<PIPE_V>();

        ReduceMin(tmpIds, expertIdsLocalFp32, maskLocalUInt32.ReinterpretCast<float>(), rsvdCnt, false);
        ReduceMax(tmpIds[BLOCK_BYTES], expertIdsLocalFp32, maskLocalUInt32.ReinterpretCast<float>(), rsvdCnt, false);
        PipeBarrier<PIPE_V>();
        SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
        int32_t minId = (int32_t)(tmpIds.GetValue(0));
        int32_t maxId = (int32_t)(tmpIds.GetValue(BLOCK_BYTES));

        Cast(inLocal, expertIdsLocalFp32, RoundMode::CAST_ROUND, rsvdCnt);
        PipeBarrier<PIPE_V>();
        int64_t duplicateNum = rsvdCnt % ONE_REPEAT_COMPARE_NUM;
        if (duplicateNum > 0) {
            int duplicateIndex = rsvdCnt - duplicateNum;
            uint64_t mask0 = UINT64_MAX;
            mask0 = mask0 << duplicateNum;
            uint64_t mask[NUM_TWO] = {mask0, 0};
            Duplicate(inLocal[duplicateIndex], INT_MAX, mask, 1, DST_BLK_STRIDE, DST_REP_STRIDE);
            Duplicate(inLocal[interVal + duplicateIndex], INT_MAX, mask, 1, DST_BLK_STRIDE, DST_REP_STRIDE);
            PipeBarrier<PIPE_V>();
        }

        uint64_t rsvdCnt2 = 0;
        int64_t curExpertIdOffset = 0;
        int64_t outOffset = 0;
        LocalTensor<int64_t> groupListOutLocal = sortedBuffer.Get<int64_t>();

        for (int32_t i = minId; i <= maxId; i++) {
            LocalTensor<int32_t> outLocal = sortDataCopyOutQueue.AllocTensor<int32_t>();
            AscendC::CompareScalar(
                maskLocalTensorUInt8, inLocal, i, AscendC::CMPMODE::EQ,
                (rsvdCnt + ONE_REPEAT_COMPARE_NUM - 1) / ONE_REPEAT_COMPARE_NUM * ONE_REPEAT_COMPARE_NUM);
            PipeBarrier<PIPE_V>();
            GatherMask(outLocal, inLocal, maskLocalUInt32, true, rsvdCnt, gatherMaskParams, rsvdCnt2);
            GatherMask(outLocal[interVal], inLocal[interVal], maskLocalUInt32, true, rsvdCnt, gatherMaskParams,
                       rsvdCnt2);
            PipeBarrier<PIPE_V>();
            SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
            groupListOutLocal.SetValue(curExpertIdOffset, i);
            groupListOutLocal.SetValue(curExpertIdOffset + 1, rsvdCnt2);
            curExpertIdOffset += NUM_TWO;

            sortDataCopyOutQueue.EnQue(outLocal);
            outLocal = sortDataCopyOutQueue.DeQue<int32_t>();

            DataCopyExtParams dataCopyParams{static_cast<uint16_t>(1),
                                             static_cast<uint32_t>(rsvdCnt2 * sizeof(int32_t)), 0, 0, 0};
            DataCopyPad(sortedexpertIdsGm[outOffset], outLocal, dataCopyParams);
            DataCopyPad(sortedRowIdsGm[outOffset], outLocal[interVal], dataCopyParams);
            outOffset += rsvdCnt2;
            sortDataCopyOutQueue.FreeTensor(outLocal);
        }

        if (curExpertIdOffset < contextInfo_->expertNum * NUM_TWO) {
            groupListOutLocal.SetValue(curExpertIdOffset, 0);
            groupListOutLocal.SetValue(curExpertIdOffset + 1, 0);
            curExpertIdOffset += NUM_TWO;
        }

        DataCopyExtParams copyParams{static_cast<uint16_t>(1),
                                     static_cast<uint32_t>(curExpertIdOffset * sizeof(int64_t)), 0, 0, 0};
        SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
        DataCopyPad(groupListGm_, groupListOutLocal, copyParams);

        sortDataCopyInQueue.FreeTensor(inLocal);
    }
    SyncAll();
}

__aicore__ inline void KernelScanGetValidExperts::CopyOutValidCount()
{
    // 个数为0，也要设置到gm上，防止脏数据
    LocalTensor<int32_t> tempTensor = tempBuffer.Get<int32_t>(BLOCK_BYTES / sizeof(int32_t));
    tempTensor.SetValue(0, totalValidCnt_);
    SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
    DataCopyExtParams copyParams1{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
    DataCopyPad(workspaceSortNumGm_[blockIdx_ * sortNumWorkSpacePerCore_], tempTensor, copyParams1);

    LocalTensor<int32_t> outLocal = sortDataCopyOutQueue.AllocTensor<int32_t>();
    outLocal.SetValue(0, totalValidCnt_);
    DataCopyParams params;
    params.blockCount = 1;
    params.blockLen = 1 * sizeof(int32_t);
    SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
    SetAtomicAdd<int32_t>();
    DataCopyPad(rsvdCntGm[0], outLocal[0], params);
    SetAtomicNone();
    sortDataCopyOutQueue.FreeTensor(outLocal);
}

__aicore__ inline void KernelScanGetValidExperts::Init(GM_ADDR tokenInfoGm, GM_ADDR workspace, GM_ADDR groupList,
                                                       const ScheduleContextInfo *contextInfo, TPipe *tPipe)
{
    this->pipe = tPipe;
    contextInfo_ = contextInfo;
    blockIdx_ = GetBlockIdx();

    F_ = contextInfo_->BS * contextInfo_->K + 1 + 1;
    BsKLenWithPading_ = contextInfo_->BS * contextInfo_->K + contextInfo_->BsKPaddingCount;
    // 这里最好改成 totalLength no pad, 后面写workspace的地方也需要修改
    this->totalLength = contextInfo_->A * BsKLenWithPading_;

    if (this->totalLength <= contextInfo_->sortLoopMaxElement) {
        needCoreNum_ = 1;
    } else {
        int64_t maxSessionPerSort = Max(1L, contextInfo_->sortLoopMaxElement / BsKLenWithPading_);
        int64_t leftSessionNum = CeilDiv(contextInfo_->A, maxSessionPerSort);
        int64_t perCoreLeftSessionNum = CeilDiv(leftSessionNum, contextInfo_->coreNum);
        needCoreNum_ = Min(CeilDiv(leftSessionNum, perCoreLeftSessionNum), contextInfo_->coreNum);
    }

    sortNumWorkSpacePerCore_ = contextInfo_->sortNumWorkSpace / needCoreNum_;
    perCoreSessionNum_ = CeilDiv(contextInfo_->A, needCoreNum_);

    curCoreSessionNum_ = (blockIdx_ == needCoreNum_ - 1) ? (contextInfo_->A - (needCoreNum_ - 1) * perCoreSessionNum_) :
                                                           perCoreSessionNum_;

    perCoreSortNum_ = perCoreSessionNum_ * BsKLenWithPading_;

    int64_t oneLoopMaxSessionNum = contextInfo_->sortLoopMaxElement / BsKLenWithPading_;
    if (oneLoopMaxSessionNum == 0) {
        int64_t BsKloops_ = BsKLenWithPading_ / contextInfo_->sortLoopMaxElement;
    } else {
        sessionLoops_ = CeilDiv(curCoreSessionNum_, oneLoopMaxSessionNum);
        perLoopSessionNum_ = Min(oneLoopMaxSessionNum, curCoreSessionNum_);
        lastLoopSessionNum_ = curCoreSessionNum_ - (sessionLoops_ - 1) * perLoopSessionNum_;

        perLoopElements_ = perLoopSessionNum_ * BsKLenWithPading_;
        lastLoopElement_ = lastLoopSessionNum_ * BsKLenWithPading_;
    }

    int64_t expertIdStartPos = contextInfo_->curMicroBatchID * F_ + 1 + 1;
    expertIdsGmFStart_.SetGlobalBuffer((__gm__ int32_t *)tokenInfoGm + contextInfo_->curMicroBatchID * F_);
    expertIdsGm.SetGlobalBuffer((__gm__ int32_t *)tokenInfoGm + expertIdStartPos +
                                blockIdx_ * perCoreSessionNum_ * contextInfo_->M * F_);
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

    groupListGm_.SetGlobalBuffer((__gm__ int64_t *)groupList, contextInfo_->expertNum * NUM_TWO);

    // key and value
    workspaceGms[0].SetGlobalBuffer((__gm__ int32_t *)workspace + OFFSET_SORTED_EXPERT_IDS +
                                        contextInfo_->sortNumWorkSpace + this->totalLength * NUM_TWO,
                                    this->totalLength * NUM_TWO);
    workspaceGms[1].SetGlobalBuffer((__gm__ int32_t *)workspace + OFFSET_SORTED_EXPERT_IDS +
                                        contextInfo_->sortNumWorkSpace + this->totalLength * (NUM_TWO + NUM_TWO),
                                    this->totalLength * NUM_TWO);

    bufferSize_ =
        Ceil(contextInfo_->sortLoopMaxElement, ONE_REPEAT_SORT_NUM) * ONE_REPEAT_SORT_NUM * sizeof(int32_t) * NUM_TWO;
    pipe->InitBuffer(sortDataCopyInQueue, 1, bufferSize_);
    pipe->InitBuffer(sortDataCopyOutQueue, 1, bufferSize_);
    pipe->InitBuffer(sortedBuffer, bufferSize_);
    pipe->InitBuffer(tempBuffer, bufferSize_);
}

__aicore__ inline void KernelScanGetValidExperts::ClearTokenInfoFlag()
{
    // 用最后一个核清理flag. 一个block一个有效数(0, int32_t). 总共需要A个block.
    if (blockIdx_ == contextInfo_->coreNum - 1) {
        int64_t perLoopANum = bufferSize_ / BLOCK_SIZE; // buffer总共可以支持的block个数
        int64_t loops = Ceil(contextInfo_->A, perLoopANum);
        int64_t lastLoopANum = contextInfo_->A - (loops - 1) * perLoopANum;
        int64_t duplicateNum = Min(static_cast<int64_t>(contextInfo_->A), perLoopANum) * 8; // 8: block num

        LocalTensor<int32_t> clearLocal = tempBuffer.Get<int32_t>();
        Duplicate<int32_t>(clearLocal, 0, duplicateNum);

        SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);

        int64_t curElementA = perLoopANum;
        for (int64_t idx = 0; idx < loops; idx++) {
            if (idx == loops - 1) {
                curElementA = lastLoopANum;
            }

            DataCopyExtParams copyOutParams{static_cast<uint16_t>(curElementA), static_cast<uint32_t>(sizeof(int32_t)),
                                            0, static_cast<uint32_t>((contextInfo_->M * F_ - 1) * sizeof(int32_t)), 0};
            DataCopyPad(expertIdsGmFStart_[idx * perLoopANum * contextInfo_->M * F_], clearLocal, copyOutParams);
        }
    }
}

__aicore__ inline void KernelScanGetValidExperts::Process()
{
    VBSProcess();
    ClearTokenInfoFlag();
    GatherOutProcess();
}
} // namespace FfnWbBatching
#endif // OP_KERNEL_FFN_WB_SCAN_GET_VALID_EXPERTS_H
