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
 * \file ffn_wb_group_listing.h
 * \brief
 */

#ifndef OP_KERNEL_FFN_WB_GROUP_LISTING_H
#define OP_KERNEL_FFN_WB_GROUP_LISTING_H

#include "ffn_wb_common.h"
#include "kernel_operator.h"

namespace FfnWbBatching {
using namespace AscendC;

class KernelFfnWBGroupListing {
public:
    __aicore__ inline KernelFfnWBGroupListing(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR groupList, GM_ADDR groupListTmp,
                                ScheduleContextInfo *scheduleContext, TPipe *tPipe, int64_t groupListingDealFlag);
    __aicore__ inline void Process(int64_t groupListingDealFlag);
    __aicore__ inline void ProcessExpertCount();

private:
    __aicore__ inline void CopyIn(int64_t loop, int64_t curLoopElements);
    __aicore__ inline void Compute(int64_t curLoopElements);
    __aicore__ inline void CopyOut();

    __aicore__ inline void expertCountCopyIn();
    __aicore__ inline void expertCountCompute();
    __aicore__ inline void expertCountCopyOut();
    __aicore__ inline void CopyInOneCore(int64_t inputNum);
    __aicore__ inline void ComputeOneCore(int64_t inputNum);

private:
    TPipe *pipe_;

    GlobalTensor<int32_t> expandedExpertIdsGm; // 排序后的专家索引（全局内存）
    GlobalTensor<int64_t> groupListGm;         // 最终专家Token计数（int64）
    GlobalTensor<int32_t> expertCountTempGm;   // 临时专家计数（int32）
    LocalTensor<int64_t> groupListOutLocal;

    TQue<QuePosition::VECIN, 1> sortedExpertIdxInQueue;  // 专家索引输入队列
    TQue<QuePosition::VECOUT, 1> expertIdxCountOutQueue; // 最终计数输出队列

    int64_t blockIdx;
    int64_t perLoopRows;
    int64_t coreNum = 0;
    int64_t expertNum = 0;

    int64_t tokenCount = 0;
    int64_t curExpertIdOffset = 0;
    int32_t lastExpertId = -1;
    int32_t firstExpertId = -1;

    int64_t curCoreElements = 0;
    int64_t curcoreLoopsNum = 0;
    int64_t curCorePerLoopElements = 0;
    int64_t curCoreLastLoopElements = 0;

    int64_t perCoreElements = 0;
    int64_t lastCoreElements = 0;
    int64_t perCoreLoopsNum = 0;
    int64_t lastCoreLoopsNum = 0;
    int64_t perCorePerLoopElements = 0;
    int64_t perCoreLastLoopElements = 0;
    int64_t lastCorePerLoopElements = 0;
    int64_t lastCoreLastLoopElements = 0;
    int64_t validGatherIdxLength = 0;
    int64_t actualExpertTotalNum = 0;
};

__aicore__ inline void KernelFfnWBGroupListing::Init(GM_ADDR x, GM_ADDR groupList, GM_ADDR groupListTmp,
                                                     ScheduleContextInfo *scheduleContext, TPipe *tPipe,
                                                     int64_t groupListingDealFlag)
{
    pipe_ = tPipe;
    blockIdx = GetBlockIdx();
    validGatherIdxLength = scheduleContext->validGatherIdxLength;

    coreNum = scheduleContext->coreNum;
    perLoopRows = 8192L;

    // 计算每个核处理数据
    perCoreElements = validGatherIdxLength / GROUP_LISTING_MULTI_AIV_NUM;
    lastCoreElements = perCoreElements + (validGatherIdxLength % GROUP_LISTING_MULTI_AIV_NUM);

    // 计算每个核循环次数
    perCoreLoopsNum = (perCoreElements + perLoopRows - 1) / perLoopRows;
    perCorePerLoopElements = perLoopRows;
    perCoreLastLoopElements = (perCoreElements % perLoopRows == 0) ? perLoopRows : (perCoreElements % perLoopRows);
    lastCoreLoopsNum = (lastCoreElements + perLoopRows - 1) / perLoopRows;
    lastCorePerLoopElements = perLoopRows;
    lastCoreLastLoopElements = (lastCoreElements % perLoopRows == 0) ? perLoopRows : (lastCoreElements % perLoopRows);

    const int64_t firstCoreIdx = coreNum - GROUP_LISTING_MULTI_AIV_NUM;
    if ((blockIdx >= firstCoreIdx) && (blockIdx < coreNum - 1)) {
        curCoreElements = perCoreElements;
        curcoreLoopsNum = perCoreLoopsNum;
        curCorePerLoopElements = perCorePerLoopElements;
        curCoreLastLoopElements = perCoreLastLoopElements;
    } else if (blockIdx == coreNum - 1) {
        curCoreElements = lastCoreElements;
        curcoreLoopsNum = lastCoreLoopsNum;
        curCorePerLoopElements = lastCorePerLoopElements;
        curCoreLastLoopElements = lastCoreLastLoopElements;
    }

    expertNum = scheduleContext->expertNum;

    // 输入global 排序好的ExpertIds
    if (groupListingDealFlag == 1) {
        expandedExpertIdsGm.SetGlobalBuffer((__gm__ int32_t *)x + (blockIdx - firstCoreIdx) * perCoreElements,
                                            curCoreElements);
    } else {
        expandedExpertIdsGm.SetGlobalBuffer((__gm__ int32_t *)x, validGatherIdxLength);
    }
    // 输出global group_list数据[id * count] int_64
    groupListGm.SetGlobalBuffer((__gm__ int64_t *)groupList, expertNum * NUM_TWO);
    // 输出global 存放原子加group_list数据[count] int_32
    expertCountTempGm.SetGlobalBuffer((__gm__ int32_t *)groupListTmp, expertNum);

    int64_t cntSize = (expertNum * NUM_TWO * sizeof(int64_t) + BLOCK_BYTES - 1) / BLOCK_BYTES * BLOCK_BYTES;
    pipe_->InitBuffer(sortedExpertIdxInQueue, 1, perLoopRows * sizeof(int32_t));
    pipe_->InitBuffer(expertIdxCountOutQueue, 1, cntSize);
}

__aicore__ inline void KernelFfnWBGroupListing::Process(int64_t groupListingDealFlag)
{
    if ((blockIdx >= coreNum - GROUP_LISTING_MULTI_AIV_NUM) && (blockIdx < coreNum) && (groupListingDealFlag == 1)) {
        for (int64_t i = 0; i < curcoreLoopsNum; i++) {
            int64_t perLoopElements = (i == (curcoreLoopsNum - 1)) ? curCoreLastLoopElements : curCorePerLoopElements;
            CopyIn(i, perLoopElements);
            Compute(perLoopElements);
            CopyOut();
        }
    }

    if ((blockIdx == coreNum - 1) && (groupListingDealFlag == 0)) {
        groupListOutLocal = expertIdxCountOutQueue.AllocTensor<int64_t>();
        curExpertIdOffset = 0;
        CopyInOneCore(validGatherIdxLength);
        ComputeOneCore(validGatherIdxLength);

        // 更新最后一个expertId和tokenCnt，并且需要在groupList后面添加一个[0， 0]
        if (curExpertIdOffset < expertNum) {
            groupListOutLocal.SetValue(curExpertIdOffset * NUM_TWO, lastExpertId);
            groupListOutLocal.SetValue(curExpertIdOffset * NUM_TWO + 1, tokenCount);
            curExpertIdOffset += 1;
        }
        // group list未满的情况下，最后补充[0， 0]
        if (curExpertIdOffset < expertNum) {
            groupListOutLocal.SetValue(curExpertIdOffset * NUM_TWO, 0);
            groupListOutLocal.SetValue(curExpertIdOffset * NUM_TWO + 1, 0);
            curExpertIdOffset += 1;
        }

        DataCopyExtParams copyParams{static_cast<uint16_t>(1),
                                     static_cast<uint32_t>(curExpertIdOffset * NUM_TWO * sizeof(int64_t)), 0, 0, 0};
        SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
        DataCopyPad(groupListGm, groupListOutLocal, copyParams);

        expertIdxCountOutQueue.FreeTensor(groupListOutLocal);
    }
}

__aicore__ inline void KernelFfnWBGroupListing::ProcessExpertCount()
{
    if (blockIdx == coreNum - 1) {
        expertCountCopyIn();
        expertCountCompute();
        expertCountCopyOut();
    }
}

__aicore__ inline void KernelFfnWBGroupListing::CopyIn(int64_t loop, int64_t curLoopElements)
{
    LocalTensor<int32_t> sortedExpertIdxInLocal = sortedExpertIdxInQueue.AllocTensor<int32_t>();
    DataCopyExtParams dataCopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(curLoopElements * sizeof(int32_t)),
                                     0, 0, 0};
    DataCopyPadExtParams dataCopyPadParams{false, 0, 0, 0};
    int64_t sortedexpertIdxOffset = loop * curCorePerLoopElements;
    DataCopyPad(sortedExpertIdxInLocal, expandedExpertIdsGm[sortedexpertIdxOffset], dataCopyParams, dataCopyPadParams);

    sortedExpertIdxInQueue.EnQue(sortedExpertIdxInLocal);
}

__aicore__ inline void KernelFfnWBGroupListing::Compute(int64_t curLoopElements)
{
    LocalTensor<int32_t> sortedExpertIdxInLocal = sortedExpertIdxInQueue.DeQue<int32_t>();
    LocalTensor<int32_t> expertCountOutLocal = expertIdxCountOutQueue.AllocTensor<int32_t>();
    Duplicate(expertCountOutLocal.ReinterpretCast<int32_t>(), static_cast<int32_t>(0), static_cast<int32_t>(expertNum));
    event_t eventIDVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIDVToS);
    WaitFlag<HardEvent::V_S>(eventIDVToS);
    lastExpertId = sortedExpertIdxInLocal.GetValue(0);
    tokenCount = 1;
    for (int64_t i = 1; i < curLoopElements; i++) {
        int32_t curExpertId = sortedExpertIdxInLocal.GetValue(i);
        tokenCount++;
        if (curExpertId > lastExpertId) {
            // 先更新expertId切换前的数据
            expertCountOutLocal.SetValue(lastExpertId, tokenCount - 1);
            tokenCount = 1;
            lastExpertId = curExpertId;
        }
    }
    expertCountOutLocal.SetValue(lastExpertId, tokenCount);

    event_t eventIDSToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
    WaitFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
    expertIdxCountOutQueue.EnQue<int32_t>(expertCountOutLocal);
    sortedExpertIdxInQueue.FreeTensor(sortedExpertIdxInLocal);
}

__aicore__ inline void KernelFfnWBGroupListing::CopyOut()
{
    LocalTensor<int32_t> expertCountOutLocal = expertIdxCountOutQueue.DeQue<int32_t>();

    DataCopyExtParams copyParams{static_cast<uint16_t>(1), static_cast<uint32_t>((expertNum) * sizeof(int32_t)), 0, 0,
                                 0};
    SetAtomicAdd<int32_t>();
    DataCopyPad(expertCountTempGm, expertCountOutLocal, copyParams);
    SetAtomicNone();

    expertIdxCountOutQueue.FreeTensor(expertCountOutLocal);
}

__aicore__ inline void KernelFfnWBGroupListing::expertCountCopyIn()
{
    LocalTensor<int32_t> expertCountTempInLocal = sortedExpertIdxInQueue.AllocTensor<int32_t>();

    DataCopyExtParams dataCopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>((expertNum) * sizeof(int32_t)), 0,
                                     0, 0};
    DataCopyPadExtParams dataCopyPadParams{false, 0, 0, 0};
    DataCopyPad(expertCountTempInLocal, expertCountTempGm, dataCopyParams, dataCopyPadParams);

    sortedExpertIdxInQueue.EnQue(expertCountTempInLocal);
}

__aicore__ inline void KernelFfnWBGroupListing::expertCountCompute()
{
    LocalTensor<int32_t> expertCountTempInLocal = sortedExpertIdxInQueue.DeQue<int32_t>();

    LocalTensor<int64_t> expertCountOutLocal = expertIdxCountOutQueue.AllocTensor<int64_t>();
    event_t eventIDMte2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(eventIDMte2ToS);
    WaitFlag<HardEvent::MTE2_S>(eventIDMte2ToS);
    for (int64_t i = 0; i < expertNum; i++) {
        int64_t expertCount = static_cast<int64_t>(expertCountTempInLocal.GetValue(i));
        if (expertCount > 0) {
            actualExpertTotalNum++;
            expertCountOutLocal.SetValue((actualExpertTotalNum - 1) * NUM_TWO, i);
            expertCountOutLocal.SetValue((actualExpertTotalNum - 1) * NUM_TWO + 1, expertCount);
        }
    }
    if (actualExpertTotalNum < expertNum) {
        expertCountOutLocal.SetValue((actualExpertTotalNum - 1) * NUM_TWO + NUM_TWO, 0);
        expertCountOutLocal.SetValue((actualExpertTotalNum - 1) * NUM_TWO + NUM_THREE, 0);
    }

    event_t eventIDSToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(eventIDSToMte3);
    WaitFlag<HardEvent::S_MTE3>(eventIDSToMte3);
    expertIdxCountOutQueue.EnQue<int64_t>(expertCountOutLocal);
    sortedExpertIdxInQueue.FreeTensor(expertCountTempInLocal);
}

__aicore__ inline void KernelFfnWBGroupListing::expertCountCopyOut()
{
    LocalTensor<int64_t> expertCountOutLocal = expertIdxCountOutQueue.DeQue<int64_t>();
    DataCopyExtParams copyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(expertNum * NUM_TWO * sizeof(int64_t)),
                                 0, 0, 0};
    DataCopyPad(groupListGm, expertCountOutLocal, copyParams);

    expertIdxCountOutQueue.FreeTensor(expertCountOutLocal);
}

__aicore__ inline void KernelFfnWBGroupListing::CopyInOneCore(int64_t inputNum)
{
    LocalTensor<int32_t> inLocal = sortedExpertIdxInQueue.AllocTensor<int32_t>();
    DataCopyExtParams dataCopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(inputNum * sizeof(int32_t)), 0, 0,
                                     0};
    DataCopyPadExtParams<int32_t> dataCopyPadParams{false, 0, 0, 0};
    DataCopyPad(inLocal, expandedExpertIdsGm, dataCopyParams, dataCopyPadParams);
    sortedExpertIdxInQueue.EnQue<int32_t>(inLocal);
}

__aicore__ inline void KernelFfnWBGroupListing::ComputeOneCore(int64_t inputNum)
{
    LocalTensor<int32_t> inLocal = sortedExpertIdxInQueue.DeQue<int32_t>();
    SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
    if (lastExpertId == -1) {
        lastExpertId = inLocal.GetValue(0);
        firstExpertId = lastExpertId;
    }
    for (int64_t i = 0; i < inputNum; i++) {
        int32_t curExpertId = inLocal.GetValue(i);
        tokenCount++;
        if (curExpertId > lastExpertId && curExpertIdOffset < expertNum) {
            // 先更新expertId切换前的数据
            groupListOutLocal.SetValue(curExpertIdOffset * NUM_TWO, lastExpertId);
            groupListOutLocal.SetValue(curExpertIdOffset * NUM_TWO + 1, tokenCount - 1);
            curExpertIdOffset += 1;
            tokenCount = 1;
            lastExpertId = curExpertId;
        }
    }
    sortedExpertIdxInQueue.FreeTensor(inLocal);
}

} // namespace FfnWbBatching
#endif // OP_KERNEL_FFN_WB_GROUP_LISTING_H
