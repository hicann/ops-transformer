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
 * \file all_gather_add.h
 * \brief
 */
#ifndef ALL_GATHER_ADD_H
#define ALL_GATHER_ADD_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "all_gather_add_tiling.h"

constexpr int32_t ADD_BUFFER_NUM = 2;

namespace AscendC {
class AllGatherAdd {
public:
    __aicore__ inline AllGatherAdd(){};
    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR gatherGM,
                                AllGatherAddTilingData* tilingData, TPipe* tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void HcclPrepare();
    __aicore__ inline void CalcAddGmAddr(int32_t commTurn);
    __aicore__ inline void CopyIn(int32_t progress);
    __aicore__ inline void CopyOut(int32_t progress);
    __aicore__ inline void Compute();
    __aicore__ inline void HcclFinalize();

private:
    AllGatherAddTilingData* tilingData_;

    Hccl<HCCL_SERVER_TYPE_AICPU> hccl_;

    TQue<QuePosition::VECIN, ADD_BUFFER_NUM> inputQueueGather;
    TQue<QuePosition::VECIN, ADD_BUFFER_NUM> inputQueueB;
    TQue<QuePosition::VECOUT, ADD_BUFFER_NUM> outputQueueC;

    GM_ADDR aGM_;
    GM_ADDR bGM_;
    GM_ADDR cGM_;
    GM_ADDR gatherGM_;

    GlobalTensor<half> gatherOutGM;
    GlobalTensor<half> inputBGM;
    GlobalTensor<half> outputCGM;

    int64_t blockElemNum_ = 0;
    int64_t tileNum_ = 0;
    uint32_t addTileElemNum_ = 0;
    int64_t blockIdx_ = 0;
    uint64_t elemNumPerRank_ = 0;

    HcclHandle handleId_{INVALID_HANDLE_ID};
};

__aicore__ inline void AllGatherAdd::Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR gatherGM,
                                          AllGatherAddTilingData* tilingData, TPipe* tPipe)
{
    aGM_ = aGM;
    bGM_ = bGM;
    cGM_ = cGM;
    gatherGM_ = gatherGM;

    tilingData_ = tilingData;
    blockElemNum_ = tilingData->blockElemNum;
    addTileElemNum_ = tilingData->addTileElemNum / ADD_BUFFER_NUM;
    tileNum_ = tilingData->tileNum;
    blockIdx_ = AscendC::GetBlockIdx();

    // 初始化hccl对象
    GM_ADDR contextGM = GetHcclContext<HCCL_GROUP_ID_0>();
    hccl_.InitV2(contextGM, tilingData);
    hccl_.SetCcTilingV2(offsetof(AllGatherAddTilingData, mc2CcTiling));

    tPipe->InitBuffer(inputQueueGather, ADD_BUFFER_NUM, addTileElemNum_ * sizeof(half));
    tPipe->InitBuffer(inputQueueB, ADD_BUFFER_NUM, addTileElemNum_ * sizeof(half));
    tPipe->InitBuffer(outputQueueC, ADD_BUFFER_NUM, addTileElemNum_ * sizeof(half));
}

__aicore__ inline void AllGatherAdd::HcclPrepare()
{
    elemNumPerRank_ = tilingData_->gatherTileElemNum *
                      tilingData_->commTurn; // 通信多轮切分，多张卡的数据拼接到gatherOutGM时，相邻数据块的起始地址偏移
    // 下发通信任务
    handleId_ = hccl_.AllGather<true>(aGM_, gatherGM_, tilingData_->gatherTileElemNum,
                                      HcclDataType::HCCL_DATA_TYPE_FP16, elemNumPerRank_, tilingData_->commTurn);
}

__aicore__ inline void AllGatherAdd::CopyIn(int32_t progress)
{
    AscendC::LocalTensor<half> gatherLocal = inputQueueGather.AllocTensor<half>();
    AscendC::LocalTensor<half> bLocal = inputQueueB.AllocTensor<half>();
    AscendC::DataCopy(gatherLocal, gatherOutGM[progress * addTileElemNum_], addTileElemNum_);
    AscendC::DataCopy(bLocal, inputBGM[progress * addTileElemNum_], addTileElemNum_);
    inputQueueGather.EnQue(gatherLocal);
    inputQueueB.EnQue(bLocal);
}

__aicore__ inline void AllGatherAdd::Compute()
{
    AscendC::LocalTensor<half> gatherLocal = inputQueueGather.DeQue<half>();
    AscendC::LocalTensor<half> bLocal = inputQueueB.DeQue<half>();
    AscendC::LocalTensor<half> cLocal = outputQueueC.AllocTensor<half>();
    AscendC::Add(cLocal, gatherLocal, bLocal, addTileElemNum_);
    outputQueueC.EnQue<half>(cLocal);
    inputQueueGather.FreeTensor(gatherLocal);
    inputQueueB.FreeTensor(bLocal);
}

__aicore__ inline void AllGatherAdd::CopyOut(int32_t progress)
{
    AscendC::LocalTensor<half> cLocal = outputQueueC.DeQue<half>();
    AscendC::DataCopy(outputCGM[progress * addTileElemNum_], cLocal, addTileElemNum_);
    outputQueueC.FreeTensor(cLocal);
}

__aicore__ inline void AllGatherAdd::HcclFinalize()
{
    AscendC::SyncAll<true>();
    hccl_.Finalize();
}

__aicore__ inline void AllGatherAdd::CalcAddGmAddr(int32_t commTurn)
{
    uint32_t commOffset = commTurn * tilingData_->gatherTileElemNum; // 1.根据通信轮次偏移单个通信数据大小
    uint32_t blockOffset = blockIdx_ / tilingData_->addCoresPerRank *
                           elemNumPerRank_; // 2.根据rank数和aivId判断当前核被分到处理哪个rank的通信数据
    uint32_t totalOffset = commOffset + blockOffset +
                           (blockIdx_ % tilingData_->addCoresPerRank) *
                               blockElemNum_; // 3.最终偏移需要再加上当前核在所处理rank数据上的偏移
    gatherOutGM.SetGlobalBuffer((__gm__ half*)gatherGM_ + totalOffset, blockElemNum_);
    inputBGM.SetGlobalBuffer((__gm__ half*)bGM_ + totalOffset, blockElemNum_);
    outputCGM.SetGlobalBuffer((__gm__ half*)cGM_ + totalOffset, blockElemNum_);
}

__aicore__ inline void AllGatherAdd::Process()
{
    HcclPrepare();
    int addLoop = tileNum_ * ADD_BUFFER_NUM;
    for (int i = 0; i < tilingData_->commTurn; i++) {
        hccl_.Wait(handleId_);
        // 根据通信轮次和rankSize计算本核需要处理数据的起始地址
        CalcAddGmAddr(i);
        // 对前一轮的通信结果进行Add计算
        for (int j = 0; j < addLoop; j++) {
            CopyIn(j);
            Compute();
            CopyOut(j);
        }
    }
    HcclFinalize();
}
} // namespace AscendC
#endif
