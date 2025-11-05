/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file elastic_receivable_test.h
 * \brief
 */
#ifndef ELASTIC_RECEIVABLE_TEST_H
#define ELASTIC_RECEIVABLE_TEST_H

#include "elastic_receivable_test.h"
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "elastic_receivable_test_tiling.h"
#include "../moe_distribute_dispatch/moe_distribute_base.h"

namespace ElasticReceivableTestImpl {
constexpr uint8_t BUFFER_NUM = 2; // 多buf
constexpr uint32_t UB_ALIGN = 32; // UB按32字节对齐
constexpr uint32_t STATE_SIZE = 512; // 每卡写入512B
constexpr uint32_t COPY_SIZE = STATE_SIZE / UB_ALIGN; // 每次拷贝长度
constexpr uint32_t DATA_SEND_TEST_SIZE = 128 * 1024;
constexpr uint32_t DATA_SEND_TEST_COPY_SIZE = DATA_SEND_TEST_SIZE / UB_ALIGN;
constexpr uint64_t SEND_TEST_SIZE = 2 * 1024 * 1024; // 单次测试发送量2M
constexpr uint64_t STATUS_MULTIPLY = 512; // 状态区大小为512 * worldSize

constexpr uint64_t WIN_STATE_OFFSET = 512 * 1024; // 状态区的偏移(A区域和B区域)
constexpr uint64_t STATE_WIN_OFFSET = 900 * 1024; // flag标记位的偏移
constexpr uint32_t MAX_AIV_NUM = 48; 
 
template<AscendC::HardEvent event>
__aicore__ inline void SyncFunc() {
    int32_t eventID = static_cast<int32_t>(GetTPipePtr()->FetchEventID(event));
    AscendC::SetFlag<event>(eventID);
    AscendC::WaitFlag<event>(eventID);
}
 
#define TemplateMC2TypeClass typename XType
#define TemplateMC2TypeFunc XType
 
using namespace AscendC;
template <TemplateMC2TypeClass>
class ElasticReceivableTest {
public:
    __aicore__ inline ElasticReceivableTest() {};
    __aicore__ inline void Init(GM_ADDR dstRank, GM_ADDR workspaceGM, TPipe *pipe, const ElasticReceivableTestTilingData *tilingData);
    __aicore__ inline void Process();
private:
    __aicore__ inline GM_ADDR GeWindowAddr(uint32_t toRankId, uint32_t curRankID);
    __aicore__ inline GM_ADDR GeWindowDataAddr(uint32_t toRankId, uint32_t curRankID);
    __aicore__ inline void ComputeRankPerAiv(int32_t targetRankIndex);
    uint32_t aivId_;
    uint32_t rankId_;
    TPipe *tpipe_{nullptr};
    uint32_t aivNum_{0};
    uint32_t rankNum_{0};
    int32_t dataState_{0};
    int32_t dataOffset_{0};
    uint32_t sendRankNum_{0};
    uint32_t startRankId_{0};
    uint32_t endRankId_{0};
    uint32_t worldSize_{0};
    uint32_t stateOffset_{0};
    __gm__ HcclOpResParam *winContext_{nullptr};
 
    LocalTensor<int32_t> statusTensor_;
    LocalTensor<int32_t> sendTestTensor_;
    GlobalTensor<float> windowInstatusFp32Tensor_;
    TBuf<> statusBuf_;
    TBuf<> sendTestBuf_;

    GM_ADDR dstRankGm_;
    GM_ADDR statusSpaceGm_;
};

template <TemplateMC2TypeClass>
__aicore__ inline GM_ADDR ElasticReceivableTest<TemplateMC2TypeFunc>::GeWindowAddr(uint32_t toRankId, uint32_t curRankID)
{
    if (toRankId == curRankID) {
        return (GM_ADDR)(winContext_->localWindowsExp + dataOffset_ + curRankID * STATE_SIZE);
    }
    return (GM_ADDR)(((HcclRankRelationResV2 *)(winContext_->remoteRes[toRankId].nextDevicePtr))->windowsExp + dataOffset_ +
        curRankID * STATE_SIZE);
}

template <TemplateMC2TypeClass>
__aicore__ inline GM_ADDR ElasticReceivableTest<TemplateMC2TypeFunc>::GeWindowDataAddr(uint32_t toRankId, uint32_t curRankID)
{
    if (toRankId == curRankID) {
        return (GM_ADDR)(winContext_->localWindowsIn);
    }
    return (GM_ADDR)(((HcclRankRelationResV2 *)(winContext_->remoteRes[toRankId].nextDevicePtr))->windowsIn);
}

template <TemplateMC2TypeClass>
__aicore__ inline void ElasticReceivableTest<TemplateMC2TypeFunc>::ComputeRankPerAiv(int32_t targetRankIndex)
{
    // 分核计算
    sendRankNum_ = rankNum_ / aivNum_; // 每个aiv需要处理的专家数
    uint32_t remainderRankNum = rankNum_ % aivNum_;
    startRankId_ = sendRankNum_ * aivId_ + targetRankIndex * rankNum_; // 每个aiv发送的起始rankid
    if (aivId_ < remainderRankNum) { // 前remainderRankNum个aiv需要多发1个卡的数据
        sendRankNum_ += 1;
        startRankId_ += aivId_;
    } else {
        startRankId_ += remainderRankNum;
    }
    endRankId_ = startRankId_ + sendRankNum_;
}

template <TemplateMC2TypeClass>
__aicore__ inline void ElasticReceivableTest<TemplateMC2TypeFunc>::Init(GM_ADDR dstRank, GM_ADDR workspaceGM,
    TPipe *pipe, const ElasticReceivableTestTilingData *tilingData)
{
    tpipe_ = pipe;
    aivId_ = GetBlockIdx();
    winContext_ = (__gm__ HcclOpResParam *)AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
    rankId_ = winContext_->localUsrRankId;
    rankNum_ = tilingData->elasticReceivableTestInfo.rankNum;
    aivNum_ = tilingData->elasticReceivableTestInfo.aivNum;
    worldSize_ = tilingData->elasticReceivableTestInfo.worldSize;
    stateOffset_ = STATE_SIZE;
    GlobalTensor<int32_t> selfDataStatusTensor;
    GM_ADDR statusDataSpaceGm = (GM_ADDR)(winContext_->localWindowsExp) + aivId_ * UB_ALIGN;

    GlobalTensor<int32_t> targetRankTensor;
    targetRankTensor.SetGlobalBuffer((__gm__ int32_t*)dstRank);
    int32_t targetRankIndex = targetRankTensor(0) / rankNum_;
    ComputeRankPerAiv(targetRankIndex);

    TBuf<> dataStateBuf_;
    tpipe_->InitBuffer(dataStateBuf_, UB_ALIGN);
    selfDataStatusTensor.SetGlobalBuffer((__gm__ int32_t*)statusDataSpaceGm);
    LocalTensor<int32_t> dataStateTensor_ = dataStateBuf_.Get<int32_t>();
    DataCopy(dataStateTensor_, selfDataStatusTensor, UB_ALIGN / sizeof(int32_t));
    PipeBarrier<PIPE_ALL>();
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    dataState_ = dataStateTensor_.GetValue(0);

    dataOffset_ = MAX_AIV_NUM * UB_ALIGN + dataState_ * (worldSize_ * STATE_SIZE);
    // 申请状态区的buffer
    uint32_t dataLen_ = worldSize_ >= UB_ALIGN / sizeof(float) ? worldSize_ : UB_ALIGN / sizeof(float);
    tpipe_->InitBuffer(statusBuf_, STATE_SIZE); // 512B
    tpipe_->InitBuffer(sendTestBuf_, DATA_SEND_TEST_SIZE); // 128K

    statusTensor_ = statusBuf_.Get<int32_t>();
    Duplicate<int32_t>(statusTensor_, rankId_ + 1, COPY_SIZE);
    sendTestTensor_ = sendTestBuf_.Get<int32_t>();
    
    dstRankGm_ = dstRank;
}
 
template <TemplateMC2TypeClass>
__aicore__ inline void ElasticReceivableTest<TemplateMC2TypeFunc>::Process()
{
    GlobalTensor<int32_t> dstRankTensor;
    GlobalTensor<int32_t> dstRankDataTensor;
    for (uint32_t rankIndex = startRankId_; rankIndex < endRankId_; ++rankIndex) {
        GM_ADDR rankDataGM = (__gm__ uint8_t*)GeWindowDataAddr(rankIndex, rankId_);
        // 向对端连续地址写入2M数据

        for (uint32_t repeat = 0; repeat < SEND_TEST_SIZE / DATA_SEND_TEST_SIZE; repeat++) {
    
            dstRankDataTensor.SetGlobalBuffer((__gm__ int32_t*)rankDataGM);
            DataCopy<int32_t>(dstRankDataTensor, sendTestTensor_, DATA_SEND_TEST_COPY_SIZE);
            rankDataGM += DATA_SEND_TEST_SIZE;
        }
        PipeBarrier<PIPE_MTE3>(); // 同步数据区写入
        // 发送状态
        GM_ADDR rankStatusGM = (__gm__ uint8_t*)GeWindowAddr(rankIndex, rankId_);
        dstRankTensor.SetGlobalBuffer((__gm__ int32_t*)rankStatusGM);
        DataCopy<int32_t>(dstRankTensor, statusTensor_, COPY_SIZE); // 拷贝512
    }
}
}
#endif