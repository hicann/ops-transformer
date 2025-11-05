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
 * \file elastic_receivable_info_collect.h
 * \brief
 */
#ifndef elastic_receivable_info_collect_H
#define elastic_receivable_info_collect_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "elastic_receivable_info_collect_tiling.h"
#include "../moe_distribute_dispatch/moe_distribute_base.h"

namespace ElasticReceivableInfoCollectImpl {
constexpr uint32_t UB_ALIGN = 32; // UB按32字节对齐
constexpr uint32_t STATUS_SIZE = 512; // 每卡写入512B

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
class ElasticReceivableInfoCollect {
public:
    __aicore__ inline ElasticReceivableInfoCollect() {};
    __aicore__ inline void Init(GM_ADDR y, GM_ADDR workspaceGM, TPipe *pipe, const ElasticReceivableInfoCollectTilingData *tilingData);
    __aicore__ inline void Process();
private:
    __aicore__ inline GM_ADDR GetWindowAddr(uint32_t toRankId, uint32_t curRankID);
    __aicore__ inline GM_ADDR GetLocalWindowAddr();
    __aicore__ inline void ComputeRankPerAiv();
    uint32_t aivId_;
    uint32_t rankId_;
    TPipe *tpipe_{nullptr};
    uint32_t aivNum_{0};
    int32_t dataState_{0};
    uint64_t dataOffset_{0};
    uint32_t sendRankNum_{0};
    uint32_t startRankId_{0};
    uint32_t endRankId_{0};
    uint32_t worldSize_{0};
    __gm__ HcclOpResParam *winContext_{nullptr};
 
    LocalTensor<int32_t> holdingTensor_;
    LocalTensor<int32_t> localStatusTensor_;
    LocalTensor<int32_t> resetTensor_;
 
    TBuf<> holdingBuf_;
    TBuf<> localStatusBuf_;
    TBuf<> resetBuf_;
    GM_ADDR yGM_;
};

template <TemplateMC2TypeClass>
__aicore__ inline GM_ADDR ElasticReceivableInfoCollect<TemplateMC2TypeFunc>::GetWindowAddr(uint32_t toRankId, uint32_t curRankID)
{
    if (toRankId == curRankID) {
        return (GM_ADDR)(winContext_->localWindowsExp) + dataOffset_;
    }
    return (GM_ADDR)(((HcclRankRelationResV2 *)(winContext_->remoteRes[toRankId].nextDevicePtr))->windowsExp + dataOffset_);
}

template <TemplateMC2TypeClass>
__aicore__ inline GM_ADDR ElasticReceivableInfoCollect<TemplateMC2TypeFunc>::GetLocalWindowAddr()
{
    return (GM_ADDR)(winContext_->localWindowsExp) + dataOffset_;
}

template <TemplateMC2TypeClass>
__aicore__ inline void ElasticReceivableInfoCollect<TemplateMC2TypeFunc>::ComputeRankPerAiv()
{
    // 分核计算
    sendRankNum_ = worldSize_ / aivNum_; // 每个aiv需要处理的卡数
    uint32_t remainderRankNum = worldSize_ % aivNum_;
    startRankId_ = sendRankNum_ * aivId_; // 每个aiv发送的起始rankid
    if (aivId_ < remainderRankNum) { // 前remainderRankNum个aiv需要多发1个卡的数据
        sendRankNum_ += 1;
        startRankId_ += aivId_;
    } else {
        startRankId_ += remainderRankNum;
    }
    endRankId_ = startRankId_ + sendRankNum_;
}

template <TemplateMC2TypeClass>
__aicore__ inline void ElasticReceivableInfoCollect<TemplateMC2TypeFunc>::Init(GM_ADDR y, GM_ADDR workspaceGM, TPipe *pipe,
    const ElasticReceivableInfoCollectTilingData *tilingData)
{
    tpipe_ = pipe;
    aivId_ = GetBlockIdx();
    winContext_ = (__gm__ HcclOpResParam *)AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
    rankId_ = winContext_->localUsrRankId;
    aivNum_ = tilingData->elasticReceivableInfoCollectInfo.aivNum;
    worldSize_ = tilingData->elasticReceivableInfoCollectInfo.worldSize;
    GlobalTensor<int32_t> selfDataStatusTensor;
    GM_ADDR statusDataSpaceGm = (GM_ADDR)(winContext_->localWindowsExp) + aivId_ * UB_ALIGN;

    // 分核计算
    ComputeRankPerAiv();

    // 申请状态区的buffer
    tpipe_->InitBuffer(holdingBuf_, UB_ALIGN * worldSize_); // 32B * worldSize
    tpipe_->InitBuffer(localStatusBuf_, worldSize_ * UB_ALIGN);
    tpipe_->InitBuffer(resetBuf_, worldSize_ * sizeof(int32_t));
 
    TBuf<> dataStateBuf;
    tpipe_->InitBuffer(dataStateBuf, UB_ALIGN);
    selfDataStatusTensor.SetGlobalBuffer((__gm__ int32_t*)statusDataSpaceGm);
    LocalTensor<int32_t> dataStateTensor_ = dataStateBuf.Get<int32_t>();
    DataCopy(dataStateTensor_, selfDataStatusTensor, UB_ALIGN / sizeof(int32_t));
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    dataState_ = dataStateTensor_.GetValue(0);
    dataStateTensor_.SetValue(0, dataState_ + 1);
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopy(selfDataStatusTensor, dataStateTensor_, UB_ALIGN / sizeof(int32_t));
    PipeBarrier<PIPE_ALL>();
    dataOffset_ = static_cast<uint64_t>(MAX_AIV_NUM * UB_ALIGN + dataState_ * (worldSize_ * STATUS_SIZE));

    holdingTensor_ = holdingBuf_.Get<int32_t>();
    localStatusTensor_ = localStatusBuf_.Get<int32_t>();
    resetTensor_ = resetBuf_.Get<int32_t>();
    Duplicate<int32_t>(resetTensor_, 0, worldSize_);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    // int32 * worldSize;
    yGM_ = y + startRankId_ * sizeof(int32_t) * worldSize_;
}

template <TemplateMC2TypeClass>
__aicore__ inline void ElasticReceivableInfoCollect<TemplateMC2TypeFunc>::Process()
{
    GlobalTensor<int32_t> recvRankTensor;
    GlobalTensor<int32_t> dstGmTensor;
    GlobalTensor<int32_t> yGMTensor;
    int32_t sumOutFlag = 0;
    int64_t offset = 0;
    DataCopyParams copyParamGMtoUB{(uint16_t)worldSize_, 1, STATUS_SIZE / UB_ALIGN - 1, 0};
    DataCopyExtParams copyParamUBtoOut{(uint16_t)worldSize_, sizeof(int32_t), 0, 0, 0};

    GM_ADDR recvStatusGM = GetLocalWindowAddr();
    recvRankTensor.SetGlobalBuffer((__gm__ int32_t*)recvStatusGM);
    DataCopy(localStatusTensor_, recvRankTensor, copyParamGMtoUB);
    SyncFunc<AscendC::HardEvent::MTE2_S>();

    for (int32_t index = startRankId_; index < endRankId_; index++) {
        // 对应位置为rank + 1说明链路通
        if (localStatusTensor_(index * UB_ALIGN / sizeof(int32_t)) == index + 1) {
            // 拷贝对端卡的windowExp
            GM_ADDR dstGM = GetWindowAddr(index, rankId_);
            dstGmTensor.SetGlobalBuffer((__gm__ int32_t*)dstGM);
            DataCopy(holdingTensor_, dstGmTensor, copyParamGMtoUB); // 同步状态拷回UB, 并整理数据
            SyncFunc<AscendC::HardEvent::MTE2_MTE3>();
            yGMTensor.SetGlobalBuffer((__gm__ int32_t*)yGM_);
            DataCopyPad(yGMTensor, holdingTensor_, copyParamUBtoOut);
            DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(yGMTensor);
            SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
        } else {
            yGMTensor.SetGlobalBuffer((__gm__ int32_t*)yGM_);
            DataCopy(yGMTensor, resetTensor_, worldSize_);
            DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(yGMTensor);
            SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
        }
        yGM_ += sizeof(int32_t) * worldSize_;
    }
}
}
#endif