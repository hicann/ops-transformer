/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_distribute_buffer_reset.h
 * \brief
 */
#ifndef MOE_DISTRIBUTE_BUFFER_RESET_H
#define MOE_DISTRIBUTE_BUFFER_RESET_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "moe_distribute_buffer_reset_tiling.h"
#include "../moe_distribute_dispatch/moe_distribute_base.h"

namespace MoeDistributeBufferResetImpl {
constexpr uint8_t BUFFER_NUM = 2;     // 多buf
constexpr uint32_t UB_ALIGN = 32;     // UB按32字节对齐
constexpr uint32_t STATE_OFFSET = 32; // 状态空间偏移地址
constexpr uint32_t STATUS_DATA_PER_COUNT = 8;
constexpr uint64_t WIN_EXP_OFFSET = 1024 * 1024; // flag标记位的偏移
constexpr uint64_t CLEAN_BUFF_SIZE = 100 * 1024;
constexpr uint64_t WIN_STATE_START_OFFSET = WIN_EXP_OFFSET - 767 * 32;
constexpr uint32_t FLOAT_PER_UB_ALIGN = 8U;

template <AscendC::HardEvent event>
__aicore__ inline void SyncFunc()
{
    int32_t eventID = static_cast<int32_t>(GetTPipePtr()->FetchEventID(event));
    AscendC::SetFlag<event>(eventID);
    AscendC::WaitFlag<event>(eventID);
}

__aicore__ inline int32_t AlignUp(int32_t a, int32_t b)
{
    if (unlikely(b == 0)) {
        return a;
    }
    return (a + b - 1) / b * b;
}

using namespace AscendC;

#define TemplateMC2TypeClass typename ElasticInfoType
#define TemplateMC2TypeFunc ElasticInfoType

template <TemplateMC2TypeClass>
class MoeDistributeBufferReset {
public:
    __aicore__ inline MoeDistributeBufferReset(){};
    __aicore__ inline void Init(GM_ADDR elasticInfo, GM_ADDR workspaceGM, TPipe *pipe,
                                const MoeDistributeBufferResetTilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline GM_ADDR GeWindowDataAddr(uint32_t toRankId, uint32_t curRankID);
    __aicore__ inline GM_ADDR GeWindowStateAddr(uint32_t toRankId, uint32_t curRankID);
    __aicore__ inline void SplitCoreForBufferClean();
    __aicore__ inline void SendStatus(bool fullSend);
    __aicore__ inline void WaitStatus(float sumRes);
    __aicore__ inline void CleanStatus();
    uint32_t aivId_;
    uint32_t rankId_;
    TPipe *tpipe_{nullptr};
    uint32_t aivNum_{0};
    uint32_t sendRankNum_{0};
    uint32_t startRankId_{0};
    uint32_t endRankId_{0};
    uint32_t worldSize_{0};
    uint32_t needSync_{0};
    uint32_t stateOffset_{0};
    uint32_t totalWindowSize_{0};
    uint32_t dataState_{0};
    uint64_t winStartOffset_{0};
    uint64_t winEndOffset_{0};
    uint64_t innerLoopCnt_{0};
    uint64_t expInnerLoopCnt_{0};
    uint64_t expTailLoopCleanSize_{0};
    uint64_t startExpOffset_{0};
    uint64_t startDataOffset_{0};
    uint64_t dataTailLoopCleanSize_{0};
    uint64_t dataInnerLoopCnt_{0};
    __gm__ HcclOpResParam *winContext_{nullptr};

    LocalTensor<float> statusFp32Tensor_;
    GlobalTensor<int32_t> elasticInfoGlobalTensor_;
    LocalTensor<int32_t> elastionInfoLocalTensor_;
    GlobalTensor<float> localWinTensorGm_;
    GlobalTensor<float> localWinExpTensorGm_;
    LocalTensor<float> sumLocalTensor_;
    LocalTensor<float> cleanLocalTensor_;
    LocalTensor<float> statusLocalTensor_;
    LocalTensor<float> elasticInfofp32Tensor_;
    TBuf<> statusBuf_;
    TBuf<> elasticInfoBuf_;
    TBuf<> cleanBuf_;
    TBuf<> srcUbBuffer_;
    TBuf<> gatherMaskOutBuf_;
    TBuf<> elasticInfofp32_;
    GM_ADDR elasticInfoGm_;
    GM_ADDR statusSpaceGm_;
};

template <TemplateMC2TypeClass>
__aicore__ inline GM_ADDR MoeDistributeBufferReset<TemplateMC2TypeFunc>::GeWindowDataAddr(uint32_t toRankId,
                                                                                          uint32_t curRankID)
{
    if (toRankId == curRankID) {
        return (GM_ADDR)(winContext_->localWindowsIn);
    }
    return (GM_ADDR)(((HcclRankRelationResV2 *)(winContext_->remoteRes[toRankId].nextDevicePtr))->windowsIn);
}

template <TemplateMC2TypeClass>
__aicore__ inline GM_ADDR MoeDistributeBufferReset<TemplateMC2TypeFunc>::GeWindowStateAddr(uint32_t toRankId,
                                                                                           uint32_t curRankID)
{
    if (toRankId == curRankID) {
        return (GM_ADDR)(winContext_->localWindowsExp);
    }
    return (GM_ADDR)(((HcclRankRelationResV2 *)(winContext_->remoteRes[toRankId].nextDevicePtr))->windowsExp);
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeBufferReset<TemplateMC2TypeFunc>::SplitCoreForBufferClean()
{
    uint64_t dataCleanPerCore = winContext_->winSize / aivNum_;
    uint64_t remainPerCore = winContext_->winSize % aivNum_;
    if (aivId_ < remainPerCore) {
        dataCleanPerCore += 1;
    }
    dataInnerLoopCnt_ = dataCleanPerCore / CLEAN_BUFF_SIZE;
    dataTailLoopCleanSize_ = dataCleanPerCore % CLEAN_BUFF_SIZE;
    startDataOffset_ = dataCleanPerCore * aivId_;

    uint64_t expCleanPerCore = WIN_STATE_START_OFFSET / aivNum_;
    uint64_t expRemainWinClean = WIN_STATE_START_OFFSET % aivNum_;
    if (aivId_ < expRemainWinClean) {
        expCleanPerCore += 1;
    }
    expInnerLoopCnt_ = expCleanPerCore / CLEAN_BUFF_SIZE;
    expTailLoopCleanSize_ = expCleanPerCore % CLEAN_BUFF_SIZE;
    startExpOffset_ = expCleanPerCore * aivId_;
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeBufferReset<TemplateMC2TypeFunc>::SendStatus(bool fullSend)
{
    GlobalTensor<float> dstWinGMTensor;
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    if (fullSend) {
        for (int toRankId = startRankId_; toRankId < endRankId_; toRankId++) {
            GM_ADDR remoteWinaddr = GeWindowStateAddr(toRankId, rankId_);
            dstWinGMTensor.SetGlobalBuffer((__gm__ float *)(remoteWinaddr + WIN_STATE_START_OFFSET));
            DataCopy(dstWinGMTensor[rankId_ * STATUS_DATA_PER_COUNT], statusLocalTensor_, STATUS_DATA_PER_COUNT);
        }
    } else {
        for (int toRankId = startRankId_; toRankId < endRankId_; toRankId++) {
            if (elasticInfofp32Tensor_(toRankId - startRankId_) == (float)1.0) {
                GM_ADDR remoteWinaddr = GeWindowStateAddr(toRankId, rankId_);
                dstWinGMTensor.SetGlobalBuffer((__gm__ float *)(remoteWinaddr + WIN_STATE_START_OFFSET));
                DataCopy(dstWinGMTensor[rankId_ * STATUS_DATA_PER_COUNT], statusLocalTensor_, STATUS_DATA_PER_COUNT);
            }
        }
    }
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeBufferReset<TemplateMC2TypeFunc>::WaitStatus(float sumRes)
{
    float sumTarget = (float)0.0;
    if (sumRes > (float)0.5) {
        while ((sumTarget < (sumRes - (float)0.5)) || (sumTarget > (sumRes + (float)0.5))) {
            DataCopyParams intriParams{static_cast<uint16_t>(sendRankNum_), 1, 0, 0};
            DataCopy(cleanLocalTensor_,
                    localWinExpTensorGm_[(WIN_STATE_START_OFFSET + startRankId_ * stateOffset_) / sizeof(float)],
                    intriParams);
            SyncFunc<AscendC::HardEvent::MTE2_V>();
            LocalTensor<float> gatherMaskOutTensor = gatherMaskOutBuf_.Get<float>();
            uint32_t mask = 1; // gatherMask + sum 相关参数
            ReduceSum(sumLocalTensor_, cleanLocalTensor_, gatherMaskOutTensor, mask, sendRankNum_, 1);
            SyncFunc<AscendC::HardEvent::V_S>();
            sumTarget = sumLocalTensor_.GetValue(0);
        }
    }
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeBufferReset<TemplateMC2TypeFunc>::CleanStatus()
{
    GlobalTensor<int32_t> countsGlobal;
    countsGlobal.SetGlobalBuffer((__gm__ int32_t *)(winContext_->localWindowsExp));
    LocalTensor<float> cleanStateTensor = cleanBuf_.GetWithOffset<float>(worldSize_, CLEAN_BUFF_SIZE - UB_ALIGN * sendRankNum_);
    DataCopy(localWinExpTensorGm_[(WIN_STATE_START_OFFSET + startRankId_ * stateOffset_) / sizeof(float)], cleanStateTensor, STATUS_DATA_PER_COUNT * sendRankNum_);
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(countsGlobal[(WIN_EXP_OFFSET - 64 * aivId_) / sizeof(int32_t)]);
}

template <TemplateMC2TypeClass>
__aicore__ inline void
MoeDistributeBufferReset<TemplateMC2TypeFunc>::Init(GM_ADDR elasticInfo, GM_ADDR workspaceGM, TPipe *pipe,
                                                    const MoeDistributeBufferResetTilingData *tilingData)
{
    tpipe_ = pipe;
    aivId_ = GetBlockIdx();
    winContext_ = (__gm__ HcclOpResParam *)AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
    rankId_ = winContext_->localUsrRankId;
    aivNum_ = tilingData->moeDistributeBufferReset.aivNum;
    worldSize_ = tilingData->moeDistributeBufferReset.worldSize;
    needSync_ = tilingData->moeDistributeBufferReset.needSync;
    sendRankNum_ = worldSize_ / aivNum_;
    uint32_t remainderRankNum = worldSize_ % aivNum_;
    stateOffset_ = STATE_OFFSET;
    elasticInfoGm_ = elasticInfo;
    // 分核计算
    startRankId_ = sendRankNum_ * aivId_; // 每个aiv发送的起始rankid
    if (aivId_ < remainderRankNum) {
      sendRankNum_ += 1;
      startRankId_ += aivId_;
    } else {
      startRankId_ += remainderRankNum;
    }
    endRankId_ = startRankId_ + sendRankNum_;
    // 申请状态区的buffer
    elasticInfoGlobalTensor_.SetGlobalBuffer((__gm__ int32_t *)(elasticInfoGm_));
    tpipe_->InitBuffer(elasticInfoBuf_,
                       UB_ALIGN + AlignUp((worldSize_ + 1) * sizeof(int32_t), UB_ALIGN));
    tpipe_->InitBuffer(cleanBuf_, CLEAN_BUFF_SIZE);
    tpipe_->InitBuffer(srcUbBuffer_, UB_ALIGN * FLOAT_PER_UB_ALIGN);
    tpipe_->InitBuffer(elasticInfofp32_, sendRankNum_ * sizeof(float));
    tpipe_->InitBuffer(gatherMaskOutBuf_, sendRankNum_ * sizeof(float));
    elastionInfoLocalTensor_ = elasticInfoBuf_.GetWithOffset<int32_t>(worldSize_, UB_ALIGN);
    sumLocalTensor_ = elasticInfoBuf_.GetWithOffset<float>(UB_ALIGN, 0);
    cleanLocalTensor_ = cleanBuf_.Get<float>();
    localWinTensorGm_.SetGlobalBuffer((__gm__ float *)(winContext_->localWindowsIn));
    localWinExpTensorGm_.SetGlobalBuffer((__gm__ float *)(winContext_->localWindowsExp));

    statusLocalTensor_ = srcUbBuffer_.Get<float>();
    elasticInfofp32Tensor_ = elasticInfofp32_.Get<float>();
    statusLocalTensor_(0) = (float)1.0;
    Duplicate<float>(cleanLocalTensor_, (float)0.0, CLEAN_BUFF_SIZE / sizeof(float));
    SyncFunc<AscendC::HardEvent::V_MTE3>();
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeBufferReset<TemplateMC2TypeFunc>::Process()
{
    if (startRankId_ < worldSize_) {
        DataCopy(elastionInfoLocalTensor_, elasticInfoGlobalTensor_[startRankId_], AlignUp(sendRankNum_, 8));
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        Cast(elasticInfofp32Tensor_, elastionInfoLocalTensor_, RoundMode::CAST_RINT, sendRankNum_);
    }

    SplitCoreForBufferClean();
    // clean data
    for (int loopCnt = 0; loopCnt < dataInnerLoopCnt_; loopCnt++) {
        DataCopy(localWinTensorGm_[(startDataOffset_ + loopCnt * CLEAN_BUFF_SIZE) / sizeof(int32_t)], cleanLocalTensor_,
                 CLEAN_BUFF_SIZE / sizeof(int32_t));
    }
    if (dataTailLoopCleanSize_ > 0) {
        DataCopyExtParams extCopyParams{1U, static_cast<uint32_t>(dataTailLoopCleanSize_), 0U, 0U, 0U};
        DataCopyPad(localWinTensorGm_[(startDataOffset_ + dataInnerLoopCnt_ * CLEAN_BUFF_SIZE) / sizeof(int32_t)],
                    cleanLocalTensor_, extCopyParams);
    }
    // clean exp
    for (int loopCnt = 0; loopCnt < expInnerLoopCnt_; loopCnt++) {
        DataCopy(localWinExpTensorGm_[(startExpOffset_ + loopCnt * CLEAN_BUFF_SIZE) / sizeof(int32_t)], cleanLocalTensor_,
                 CLEAN_BUFF_SIZE / sizeof(int32_t));
    }
    if (expTailLoopCleanSize_ > 0) {
        DataCopyExtParams extCopyParams{1U, static_cast<uint32_t>(expTailLoopCleanSize_), 0U, 0U, 0U};
        DataCopyPad(localWinExpTensorGm_[(startExpOffset_ + expInnerLoopCnt_ * CLEAN_BUFF_SIZE) / sizeof(int32_t)],
                    cleanLocalTensor_, extCopyParams);
    }
    pipe_barrier(PIPE_ALL);
    if(needSync_) {
        SyncAll<true>();
        if (startRankId_ < worldSize_) {
            SumParams sumParams{1, sendRankNum_, sendRankNum_};
            Sum(sumLocalTensor_, elasticInfofp32Tensor_, sumParams);
            SyncFunc<AscendC::HardEvent::V_S>();
            float sumRes = sumLocalTensor_.GetValue(0);
            float targetSum = (float)1.0 * sendRankNum_;
            bool fullSend = false;
            if ((float)targetSum - (float)0.5 < sumRes || (float)targetSum + (float)0.5 > sumRes) {
                fullSend = true;
            }
            // send status
            SendStatus(fullSend);
            // wait status
            WaitStatus(sumRes);
            // clean 清状态
            CleanStatus();
        }
    }
}
} // namespace MoeDistributeBufferResetImpl
#endif