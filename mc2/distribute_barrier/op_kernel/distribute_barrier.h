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
 * \file distribute_barrier.h
 * \brief
 */
#ifndef DISTRIBUTE_BARRIER_H
#define DISTRIBUTE_BARRIER_H

#include "distribute_barrier_tiling.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "kernel_tiling/kernel_tiling.h"
#if __has_include("../common/moe_distribute_base.h")
#include "../common/mc2_moe_context.h"
#include "../common/moe_distribute_base.h"
#include "../common/mc2_kernel_utils.h"
#else
#include "../../common/op_kernel/mc2_moe_context.h"
#include "../../common/op_kernel/moe_distribute_base.h"
#include "../../common/op_kernel/mc2_kernel_utils.h"
#endif

namespace DistributeBarrierImpl {
constexpr uint8_t BUFFER_NUM = 2;      // 多buf
constexpr uint32_t UB_ALIGN = 32;      // UB按32字节对齐
constexpr uint32_t STATE_OFFSET = 512; // 状态空间偏移地址
constexpr uint32_t ELASTIC_METAINFO_OFFSET = 4U;
constexpr uint32_t RANK_LIST_NUM = 2U;
constexpr uint32_t LOCAL_STATUS_PADDING = 3U;
constexpr uint64_t WIN_STATE_OFFSET = 512 * 1024; // 状态区的偏移(A区域和B区域)
constexpr uint64_t STATE_WIN_OFFSET = 900 * 1024; // flag标记位的偏移
constexpr uint64_t CYCLES_PER_US = 50UL;

#define TemplateDistributeBarrierTypeClass typename XType
#define TemplateDistributeBarrierTypeFunc XType

using namespace AscendC;
using namespace Mc2Aclnn;
template <TemplateDistributeBarrierTypeClass>
class DistributeBarrier {
public:
    __aicore__ inline DistributeBarrier(){};
    __aicore__ inline void Init(GM_ADDR mc2Context, GM_ADDR timeOut, GM_ADDR elasticInfo, GM_ADDR workspaceGM,
                                TPipe *pipe, const DistributeBarrierTilingData *tilingData);
    __aicore__ inline void TimeOutTest();
    __aicore__ inline void BarrierProcess();
    __aicore__ inline void Process();

private:
    __aicore__ inline GM_ADDR GeWindowAddr(uint32_t toRankId, uint32_t curRankID);
    __aicore__ inline void InitElasticInfo(GM_ADDR elasticInfo);
    __aicore__ inline void InitContextAndRankInfo(GM_ADDR mc2Context, const DistributeBarrierTilingData *tilingData);
    __aicore__ inline void InitStatus();
    __aicore__ inline void SplitToCore();
    __aicore__ inline void SetStatus();
    __aicore__ inline void WaitStatus();
    __aicore__ inline void CleanStatus();
    uint32_t aivId_;
    uint32_t sendAivId_;
    uint32_t rankIdOriginal_;
    uint32_t rankId_;
    TPipe *tpipe_{nullptr};
    uint32_t aivNum_{0};
    uint32_t sendAivNum_{0};
    uint32_t sendRankNum_{0};
    uint32_t startRankId_{0};
    uint32_t endRankId_{0};
    uint32_t worldSizeOriginal_{0};
    uint32_t worldSize_{0};
    uint32_t stateOffset_{0};
    uint32_t dataState_{0};
    uint32_t timeOut_{0};
    bool isInputTimeout_{false};
    bool isInputElasticInfo_{false};
    bool isElasticTrueFlag_{false};
    bool isMc2Context_ = {false};

    __gm__ Mc2Kernel::HcclOpParam *winContext_{nullptr};
    __gm__ Mc2MoeContext *mc2Context_{nullptr};

    LocalTensor<float> statusFp32Tensor_;
    LocalTensor<int32_t> elasticInfoTensor_;
    GlobalTensor<float> windowInstatusFp32Tensor_;
    TBuf<> statusBuf_;
    TBuf<> gatherMaskOutBuf_; // gather mask输出buf
    TBuf<> scalarBuf_;        // 辅助gather tensor定义
    TBuf<> waitStatusBuf_;
    TBuf<> elasticInfoBuf_;
    TBuf<> timeoutInfoBuf_;
    TBuf<> timeoutMaskOutBuf_;

    GM_ADDR statusSpaceGm_;
    GM_ADDR finishGm_;
    GM_ADDR statusDataSpaceGm_;
};

template <TemplateDistributeBarrierTypeClass>
__aicore__ inline void DistributeBarrier<TemplateDistributeBarrierTypeFunc>::TimeOutTest()
{
    uint64_t systemCntBegin = static_cast<uint64_t>(GetSystemCycle());
    float curRank = 0.0;
    uint32_t usedAivNum = sendRankNum_ > 0 ? sendAivNum_ : worldSize_;
    GlobalTensor<float> finishTensor;
    finishTensor.SetGlobalBuffer((__gm__ float *)finishGm_);
    LocalTensor<float> timeoutTensor;
    timeoutTensor = timeoutInfoBuf_.Get<float>();
    uint32_t mask = 1;
    LocalTensor<float> gatherMaskOutTensor;
    float targetNum = usedAivNum * (float)1.0;
    LocalTensor<float> statusSumOutTensor = scalarBuf_.GetWithOffset<float>(UB_ALIGN / sizeof(float), UB_ALIGN);
    LocalTensor<float> timeoutMaskOutTensor;
    timeoutMaskOutTensor = timeoutMaskOutBuf_.Get<float>();
    while (curRank < targetNum) {
        DataCopyParams copyParams{(uint16_t)usedAivNum, 1, LOCAL_STATUS_PADDING, 0U};
        DataCopy(timeoutTensor, finishTensor, copyParams);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        ReduceSum(statusSumOutTensor, timeoutTensor, timeoutMaskOutTensor, mask, usedAivNum, 1);
        SyncFunc<AscendC::HardEvent::V_S>();
        curRank = statusSumOutTensor.GetValue(0);
        if (isInputTimeout_) {
            uint64_t systemCntEnd = static_cast<uint64_t>(GetSystemCycle());
            uint64_t duration = (systemCntEnd - systemCntBegin) / CYCLES_PER_US;
            if (duration >= timeOut_) {
                // 超时后做dfx，通过assert做aicore退出处理
                PipeBarrier<PIPE_ALL>();
                assert((duration < timeOut_) && "Operator execution timed out");
                PipeBarrier<PIPE_ALL>();
            }
        }
    }
    DataCopyParams intriOutParams{static_cast<uint16_t>(usedAivNum), 1, 0U, LOCAL_STATUS_PADDING};
    LocalTensor<int32_t> cleanStateTensor = waitStatusBuf_.Get<int32_t>();
    SyncFunc<AscendC::HardEvent::S_V>();
    Duplicate<int32_t>(cleanStateTensor, 0, usedAivNum * UB_ALIGN / sizeof(float));
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(finishTensor, cleanStateTensor.ReinterpretCast<float>(), intriOutParams);
}

template <TemplateDistributeBarrierTypeClass>
__aicore__ inline GM_ADDR DistributeBarrier<TemplateDistributeBarrierTypeFunc>::GeWindowAddr(uint32_t toRankId,
                                                                                             uint32_t curRankID)
{
    if (isMc2Context_) {
        return (GM_ADDR)mc2Context_->epHcclBuffer_[toRankId] + dataState_ * WIN_STATE_OFFSET;
    } else {
        return Mc2Kernel::GetBaseWindStateAddrByRankId(winContext_, toRankId, curRankID) +
               dataState_ * WIN_STATE_OFFSET;
    }
}

template <TemplateDistributeBarrierTypeClass>
__aicore__ inline void DistributeBarrier<TemplateDistributeBarrierTypeFunc>::InitElasticInfo(GM_ADDR elasticInfo)
{
    tpipe_->InitBuffer(
        elasticInfoBuf_,
        Ceil((ELASTIC_METAINFO_OFFSET + RANK_LIST_NUM * worldSizeOriginal_) * sizeof(uint32_t), UB_ALIGN) * UB_ALIGN);
    elasticInfoTensor_ = elasticInfoBuf_.Get<int32_t>();
    GlobalTensor<int32_t> elasticInfoGMTensor;
    elasticInfoGMTensor.SetGlobalBuffer((__gm__ int32_t *)(elasticInfo));
    DataCopyExtParams elasticInfoParams = {
        1U, static_cast<uint32_t>((ELASTIC_METAINFO_OFFSET + RANK_LIST_NUM * worldSizeOriginal_) * sizeof(uint32_t)),
        0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> elasticInfoCopyPadParams{false, 0U, 0U, 0U};
    DataCopyPad(elasticInfoTensor_, elasticInfoGMTensor, elasticInfoParams, elasticInfoCopyPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    if (elasticInfoTensor_.GetValue(0) == 1) {
        isElasticTrueFlag_ = true;
        worldSize_ = elasticInfoTensor_.GetValue(1);
        if (rankId_ < worldSizeOriginal_) {
            rankId_ = elasticInfoTensor_.GetValue(ELASTIC_METAINFO_OFFSET + rankId_);
        }
    }
}

template <TemplateDistributeBarrierTypeClass>
__aicore__ inline void DistributeBarrier<TemplateDistributeBarrierTypeFunc>::InitStatus()
{
    GlobalTensor<int32_t> selfDataStatusTensor;
    // 获取flag标记，flag标记写在win状态区的STATE_WIN_OFFSET偏移的位置
    selfDataStatusTensor.SetGlobalBuffer((__gm__ int32_t *)(statusDataSpaceGm_ + STATE_WIN_OFFSET));
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
        selfDataStatusTensor[aivId_ * UB_ALIGN]);
    dataState_ = selfDataStatusTensor(aivId_ * UB_ALIGN);
    if (dataState_ == 0) {
        selfDataStatusTensor(aivId_ * UB_ALIGN) = 1;
    } else {
        selfDataStatusTensor(aivId_ * UB_ALIGN) = 0;
    }
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
        selfDataStatusTensor[aivId_ * UB_ALIGN]);
}

template <TemplateDistributeBarrierTypeClass>
__aicore__ inline void DistributeBarrier<TemplateDistributeBarrierTypeFunc>::InitContextAndRankInfo(
    GM_ADDR mc2Context, const DistributeBarrierTilingData *tilingData)
{
    if (tilingData->distributeBarrierInfo.isMc2Context) {
        isMc2Context_ = true;
    }

    if (isMc2Context_) {
        mc2Context_ = (__gm__ Mc2MoeContext *)mc2Context;
        rankIdOriginal_ = mc2Context_->epRankId;
        rankId_ = mc2Context_->epRankId;
        statusDataSpaceGm_ = (GM_ADDR)(mc2Context_->epHcclBuffer_[rankId_]);
    } else {
        winContext_ = (__gm__ Mc2Kernel::HcclOpParam *)AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
        rankIdOriginal_ = Mc2Kernel::GetRankId(winContext_);
        rankId_ = Mc2Kernel::GetRankId(winContext_);
        statusDataSpaceGm_ = Mc2Kernel::GetStatusDataSpaceGm(winContext_);
    }
}

template <TemplateDistributeBarrierTypeClass>
__aicore__ inline void DistributeBarrier<TemplateDistributeBarrierTypeFunc>::SplitToCore()
{
    sendRankNum_ = worldSize_ / sendAivNum_; // 每个aiv需要处理的专家数
    uint32_t remainderRankNum = worldSize_ % sendAivNum_;
    if (aivId_ > 0) {
        sendAivId_ = aivId_ - 1;
        startRankId_ = sendRankNum_ * sendAivId_; // + sharedExpertRankNum_, 每个aiv发送的起始rankid
        if (sendAivId_ < remainderRankNum) {      // 前remainderRankNum个aiv需要多发1个卡的数据
            sendRankNum_ += 1;
            startRankId_ += sendAivId_;
        } else {
            startRankId_ += remainderRankNum;
        }
        endRankId_ = startRankId_ + sendRankNum_;
    }
}

template <TemplateDistributeBarrierTypeClass>
__aicore__ inline void DistributeBarrier<TemplateDistributeBarrierTypeFunc>::Init(
    GM_ADDR mc2Context, GM_ADDR timeOut, GM_ADDR elasticInfo, GM_ADDR workspaceGM, TPipe *pipe,
    const DistributeBarrierTilingData *tilingData)
{
    tpipe_ = pipe;
    aivId_ = GetBlockIdx();
    InitContextAndRankInfo(mc2Context, tilingData);
    aivNum_ = tilingData->distributeBarrierInfo.aivNum;
    worldSizeOriginal_ = tilingData->distributeBarrierInfo.worldSize;
    worldSize_ = tilingData->distributeBarrierInfo.worldSize;
    isInputTimeout_ = tilingData->distributeBarrierInfo.isInputTimeOut;
    isInputElasticInfo_ = tilingData->distributeBarrierInfo.isInputElasticInfo;
    if (isInputTimeout_) {
        GlobalTensor<int32_t> timeOutGMTensor;
        timeOutGMTensor.SetGlobalBuffer((__gm__ int32_t *)(timeOut));
        timeOut_ = timeOutGMTensor.GetValue(0);
    }

    if (isInputElasticInfo_) {
        InitElasticInfo(elasticInfo);
    }

    sendAivNum_ = aivNum_ - 1;
    stateOffset_ = STATE_OFFSET;
    InitStatus();

    // 分核计算，获取当前核处理的RankId范围
    SplitToCore();

    // 申请状态区的buffer
    uint32_t dataLen_ = worldSize_ >= UB_ALIGN / sizeof(float) ? worldSize_ : UB_ALIGN / sizeof(float);
    tpipe_->InitBuffer(statusBuf_, dataLen_ * UB_ALIGN);        // expertNum * 32B
    tpipe_->InitBuffer(gatherMaskOutBuf_, dataLen_ * UB_ALIGN); // worldsize * 4B
    tpipe_->InitBuffer(scalarBuf_, UB_ALIGN * 2);               // 72B

    int32_t statusBufSize = sendRankNum_;
    if (aivId_ == 0) {
        tpipe_->InitBuffer(timeoutInfoBuf_, UB_ALIGN * aivNum_);
        uint64_t gatherMaskOutSize = Ceil(sendAivNum_ * sizeof(float), UB_ALIGN) * UB_ALIGN;
        tpipe_->InitBuffer(timeoutMaskOutBuf_, gatherMaskOutSize);
        statusBufSize = sendRankNum_ > 0 ? sendAivNum_ : worldSize_;
    }
    tpipe_->InitBuffer(waitStatusBuf_, statusBufSize * UB_ALIGN);

    statusFp32Tensor_ = waitStatusBuf_.Get<float>();
    Duplicate<float>(statusFp32Tensor_, 0, sendRankNum_ * UB_ALIGN / sizeof(float));
    statusFp32Tensor_ = statusBuf_.Get<float>();
    Duplicate<float>(statusFp32Tensor_, 1.0f, dataLen_ * UB_ALIGN / sizeof(float));

    statusSpaceGm_ = GeWindowAddr(rankIdOriginal_, rankIdOriginal_);
    finishGm_ = statusSpaceGm_ + STATE_WIN_OFFSET + aivNum_ * UB_ALIGN * sizeof(float);
    windowInstatusFp32Tensor_.SetGlobalBuffer((__gm__ float *)(statusSpaceGm_));
}

template <TemplateDistributeBarrierTypeClass>
__aicore__ inline void DistributeBarrier<TemplateDistributeBarrierTypeFunc>::SetStatus()
{
    GlobalTensor<float> rankGMTensor;
    uint32_t offset = stateOffset_ * rankId_;
    for (uint32_t rankIndex = startRankId_; rankIndex < endRankId_; ++rankIndex) {
        if (rankIndex < worldSize_) {
            uint32_t toRankId = rankIndex;
            if (isElasticTrueFlag_) {
                toRankId = elasticInfoTensor_.GetValue(ELASTIC_METAINFO_OFFSET + worldSizeOriginal_ + rankIndex);
            }
            GM_ADDR rankGM = (__gm__ uint8_t *)(GeWindowAddr(toRankId, rankIdOriginal_) + offset); // 计算地址偏移
            SyncFunc<AscendC::HardEvent::V_MTE3>();
            rankGMTensor.SetGlobalBuffer((__gm__ float *)rankGM);
            DataCopy<float>(rankGMTensor, statusFp32Tensor_, UB_ALIGN / sizeof(float)); // 8时数据大小，按32对齐拷贝
        }
    }
}

template <TemplateDistributeBarrierTypeClass>
__aicore__ inline void DistributeBarrier<TemplateDistributeBarrierTypeFunc>::WaitStatus()
{
    LocalTensor<float> gatherMaskOutTensor = gatherMaskOutBuf_.Get<float>();
    LocalTensor<float> statusSumOutTensor = scalarBuf_.GetWithOffset<float>(UB_ALIGN / sizeof(float), UB_ALIGN);
    statusFp32Tensor_ = waitStatusBuf_.Get<float>();
    float compareTarget = static_cast<float>(1.0) * sendRankNum_ * UB_ALIGN / sizeof(float);
    float sumOfFlag = static_cast<float>(-1.0);
    DataCopyParams intriParams{static_cast<uint16_t>(sendRankNum_), 1, (STATE_OFFSET - UB_ALIGN) / UB_ALIGN, 0};
    uint64_t systemCntBegin = static_cast<uint64_t>(GetSystemCycle());
    SyncFunc<AscendC::HardEvent::S_V>();
    while (sumOfFlag != compareTarget) {
        DataCopy(statusFp32Tensor_, windowInstatusFp32Tensor_[startRankId_ * stateOffset_ / sizeof(float)],
                 intriParams);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        ReduceSum(statusSumOutTensor, statusFp32Tensor_, gatherMaskOutTensor, sendRankNum_ * UB_ALIGN / sizeof(float));
        SyncFunc<AscendC::HardEvent::V_S>();
        sumOfFlag = statusSumOutTensor.GetValue(0);
    }
}

template <TemplateDistributeBarrierTypeClass>
__aicore__ inline void DistributeBarrier<TemplateDistributeBarrierTypeFunc>::CleanStatus()
{
    GlobalTensor<float> finishTensor;
    finishTensor.SetGlobalBuffer((__gm__ float *)finishGm_);
    finishTensor(sendAivId_ * UB_ALIGN) = 1.0;
    DataCacheCleanAndInvalid<float, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
        finishTensor[sendAivId_ * UB_ALIGN]);

    DataCopyParams intriOutParams{static_cast<uint16_t>(sendRankNum_), 1, 0, (STATE_OFFSET - UB_ALIGN) / UB_ALIGN};
    LocalTensor<int32_t> cleanStateTensor = waitStatusBuf_.Get<int32_t>();
    SyncFunc<AscendC::HardEvent::S_V>();
    Duplicate<int32_t>(cleanStateTensor, 0, sendRankNum_ * UB_ALIGN / sizeof(float));
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(windowInstatusFp32Tensor_[startRankId_ * stateOffset_ / sizeof(float)],
             cleanStateTensor.ReinterpretCast<float>(), intriOutParams);
    SyncFunc<AscendC::HardEvent::MTE3_S>();
}

template <TemplateDistributeBarrierTypeClass>
__aicore__ inline void DistributeBarrier<TemplateDistributeBarrierTypeFunc>::BarrierProcess()
{
    // 对当前核分到的卡发状态
    SetStatus();
    PipeBarrier<PIPE_ALL>();
    if (startRankId_ >= worldSize_) {
        return;
    }
    // 当前核循环读取状态空间，确定收到该收的数据总数一致
    WaitStatus();
    // 为确保后续再次调用算子读取状态位的正确性，需要清理状态区空间
    CleanStatus();
}

template <TemplateDistributeBarrierTypeClass>
__aicore__ inline void DistributeBarrier<TemplateDistributeBarrierTypeFunc>::Process()
{
    if (aivId_ == 0) {
        TimeOutTest();
    } else {
        BarrierProcess();
    }
    SyncAll<true>(); // 保证所有核退出的时间一致
}
} // namespace DistributeBarrierImpl
#endif
