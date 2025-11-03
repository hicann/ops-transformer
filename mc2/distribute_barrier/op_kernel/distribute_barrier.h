/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file distribute_barrier.h
 * \brief
 */
#ifndef DISTRIBUTE_BARRIER_H
#define DISTRIBUTE_BARRIER_H

#include "distribute_barrier_tiling.h"
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "moe_distribute_base.h"

namespace DistributeBarrierImpl {
constexpr uint8_t BUFFER_NUM = 2;       // 多buf
constexpr uint32_t UB_ALIGN = 32;       // UB按32字节对齐
constexpr uint32_t STATE_OFFSET = 512;  // 状态空间偏移地址
constexpr uint32_t ELASTIC_METAINFO_OFFSET = 4U;
constexpr uint32_t RANK_LIST_NUM = 2U;
constexpr uint64_t WIN_STATE_OFFSET = 512 * 1024;  // 状态区的偏移(A区域和B区域)
constexpr uint64_t STATE_WIN_OFFSET = 900 * 1024;  // flag标记位的偏移
constexpr uint64_t CYCLES_PER_US = 50UL;

template <AscendC::HardEvent event>
__aicore__ inline void SyncFunc() {
  int32_t eventID = static_cast<int32_t>(GetTPipePtr()->FetchEventID(event));
  AscendC::SetFlag<event>(eventID);
  AscendC::WaitFlag<event>(eventID);
}

#define TemplateMC2TypeClass typename XType
#define TemplateMC2TypeFunc XType

using namespace AscendC;
template <TemplateMC2TypeClass>
class DistributeBarrier {
 public:
  __aicore__ inline DistributeBarrier(){};
  __aicore__ inline void Init(GM_ADDR timeOut, GM_ADDR elasticInfo, GM_ADDR workspaceGM, TPipe *pipe,
                              const DistributeBarrierTilingData *tilingData);
  __aicore__ inline void Process();

 private:
  __aicore__ inline GM_ADDR GeWindowAddr(uint32_t toRankId, uint32_t curRankID);
  __aicore__ inline void InitElasticInfo(GM_ADDR elasticInfo);
  __aicore__ inline void SplitToCore();
  __aicore__ inline void SetStatus();
  __aicore__ inline void WaitStatus();
  __aicore__ inline void CleanStatus();
  uint32_t aivId_;
  uint32_t rankIdOriginal_;
  uint32_t rankId_;
  TPipe *tpipe_{nullptr};
  uint32_t aivNum_{0};
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
  __gm__ HcclOpResParam *winContext_{nullptr};

  LocalTensor<float> statusFp32Tensor_;
  LocalTensor<int32_t> elasticInfoTensor_;
  GlobalTensor<float> windowInstatusFp32Tensor_;
  TBuf<> statusBuf_;
  TBuf<> gatherMaskOutBuf_;  // gather mask输出buf
  TBuf<> scalarBuf_;         // 辅助gather tensor定义
  TBuf<> waitStatusBuf_;
  TBuf<> elasticInfoBuf_;

  GM_ADDR statusSpaceGm_;
};

template <TemplateMC2TypeClass>
__aicore__ inline GM_ADDR DistributeBarrier<TemplateMC2TypeFunc>::GeWindowAddr(
    uint32_t toRankId, uint32_t curRankID) {
  if (toRankId == curRankID) {
    return (GM_ADDR)(winContext_->localWindowsExp) +
           dataState_ * WIN_STATE_OFFSET;
  }
  return (GM_ADDR)(((HcclRankRelationResV2 *)(winContext_->remoteRes[toRankId]
                                                  .nextDevicePtr))
                       ->windowsExp) +
         dataState_ * WIN_STATE_OFFSET;
}

template <TemplateMC2TypeClass>
__aicore__ inline void DistributeBarrier<TemplateMC2TypeFunc>::InitElasticInfo(GM_ADDR elasticInfo) {
  tpipe_->InitBuffer(elasticInfoBuf_, Ceil((ELASTIC_METAINFO_OFFSET + RANK_LIST_NUM * worldSizeOriginal_) * sizeof(uint32_t), UB_ALIGN) * UB_ALIGN);
  elasticInfoTensor_ = elasticInfoBuf_.Get<int32_t>();
  GlobalTensor<int32_t> elasticInfoGMTensor;
  elasticInfoGMTensor.SetGlobalBuffer((__gm__ int32_t*)(elasticInfo));
  DataCopyExtParams elasticInfoParams = {1U, static_cast<uint32_t>((ELASTIC_METAINFO_OFFSET + RANK_LIST_NUM * worldSizeOriginal_) * sizeof(uint32_t)), 0U, 0U, 0U};
  DataCopyPadExtParams<int32_t> elasticInfoCopyPadParams{false, 0U, 0U, 0U};
  DataCopyPad(elasticInfoTensor_, elasticInfoGMTensor, elasticInfoParams, elasticInfoCopyPadParams);
  SyncFunc<AscendC::HardEvent::MTE2_S>();
  if (elasticInfoTensor_.GetValue(0) == 1) {
    isElasticTrueFlag_ = true;
    worldSize_ = elasticInfoTensor_.GetValue(1);
    rankId_ = elasticInfoTensor_.GetValue(ELASTIC_METAINFO_OFFSET + rankId_);
  }
}

template <TemplateMC2TypeClass>
__aicore__ inline void DistributeBarrier<TemplateMC2TypeFunc>::SplitToCore()
{
  sendRankNum_ = worldSize_ / aivNum_;  // 每个aiv需要处理的专家数
  uint32_t remainderRankNum = worldSize_ % aivNum_;
  startRankId_ = sendRankNum_ * aivId_;  // + sharedExpertRankNum_, 每个aiv发送的起始rankid
  if (aivId_ < remainderRankNum) {  // 前remainderRankNum个aiv需要多发1个卡的数据
    sendRankNum_ += 1;
    startRankId_ += aivId_;
  } else {
    startRankId_ += remainderRankNum;
  }
  endRankId_ = startRankId_ + sendRankNum_;
}

template <TemplateMC2TypeClass>
__aicore__ inline void DistributeBarrier<TemplateMC2TypeFunc>::Init(
    GM_ADDR timeOut, GM_ADDR elasticInfo, GM_ADDR workspaceGM, TPipe *pipe,
    const DistributeBarrierTilingData *tilingData) {
  tpipe_ = pipe;
  aivId_ = GetBlockIdx();
  winContext_ = (__gm__ HcclOpResParam *)AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
  rankIdOriginal_ = winContext_->localUsrRankId;
  rankId_ = winContext_->localUsrRankId;
  aivNum_ = tilingData->distributeBarrierInfo.aivNum;
  worldSizeOriginal_ = tilingData->distributeBarrierInfo.worldSize;
  worldSize_ = tilingData->distributeBarrierInfo.worldSize;
  isInputTimeout_ = tilingData->distributeBarrierInfo.isInputTimeOut;
  isInputElasticInfo_ = tilingData->distributeBarrierInfo.isInputElasticInfo;
  if (isInputTimeout_) {
    GlobalTensor<int32_t> timeOutGMTensor;
    timeOutGMTensor.SetGlobalBuffer((__gm__ int32_t*)(timeOut));
    timeOut_ = timeOutGMTensor.GetValue(0);
  }
  if (isInputElasticInfo_) {
    InitElasticInfo(elasticInfo);
  }
  stateOffset_ = STATE_OFFSET;
  GlobalTensor<int32_t> selfDataStatusTensor;
  GM_ADDR statusDataSpaceGm = (GM_ADDR)(winContext_->localWindowsExp);
  // 获取flag标记，flag标记写在win状态区的STATE_WIN_OFFSET偏移的位置
  selfDataStatusTensor.SetGlobalBuffer(
      (__gm__ int32_t *)(statusDataSpaceGm + STATE_WIN_OFFSET));
  DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE,
                           DcciDst::CACHELINE_OUT>(selfDataStatusTensor[aivId_ * UB_ALIGN]);
  dataState_ = selfDataStatusTensor(aivId_ * UB_ALIGN);
  if (dataState_ == 0) {
    selfDataStatusTensor(aivId_ * UB_ALIGN) = 1;
  } else {
    selfDataStatusTensor(aivId_ * UB_ALIGN) = 0;
  }
  DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
    selfDataStatusTensor[aivId_ * UB_ALIGN]);

  // 分核计算，获取当前核处理的RankId范围
  SplitToCore();

  // 申请状态区的buffer
  uint32_t dataLen_ = worldSize_ >= UB_ALIGN / sizeof(float)
                          ? worldSize_
                          : UB_ALIGN / sizeof(float);
  tpipe_->InitBuffer(statusBuf_, dataLen_ * UB_ALIGN);  // expertNum * 32B
  tpipe_->InitBuffer(gatherMaskOutBuf_, dataLen_ * UB_ALIGN);  // worldsize * 4B
  tpipe_->InitBuffer(scalarBuf_, UB_ALIGN * 2);                // 72B
  tpipe_->InitBuffer(waitStatusBuf_, sendRankNum_ * UB_ALIGN);

  statusFp32Tensor_ = waitStatusBuf_.Get<float>();
  Duplicate<float>(statusFp32Tensor_, 0,
                   sendRankNum_ * UB_ALIGN / sizeof(float));
  statusFp32Tensor_ = statusBuf_.Get<float>();
  Duplicate<float>(statusFp32Tensor_, 1.0f,
                   dataLen_ * UB_ALIGN / sizeof(float));

  statusSpaceGm_ = GeWindowAddr(rankIdOriginal_, rankIdOriginal_);
  windowInstatusFp32Tensor_.SetGlobalBuffer((__gm__ float *)(statusSpaceGm_));
}

template <TemplateMC2TypeClass>
__aicore__ inline void DistributeBarrier<TemplateMC2TypeFunc>::SetStatus() {
  GlobalTensor<float> rankGMTensor;
  uint32_t offset = stateOffset_ * rankId_;
  for (uint32_t rankIndex = startRankId_; rankIndex < endRankId_; ++rankIndex) {
    if (rankIndex < worldSize_) {
      uint32_t toRankId = rankIndex;
      if (isElasticTrueFlag_) {
        toRankId = elasticInfoTensor_.GetValue(ELASTIC_METAINFO_OFFSET + worldSizeOriginal_ + rankIndex);
      }
      GM_ADDR rankGM = (__gm__ uint8_t *)(GeWindowAddr(toRankId, rankIdOriginal_) + offset);  // 计算地址偏移
      rankGMTensor.SetGlobalBuffer((__gm__ float *)rankGM);
      DataCopy<float>(rankGMTensor, statusFp32Tensor_, UB_ALIGN / sizeof(float));  // 8时数据大小，按32对齐拷贝
    }
  }
}

template <TemplateMC2TypeClass>
__aicore__ inline void DistributeBarrier<TemplateMC2TypeFunc>::WaitStatus() {
  LocalTensor<float> gatherMaskOutTensor = gatherMaskOutBuf_.Get<float>();
  LocalTensor<float> statusSumOutTensor =
      scalarBuf_.GetWithOffset<float>(UB_ALIGN / sizeof(float), UB_ALIGN);
  statusFp32Tensor_ = waitStatusBuf_.Get<float>();
  float compareTarget =
      static_cast<float>(1.0) * sendRankNum_ * UB_ALIGN / sizeof(float);
  float sumOfFlag = static_cast<float>(-1.0);
  DataCopyParams intriParams{static_cast<uint16_t>(sendRankNum_), 1,
                             (STATE_OFFSET - UB_ALIGN) / UB_ALIGN, 0};
  uint64_t systemCntBegin = static_cast<uint64_t>(GetSystemCycle());
  SyncFunc<AscendC::HardEvent::S_V>();
  while (sumOfFlag != compareTarget) {
    DataCopy(
        statusFp32Tensor_,
        windowInstatusFp32Tensor_[startRankId_ * stateOffset_ / sizeof(float)],
        intriParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    ReduceSum(statusSumOutTensor, statusFp32Tensor_, gatherMaskOutTensor,
              sendRankNum_ * UB_ALIGN / sizeof(float));
    SyncFunc<AscendC::HardEvent::V_S>();
    sumOfFlag = statusSumOutTensor.GetValue(0);
    if (isInputTimeout_) {
      uint64_t systemCntEnd = static_cast<uint64_t>(GetSystemCycle());
      uint64_t duration = (systemCntEnd - systemCntBegin) / CYCLES_PER_US;
      assert(duration < timeOut_);
    }
  }
}

template <TemplateMC2TypeClass>
__aicore__ inline void DistributeBarrier<TemplateMC2TypeFunc>::CleanStatus() {
  DataCopyParams intriOutParams{static_cast<uint16_t>(sendRankNum_), 1, 0,
                                (STATE_OFFSET - UB_ALIGN) / UB_ALIGN};
  LocalTensor<int32_t> cleanStateTensor = waitStatusBuf_.Get<int32_t>();
  SyncFunc<AscendC::HardEvent::S_V>();
  Duplicate<int32_t>(cleanStateTensor, 0,
                     sendRankNum_ * UB_ALIGN / sizeof(float));
  SyncFunc<AscendC::HardEvent::V_MTE3>();
  DataCopy(
      windowInstatusFp32Tensor_[startRankId_ * stateOffset_ / sizeof(float)],
      cleanStateTensor.ReinterpretCast<float>(), intriOutParams);
  SyncFunc<AscendC::HardEvent::MTE3_S>();
}

template <TemplateMC2TypeClass>
__aicore__ inline void DistributeBarrier<TemplateMC2TypeFunc>::Process() {
  // 对当前核分到的卡发状态
  SetStatus();
  if (startRankId_ >= worldSize_) {
    return;
  }
  // 当前核循环读取状态空间，确定收到该收的数据总数一致
  WaitStatus();
  // 为确保后续再次调用算子读取状态位的正确性，需要清理状态区空间
  CleanStatus();
}
}  // namespace DistributeBarrierImpl
#endif