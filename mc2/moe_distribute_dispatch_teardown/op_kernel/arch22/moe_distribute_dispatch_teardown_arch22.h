/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_distribute_dispatch_teardown_arch22.h
 * \brief
 */

#ifndef MOE_DISTRIBUTE_DISPATCH_TEARDOWN_H
#define MOE_DISTRIBUTE_DISPATCH_TEARDOWN_H

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_vec_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "kernel_tiling/kernel_tiling.h"
#if __has_include("../../moe_distribute_dispatch_setup/moe_distribute_dispatch_setup_base.h")
#include "../../moe_distribute_dispatch_setup/moe_distribute_dispatch_setup_base.h"
#else
#include "../../../moe_distribute_dispatch_setup/op_kernel/moe_distribute_dispatch_setup_base.h"
#endif
#include "../moe_distribute_dispatch_teardown_tiling.h"

namespace Mc2Kernel {
constexpr uint8_t BUFFER_NUM = 2;       // 多buf
constexpr uint32_t STATE_OFFSET = 512U; // 状态空间偏移地址
constexpr uint32_t UB_ALIGN = 32U;      // UB按32字节对齐
constexpr uint64_t WIN_STATE_OFFSET = 350UL * 1024UL;
constexpr uint64_t STATE_WIN_OFFSET = 950UL * 1024UL;
constexpr uint32_t WORKSPACE_ELEMENT_OFFSET = 512U;
constexpr uint64_t WIN_ADDR_ALIGN = 512UL;
constexpr float EPSILON = 1e-6f;

template <AscendC::HardEvent event>
__aicore__ inline void SyncFunc()
{
    int32_t eventID = static_cast<int32_t>(GetTPipePtr()->FetchEventID(event));
    AscendC::SetFlag<event>(eventID);
    AscendC::WaitFlag<event>(eventID);
}

#define TemplateMC2TypeClass \
    typename XType, typename ExpandXOutType, bool StaticQuant, bool DynamicQuant, bool IsSmoothScaleExist, \
        bool IsShareExpertRank
#define TemplateMC2TypeFunc XType, ExpandXOutType, StaticQuant, DynamicQuant, IsSmoothScaleExist, IsShareExpertRank

using namespace AscendC;
template <TemplateMC2TypeClass>
class MoeDistributeDispatchTeardown {
public:
    __aicore__ inline MoeDistributeDispatchTeardown(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR expertIds, GM_ADDR commCmdInfo, GM_ADDR expandXOut,
                                GM_ADDR dynamicScalesOut, GM_ADDR assistInfoForCombineOut, GM_ADDR expertTokenNumsOut,
                                GM_ADDR workspaceGM, TPipe *pipe,
                                const MoeDistributeDispatchTeardownTilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void LocalWindowCopy();
    __aicore__ inline void WaitDispatch();
    __aicore__ inline void GetCumSum(LocalTensor<int32_t> &outLocal, uint32_t totalCount);
    __aicore__ inline void UpdataTokenNumsOut();
    __aicore__ inline void SplitToCore(uint32_t curSendCnt, uint32_t curUseAivNum, uint32_t &startTokenId,
                                       uint32_t &endTokenId, uint32_t &sendTokenNum, bool isFront = true);
    __aicore__ inline void SyncCntOnCore(LocalTensor<float> &gatherMaskOutTensor,
                                         LocalTensor<uint32_t> &gatherTmpTensor,
                                         LocalTensor<float> &statusSumOutTensor);
    __aicore__ inline GM_ADDR GetWindAddrByRankId(const int32_t rankId)
    {
        if (epRankId_ == rankId) {
            return (GM_ADDR)(winContext_->localWindowsIn) + winDataSizeOffset_;
        }
        return (GM_ADDR)(((HcclRankRelationResV2 *)(winContext_->remoteRes[rankId].nextDevicePtr))->windowsIn) +
               winDataSizeOffset_;
    }

    __aicore__ inline GM_ADDR GetWindStateAddrByRankId(const int32_t rankId)
    {
        if (epRankId_ == rankId) {
            return (GM_ADDR)(winContext_->localWindowsExp) + dataState_ * WIN_STATE_OFFSET;
        }
        return (GM_ADDR)(((HcclRankRelationResV2 *)(winContext_->remoteRes[rankId].nextDevicePtr))->windowsExp) +
               dataState_ * WIN_STATE_OFFSET;
    }

    __aicore__ inline uint32_t MIN(uint32_t x, uint32_t y)
    {
        return (x < y) ? x : y;
    }

    __aicore__ inline int32_t ReduceSumWorkNeedSize(int32_t calCnt)
    {
        int typeSize = static_cast<int>(sizeof(int32_t));
        int32_t elementsPerBlock = static_cast<int32_t>(UB_ALIGN / typeSize);
        int32_t iter1OutputCount = calCnt;
        int32_t iter1AlignEnd = ((iter1OutputCount + elementsPerBlock - 1) / elementsPerBlock) * elementsPerBlock;
        return iter1AlignEnd;
    }

    TPipe *tpipe_{nullptr};
    GlobalTensor<XType> xGMTensor_;
    GlobalTensor<ExpandXOutType> expandXOutGMTensor_;
    GlobalTensor<float> dynamicScalesOutGMTensor_;
    GlobalTensor<int64_t> expertTokenNumsOutGMTensor_;
    GlobalTensor<float> windowInstatusFp32Tensor_;

    LocalTensor<ExpandXOutType> xTmpTensor_;
    LocalTensor<float> xOutFp32Tensor_;
    LocalTensor<int32_t> statusTensor_;
    LocalTensor<float> statusFp32Tensor_;
    TBuf<> statusBuf_;
    TBuf<> gatherMaskOutBuf_; // gather mask输出buf
    TBuf<> sumCoreBuf_;
    TBuf<> sumLocalBuf_;
    TBuf<> sumContinueBuf_;
    TBuf<> scalarBuf_; // 辅助gather tensor定义
    TBuf<> waitStatusBuf_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> xQueue_; // 非量化使用，量化场景接收也可使用

    GM_ADDR expandXOutGM_;
    GM_ADDR sendCountsOutGM_;
    GM_ADDR statusSpaceGm_;
    GM_ADDR windowGM_;
    GM_ADDR recvCntWorkspaceGM_;

    // tiling侧已确保数据上限，相乘不会越界，因此统一采用uint32_t进行处理
    uint32_t axisBS_{0};
    uint32_t axisMaxBS_{0};
    uint32_t axisH_{0};
    uint32_t axisK_{0};
    uint32_t aivNum_{0};
    uint32_t sharedUsedAivNum_{0};
    uint32_t moeUsedAivNum_{0};
    uint32_t epWorldSize_{0};
    uint32_t epRankId_{0};
    uint32_t aivId_{0}; // aiv id
    uint32_t sharedExpertNum_{0};
    uint32_t sharedExpertRankNum_{0};    // 共享专家卡数
    uint32_t rankNumPerSharedExpert_{0}; // 部署单个共享专家所用的卡数
    uint32_t moeExpertNum_{0};
    uint32_t moeExpertRankNum_{0}; // moe专家卡数，等于epWorldSize_ - sharedExpertRankNum_
    uint32_t moeExpertNumPerRank_{0};
    uint32_t dealRankPerCore_{0};
    uint32_t hOutSize_{0};
    uint32_t hAlignWinSize_{0};
    uint32_t hAlignWinCnt_{0};
    uint32_t hOutAlignUbSize_{0};
    uint32_t hOutSizeAlign_{0};
    uint32_t startExpertId_;
    uint32_t endExpertId_;
    uint32_t sendExpertNum_;
    uint32_t totalCnt_;
    uint32_t lastCore_{0};
    uint32_t dataState_{0};
    uint64_t winDataSizeOffset_{0};
    uint64_t expertPerSizeOnWin_{0};
    uint64_t recvWinBlockNum_; // 接收Win区块数
    float sumTarget_;
    uint32_t gatherCount_{0};
    uint32_t expertTokenNumsType_{1};
    uint32_t stateOffset_{0};
    uint32_t recStatusNumPerCore_{0};
    int32_t expertIdsCnt_{0};
    uint32_t rscvStatusNum_{0};
    uint32_t remainderRankNum_{0};
    uint32_t startStatusIndex_{0};
    __gm__ HcclOpResParam *winContext_{nullptr};

    DataCopyExtParams floatDataCopyParams_;
    DataCopyExtParams expandXCopyParams_;
    DataCopyExtParams hCommuCopyOutParams_;
};

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeDispatchTeardown<TemplateMC2TypeFunc>::Init(
    GM_ADDR x, GM_ADDR y, GM_ADDR expertIds, GM_ADDR commCmdInfo, GM_ADDR expandXOut, GM_ADDR dynamicScalesOut,
    GM_ADDR assistInfoForCombineOut, GM_ADDR expertTokenNumsOut, GM_ADDR workspaceGM, TPipe *pipe,
    const MoeDistributeDispatchTeardownTilingData *tilingData)
{
    tpipe_ = pipe;
    aivId_ = GetBlockIdx();
    aivNum_ = tilingData->moeDistributeDispatchTeardownInfo.aivNum;
    epRankId_ = tilingData->moeDistributeDispatchTeardownInfo.epRankId;
    auto contextGM0 = AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
    winContext_ = (__gm__ HcclOpResParam *)AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
    GlobalTensor<int32_t> selfDataStatusTensor;
    GM_ADDR statusDataSpaceGm = (GM_ADDR)(winContext_->localWindowsExp);
    selfDataStatusTensor.SetGlobalBuffer(
        (__gm__ int32_t *)(statusDataSpaceGm + STATE_WIN_OFFSET + aivId_ * WIN_ADDR_ALIGN));

    axisBS_ = tilingData->moeDistributeDispatchTeardownInfo.bs;
    axisH_ = tilingData->moeDistributeDispatchTeardownInfo.h;
    epWorldSize_ = tilingData->moeDistributeDispatchTeardownInfo.epWorldSize;
    axisMaxBS_ = tilingData->moeDistributeDispatchTeardownInfo.globalBs / epWorldSize_;
    moeExpertNum_ = tilingData->moeDistributeDispatchTeardownInfo.moeExpertNum;
    sharedExpertNum_ = tilingData->moeDistributeDispatchTeardownInfo.sharedExpertNum;
    sharedExpertRankNum_ = tilingData->moeDistributeDispatchTeardownInfo.sharedExpertRankNum;
    if (sharedExpertNum_ > 0) {
        rankNumPerSharedExpert_ = sharedExpertRankNum_ / sharedExpertNum_;
    }
    expertTokenNumsType_ = tilingData->moeDistributeDispatchTeardownInfo.expertTokenNumsType;
    moeExpertRankNum_ = epWorldSize_ - sharedExpertRankNum_;
    moeExpertNumPerRank_ = moeExpertNum_ / moeExpertRankNum_;
    axisK_ = tilingData->moeDistributeDispatchTeardownInfo.k;
    xGMTensor_.SetGlobalBuffer((__gm__ XType *)x);
    expandXOutGMTensor_.SetGlobalBuffer((__gm__ ExpandXOutType *)expandXOut);
    dynamicScalesOutGMTensor_.SetGlobalBuffer((__gm__ float *)dynamicScalesOut);
    expertTokenNumsOutGMTensor_.SetGlobalBuffer((__gm__ int64_t *)expertTokenNumsOut);
    expandXOutGM_ = expandXOut;
    sendCountsOutGM_ = assistInfoForCombineOut; // 无GlobalTensor
    recvCntWorkspaceGM_ = workspaceGM;

    uint32_t hSize = axisH_ * sizeof(XType);
    hOutSize_ = axisH_ * sizeof(ExpandXOutType);
    hOutSizeAlign_ = Ceil(hOutSize_, UB_ALIGN) * UB_ALIGN; // scale起始放置偏移
    uint32_t hScaleSizeAlign = hOutSizeAlign_ + UB_ALIGN;  // 实际搬运大小，搬运token_align32B + 32B(float)
    hAlignWinSize_ = Ceil(hScaleSizeAlign, WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN; // win区token起始地址对齐512
    hAlignWinCnt_ = hAlignWinSize_ / sizeof(ExpandXOutType);
    expertPerSizeOnWin_ = axisMaxBS_ * hAlignWinSize_;
    if (sharedExpertRankNum_ != 0U) {
        sharedUsedAivNum_ = (aivNum_ * sharedExpertNum_) / (axisK_ + sharedExpertNum_);
        if (sharedUsedAivNum_ == 0) {
            sharedUsedAivNum_ = 1;
        }
    }
    expertIdsCnt_ = axisBS_ * axisK_;
    recvWinBlockNum_ = epWorldSize_ * moeExpertNumPerRank_;
    moeUsedAivNum_ = aivNum_ - sharedUsedAivNum_;
    dealRankPerCore_ = (recvWinBlockNum_ + aivNum_ - 1) / aivNum_;
    stateOffset_ = ((recvWinBlockNum_ > 512) ? (STATE_OFFSET / 2) : STATE_OFFSET);
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(selfDataStatusTensor);
    dataState_ = selfDataStatusTensor(0);
    if (dataState_ == 0) {
        selfDataStatusTensor(0) = 1;
    } else {
        selfDataStatusTensor(0) = 0;
    }
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(selfDataStatusTensor);
    if constexpr (IsShareExpertRank) {
        rscvStatusNum_ = epWorldSize_;
    } else {
        rscvStatusNum_ = recvWinBlockNum_;
    }
    recStatusNumPerCore_ = rscvStatusNum_ / aivNum_; // 每个aiv需要处理的专家数
    remainderRankNum_ = rscvStatusNum_ % aivNum_;
    startStatusIndex_ = recStatusNumPerCore_ * aivId_; // + sharedExpertRankNum_, 每个aiv发送的
    if (aivId_ < remainderRankNum_) {                  // 前remainderRankNum个aiv需要多发1个卡的数据
        recStatusNumPerCore_ += 1;
        startStatusIndex_ += aivId_;
    } else {
        startStatusIndex_ += remainderRankNum_;
    }
    uint32_t waitStatusBufSize = (((recStatusNumPerCore_ * UB_ALIGN) > 256) ? (recStatusNumPerCore_ * UB_ALIGN) : 256);
    tpipe_->InitBuffer(waitStatusBuf_, waitStatusBufSize);      // 1024/24 * 32B = 43 * 32B
    uint32_t statusBufCntAlign = Ceil(recvWinBlockNum_, 8) * 8; // 8 = UB_ALIGN / sizeof(int32_t)
    tpipe_->InitBuffer(statusBuf_, statusBufCntAlign * UB_ALIGN);
    statusTensor_ = statusBuf_.Get<int32_t>(); // 保存发送数据量及flag，同时用于计算windows中的偏移
    statusSpaceGm_ = GetWindStateAddrByRankId(epRankId_);
    sumTarget_ = static_cast<float>(1.0);
    winDataSizeOffset_ = dataState_ * (tilingData->moeDistributeDispatchTeardownInfo.totalWinSize / 2);
    windowGM_ = GetWindAddrByRankId(epRankId_);
    windowInstatusFp32Tensor_.SetGlobalBuffer((__gm__ float *)(statusSpaceGm_));
    hOutAlignUbSize_ = Ceil(hScaleSizeAlign, UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(xQueue_, BUFFER_NUM, hOutAlignUbSize_); // 7k*2 + 32 + 6
    expertIdsCnt_ = axisBS_ * axisK_;
    uint32_t expertIdsSize = axisBS_ * axisK_ * sizeof(int32_t); // 约束32对齐

    tpipe_->InitBuffer(gatherMaskOutBuf_, recvWinBlockNum_ * sizeof(float)); // 1024 * 4B
    tpipe_->InitBuffer(sumCoreBuf_, aivNum_ * UB_ALIGN);                     // 48 * 32B
    tpipe_->InitBuffer(sumLocalBuf_, aivNum_ * UB_ALIGN);                    // 48 * 32B
    tpipe_->InitBuffer(sumContinueBuf_, aivNum_ * sizeof(float));            // 48 * 4B
    tpipe_->InitBuffer(scalarBuf_, UB_ALIGN * 3);                            // 96B

    uint32_t axisHCommu = hScaleSizeAlign / sizeof(ExpandXOutType); // 有效搬运长度
    floatDataCopyParams_ = {1U, sizeof(float), 0U, 0U, 0U};
    hCommuCopyOutParams_ = {1U, static_cast<uint32_t>(axisHCommu * sizeof(ExpandXOutType)), 0U, 0U, 0U};
    expandXCopyParams_ = {1U, static_cast<uint32_t>(axisH_ * sizeof(ExpandXOutType)), 0U, 0U, 0U};
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeDispatchTeardown<TemplateMC2TypeFunc>::SplitToCore(
    uint32_t curSendCnt, uint32_t curUseAivNum, uint32_t &startTokenId, uint32_t &endTokenId, uint32_t &sendTokenNum,
    bool isFront)
{
    sendTokenNum = curSendCnt / curUseAivNum;               // 每个aiv需要发送的token数
    uint32_t remainderTokenNum = curSendCnt % curUseAivNum; // 余数
    uint32_t newAivId;
    if (isFront) {
        newAivId = aivId_;
    } else {
        newAivId = aivId_ - moeUsedAivNum_; // 由于是后面的核作为发送的共享专家，因此需要换算
    }
    startTokenId = sendTokenNum * newAivId; // 每个aiv发送时的起始rankid
    if (newAivId < remainderTokenNum) {     // 前remainderRankNum个aiv需要多发1个卡的数据
        sendTokenNum += 1;
        startTokenId += newAivId;
    } else {
        startTokenId += remainderTokenNum;
    }
    endTokenId = startTokenId + sendTokenNum;
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeDispatchTeardown<TemplateMC2TypeFunc>::SyncCntOnCore(
    LocalTensor<float> &gatherMaskOutTensor, LocalTensor<uint32_t> &gatherTmpTensor,
    LocalTensor<float> &statusSumOutTensor)
{
    gatherTmpTensor.SetValue(0, 2); // 源操作数每个datablock取下标为1的元素
    uint32_t mask = 2;              // 源操作数每个datablock只需要处理两个元素
    SyncFunc<AscendC::HardEvent::S_V>();

    // 将当前核对应的专家recv cnt收集到gatherMaskOutTensor
    uint64_t rsvdCnt = 0;
    GatherMask(gatherMaskOutTensor, statusFp32Tensor_, gatherTmpTensor, true, mask,
               {1, (uint16_t)recStatusNumPerCore_, 1, 0}, rsvdCnt);
    PipeBarrier<PIPE_V>();

    // 对当前核对应的专家recv cnt求和
    SumParams sumParams{1, recStatusNumPerCore_, recStatusNumPerCore_};
    Sum(statusSumOutTensor, gatherMaskOutTensor, sumParams);
    SyncFunc<AscendC::HardEvent::V_S>();
    int32_t sumOfRecvCnt = statusSumOutTensor.ReinterpretCast<int32_t>().GetValue(0);

    // 把当前核的所有专家的recv cnt之和写到workspace
    uint64_t coreOffset = WORKSPACE_ELEMENT_OFFSET * aivNum_;
    GM_ADDR wAddr = (__gm__ uint8_t *)(recvCntWorkspaceGM_) + coreOffset * aivId_; // 写workspace需要按照512字节对齐
    GlobalTensor<int32_t> sumTensor;
    sumTensor.SetGlobalBuffer((__gm__ int32_t *)wAddr);
    uint16_t workCoreNum = MIN(recvWinBlockNum_, aivNum_);
    // 每个核把sumOfRecvCnt重复写workCoreNum份
    LocalTensor<int32_t> sumCoreTensor = sumCoreBuf_.Get<int32_t>();
    // 仅处理每个datablock的首元素（对应maskArray[0]的bit0）。操作数为32bit情况下，maskArray只有第0个元素有效
    // 每个元素占4字节，每个32字节处理8份，mask中每8个bit的填充第1位
    uint64_t maskArray[2] = {0x0101010101010101, 0};
    // 每个核一个datablock，总共需要处理workCoreNum个核。每个repeat总共256字节，可以处理8个datablock
    uint8_t repeatTimes = static_cast<uint8_t>(workCoreNum + 7) / 8;
    // 1代表单个repeat内不同的datablock连续，没有跳过
    // 8代表不同repeat的首元素间隔8个datablock
    Duplicate<int32_t>(sumCoreTensor, sumOfRecvCnt, maskArray, repeatTimes, 1, 8);
    DataCopyParams sumIntriParams{static_cast<uint16_t>(workCoreNum), 1, 0, 15};
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(sumTensor, sumCoreTensor, sumIntriParams);
    PipeBarrier<PIPE_ALL>();
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeDispatchTeardown<TemplateMC2TypeFunc>::WaitDispatch()
{
    startExpertId_ = startStatusIndex_; // 后面LocalWinCopy分核与此处保持一致
    endExpertId_ = startExpertId_ + recStatusNumPerCore_;
    sendExpertNum_ = recStatusNumPerCore_;
    if (unlikely(startStatusIndex_ >= rscvStatusNum_)) {
        return;
    }
    LocalTensor<float> gatherMaskOutTensor = gatherMaskOutBuf_.Get<float>();
    LocalTensor<uint32_t> gatherTmpTensor = scalarBuf_.GetWithOffset<uint32_t>(UB_ALIGN / sizeof(uint32_t), 0);
    gatherTmpTensor.SetValue(0, 1);
    LocalTensor<float> statusSumOutTensor = scalarBuf_.GetWithOffset<float>(UB_ALIGN / sizeof(float), UB_ALIGN);
    statusFp32Tensor_ = waitStatusBuf_.Get<float>();
    uint32_t mask = 1; // gatherMask + sum 相关参数
    float compareTarget = sumTarget_ * recStatusNumPerCore_;
    float sumOfFlag = static_cast<float>(-1.0);
    DataCopyParams intriParams{static_cast<uint16_t>(recStatusNumPerCore_), 1,
                               static_cast<uint16_t>((recvWinBlockNum_ > 512) ? 7 : 15), 0};
    SyncFunc<AscendC::HardEvent::S_V>();
    while (std::abs(sumOfFlag - compareTarget) > EPSILON) {
        DataCopy(statusFp32Tensor_, windowInstatusFp32Tensor_[startStatusIndex_ * stateOffset_ / sizeof(float)],
                 intriParams);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        ReduceSum(statusSumOutTensor, statusFp32Tensor_, gatherMaskOutTensor, mask, recStatusNumPerCore_, 1);
        SyncFunc<AscendC::HardEvent::V_S>();
        sumOfFlag = statusSumOutTensor.GetValue(0);
    }
    // 清状态
    SyncFunc<AscendC::HardEvent::MTE3_S>();
    DataCopyParams intriOutParams{static_cast<uint16_t>(recStatusNumPerCore_), 1, 0,
                                  static_cast<uint16_t>((recvWinBlockNum_ > 512) ? 7 : 15)};
    uint64_t duplicateMask[2] = {0x101010101010101, 0}; // 一次性操作256字节，也是64个int32_t，每8个数将首个设置为0
    LocalTensor<int32_t> cleanStateTensor = waitStatusBuf_.Get<int32_t>();
    SyncFunc<AscendC::HardEvent::S_V>();
    Duplicate<int32_t>(cleanStateTensor, 0, duplicateMask, Ceil(recStatusNumPerCore_, 8), 1, 8);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(windowInstatusFp32Tensor_[startStatusIndex_ * stateOffset_ / sizeof(float)],
             cleanStateTensor.ReinterpretCast<float>(), intriOutParams);
    SyncFunc<AscendC::HardEvent::MTE3_S>();

    // 核间同步token cnt
    SyncCntOnCore(gatherMaskOutTensor, gatherTmpTensor, statusSumOutTensor);
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeDispatchTeardown<TemplateMC2TypeFunc>::GetCumSum(LocalTensor<int32_t> &outLocal,
                                                                                     uint32_t totalCount)
{
    outLocal = gatherMaskOutBuf_.Get<int32_t>();
    // 获取workspace中每个核的recvcnt
    GM_ADDR wAddr = (__gm__ uint8_t *)(recvCntWorkspaceGM_) + WORKSPACE_ELEMENT_OFFSET * aivId_;
    GlobalTensor<int32_t> sumTensor;
    sumTensor.SetGlobalBuffer((__gm__ int32_t *)wAddr);

    // 不支持allgather场景，只需要拷贝totalCount个核的recv cnt
    uint16_t copySumNum = totalCount;
    uint16_t copyStride = 16 * aivNum_ - 1;
    DataCopyParams sumIntriParams{static_cast<uint16_t>(copySumNum), 1, copyStride, 0};
    LocalTensor<int32_t> sumLocalTensor = sumLocalBuf_.Get<int32_t>();
    DataCopy(sumLocalTensor, sumTensor, sumIntriParams);
    LocalTensor<uint32_t> gatherSumPattern = scalarBuf_.GetWithOffset<uint32_t>(UB_ALIGN / sizeof(uint32_t), 0);
    gatherSumPattern.SetValue(0, 1);
    uint32_t mask = 1;
    uint64_t rsvdCnt = 0;
    LocalTensor<int32_t> sumContinueTensor = sumContinueBuf_.Get<int32_t>();
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    SyncFunc<AscendC::HardEvent::S_V>();
    GatherMask(sumContinueTensor, sumLocalTensor, gatherSumPattern, true, mask, {1, copySumNum, 1, 0}, rsvdCnt);
    // height, width(按照32字节对齐padding后总元素个数), nNum，结果矩阵第一列为对应行的求和结果
    uint32_t innerSumParams = (copySumNum * sizeof(float) + UB_ALIGN - 1) / UB_ALIGN * UB_ALIGN / sizeof(float);
    LocalTensor<float> recvCntSumOutTensor = scalarBuf_.GetWithOffset<float>(UB_ALIGN / sizeof(float), UB_ALIGN);
    PipeBarrier<PIPE_V>();
    LocalTensor<float> tmpFp32 = sumContinueTensor.ReinterpretCast<float>();

    // 0核前面所有核recv cnt总和是0
    if (totalCount == 0) {
        outLocal.SetValue(0, 0);
        return;
    }
    SumParams sumParams{1, innerSumParams, totalCount};
    Sum(recvCntSumOutTensor, tmpFp32, sumParams);
    SyncFunc<AscendC::HardEvent::V_S>();
    // 最终输出outLocal第0个元素是当前核前面所有核recv cnt总和
    outLocal.SetValue(0, recvCntSumOutTensor.ReinterpretCast<int32_t>().GetValue(0));
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeDispatchTeardown<TemplateMC2TypeFunc>::LocalWindowCopy()
{
    LocalTensor<int32_t> outCountLocal;
    if (startExpertId_ >= rscvStatusNum_) { // 分核已与前面的waitDispatch里保持一致
        return;
    }
    GetCumSum(outCountLocal, aivId_);
    uint32_t beginIdx = outCountLocal.GetValue(0);
    statusTensor_ = waitStatusBuf_.Get<int32_t>();
    DataCopyPadExtParams<ExpandXOutType> copyPadExtParams{false, 0U, 0U, 0U};
    DataCopyExtParams dataCopyOutParams{1U, static_cast<uint32_t>(sendExpertNum_ * sizeof(int32_t)), 0U, 0U, 0U};
    for (uint32_t index = startExpertId_; index < endExpertId_; index++) {
        uint32_t i = index - startExpertId_;
        uint32_t count = statusTensor_.GetValue(i * 8 + 1);
        outCountLocal.SetValue(i, beginIdx + count);
        uint32_t winOffset = index;
        if constexpr (!IsShareExpertRank) {
            if (moeExpertNumPerRank_ > 1) { // moe专家卡且一卡多专家场景 转换成数据区的排布偏移
                winOffset = index % epWorldSize_ * moeExpertNumPerRank_ + index / epWorldSize_;
            }
        }
        GM_ADDR wAddr = (__gm__ uint8_t *)(windowGM_) + winOffset * expertPerSizeOnWin_;
        GlobalTensor<ExpandXOutType> tokGlobal;
        GlobalTensor<ExpandXOutType> expandXOutGlobal;
        tokGlobal.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        for (uint32_t j = 0; j < count; j++) {
            tokGlobal.SetGlobalBuffer((__gm__ ExpandXOutType *)(wAddr + j * hAlignWinSize_));
            // 将数据从Window拷贝到UB
            xTmpTensor_ = xQueue_.AllocTensor<ExpandXOutType>();
            DataCopyPad(xTmpTensor_, tokGlobal, hCommuCopyOutParams_, copyPadExtParams);
            xQueue_.EnQue(xTmpTensor_);
            xTmpTensor_ = xQueue_.DeQue<ExpandXOutType>();
            if constexpr (DynamicQuant || StaticQuant) {
                xOutFp32Tensor_ = xTmpTensor_.template ReinterpretCast<float>();
                DataCopyPad(dynamicScalesOutGMTensor_[beginIdx + j], xOutFp32Tensor_[hOutSizeAlign_ / sizeof(float)],
                            floatDataCopyParams_);
            }
            expandXOutGlobal.SetGlobalBuffer((__gm__ ExpandXOutType *)(expandXOutGM_) + (beginIdx + j) * axisH_,
                                             axisH_);
            DataCopyPad(expandXOutGlobal, xTmpTensor_, expandXCopyParams_);
            xQueue_.FreeTensor(xTmpTensor_);
        }
        beginIdx += count;
    }
    totalCnt_ = beginIdx;
    lastCore_ = MIN(rscvStatusNum_, aivNum_) - 1;

    GlobalTensor<int32_t> sendCountsGlobal;
    sendCountsGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(sendCountsOutGM_));
    DataCopyPad(sendCountsGlobal[startExpertId_], outCountLocal, dataCopyOutParams);
    PipeBarrier<PIPE_MTE3>();
}
// 更新tokenNumsOut tensor
template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeDispatchTeardown<TemplateMC2TypeFunc>::UpdataTokenNumsOut()
{
    // 最后一个核做更新，Moe专家只有最后一个核有计算出所有 sendCountsGlobal
    if constexpr (!IsShareExpertRank) {
        if (moeExpertNumPerRank_ > 1) {
            return;
        }
    }

    if (aivId_ == lastCore_) {
        // Moe专家token总数在Cumsum内计算得出
        uint32_t tokenNum = totalCnt_;
        expertTokenNumsOutGMTensor_.SetValue(0, tokenNum);
        DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
            expertTokenNumsOutGMTensor_);
        // moe一卡多专家场景下更新moe专家卡对应expertTokenNums数据
        if (moeExpertNumPerRank_ != 1) {
            if constexpr (!IsShareExpertRank) {
                uint32_t tokenSums = 0;
                SyncFunc<AscendC::HardEvent::MTE3_S>();
                GlobalTensor<int32_t> sendCountsGlobal;
                sendCountsGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(sendCountsOutGM_));
                DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
                    sendCountsGlobal[epWorldSize_ - 1]);

                uint32_t firstMoeCnt = sendCountsGlobal.GetValue(epWorldSize_ - 1);
                tokenSums = firstMoeCnt + gatherCount_;
                expertTokenNumsOutGMTensor_.SetValue(0, tokenSums);
                DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
                    expertTokenNumsOutGMTensor_[0]);
                for (uint32_t localMoeIndex = 1; localMoeIndex < moeExpertNumPerRank_; ++localMoeIndex) {
                    uint32_t preOffset = epWorldSize_ * (localMoeIndex - 1) + epWorldSize_ - 1;
                    uint32_t curOffset = epWorldSize_ * localMoeIndex + epWorldSize_ - 1;
                    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
                        sendCountsGlobal[preOffset]);
                    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
                        sendCountsGlobal[curOffset]);
                    uint32_t preMoeIndexCnt = sendCountsGlobal.GetValue(preOffset);
                    uint32_t curMoeIndexCnt = sendCountsGlobal.GetValue(curOffset);
                    tokenSums = ((expertTokenNumsType_ == 0) ? tokenSums : 0) + (curMoeIndexCnt - preMoeIndexCnt) +
                                gatherCount_;
                    expertTokenNumsOutGMTensor_.SetValue(localMoeIndex, tokenSums);
                    DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
                        expertTokenNumsOutGMTensor_[localMoeIndex]);
                }
            }
        }
    }
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeDispatchTeardown<TemplateMC2TypeFunc>::Process()
{
    if ASCEND_IS_AIV { // 全aiv处理
        WaitDispatch();
        SyncAll<true>();
        LocalWindowCopy();
        UpdataTokenNumsOut();
    }
}

} // namespace Mc2Kernel
#endif // MOE_DISTRIBUTE_DISPATCH_TEARDOWN_H
