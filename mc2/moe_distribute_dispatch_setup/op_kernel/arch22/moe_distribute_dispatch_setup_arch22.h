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
 * \file moe_distribute_dispatch_setup_arch22.h
 * \brief
 */

#ifndef MOE_DISTRIBUTE_DISPATCH_SETUP_H
#define MOE_DISTRIBUTE_DISPATCH_SETUP_H

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_vec_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "kernel_tiling/kernel_tiling.h"
#include "../moe_distribute_dispatch_setup_base.h"
#include "../moe_distribute_dispatch_setup_tiling.h"

namespace Mc2Kernel {
constexpr uint8_t BUFFER_NUM = 2;       // 多buf
constexpr uint32_t STATE_OFFSET = 512U; // 状态空间偏移地址
constexpr uint32_t UB_ALIGN = 32U;      // UB按32字节对齐
constexpr uint64_t WIN_STATE_OFFSET = 350UL * 1024UL;
constexpr uint64_t STATE_WIN_OFFSET = 950UL * 1024UL;
constexpr uint64_t WIN_ADDR_ALIGN = 512UL;
constexpr uint32_t SQE_START_OFFSET = 10U * 1024U * 1024U;

struct BatchWriteItem {
    uint64_t type;
    uint32_t res1[5];
    uint32_t length;
    uint32_t srcAddrLow;
    uint32_t srcAddrHigh;
    uint32_t dstAddrLow;
    uint32_t dstAddrHigh;
    uint32_t res2[4];
};

template <AscendC::HardEvent event>
__aicore__ inline void SyncFunc()
{
    int32_t eventID = static_cast<int32_t>(GetTPipePtr()->FetchEventID(event));
    AscendC::SetFlag<event>(eventID);
    AscendC::WaitFlag<event>(eventID);
}

#define TemplateMC2TypeClass \
    typename XType, typename YOutType, bool StaticQuant, bool DynamicQuant, bool IsSmoothScaleExist
#define TemplateMC2TypeFunc XType, YOutType, StaticQuant, DynamicQuant, IsSmoothScaleExist

using namespace AscendC;
template <TemplateMC2TypeClass>
class MoeDistributeDispatchSetup {
public:
    __aicore__ inline MoeDistributeDispatchSetup(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR xActiveMask, GM_ADDR YOut,
                                GM_ADDR expandIdxOut, GM_ADDR commCmdInfoOut, GM_ADDR workspaceGM, TPipe *pipe,
                                const MoeDistributeDispatchSetupTilingData *tilingData, __gm__ void *mc2InitTiling,
                                __gm__ void *mc2CcTiling);
    __aicore__ inline void Process();

private:
    __aicore__ inline void SendToSharedExpert();
    __aicore__ inline void SendToMoeExpert();
    __aicore__ inline void AlltoAllDispatch();
    __aicore__ inline void ActiveMaskCalCnt();
    __aicore__ inline void ReduceMaxInplace(const LocalTensor<float> &srcLocal, uint32_t count);
    __aicore__ inline void QuantProcess(uint32_t expertIndex);
    __aicore__ inline void SetStatus();
    __aicore__ inline void QuantInit(GM_ADDR scales);
    __aicore__ inline void SplitToCore(uint32_t curSendCnt_, uint32_t curUseAivNum, uint32_t &startTokenId_,
                                       uint32_t &endTokenId_, uint32_t &sendTokenNum_, bool isFront = true);
    __aicore__ inline void CalTokenSendExpertCnt(uint32_t dstExpertId, int32_t calCnt, int32_t &curExpertCnt);
    __aicore__ inline GM_ADDR GetWindAddrByRankId(const int32_t rankId)
    {
        if (epRankId_ == rankId) {
            return (GM_ADDR)(hcclContext_->localWindowsIn) + winDataSizeOffset_;
        }
        return (GM_ADDR)(((HcclRankRelationResV2 *)(hcclContext_->remoteRes[rankId].nextDevicePtr))->windowsIn) +
               winDataSizeOffset_;
    }

    __aicore__ inline GM_ADDR GetWindStateAddrByRankId(const int32_t rankId)
    {
        // A区给512Byte
        if (rankId == epRankId_) {
            return (GM_ADDR)(hcclContext_->localWindowsExp) + dataState_ * WIN_STATE_OFFSET;
        }
        return (GM_ADDR)(((HcclRankRelationResV2 *)(hcclContext_->remoteRes[rankId].nextDevicePtr))->windowsExp) +
               dataState_ * WIN_STATE_OFFSET;
    }

    __aicore__ inline uint32_t MIN(uint32_t x, uint32_t y)
    {
        return (x < y) ? x : y;
    }

    __aicore__ inline int32_t ReduceSumWorkNeedSize(int32_t calCnt)
    {
        int typeSize = static_cast<int>(sizeof(int32_t));
        int32_t elementsPerBlock = 32 / typeSize;
        int32_t iter1OutputCount = calCnt;
        int32_t iter1AlignEnd = ((iter1OutputCount + elementsPerBlock - 1) / elementsPerBlock) * elementsPerBlock;
        return iter1AlignEnd;
    }

    TPipe *tpipe_{nullptr};
    GlobalTensor<XType> xGMTensor_;
    GlobalTensor<int32_t> expertIdsGMTensor_;
    GlobalTensor<float> scalesGMTensor_;
    GlobalTensor<YOutType> expandXOutGMTensor_;
    GlobalTensor<bool> xActiveMaskGMTensor_;
    GlobalTensor<float> windowInstatusFp32Tensor_;

    LocalTensor<YOutType> xTmpTensor_;
    LocalTensor<XType> xInTensor_;
    LocalTensor<YOutType> xOutTensor_;
    LocalTensor<int32_t> expertCountTensor_;
    LocalTensor<int32_t> expertIdsTensor_;
    LocalTensor<float> rowMaxTensor_;
    LocalTensor<int32_t> statusTensor_;
    LocalTensor<float> smoothScalesTensor_;
    LocalTensor<int32_t> dstExpIdTensor_;
    LocalTensor<int32_t> subExpIdTensor_;
    LocalTensor<float> workLocalTensor_;
    TBuf<> expertIdsBuf_;
    TBuf<> statusBuf_;
    TBuf<> rowMaxBuf_;
    TBuf<> expertCountBuf_;
    TBuf<> receiveDataCastFloatBuf_;
    TBuf<> smoothScalesBuf_;
    TBuf<> dstExpBuf_;
    TBuf<> subExpBuf_;
    TBuf<> workLocalBuf_;
    TBuf<> batchItemInfoBuf_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> xQueue_; // 非量化使用，量化场景接收也可使用
    TQue<QuePosition::VECIN, 1> xInQueue_;
    TQue<QuePosition::VECOUT, 1> xOutQueue_;
    TQue<QuePosition::VECIN, 1> smoothScalesInQueue_;

    GM_ADDR yOutGM_;
    GM_ADDR expandIdxOutGM_;
    GM_ADDR statusSpaceGm_;
    GM_ADDR windowGM_;
    GM_ADDR workspaceGM_;
    GM_ADDR commCmdInfoOutGM_;
    GM_ADDR statusDataSpaceGM_;
    __gm__ BatchWriteItem *sendInfo_;
    __gm__ BatchWriteItem *sharedSendInfo_;
    __gm__ BatchWriteItem *statusSendInfo_;

    DataCopyExtParams dataCopyParamsFloat_;

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
    uint32_t totalExpertNum_{0};
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
    uint32_t activeMaskBsCnt_{0};
    uint32_t axisBsAlignSize_{0};
    uint64_t winDataSizeOffset_{0};
    uint64_t expertPerSizeOnWin_{0};
    uint64_t recvWinBlockNum_; // 接收Win区块数
    bool isInputActiveMaskFlag_ = false;
    uint32_t stateOffset_{0};
    uint32_t recStatusNumPerCore_{0};
    int32_t expertIdsCnt_{0};
    int32_t expertIdsActCnt_{0};
    uint32_t rscvStatusNum_{0};
    uint32_t remainderRankNum_{0};
    uint32_t startStatusIndex_{0};
    uint32_t axisHCommu{0};
    uint32_t sdmaUsedStreamPerCore_{0};
    uint32_t startTokenId_{0};
    uint32_t endTokenId_{0};
    uint32_t sendTokenNum_{0};
    uint32_t curSendCnt_{0};
    bool isSendShared_{false};
    Hccl<HCCL_SERVER_TYPE_AICPU> hccl;
    __gm__ HcclOpResParam *hcclContext_ = nullptr;
    DataCopyExtParams expandXCopyParams_;
    DataCopyExtParams xCopyParams_;
    DataCopyExtParams hCommuCopyOutParams_;
};

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeDispatchSetup<TemplateMC2TypeFunc>::Init(
    GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR xActiveMask, GM_ADDR YOut, GM_ADDR expandIdxOut,
    GM_ADDR commCmdInfoOut, GM_ADDR workspaceGM, TPipe *pipe, const MoeDistributeDispatchSetupTilingData *tilingData,
    __gm__ void *mc2InitTiling, __gm__ void *mc2CcTiling)
{
    tpipe_ = pipe;
    workspaceGM_ = workspaceGM;
    aivId_ = GetBlockIdx();
    aivNum_ = tilingData->moeDistributeDispatchSetupInfo.aivNum;
    if (aivId_ >= aivNum_) {
        return;
    }
    epRankId_ = tilingData->moeDistributeDispatchSetupInfo.epRankId;
    hcclContext_ = (__gm__ HcclOpResParam *)AscendC::GetHcclContext<HCCL_GROUP_ID_0>();

    hccl.Init((GM_ADDR)hcclContext_, mc2InitTiling);
    hccl.SetCcTiling(mc2CcTiling);
    GlobalTensor<int32_t> selfDataStatusTensor;
    statusDataSpaceGM_ = (GM_ADDR)(hcclContext_->localWindowsExp);
    selfDataStatusTensor.SetGlobalBuffer(
        (__gm__ int32_t *)(statusDataSpaceGM_ + STATE_WIN_OFFSET + aivId_ * WIN_ADDR_ALIGN));
    axisBS_ = tilingData->moeDistributeDispatchSetupInfo.bs;
    axisH_ = tilingData->moeDistributeDispatchSetupInfo.h;
    epWorldSize_ = tilingData->moeDistributeDispatchSetupInfo.epWorldSize;
    axisMaxBS_ = tilingData->moeDistributeDispatchSetupInfo.globalBs / epWorldSize_;
    moeExpertNum_ = tilingData->moeDistributeDispatchSetupInfo.moeExpertNum;
    sharedExpertNum_ = tilingData->moeDistributeDispatchSetupInfo.sharedExpertNum;
    sharedExpertRankNum_ = tilingData->moeDistributeDispatchSetupInfo.sharedExpertRankNum;
    if (sharedExpertNum_ > 0) {
        rankNumPerSharedExpert_ = sharedExpertRankNum_ / sharedExpertNum_;
    }
    moeExpertRankNum_ = epWorldSize_ - sharedExpertRankNum_;
    moeExpertNumPerRank_ = moeExpertNum_ / moeExpertRankNum_;
    isInputActiveMaskFlag_ = tilingData->moeDistributeDispatchSetupInfo.isActiveMask;
    axisK_ = tilingData->moeDistributeDispatchSetupInfo.k;
    sdmaUsedStreamPerCore_ = tilingData->moeDistributeDispatchSetupInfo.sdmaUsedStreamPerCore;

    xGMTensor_.SetGlobalBuffer((__gm__ XType *)x);
    xActiveMaskGMTensor_.SetGlobalBuffer((__gm__ bool *)xActiveMask);
    expertIdsGMTensor_.SetGlobalBuffer((__gm__ int32_t *)expertIds);
    expandXOutGMTensor_.SetGlobalBuffer((__gm__ YOutType *)YOut);
    yOutGM_ = YOut;
    expandIdxOutGM_ = expandIdxOut;
    commCmdInfoOutGM_ = commCmdInfoOut;

    hOutSize_ = axisH_ * sizeof(YOutType);
    hOutSizeAlign_ = Ceil(hOutSize_, UB_ALIGN) * UB_ALIGN; // scale起始放置偏移
    uint32_t hScaleSizeAlign = hOutSizeAlign_ + UB_ALIGN;  // 实际搬运大小，搬运token_align32B + 32B(float)
    hAlignWinSize_ = Ceil(hScaleSizeAlign, WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN; // win区token起始地址对齐512
    hAlignWinCnt_ = hAlignWinSize_ / sizeof(YOutType);
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
    stateOffset_ = ((recvWinBlockNum_ > 512) ? (STATE_OFFSET / 2) : STATE_OFFSET);
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(selfDataStatusTensor);
    dataState_ = selfDataStatusTensor(0);
    rscvStatusNum_ = recvWinBlockNum_;
    recStatusNumPerCore_ = rscvStatusNum_ / aivNum_; // 每个aiv需要处理的专家数
    remainderRankNum_ = rscvStatusNum_ % aivNum_;
    startStatusIndex_ = recStatusNumPerCore_ * aivId_; // + sharedExpertRankNum_, 每个aiv发送的
    if (aivId_ < remainderRankNum_) {                  // 前remainderRankNum个aiv需要多发1个卡的数据
        recStatusNumPerCore_ += 1;
        startStatusIndex_ += aivId_;
    } else {
        startStatusIndex_ += remainderRankNum_;
    }
    uint32_t statusBufCntAlign = Ceil(recvWinBlockNum_, 8) * 8; // 8 = UB_ALIGN / sizeof(int32_t)
    tpipe_->InitBuffer(statusBuf_, statusBufCntAlign * UB_ALIGN);
    statusTensor_ = statusBuf_.Get<int32_t>(); // 保存发送数据量及flag，同时用于计算windows中的偏移
    Duplicate<int32_t>(statusTensor_, 0, recvWinBlockNum_ * 8); // 8 = UB_ALIGN / sizeof(int32_t)
    statusSpaceGm_ = GetWindStateAddrByRankId(epRankId_);
    uint64_t mask[2] = {0x101010101010101, 0}; // 一次性操作256字节，也是64个int32_t，每8个数将首个设置为0x3F800000
    PipeBarrier<PIPE_V>();
    Duplicate<int32_t>(statusTensor_, 0x3F800000, mask, statusBufCntAlign / 8, 1, 8); // 0x3F800000是float的1

    // 当前win区划分为前后区，dispatch/combine不再划分区域
    winDataSizeOffset_ = dataState_ * (tilingData->moeDistributeDispatchSetupInfo.totalWinSize / 2);
    windowGM_ = GetWindAddrByRankId(epRankId_);
    windowInstatusFp32Tensor_.SetGlobalBuffer((__gm__ float *)(statusSpaceGm_));
    hOutAlignUbSize_ = Ceil(hScaleSizeAlign, UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(xQueue_, BUFFER_NUM, hOutAlignUbSize_); // 7k*2 + 32 + 6
    expertIdsCnt_ = axisBS_ * axisK_;
    if constexpr (DynamicQuant || StaticQuant) {
        QuantInit(scales);
        dstExpBuf_ = receiveDataCastFloatBuf_; // 内存复用
        subExpBuf_ = smoothScalesBuf_;         // 内存复用
    } else {
        uint32_t expertIdxSize = expertIdsCnt_ * sizeof(int32_t);
        tpipe_->InitBuffer(dstExpBuf_, expertIdxSize); // BS * K * 4 = 32K
        tpipe_->InitBuffer(subExpBuf_, expertIdxSize); // BS * K * 4 = 32K
    }
    dstExpIdTensor_ = dstExpBuf_.Get<int32_t>();
    subExpIdTensor_ = subExpBuf_.Get<int32_t>();

    uint32_t expertIdsSize = axisBS_ * axisK_ * sizeof(int32_t); // 约束32对齐
    tpipe_->InitBuffer(expertIdsBuf_, expertIdsSize);
    expertIdsTensor_ = expertIdsBuf_.Get<int32_t>();
    tpipe_->InitBuffer(expertCountBuf_, expertIdsSize); // BS * K * 4
    expertCountTensor_ = expertCountBuf_.Get<int32_t>();
    // reduceSum计算所需的Tensor空间，取最大统一前面申请
    int32_t reduceSumWorkNeedSize = ReduceSumWorkNeedSize(expertIdsCnt_);
    tpipe_->InitBuffer(workLocalBuf_, reduceSumWorkNeedSize * sizeof(int32_t));
    workLocalTensor_ = workLocalBuf_.Get<float>();

    axisHCommu = hScaleSizeAlign / sizeof(YOutType); // 有效搬运长度
    xCopyParams_ = {1U, static_cast<uint32_t>(axisH_ * sizeof(XType)), 0U, 0U, 0U};
    hCommuCopyOutParams_ = {1U, static_cast<uint32_t>(axisHCommu * sizeof(YOutType)), 0U, 0U, 0U};
    expandXCopyParams_ = {1U, static_cast<uint32_t>(axisH_ * sizeof(YOutType)), 0U, 0U, 0U};
    sendInfo_ = reinterpret_cast<__gm__ BatchWriteItem *>(commCmdInfoOutGM_);
    sharedSendInfo_ =
        reinterpret_cast<__gm__ BatchWriteItem *>(commCmdInfoOutGM_ + sizeof(BatchWriteItem) * axisBS_ * axisK_);
    statusSendInfo_ = sharedSendInfo_ + axisBS_ * axisK_;
    activeMaskBsCnt_ = axisBS_;
    if (isInputActiveMaskFlag_) {
        ActiveMaskCalCnt();
    }

    // 发送数据分核
    isSendShared_ = (aivId_ >= moeUsedAivNum_) && (sharedExpertRankNum_ != 0);
    if (isSendShared_) {
        curSendCnt_ = activeMaskBsCnt_ * sharedExpertNum_;
        SplitToCore(curSendCnt_, sharedUsedAivNum_, startTokenId_, endTokenId_, sendTokenNum_, false);
    } else {
        curSendCnt_ = activeMaskBsCnt_ * axisK_;
        SplitToCore(curSendCnt_, moeUsedAivNum_, startTokenId_, endTokenId_, sendTokenNum_);
    }
    // 发送状态分核
    totalExpertNum_ = sharedExpertRankNum_ + moeExpertNum_;
    SplitToCore(totalExpertNum_, aivNum_, startExpertId_, endExpertId_, sendExpertNum_);
    int32_t maxUseItemNums = ((sendExpertNum_ >= sendTokenNum_) ? sendExpertNum_ : sendTokenNum_);
    if (maxUseItemNums > 0) {
        tpipe_->InitBuffer(batchItemInfoBuf_, maxUseItemNums * sizeof(BatchWriteItem));
        LocalTensor<uint32_t> batchItemTensor = batchItemInfoBuf_.Get<uint32_t>();
        Duplicate<uint32_t>(batchItemTensor, 0, maxUseItemNums * sizeof(BatchWriteItem) / sizeof(uint32_t));
    }
    PipeBarrier<PIPE_ALL>();
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeDispatchSetup<TemplateMC2TypeFunc>::QuantInit(GM_ADDR scales)
{
    uint32_t hAlignSize = Ceil(axisH_ * sizeof(XType), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(xInQueue_, BUFFER_NUM, hAlignSize);        // 14K * 2
    tpipe_->InitBuffer(xOutQueue_, BUFFER_NUM, hOutAlignUbSize_); // 7K * 2 + 32 + 6
    scalesGMTensor_.SetGlobalBuffer((__gm__ float *)scales);
    if constexpr (DynamicQuant) {
        tpipe_->InitBuffer(rowMaxBuf_, UB_ALIGN); // 32B
    }
    uint32_t hFp32Size = axisH_ * sizeof(float);
    tpipe_->InitBuffer(receiveDataCastFloatBuf_, hFp32Size);
    tpipe_->InitBuffer(smoothScalesBuf_, hFp32Size);
    tpipe_->InitBuffer(smoothScalesInQueue_, BUFFER_NUM, hFp32Size);
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeDispatchSetup<TemplateMC2TypeFunc>::SendToSharedExpert()
{
    if (startTokenId_ >= curSendCnt_) {
        return;
    }
    uint32_t idInSharedGroup = epRankId_ % rankNumPerSharedExpert_; // 计算目的共享专家卡在其所在共享专家组的id

    GlobalTensor<YOutType> yOutGMTensor;
    GlobalTensor<YOutType> dstWinGMTensor;
    LocalTensor<uint32_t> batchItemInfo = batchItemInfoBuf_.Get<uint32_t>();
    LocalTensor<uint64_t> batchItemInfoU64 = batchItemInfoBuf_.Get<uint64_t>();
    uint32_t itemSizeInt32Cnt = sizeof(BatchWriteItem) / sizeof(uint32_t);
    uint32_t itemSizeInt64Cnt = sizeof(BatchWriteItem) / sizeof(uint64_t);
    DataCopyPadExtParams<XType> copyPadExtParams{false, 0U, 0U, 0U};
    DataCopyExtParams batchItemCopyOutParams{1U, sizeof(BatchWriteItem), 0U, 0U, 0U};
    for (uint32_t virtualTokenIndex = startTokenId_; virtualTokenIndex < endTokenId_; ++virtualTokenIndex) {
        uint32_t toSharedExpertIndex = virtualTokenIndex / activeMaskBsCnt_;
        uint32_t toRankId = idInSharedGroup + toSharedExpertIndex * rankNumPerSharedExpert_;
        yOutGMTensor.SetGlobalBuffer((__gm__ YOutType *)(yOutGM_ + axisBS_ * axisK_ * hAlignWinSize_));
        dstWinGMTensor.SetGlobalBuffer(
            (__gm__ YOutType *)(GetWindAddrByRankId(toRankId) + expertPerSizeOnWin_ * epRankId_));
        uint32_t tokenIndex = virtualTokenIndex % activeMaskBsCnt_;

        if constexpr (DynamicQuant || StaticQuant) {
            xInTensor_ = xInQueue_.AllocTensor<XType>();
            DataCopyPad(xInTensor_, xGMTensor_[tokenIndex * axisH_], xCopyParams_, copyPadExtParams);
            xInQueue_.EnQue(xInTensor_);
            xInTensor_ = xInQueue_.DeQue<XType>();
            xOutTensor_ = xOutQueue_.AllocTensor<YOutType>();
            QuantProcess(toSharedExpertIndex);
            xOutQueue_.EnQue(xOutTensor_);
            xOutTensor_ = xOutQueue_.DeQue<YOutType>();
            DataCopyPad(yOutGMTensor[virtualTokenIndex * hAlignWinCnt_], xOutTensor_, hCommuCopyOutParams_);
            xOutQueue_.FreeTensor(xOutTensor_);
        } else {
            xTmpTensor_ = xQueue_.AllocTensor<YOutType>();
            DataCopyPad(xTmpTensor_, xGMTensor_[tokenIndex * axisH_], expandXCopyParams_, copyPadExtParams);
            xQueue_.EnQue(xTmpTensor_);
            xTmpTensor_ = xQueue_.DeQue<YOutType>();
            DataCopyPad(yOutGMTensor[virtualTokenIndex * hAlignWinCnt_], xTmpTensor_, hCommuCopyOutParams_);
            xQueue_.FreeTensor<YOutType>(xTmpTensor_);
        }

        GlobalTensor<uint32_t> itemTensor;
        itemTensor.SetGlobalBuffer((__gm__ uint32_t *)(sharedSendInfo_ + virtualTokenIndex));
        uint32_t index = virtualTokenIndex - startTokenId_;
        if constexpr (DynamicQuant || StaticQuant) {
            batchItemInfoU64(index * itemSizeInt64Cnt) = static_cast<uint64_t>(HcclDataType::HCCL_DATA_TYPE_INT8);
        } else {
            batchItemInfoU64(index * itemSizeInt64Cnt) = static_cast<uint64_t>(HcclDataType::HCCL_DATA_TYPE_FP16);
        }
        batchItemInfo(index * itemSizeInt32Cnt + 7) = axisHCommu;
        batchItemInfoU64(index * itemSizeInt64Cnt + 4) = reinterpret_cast<uint64_t>(
            yOutGM_ + axisBS_ * axisK_ * hAlignWinSize_ + virtualTokenIndex * hAlignWinSize_);
        batchItemInfoU64(index * itemSizeInt64Cnt + 5) =
            reinterpret_cast<uint64_t>(dstWinGMTensor.GetPhyAddr() + tokenIndex * hAlignWinSize_);
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        DataCopyPad(itemTensor, batchItemInfo[index * itemSizeInt32Cnt], batchItemCopyOutParams);
    }
    PipeBarrier<PIPE_ALL>();
    // 一个核4个stream
    int32_t length = sendTokenNum_ / sdmaUsedStreamPerCore_;
    int32_t remain = sendTokenNum_ % sdmaUsedStreamPerCore_;
    int32_t curOffset = 0;
    for (uint32_t i = 0; i < sdmaUsedStreamPerCore_; i++) {
        int newLength = length + (i < remain ? 1 : 0);
        if (newLength <= 0) {
            continue;
        }
        curOffset = startTokenId_ + length * i + (i < remain ? i : remain);
        auto handleId =
            hccl.BatchWrite<true>(reinterpret_cast<__gm__ uint8_t *>(sharedSendInfo_ + curOffset), newLength, i);
    }
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeDispatchSetup<TemplateMC2TypeFunc>::CalTokenSendExpertCnt(uint32_t dstExpertId,
                                                                                              int32_t calCnt,
                                                                                              int32_t &curExpertCnt)
{
    Duplicate<int32_t>(dstExpIdTensor_, dstExpertId, calCnt);
    PipeBarrier<PIPE_V>();
    Sub(subExpIdTensor_, expertIdsTensor_, dstExpIdTensor_, calCnt);
    PipeBarrier<PIPE_V>();
    LocalTensor<float> tmpFp32 = subExpIdTensor_.ReinterpretCast<float>();
    LocalTensor<float> tmpoutFp32 = dstExpIdTensor_.ReinterpretCast<float>();
    Abs(tmpoutFp32, tmpFp32, calCnt);
    PipeBarrier<PIPE_V>();
    Mins(subExpIdTensor_, dstExpIdTensor_, 1, calCnt);
    PipeBarrier<PIPE_V>();
    ReduceSum<float>(tmpoutFp32, tmpFp32, workLocalTensor_, calCnt);
    SyncFunc<AscendC::HardEvent::V_S>();
    int32_t curOtherExpertCnt = dstExpIdTensor_(0);
    if (calCnt > curOtherExpertCnt) {
        curExpertCnt = calCnt - curOtherExpertCnt;
    }
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeDispatchSetup<TemplateMC2TypeFunc>::SplitToCore(
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
__aicore__ inline void MoeDistributeDispatchSetup<TemplateMC2TypeFunc>::SendToMoeExpert()
{
    if (startTokenId_ >= curSendCnt_) {
        return;
    }
    GlobalTensor<YOutType> dstWinGMTensor;
    GlobalTensor<YOutType> yOutGMTensor;
    LocalTensor<uint32_t> batchItemInfo = batchItemInfoBuf_.Get<uint32_t>();
    LocalTensor<uint64_t> batchItemInfoU64 = batchItemInfoBuf_.Get<uint64_t>();
    uint32_t itemSizeInt32Cnt = sizeof(BatchWriteItem) / sizeof(uint32_t);
    uint32_t itemSizeInt64Cnt = sizeof(BatchWriteItem) / sizeof(uint64_t);
    DataCopyPadExtParams<XType> copyPadExtParams{false, 0U, 0U, 0U};
    DataCopyExtParams batchItemCopyOutParams{1U, sizeof(BatchWriteItem), 0U, 0U, 0U};
    for (int32_t tokenIndex = startTokenId_; tokenIndex < endTokenId_; ++tokenIndex) {
        uint32_t dstExpertId = expertIdsTensor_(tokenIndex);
        int32_t curExpertCnt = 0;
        if (likely(tokenIndex > 0)) {
            CalTokenSendExpertCnt(dstExpertId, tokenIndex, curExpertCnt);
        }
        expertCountTensor_(tokenIndex - startTokenId_) = curExpertCnt;
        uint32_t tempRankId = dstExpertId / moeExpertNumPerRank_ + sharedExpertRankNum_;
        GM_ADDR rankGM = (__gm__ uint8_t *)(GetWindAddrByRankId(tempRankId) +
                                            (expertPerSizeOnWin_ *
                                             (epRankId_ * moeExpertNumPerRank_ + dstExpertId % moeExpertNumPerRank_)) +
                                            hAlignWinSize_ * curExpertCnt); // 计算地址偏移
        dstWinGMTensor.SetGlobalBuffer((__gm__ YOutType *)rankGM);
        yOutGMTensor.SetGlobalBuffer((__gm__ YOutType *)(yOutGM_));
        if constexpr (DynamicQuant || StaticQuant) {
            xInTensor_ = xInQueue_.AllocTensor<XType>();
            DataCopyPad(xInTensor_, xGMTensor_[tokenIndex / axisK_ * axisH_], xCopyParams_, copyPadExtParams);
            xInQueue_.EnQue(xInTensor_);
            xInTensor_ = xInQueue_.DeQue<XType>();
            xOutTensor_ = xOutQueue_.AllocTensor<YOutType>();
            QuantProcess(dstExpertId + sharedExpertNum_);
            xOutQueue_.EnQue(xOutTensor_);
            xOutTensor_ = xOutQueue_.DeQue<YOutType>();
            DataCopyPad(yOutGMTensor[tokenIndex * hAlignWinCnt_], xOutTensor_, hCommuCopyOutParams_);
            xOutQueue_.FreeTensor(xOutTensor_);
        } else {
            xTmpTensor_ = xQueue_.AllocTensor<YOutType>();
            DataCopyPad(xTmpTensor_, xGMTensor_[tokenIndex / axisK_ * axisH_], xCopyParams_, copyPadExtParams);
            xQueue_.EnQue(xTmpTensor_);
            xTmpTensor_ = xQueue_.DeQue<YOutType>();
            DataCopyPad(yOutGMTensor[tokenIndex * hAlignWinCnt_], xTmpTensor_, hCommuCopyOutParams_);
            xQueue_.FreeTensor<YOutType>(xTmpTensor_);
        }

        GlobalTensor<uint32_t> itemTensor;
        itemTensor.SetGlobalBuffer((__gm__ uint32_t *)(sendInfo_ + tokenIndex));
        uint32_t index = tokenIndex - startTokenId_;
        if constexpr (DynamicQuant || StaticQuant) {
            batchItemInfoU64(index * itemSizeInt64Cnt) = static_cast<uint64_t>(HcclDataType::HCCL_DATA_TYPE_INT8);
        } else {
            batchItemInfoU64(index * itemSizeInt64Cnt) = static_cast<uint64_t>(HcclDataType::HCCL_DATA_TYPE_FP16);
        }
        batchItemInfo(index * itemSizeInt32Cnt + 7) = axisHCommu;
        batchItemInfoU64(index * itemSizeInt64Cnt + 4) =
            reinterpret_cast<uint64_t>(yOutGM_ + tokenIndex * hAlignWinSize_);
        batchItemInfoU64(index * itemSizeInt64Cnt + 5) = reinterpret_cast<uint64_t>(rankGM);
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        DataCopyPad(itemTensor, batchItemInfo[index * itemSizeInt32Cnt], batchItemCopyOutParams);
    }
    PipeBarrier<PIPE_ALL>();
    // 一个核4个stream
    int32_t length = sendTokenNum_ / sdmaUsedStreamPerCore_;
    int32_t remain = sendTokenNum_ % sdmaUsedStreamPerCore_;
    int32_t curOffset = 0;
    for (uint32_t i = 0; i < sdmaUsedStreamPerCore_; i++) {
        int newLength = length + (i < remain ? 1 : 0);
        if (newLength <= 0) {
            continue;
        }
        curOffset = startTokenId_ + length * i + (i < remain ? i : remain);
        auto handleId = hccl.BatchWrite<true>(reinterpret_cast<__gm__ uint8_t *>(sendInfo_ + curOffset), newLength, i);
    }
    GlobalTensor<int32_t> expandIdxGMTensor;
    expandIdxGMTensor.SetGlobalBuffer((__gm__ int32_t *)(expandIdxOutGM_ + startTokenId_ * sizeof(int32_t)));
    DataCopyExtParams expertIdsCntParams = {1U, static_cast<uint32_t>(sendTokenNum_ * sizeof(uint32_t)), 0U, 0U, 0U};
    DataCopyPad(expandIdxGMTensor, expertCountTensor_, expertIdsCntParams);
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeDispatchSetup<TemplateMC2TypeFunc>::ActiveMaskCalCnt()
{
    // 搬运x_active_mask, 当前仅用于计算有效token总数
    LocalTensor<half> tempTensor;
    LocalTensor<half> sumOutTensor;
    LocalTensor<bool> xActiveMaskTensor;
    axisBsAlignSize_ = Ceil(axisBS_ * sizeof(bool), UB_ALIGN) * UB_ALIGN;
    xActiveMaskTensor = dstExpBuf_.Get<bool>();
    tempTensor = subExpBuf_.Get<half>();
    sumOutTensor = expertIdsBuf_.Get<half>();
    DataCopyExtParams xActiveMaskParams = {1U, static_cast<uint32_t>(axisBS_ * sizeof(bool)), 0U, 0U, 0U};
    DataCopyPadExtParams<bool> xActiveMaskCopyPadParams{false, 0U, 0U, 0U};
    DataCopyPad(xActiveMaskTensor, xActiveMaskGMTensor_, xActiveMaskParams, xActiveMaskCopyPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    LocalTensor<int8_t> xActiveMaskInt8Tensor = xActiveMaskTensor.ReinterpretCast<int8_t>();
    Cast(tempTensor, xActiveMaskInt8Tensor, RoundMode::CAST_NONE, axisBS_);
    PipeBarrier<PIPE_V>();
    SumParams params{1, axisBsAlignSize_, axisBS_};
    Sum(sumOutTensor, tempTensor, params);
    SyncFunc<AscendC::HardEvent::V_S>();
    activeMaskBsCnt_ = static_cast<int32_t>(sumOutTensor.GetValue(0));
}

/*
共享专家卡：所有核用于给moe专家发送数据
moe专家卡：部分核用于给共享专家发送数据，部分核用于给moe专家发送数据
*/
template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeDispatchSetup<TemplateMC2TypeFunc>::AlltoAllDispatch()
{
    if (activeMaskBsCnt_ == 0) {
        return;
    }
    expertIdsActCnt_ = activeMaskBsCnt_ * axisK_;

    // 后面的核向共享专家发数据
    if (isSendShared_) {
        SendToSharedExpert();
    }

    DataCopyExtParams expertIdsCntParams = {1U, static_cast<uint32_t>(expertIdsCnt_ * sizeof(uint32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> expertIdsCntCopyPadParams{false, 0U, 0U, 0U};
    DataCopyPad(expertIdsTensor_, expertIdsGMTensor_, expertIdsCntParams, expertIdsCntCopyPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    if (isSendShared_) { // 用于send共享专家数据的核，也需要搬运expertIds，后续会重新分核写状态位置，该核可能用于写moe专家flag
        return;
    }
    SendToMoeExpert();
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeDispatchSetup<TemplateMC2TypeFunc>::SetStatus()
{
    if (activeMaskBsCnt_ > 0) { // 需要发送的再算
        for (uint32_t curExpertId = startExpertId_; curExpertId < endExpertId_; ++curExpertId) {
            int32_t curExpertCnt = 0;
            int32_t cntPosIndex =
                curExpertId * 8 + 1; // 一个block有8个int32的元素，第一个元素为flag位，第二个为发送token数
            if (curExpertId <
                sharedExpertRankNum_) { // 当前处理专家为共享专家 shared:Cnt -> win -> LocalCopy(moe+share)
                if (curExpertId % rankNumPerSharedExpert_ == epRankId_ % rankNumPerSharedExpert_) {
                    curExpertCnt = activeMaskBsCnt_;
                }
            } else { // 当前处理卡为moe专家卡
                int32_t curMoeExpertId = curExpertId - sharedExpertRankNum_;
                CalTokenSendExpertCnt(curMoeExpertId, expertIdsActCnt_, curExpertCnt);
            }
            statusTensor_(cntPosIndex) = curExpertCnt;
        }
    }
    PipeBarrier<PIPE_ALL>();

    if (startExpertId_ >= totalExpertNum_) {
        return;
    }
    GlobalTensor<int32_t> windowExpGMTensor;
    LocalTensor<uint32_t> batchItemInfo = batchItemInfoBuf_.Get<uint32_t>();
    LocalTensor<uint64_t> batchItemInfoU64 = batchItemInfoBuf_.Get<uint64_t>();
    uint32_t itemSizeInt32Cnt = sizeof(BatchWriteItem) / sizeof(uint32_t);
    uint32_t itemSizeInt64Cnt = sizeof(BatchWriteItem) / sizeof(uint64_t);
    DataCopyExtParams batchItemCopyOutParams{1U, sizeof(BatchWriteItem), 0U, 0U, 0U};
    uint32_t offset = stateOffset_ * epRankId_;
    // 状态区拷贝到windowOut
    windowExpGMTensor.SetGlobalBuffer((__gm__ int32_t *)(statusDataSpaceGM_ + 2 * WIN_STATE_OFFSET +
                                                         axisBS_ * axisK_ * hAlignWinSize_ +
                                                         startExpertId_ * 8 * sizeof(int32_t)));
    DataCopy<int32_t>(windowExpGMTensor, statusTensor_[startExpertId_ * 8],
                      sendExpertNum_ * 8); // 8时数据大小，按32对齐拷贝
    PipeBarrier<PIPE_ALL>();
    for (uint32_t rankIndex = startExpertId_; rankIndex < endExpertId_; ++rankIndex) {
        uint32_t dstRankId = rankIndex;
        uint32_t index = rankIndex - startExpertId_;
        if (moeExpertNumPerRank_ > 1 && (rankIndex >= sharedExpertRankNum_)) {
            dstRankId = ((rankIndex - sharedExpertRankNum_) / moeExpertNumPerRank_ + sharedExpertRankNum_);
            offset =
                (epRankId_ + (rankIndex - sharedExpertRankNum_) % moeExpertNumPerRank_ * epWorldSize_) * stateOffset_;
        }
        GlobalTensor<uint32_t> itemTensor;
        itemTensor.SetGlobalBuffer((__gm__ uint32_t *)(statusSendInfo_ + rankIndex));
        GM_ADDR rankGM = (__gm__ uint8_t *)(GetWindStateAddrByRankId(dstRankId) + offset); // 计算地址偏移
#if defined(ASCENDC_OOM) && ASCENDC_OOM == 1
        OOMCheckAddrRange<int32_t>((__gm__ int32_t *)(GetWindStateAddrByRankId(dstRankId)), STATE_SIZE);
#endif
        batchItemInfoU64(index * itemSizeInt64Cnt) = static_cast<uint64_t>(HcclDataType::HCCL_DATA_TYPE_INT32);
        batchItemInfo(index * itemSizeInt32Cnt + 7) = 8;
        batchItemInfoU64(index * itemSizeInt64Cnt + 4) =
            reinterpret_cast<uint64_t>(statusDataSpaceGM_ + 2 * WIN_STATE_OFFSET + axisBS_ * axisK_ * hAlignWinSize_ +
                                       startExpertId_ * 8 * sizeof(uint32_t));
        batchItemInfoU64(index * itemSizeInt64Cnt + 5) = reinterpret_cast<uint64_t>(rankGM);
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        DataCopyPad(itemTensor, batchItemInfo[index * itemSizeInt32Cnt], batchItemCopyOutParams);
    }
    PipeBarrier<PIPE_ALL>();
    // 一个核4个stream
    int32_t length = sendExpertNum_ / sdmaUsedStreamPerCore_;
    int32_t remain = sendExpertNum_ % sdmaUsedStreamPerCore_;
    int32_t curOffset = 0;
    for (uint32_t i = 0; i < sdmaUsedStreamPerCore_; i++) {
        int newLength = length + (i < remain ? 1 : 0);
        if (newLength <= 0) {
            continue;
        }
        curOffset = startExpertId_ + length * i + (i < remain ? i : remain);
        auto handleId =
            hccl.BatchWrite<true>(reinterpret_cast<__gm__ uint8_t *>(statusSendInfo_ + curOffset), newLength, i);
    }
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeDispatchSetup<TemplateMC2TypeFunc>::ReduceMaxInplace(
    const LocalTensor<float> &srcLocal, uint32_t count)
{
    uint64_t repsFp32 = count >> 6;       // 6 is count / elemPerRefFp32
    uint64_t offsetsFp32 = repsFp32 << 6; // 6 is repsFp32 * elemPerRefFp32
    uint64_t remsFp32 = count & 0x3f;     // 0x3f 63, count % elemPerRefFp32
    const uint64_t elemPerRefFp32 = 64UL; // 256 bit / sizeof(float)
    if (likely(repsFp32 > 1)) {
        // 8 is rep stride
        Max(srcLocal, srcLocal[elemPerRefFp32], srcLocal, elemPerRefFp32, repsFp32 - 1, {1, 1, 1, 0, 8, 0});
        PipeBarrier<PIPE_V>();
    }
    if (unlikely(remsFp32 > 0) && unlikely(offsetsFp32 > 0)) {
        Max(srcLocal, srcLocal[offsetsFp32], srcLocal, remsFp32, 1, {1, 1, 1, 0, 8, 0});
        PipeBarrier<PIPE_V>();
    }
    uint32_t mask = (repsFp32 > 0) ? elemPerRefFp32 : count;
    // 8 is rep stride
    WholeReduceMax(srcLocal, srcLocal, mask, 1, 8, 1, 8);
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeDispatchSetup<TemplateMC2TypeFunc>::QuantProcess(uint32_t expertIndex)
{
    float dynamicScale = 0.0;
    LocalTensor<float> floatLocalTemp;
    floatLocalTemp = receiveDataCastFloatBuf_.Get<float>();

    Cast(floatLocalTemp, xInTensor_, RoundMode::CAST_NONE, axisH_);
    xInQueue_.FreeTensor<XType>(xInTensor_);
    PipeBarrier<PIPE_V>();
    if constexpr (IsSmoothScaleExist) {
        DataCopyExtParams scalesCopyInParams{1U, static_cast<uint32_t>(axisH_ * sizeof(float)), 0U, 0U, 0U};
        DataCopyPadExtParams<float> copyPadExtParams{false, 0U, 0U, 0U};
        smoothScalesTensor_ = smoothScalesInQueue_.AllocTensor<float>();
        DataCopyPad(smoothScalesTensor_, scalesGMTensor_[expertIndex * axisH_], scalesCopyInParams, copyPadExtParams);
        smoothScalesInQueue_.EnQue(smoothScalesTensor_);
        smoothScalesTensor_ = smoothScalesInQueue_.DeQue<float>();
        Mul(floatLocalTemp, floatLocalTemp, smoothScalesTensor_, axisH_);
        PipeBarrier<PIPE_V>();
        smoothScalesInQueue_.FreeTensor(smoothScalesTensor_);
    }

    if constexpr (DynamicQuant) {
        LocalTensor<float> floatLocalAbsTemp = smoothScalesBuf_.Get<float>();
        rowMaxTensor_ = rowMaxBuf_.Get<float>();

        Abs(floatLocalAbsTemp, floatLocalTemp, axisH_);
        PipeBarrier<PIPE_V>();
        ReduceMaxInplace(floatLocalAbsTemp, axisH_);

        SyncFunc<AscendC::HardEvent::V_S>();
        dynamicScale = float(127.0) / floatLocalAbsTemp.GetValue(0);
        SyncFunc<AscendC::HardEvent::S_V>();
        Muls(floatLocalTemp, floatLocalTemp, dynamicScale, axisH_);
        PipeBarrier<PIPE_V>();
    }
    LocalTensor<half> halfLocalTemp = floatLocalTemp.ReinterpretCast<half>();
    LocalTensor<int32_t> int32LocalTemp = floatLocalTemp.ReinterpretCast<int32_t>();
    Cast(int32LocalTemp, floatLocalTemp, RoundMode::CAST_RINT, axisH_);
    PipeBarrier<PIPE_V>();
    SetDeqScale((half)1.000000e+00f);
    PipeBarrier<PIPE_V>();

    Cast(halfLocalTemp, int32LocalTemp, RoundMode::CAST_ROUND, axisH_);

    PipeBarrier<PIPE_V>();
    Cast(xOutTensor_, halfLocalTemp, RoundMode::CAST_TRUNC, axisH_);

    floatLocalTemp = xOutTensor_.template ReinterpretCast<float>();
    floatLocalTemp.SetValue(hOutSizeAlign_ / sizeof(float), float(1.0) / dynamicScale); // int8->float32
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeDispatchSetup<TemplateMC2TypeFunc>::Process()
{
    if ASCEND_IS_AIV { // 全aiv处理
        AlltoAllDispatch();
        for (int i = 0; i < sdmaUsedStreamPerCore_; i++) {
            hccl.QueueBarrier(i);
        }
        SetStatus();
        hccl.Finalize<false>();
        SyncAll<true>();
    }
}

} // namespace Mc2Kernel
#endif // MOE_DISTRIBUTE_DISPATCH_SETUP_H
