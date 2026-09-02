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
 * \file moe_distribute_combine_setup.h
 * \brief
 */
#ifndef MOE_DISTRIBUTE_COMBINE_SETUP_H
#define MOE_DISTRIBUTE_COMBINE_SETUP_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "kernel_tiling/kernel_tiling.h"
#if __has_include("../../common/moe_distribute_base.h")
#include "../../common/moe_distribute_base.h"
#else
#include "../../../common/op_kernel/moe_distribute_base.h"
#endif
#include "moe_distribute_combine_setup_tiling.h"

namespace MoeDistributeCombineSetupImpl {

template <AscendC::HardEvent event>
__aicore__ inline void SyncFunc()
{
    int32_t eventID = static_cast<int32_t>(GetTPipePtr()->FetchEventID(event));
    AscendC::SetFlag<event>(eventID);
    AscendC::WaitFlag<event>(eventID);
}

#define TemplateMC2TypeClass typename ExpandXType, typename ExpandIdxType
#define TemplateMC2TypeFunc ExpandXType, ExpandIdxType

using namespace AscendC;
template <TemplateMC2TypeClass>
class MoeDistributeCombineSetup {
    constexpr static uint8_t BUFFER_NUM = 2;              // 多buf
    constexpr static uint32_t STATE_OFFSET = 512U;        // 状态空间偏移地址
    constexpr static uint32_t STATE_SIZE = 1024U * 1024U; // 1M
    constexpr static uint32_t UB_ALIGN = 32U;             // UB按32字节对齐
    constexpr static uint64_t WIN_STATE_OFFSET = 350U * 1024U;
    constexpr static uint64_t STATE_WIN_OFFSET = 950U * 1024U;
    constexpr static uint32_t COMBINE_STATE_OFFSET = 0U; // 本卡状态空间偏移地址，前面的地址给dispatch用
    constexpr static uint64_t SDMA_STATUS_WIN_OFFSET = 1000UL * 1024U;
    constexpr static uint32_t LOCAL_STREAM_MAX_NUM = 40U;

public:
    __aicore__ inline MoeDistributeCombineSetup(){};
    __aicore__ inline void Init(GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR assistInfoForCombine, GM_ADDR quantExpandX,
                                GM_ADDR commCmdInfoOut, GM_ADDR workspaceGM, TPipe *pipe,
                                const MoeDistributeCombineSetupTilingData *tilingData, __gm__ void *mc2InitTiling,
                                __gm__ void *mc2CcTiling);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitStatusTargetSum();
    __aicore__ inline void SplitCoreCal();
    __aicore__ inline void AlltoAllBuffInit();
    __aicore__ inline void Dispatch();
    __aicore__ inline void BuffInit();
    __aicore__ inline void SetStatus();
    __aicore__ inline void ExpertAlltoAllDispatch(uint32_t tokenNumLoop, uint32_t srcStartTokenIdx, uint32_t ep,
                                                  uint32_t expertIdx);

    __aicore__ GM_ADDR GetWinAddrByRankId(const uint32_t rankId, const uint8_t expertLocalId = 0U)
    {
        return (GM_ADDR)((epRankId_ == rankId) ?
                             epWinContext_->localWindowsIn :
                             ((HcclRankRelationResV2 *)(epWinContext_->remoteRes[rankId].nextDevicePtr))->windowsIn) +
               winDataSizeOffset_ + expertLocalId * expertPerSizeOnWin_;
    }

    __aicore__ GM_ADDR GetLocalWinOutAddrByRankId(const uint8_t expertLocalId = 0U)
    {
        return (GM_ADDR)(epWinContext_->localWindowsOut + expertLocalId * expertPerSizeOnWin_);
    }

    __aicore__ GM_ADDR GetWinStateAddrByRankId(uint32_t rankId)
    {
        return (GM_ADDR)((epRankId_ == rankId) ?
                             epWinContext_->localWindowsExp :
                             ((HcclRankRelationResV2 *)(epWinContext_->remoteRes[rankId].nextDevicePtr))->windowsExp) +
               COMBINE_STATE_OFFSET + dataState_ * WIN_STATE_OFFSET;
    }

    TPipe *tpipe_{nullptr};
    GlobalTensor<ExpandIdxType> assistInfoForCombineGM_;
    GlobalTensor<int32_t> sdmaStatusGlobal_;
    GM_ADDR dataCmdInfoGM_;
    GM_ADDR epWindowGM_;
    GM_ADDR epStatusSpaceGm_;
    GM_ADDR stateGM_;
    GM_ADDR sdmaStateGM_;
    GM_ADDR expandXGM_;
    GM_ADDR commCmdInfoOutGM_;
    LocalTensor<ExpandIdxType> assistInfoForCombineLocal_;

    // tiling侧已确保数据上限， 相乘不会越界，因此统一采用uin32_t进行处理
    uint32_t axisMaxBS_{0};
    uint32_t axisH_{0};
    uint32_t axisA_{0};
    uint32_t aivNum_{0};
    uint32_t worldSize_{0};
    uint32_t epRankId_{0};
    uint32_t coreIdx_{0};             // aiv id
    uint32_t sharedExpertRankNum_{0}; // 共享专家卡数
    uint32_t moeExpertPerRankNum_{0}; // 每张卡部署的moe专家数
    uint32_t moeSendNum_{0};          // moeExpertPerRankNum_ * worldSize_
    __gm__ HcclOpResParam *epWinContext_{nullptr};
    uint32_t epDataOffsetOnWin_{0};
    uint32_t epStateOffsetOnWin_{0};
    uint32_t axisHExpandXTypeSize_{0};
    uint32_t startRankId_{0};
    uint32_t endRankId_{0};
    uint32_t sendRankNum_{0};
    uint32_t dataState_{0};
    uint32_t stateOffset_{0};
    uint64_t winDataSizeOffset_{0};
    uint64_t expertPerSizeOnWin_{0};
    uint64_t totalWinSize_{0};
    uint32_t queuePerCore_{0};
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> gmTpSendCountQueue_;
    TBuf<> readStateBuf_;
    TBuf<> sendCountBuf_;
    TBuf<> batchItemInfoBuf_;
    int32_t epStateValue_;
    bool isShardExpert_{false};
    uint32_t batchWriteInfoCnt_{0}; // 记录当前核使用的batchWriteInfo数量

    __gm__ HcclOpResParam *hcclContext_ = nullptr;
    Hccl<HCCL_SERVER_TYPE_AICPU> hccl_;
};

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineSetup<TemplateMC2TypeFunc>::Init(
    GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR assistInfoForCombine, GM_ADDR quantExpandX, GM_ADDR commCmdInfoOut,
    GM_ADDR workspaceGM, TPipe *pipe, const MoeDistributeCombineSetupTilingData *tilingData, __gm__ void *mc2InitTiling,
    __gm__ void *mc2CcTiling)
{
    tpipe_ = pipe;
    coreIdx_ = GetBlockIdx();
    epRankId_ = tilingData->moeDistributeCombineSetupInfo.epRankId;
    hcclContext_ = (__gm__ HcclOpResParam *)AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
    auto contextGM = AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
    epWinContext_ = (__gm__ HcclOpResParam *)contextGM;
    GlobalTensor<int32_t> selfDataStatusTensor;
    GM_ADDR statusDataSpaceGm = (GM_ADDR)epWinContext_->localWindowsExp;
    selfDataStatusTensor.SetGlobalBuffer((__gm__ int32_t *)(statusDataSpaceGm + STATE_WIN_OFFSET + coreIdx_ * 512));
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(selfDataStatusTensor);
    sdmaStateGM_ = statusDataSpaceGm + SDMA_STATUS_WIN_OFFSET;
    sdmaStatusGlobal_.SetGlobalBuffer((__gm__ int32_t *)sdmaStateGM_);
    InitStatusTargetSum();

    dataState_ = selfDataStatusTensor(0);
    commCmdInfoOutGM_ = commCmdInfoOut;
    expandXGM_ = expandX;
    assistInfoForCombineGM_.SetGlobalBuffer((__gm__ int32_t *)assistInfoForCombine);
    axisH_ = tilingData->moeDistributeCombineSetupInfo.h;
    axisA_ = tilingData->moeDistributeCombineSetupInfo.a;
    aivNum_ = tilingData->moeDistributeCombineSetupInfo.aivNum;
    sharedExpertRankNum_ = tilingData->moeDistributeCombineSetupInfo.sharedExpertRankNum;
    moeExpertPerRankNum_ = tilingData->moeDistributeCombineSetupInfo.moeExpertPerRankNum;
    worldSize_ = tilingData->moeDistributeCombineSetupInfo.epWorldSize;
    axisMaxBS_ = tilingData->moeDistributeCombineSetupInfo.globalBs / worldSize_;
    moeSendNum_ = worldSize_ * moeExpertPerRankNum_;
    totalWinSize_ = tilingData->moeDistributeCombineSetupInfo.totalWinSize;
    stateOffset_ = (moeSendNum_ > 512) ? (STATE_OFFSET / 2) : STATE_OFFSET;
    expertPerSizeOnWin_ =
        static_cast<uint64_t>(axisMaxBS_) * static_cast<uint64_t>(axisH_) * static_cast<uint64_t>(sizeof(ExpandXType));
    winDataSizeOffset_ =
        static_cast<uint64_t>(dataState_) * (tilingData->moeDistributeCombineSetupInfo.totalWinSize / 2UL);
    epWindowGM_ = GetWinAddrByRankId(epRankId_);
    epStatusSpaceGm_ = GetWinStateAddrByRankId(epRankId_);
#if defined(ASCENDC_OOM) && ASCENDC_OOM == 1
    OOMCheckAddrRange<ExpandXType>((__gm__ ExpandXType *)(epWindowGM_), totalWinSize_);
    OOMCheckAddrRange<float>((__gm__ float *)(epStatusSpaceGm_), STATE_SIZE);
#endif
    epDataOffsetOnWin_ = epRankId_ * moeExpertPerRankNum_ * static_cast<uint32_t>(expertPerSizeOnWin_);
    epStateOffsetOnWin_ = epRankId_ * stateOffset_;
    isShardExpert_ = (epRankId_ < sharedExpertRankNum_);
    axisHExpandXTypeSize_ = axisH_ * static_cast<uint32_t>(sizeof(ExpandXType));
    queuePerCore_ = LOCAL_STREAM_MAX_NUM / aivNum_;

    hccl_.Init((GM_ADDR)hcclContext_, mc2InitTiling);
    hccl_.SetCcTiling(mc2CcTiling);

    SplitCoreCal();
}

// 在1M中选择512K偏移后的1.5k空间记录本卡历史状态
// 在setup内进行状态切换，到teardown内时，state即目标flag
template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineSetup<TemplateMC2TypeFunc>::InitStatusTargetSum()
{
    epStateValue_ = 0x3F800000;
    sdmaStatusGlobal_.SetValue(0, epStateValue_);
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(sdmaStatusGlobal_);
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineSetup<TemplateMC2TypeFunc>::SplitCoreCal()
{
    // 对worldSize按卡分核，得到每个核上处理的卡的数量
    uint32_t coreIdxNew = (coreIdx_ + epRankId_) % aivNum_; // 按照卡去进行偏移
    sendRankNum_ = worldSize_ / aivNum_;
    uint32_t remainderRankNum = worldSize_ % aivNum_;
    startRankId_ = sendRankNum_ * coreIdxNew;
    if (coreIdxNew < remainderRankNum) {
        sendRankNum_++;
        startRankId_ += coreIdxNew;
    } else {
        startRankId_ += remainderRankNum;
    }
    endRankId_ = startRankId_ + sendRankNum_;
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineSetup<TemplateMC2TypeFunc>::BuffInit()
{
    tpipe_->Reset();
    tpipe_->InitBuffer(readStateBuf_, UB_ALIGN);                                      // 32
    uint32_t sendNumAlign = Ceil(moeSendNum_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN; // 32B向上取整
    tpipe_->InitBuffer(sendCountBuf_, sendNumAlign); // worldSize_ * moeExpertPerRankNum_ * 4，且32B向上取整
    tpipe_->InitBuffer(gmTpSendCountQueue_, BUFFER_NUM, axisHExpandXTypeSize_); // 28K
    assistInfoForCombineLocal_ = sendCountBuf_.Get<int32_t>();
    tpipe_->InitBuffer(batchItemInfoBuf_, 2 * sizeof(BatchWriteItem));
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineSetup<TemplateMC2TypeFunc>::AlltoAllBuffInit()
{
    tpipe_->Reset();
    tpipe_->InitBuffer(readStateBuf_, UB_ALIGN); // 32 * moeExpertPerRankNum_
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineSetup<TemplateMC2TypeFunc>::Dispatch()
{
    if (startRankId_ >= worldSize_) { // 空闲核，直接返回
        return;
    }
    uint32_t curRankExpertNum = 0U;
    DataCopyExtParams epSendCntParams;
    if (isShardExpert_) {
        curRankExpertNum = 1U; // 对于共享专家来说assistInfoForCombine输入维度为epWordSize个
        epSendCntParams = {1U, static_cast<uint32_t>(worldSize_ * sizeof(uint32_t)), 0U, 0U, 0U};
    } else {
        curRankExpertNum = moeExpertPerRankNum_;
        epSendCntParams = {1U, static_cast<uint32_t>(moeSendNum_ * sizeof(uint32_t)), 0U, 0U, 0U};
    }
    DataCopyPadExtParams<int32_t> copyPadParams{false, 0U, 0U, 0U};
    DataCopyPad(assistInfoForCombineLocal_, assistInfoForCombineGM_, epSendCntParams, copyPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    uint32_t preCount = 0U;
    uint32_t startTokenIdx = 0U;
    uint32_t curTokenNum = 0U;
    for (uint32_t expertIdx = 0U; expertIdx < curRankExpertNum; expertIdx++) {
        dataCmdInfoGM_ = commCmdInfoOutGM_ + (expertIdx * worldSize_ + startRankId_) * sizeof(BatchWriteItem);
        for (uint32_t ep = startRankId_; ep < endRankId_; ep++) {
            uint32_t batchWriteItemStartCnt = batchWriteInfoCnt_;
            if ((ep > 0U) || (expertIdx > 0U)) {
                preCount = assistInfoForCombineLocal_.GetValue(expertIdx * worldSize_ + ep - 1);
            }
            curTokenNum = assistInfoForCombineLocal_.GetValue(expertIdx * worldSize_ + ep) - preCount;
            if (curTokenNum == 0U) {
                continue;
            }
            startTokenIdx = preCount * axisH_;
            ExpertAlltoAllDispatch(curTokenNum, preCount, ep, expertIdx);

            // 确保cache中的数据已刷新到GM地址上
            __gm__ BatchWriteItem *sendInfo = reinterpret_cast<__gm__ BatchWriteItem *>(
                dataCmdInfoGM_ + batchWriteItemStartCnt * sizeof(BatchWriteItem));
            GlobalTensor<int64_t> tempTensor;
            tempTensor.SetGlobalBuffer((__gm__ int64_t *)sendInfo);
            DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(tempTensor);

            PipeBarrier<PIPE_ALL>();
            hccl_.BatchWrite<true>((GM_ADDR)sendInfo, (batchWriteInfoCnt_ - batchWriteItemStartCnt),
                                   ep % queuePerCore_);
        }
    }
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineSetup<TemplateMC2TypeFunc>::ExpertAlltoAllDispatch(uint32_t tokenNumLoop,
                                                                                              uint32_t srcStartTokenIdx,
                                                                                              uint32_t ep,
                                                                                              uint32_t expertIdx)
{
    // 获取对应卡上 window 的首地址
    GM_ADDR rankGM = GetWinAddrByRankId(ep, expertIdx) + epDataOffsetOnWin_;
    uint32_t dataCnt = axisH_;
    GM_ADDR batchWriteItemAddr = dataCmdInfoGM_ + batchWriteInfoCnt_ * sizeof(BatchWriteItem);
    DataCopyExtParams batchItemCopyOutParams{1U, sizeof(BatchWriteItem), 0U, 0U, 0U};

    // SDMA搬运
    uint64_t sourceAddr = (uint64_t)(expandXGM_ + srcStartTokenIdx * axisHExpandXTypeSize_);
    uint64_t dstAddr = (uint64_t)(rankGM);

    LocalTensor<uint32_t> batchItemInfo = batchItemInfoBuf_.Get<uint32_t>();
    LocalTensor<uint64_t> batchItemInfoU64 = batchItemInfoBuf_.Get<uint64_t>();

    if constexpr (AscendC::IsSameType<ExpandXType, bfloat16_t>::value) {
        batchItemInfo(0) = static_cast<uint32_t>(HcclDataType::HCCL_DATA_TYPE_BFP16);
    } else {
        batchItemInfo(0) = static_cast<uint32_t>(HcclDataType::HCCL_DATA_TYPE_FP16);
    }
    batchItemInfo(7) = static_cast<uint32_t>(tokenNumLoop * dataCnt * sizeof(ExpandXType));
    batchItemInfoU64(4) = reinterpret_cast<uint64_t>(sourceAddr);
    batchItemInfoU64(5) = reinterpret_cast<uint64_t>(dstAddr);
    GlobalTensor<uint32_t> itemTensor;
    itemTensor.SetGlobalBuffer((__gm__ uint32_t *)(batchWriteItemAddr));
    DataCopyPad(itemTensor, batchItemInfo, batchItemCopyOutParams);
    batchWriteInfoCnt_ += 1U;
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineSetup<TemplateMC2TypeFunc>::SetStatus()
{
    if (startRankId_ >= worldSize_) { // 空闲核，直接返回
        return;
    }
    uint32_t i = 0;
    GM_ADDR batchWriteItemAddr = dataCmdInfoGM_ + (axisA_ + startRankId_) * sizeof(BatchWriteItem);

    // SDMA搬运填状态区
    for (uint32_t epIdx = startRankId_; epIdx < endRankId_; epIdx++) {
        stateGM_ = GetWinStateAddrByRankId(epIdx) + epStateOffsetOnWin_;

        // batchwrite填写地址
        GlobalTensor<uint32_t> localBufTensorLow_;
        GlobalTensor<uint32_t> localBufTensorHigh_;
        GlobalTensor<uint32_t> remoteBufTensorLow_;
        GlobalTensor<uint32_t> remoteBufTensorHigh_;
        GlobalTensor<uint32_t> lengthTensor_;
        GlobalTensor<uint64_t> dataTypeTensor_;

        dataTypeTensor_.SetGlobalBuffer((__gm__ uint64_t *)(batchWriteItemAddr + i * sizeof(BatchWriteItem)));

        lengthTensor_.SetGlobalBuffer(
            (__gm__ uint32_t *)(batchWriteItemAddr + i * sizeof(BatchWriteItem) + sizeof(uint32_t) * 7));

        localBufTensorLow_.SetGlobalBuffer(
            (__gm__ uint32_t *)(batchWriteItemAddr + i * sizeof(BatchWriteItem) + sizeof(uint32_t) * 8));
        localBufTensorHigh_.SetGlobalBuffer(
            (__gm__ uint32_t *)(batchWriteItemAddr + i * sizeof(BatchWriteItem) + sizeof(uint32_t) * 9));

        remoteBufTensorLow_.SetGlobalBuffer(
            (__gm__ uint32_t *)(batchWriteItemAddr + i * sizeof(BatchWriteItem) + sizeof(uint32_t) * 10));
        remoteBufTensorHigh_.SetGlobalBuffer(
            (__gm__ uint32_t *)(batchWriteItemAddr + i * sizeof(BatchWriteItem) + sizeof(uint32_t) * 11));

        // 可以组装多个通信任务，实现批量发送
        uint64_t sourceAddr = (uint64_t)(sdmaStateGM_);
        uint64_t dstAddr = (uint64_t)(stateGM_);

        localBufTensorLow_.SetValue(0, (uint32_t)sourceAddr);
        localBufTensorHigh_.SetValue(0, (uint32_t)(sourceAddr >> 32U));

        remoteBufTensorLow_.SetValue(0, (uint32_t)dstAddr);
        remoteBufTensorHigh_.SetValue(0, (uint32_t)(dstAddr >> 32U));

        lengthTensor_.SetValue(0, (uint32_t)(sizeof(float)));

        dataTypeTensor_.SetValue(0, HcclDataType::HCCL_DATA_TYPE_FP32);

        // 确保cache中的数据已刷新到GM地址上
        __gm__ BatchWriteItem *sendInfo =
            reinterpret_cast<__gm__ BatchWriteItem *>(batchWriteItemAddr + i * sizeof(BatchWriteItem));
        GlobalTensor<int64_t> tempTensor;
        tempTensor.SetGlobalBuffer((__gm__ int64_t *)sendInfo);
        DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(tempTensor);
        hccl_.BatchWrite<true>((GM_ADDR)sendInfo, 1,
                               epIdx % queuePerCore_); // 每个目标rank对应的queueid要与发数据时的保持一致
        i++;
    }
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineSetup<TemplateMC2TypeFunc>::Process()
{
    BuffInit();
    Dispatch();
    AlltoAllBuffInit();
    SetStatus();
    SyncAll<true>();
    hccl_.Finalize<false>();
}
} // namespace MoeDistributeCombineSetupImpl
#endif // MOE_DISTRIBUTE_COMBINE_IMPL_H
