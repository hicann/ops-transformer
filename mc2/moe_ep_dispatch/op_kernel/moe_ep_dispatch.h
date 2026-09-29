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
 * \file moe_ep_dispatch.h
 * \brief
 */

#ifndef MOE_EP_DISPATCH_H
#define MOE_EP_DISPATCH_H

#include <cstddef>

#if __has_include("version/asc_devkit_version.h") && __has_include("version/hcomm_version.h")
#include "version/asc_devkit_version.h"
#include "version/hcomm_version.h"

#if (ASC_DEVKIT_MAJOR > 9 || (ASC_DEVKIT_MAJOR == 9 && ASC_DEVKIT_MINOR > 0)) && \
    (HCOMM_MAJOR > 9 || (HCOMM_MAJOR == 9 && HCOMM_MINOR > 0))
#define ENABLE_MOE_EP_KERNEL
#endif

#endif

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#include "kernel_tiling/kernel_tiling.h"
#include "adv_api/hccl/hccl.h"
#include "adv_api/reduce/reduce.h"
#include "adv_api/reduce/sum.h"
#if __has_include("adv_api/hcomm/hcomm.h")
#include "adv_api/hcomm/hcomm.h"
#endif
#include "moe_ep_dispatch_tiling.h"
#include "moe_ep_dispatch_base.h"
#include "moe_ep_dispatch_vf.h"

#include "../../common/op_kernel/moe_distribute_base.h"
#include "../../common/op_kernel/mc2_kernel_utils.h"
#include "../../common/op_kernel/mc2_moe_context.h"
#include "../../common/op_kernel/moe_ep_exception_dump_writer.h"

namespace MoeEpDispatchImpl {

#define TemplateMoeEpDispatchTypeClass \
    typename XType, typename ScalesType, bool DoCpuSync, bool IsCached, bool IsTopkWeights, uint8_t NetworkMode
#define TemplateMoeEpDispatchTypeFunc XType, ScalesType, DoCpuSync, IsCached, IsTopkWeights, NetworkMode

#if defined(ENABLE_MOE_EP_KERNEL)

using namespace AscendC;
using namespace Mc2Kernel;

constexpr uint32_t UB_ALIGN = 32U;
constexpr uint32_t WIN_ADDR_ALIGN = 512U;
constexpr uint32_t ALIGNED_LEN_256 = 256U;
constexpr uint32_t TOPK_INFO_SIZE = 4U;  // sizeof(int32_t)=sizeof(float)=4B
constexpr uint32_t UB_STRIDE = 8U;       // UB_ALIGN/sizeof(int32_t)=8
constexpr uint32_t INT64_UB_STRIDE = 4U; // UB_ALIGN/sizeof(int64_t)=4
constexpr uint32_t BITS_PER_BYTE = 8U;
constexpr uint32_t STATE_STRIDE = 15U; // (WIN_ADDR_ALIGN-UB_ALIGN)/UB_ALIGN=15
constexpr uint32_t HCOMM_INIT_SIZE = 512U;
constexpr uint32_t PER_GROUP_SIZE = 32 * 1024U; // 计算count 32KB per group
constexpr uint8_t BUFFER_NUM = 2;
constexpr uint32_t NOTIFY_VAL_SHIFT = 32U;                        // notify 高32位为 sendCnt, 低32位为 state(=1)
constexpr uint32_t MAX_BATCH_SIZE = 160 * 1024U;                  // srcTokenIdx 160KB
constexpr uint32_t SGE_PER_SLOT = 2U;                             // 每 slot 两个 SGE: x 与 meta
constexpr uint32_t SGE_PER_SQE = 12U;                             // 单 SQE 的 SGE 数
constexpr uint32_t SLOT_NUM_PER_SQE = SGE_PER_SQE / SGE_PER_SLOT; // 单 SQE 的 slot 数
constexpr uint32_t HCOMM_BATCH_CAPACITY = 128U;                   // 通信敲 db 间隔
constexpr uint32_t HCOMM_SGE_BYTES = 16U;
constexpr uint32_t HCOMM_SQE_BYTES = 48U;
constexpr uint32_t HCOMM_WQE_BYTES = 64U;
constexpr struct UrmaWqeEntry DATA_CFG = {.odr = 5, .fence = 1, .se = 0, .cqe = 0, .inlineEn = 0};
constexpr struct UrmaWqeEntry LAST_DATA_CFG = {.odr = 5, .fence = 1, .se = 0, .cqe = 1, .inlineEn = 0};
constexpr struct UrmaWqeEntry NOTIFY_CFG = {.odr = 6, .fence = 1, .se = 0, .cqe = 0, .inlineEn = 0};

template <TemplateMoeEpDispatchTypeClass>
class MoeEpDispatch {
public:
    __aicore__ inline MoeEpDispatch(){};
    __aicore__ inline void Init(GM_ADDR context, GM_ADDR x, GM_ADDR topkIdx, GM_ADDR topkWeights, GM_ADDR scales,
                                GM_ADDR cachedSlotIdx, GM_ADDR numRecvPerRank, GM_ADDR numRecvPerExpert,
                                GM_ADDR dstBufferSlotIdx, GM_ADDR workspaceGM, GM_ADDR tilingGM, TPipe* pipe,
                                const MoeEpDispatchTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void BufferInit();
    __aicore__ inline void ResetCounters();
    __aicore__ inline void CalSendCnt();
    __aicore__ inline void Communication();
    __aicore__ inline void GetRecvCount();
    __aicore__ inline void SetRecvNumPerExpert();
    __aicore__ inline void SetRecvNumPerRank(LocalTensor<int32_t> recvTmpTensor);
    __aicore__ inline void WriteToRemoteWindow();
    __aicore__ inline void CommBufferInit();
    __aicore__ inline void SendTokenByChannel(uint32_t tokenStart, uint32_t tokenNum, uint32_t dstRankId,
                                              uint64_t commHandle, GM_ADDR notifyAddr);
    __aicore__ inline void SetSrcTokenIdx(uint32_t topkOffset, uint32_t calCnt, uint32_t calCntAlign,
                                          uint32_t& sendLocalCnt, LocalTensor<int32_t>& cntLocalTensor);
    __aicore__ inline void ProcessMetaSlot(uint32_t startId, uint32_t tokenCnt, uint32_t sendLocalCnt,
                                           uint32_t& curLocalSlot);
    __aicore__ inline void SendPhase();
    __aicore__ inline void SendPhaseCached();
    __aicore__ inline void GetLocalSlotStart(uint32_t sendCount, uint32_t maxBatchCnt, uint32_t alignStride,
                                             uint32_t& curLocalSlot, DataCopyExtParams& srcTokenIdxCopyParams);
    __aicore__ inline void SetLocalStatusAndBufferInitCached();
    __aicore__ inline void CommunicationCached();
    __aicore__ inline void WaitStatusCached();
    __aicore__ inline void GetSlotStartNum();
    __aicore__ inline void CopyInMetaInfo(uint32_t startTokenId, uint32_t batchCnt);
    __aicore__ inline void InitMetaScatter();
    __aicore__ inline void ScatterMetaRouteInfo(uint32_t batchStartId, uint32_t batchCnt);
    __aicore__ inline void SetStatus();
    __aicore__ inline void SplitToCore(uint32_t curSendCnt, uint32_t curUseAivNum, uint32_t& startId, uint32_t& endId,
                                       uint32_t& sendNum);
    __aicore__ inline bool GetCoreAssignment(uint32_t& copyRankStart, uint32_t& copyRankNum, uint32_t& channelIndex,
                                             uint32_t& coreIndexInGroup, uint32_t& groupSize);

    TPipe* tpipe_{nullptr};
    __gm__ Mc2Aclnn::MoeCommContext* mc2Context_{nullptr};
    AscendC::Hcomm<COMM_PROTOCOL_UBC_CTP> hcomm_; // 通信上下文
    MoeEpExceptionDump::MoeEpCoreDiagWriter diagWriter_;

    GlobalTensor<int32_t> topkIdxGMTensor_;
    GlobalTensor<float> topkWeightsGMTensor_;
    GlobalTensor<int32_t> dstSlotIdxGMTensor_; // direct: [ep][bs + 1], row = [sendCount, srcTokenId...]
    GlobalTensor<int32_t> numRecvPerRankGMTensor_;
    GlobalTensor<int64_t> numRecvPerExpertGMTensor_;
    GlobalTensor<int32_t> cachedSlotIdxGMTensor_; // direct: [ep][bs + 1]
    GlobalTensor<int32_t> scaleupCounterGMTensor_;
    GlobalTensor<int32_t> recvCounterGMTensor_;
    GlobalTensor<int32_t> sendCntPerRankGMTensor_;
    GlobalTensor<int32_t> sendCntPerExpertGMTensor_;
    GlobalTensor<int16_t> dstRankGMTensor_;
    GlobalTensor<int32_t> metaSlotGMTensor_;
    GlobalTensor<int32_t> localMetaSlotGMTensor_;
    GlobalTensor<ScalesType> scalesGMTensor_;

    LocalTensor<int32_t> metaLocalTensor_;
    LocalTensor<int32_t> numRecvPerRankTensor_;
    LocalTensor<int64_t> numRecvPerExpertTensor_;
    LocalTensor<int32_t> sendCntPerExpertTensor_;
    LocalTensor<int32_t> sendCntPerRankTensor_;
    LocalTensor<int32_t> slotIdxPerRankTensor_;
    LocalTensor<int16_t> dstRankTensor_;
    LocalTensor<int16_t> dstRankIdTensor_;
    LocalTensor<uint8_t> hcommTensor_;
    LocalTensor<uint8_t> hcommBatchTensor_;
    LocalTensor<int32_t> rankValueTensor_;
    LocalTensor<int32_t> tokenValueTensor_;
    LocalTensor<uint32_t> rankOffsetTensor_;
    LocalTensor<uint32_t> tokenOffsetTensor_;

    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> metaSlotQueue_;
    TBuf<> topkIdsBuf_;
    TBuf<> tempBuf_;
    TBuf<> dstExpBuf_;
    TBuf<> maskBuf_;
    TBuf<> hcommBuf_; // 通信
    TBuf<> sendCntRankBuf_;
    TBuf<> sendCntExpertBuf_;
    TBuf<> numRecvPerRankBuf_;
    TBuf<> numRecvPerExpertBuf_;
    TBuf<> metaScatterBuf_;

    GM_ADDR workspaceGM_{nullptr};
    GM_ADDR hostPinnedCounterAddrGM_{nullptr};
    GM_ADDR xAddr_{nullptr};
    GM_ADDR sendSrcTokenIdxAddr_{nullptr}; // 发送表基址: 非cache=dst_buffer_slot_idx输出, cache=cached_dst_slot_idx输入
    GM_ADDR scaleupCounterAddr_{nullptr};
    GM_ADDR sendCntWorkspaceAddr_{nullptr};
    GM_ADDR dstRankInfoAddr_{nullptr};
    GM_ADDR localCntStateWinAddr_{nullptr};
    GM_ADDR localSlotStateWinAddr_{nullptr};
    GM_ADDR localSlotWinAddr_{nullptr};
    GM_ADDR payloadStashWinAddr_{nullptr};
    GM_ADDR payloadStashStateWinAddr_{nullptr};
    GM_ADDR cachedStateAddr_{nullptr};

    uint32_t axisBS_{0};
    uint32_t axisH_{0};
    uint32_t axisK_{0};
    uint32_t epWorldSize_{0};
    uint32_t channelsPerRank_{1};
    uint32_t moeExpertNumPerRank_{0};
    uint32_t axisMaxBS_{0};
    uint32_t scalesBytes_{0};
    uint32_t perSlotBytes_{0};
    uint32_t moeExpertNum_{0};
    uint32_t aivNum_{0};
    uint32_t epRankId_{0};
    uint32_t aivId_{0};
    uint32_t startTokenId_{0};
    uint32_t endTokenId_{0};
    uint32_t sendTokenNum_{0};
    uint32_t perGroupTokenNum_{0};
    uint32_t startRankId_{0};
    uint32_t endRankId_{0};
    uint32_t rankNumPerCore_{0};
    uint32_t copyRankStart_{0};
    uint32_t copyRankNum_{0};
    uint32_t channelIndex_{0};
    uint32_t coreIndexInGroup_{0};
    uint32_t groupSize_{1};
    uint32_t hcommBatchBufferBytes_{0};
    uint32_t dstSlotStride_{0};
    uint32_t tokenSize_{0};
    uint32_t kAlignSize_{0};
    uint32_t axisKAlign_{0};
    uint32_t epWorldSizeAlign_{0};
    uint32_t epWorldSizeAlign512_{0};
    uint32_t counterCnt_{0};
    uint32_t counterAlign512_{0};
    uint32_t perGroupSizeAlign_{0};
    uint32_t maskBytesAlign_{0};
    uint32_t sendCntRankSizeAlign_{0};
    uint32_t moeNumPerRankSize_{0};
    uint32_t moeNumPerRankAlign_{0};
    uint32_t expertPerRankAlign_{0};
    uint32_t moeExpertNumAlign_{0};
    uint32_t moeNumPerRankAlign512_{0};
    uint32_t moeExpertNumAlign512_{0};
    uint32_t cntFlagWinSize_{0};
    uint32_t dispatchNotifyCount_{1};
    uint32_t metaSlotBytes_{0};
    uint32_t metaBatch_{1};
    uint32_t topkCopyAlign_{0};
    uint32_t metaSlotStride_{0};
    uint64_t dstRankInfoOffset_{0};
    uint64_t cntWinStateOffset_{0};
    uint64_t slotWinStateOffset_{0};
    uint64_t payloadWinStateOffset_{0};
    uint64_t winDataOffset_{0};
    uint64_t payloadStashWinOffset_{0};
    uint64_t dstRankWinOffset_{0};
    uint64_t dstRankStateOffset_{0};
    uint64_t dstStateWinOffset_{0};
    bool coreAssigned_{false};

    DataCopyParams statusCopyParams_;
    DataCopyParams clearStatusCopyParams_;
    DataCopyParams sendPerExpertParams_;
    DataCopyParams sendPerRankParams_;
    DataCopyParams metaSlotCopyParams_; // stash 元数据槽整槽拷贝参数
    DataCopyParams topkCopyParams_;
    DataCopyParams scalesCopyParams_;
    DataCopyPadParams padParams_;
    DataCopyPadExtParams<int32_t> routePadParams_;
};

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::Init(
    GM_ADDR context, GM_ADDR x, GM_ADDR topkIdx, GM_ADDR topkWeights, GM_ADDR scales, GM_ADDR cachedSlotIdx,
    GM_ADDR numRecvPerRank, GM_ADDR numRecvPerExpert, GM_ADDR dstBufferSlotIdx, GM_ADDR workspaceGM, GM_ADDR tilingGM,
    TPipe* pipe, const MoeEpDispatchTilingData* tilingData)
{
    tpipe_ = pipe;
    aivId_ = GetBlockIdx();
    workspaceGM_ = workspaceGM;
    mc2Context_ = reinterpret_cast<__gm__ Mc2Aclnn::MoeCommContext*>(context);
    epRankId_ = mc2Context_->epRankId;
    channelsPerRank_ = mc2Context_->channelsPerRank;
    constexpr size_t metadataOffset =
        offsetof(MoeEpDispatchTilingData, moeEpDispatchInfo) + offsetof(MoeEpDispatchInfo, dumpMetadata);
    MoeEpExceptionDump::WriteMetadata(context, tilingGM + metadataOffset);
    diagWriter_.Init(context, MOE_EP_CORE_DIAG_DISPATCH, tpipe_);

    const auto& info = tilingData->moeEpDispatchInfo;
    xAddr_ = x;
    metaSlotBytes_ = info.metaSlotBytes;
    metaBatch_ = info.metaBatch == 0U ? 1U : info.metaBatch;
    axisBS_ = info.cfg.numTokens;
    axisH_ = info.cfg.hidden;
    axisK_ = info.cfg.topK;
    epWorldSize_ = info.cfg.epWorldSize;
    moeExpertNum_ = info.cfg.numExperts;
    moeExpertNumPerRank_ = info.cfg.numLocalExperts;
    axisMaxBS_ = info.cfg.numMaxTokensPerRank;
    scalesBytes_ = info.scalesBytes;
    perSlotBytes_ = info.perSlotBytes;
    aivNum_ = info.aivNum;
    dstRankInfoOffset_ = info.workspace.dstRankInfoOffset;
    cntWinStateOffset_ = info.window.cntWinStateOffset;
    slotWinStateOffset_ = info.window.slotWinStateOffset;
    payloadWinStateOffset_ = info.window.payloadWinStateOffset;
    winDataOffset_ = info.window.winDataOffset;
    payloadStashWinOffset_ = info.window.payloadStashWinOffset;
    dispatchNotifyCount_ = info.dispatchNotifyCount;
    hostPinnedCounterAddrGM_ = reinterpret_cast<GM_ADDR>(info.hostPinnedCounterAddr);

    tokenSize_ = axisH_ * sizeof(XType);
    kAlignSize_ = Ceil(axisK_ * TOPK_INFO_SIZE, UB_ALIGN) * UB_ALIGN;
    axisKAlign_ = kAlignSize_ / TOPK_INFO_SIZE;
    metaSlotStride_ = metaSlotBytes_ / sizeof(int32_t);
    dstSlotStride_ = axisBS_ + 1;
    epWorldSizeAlign_ = Ceil(epWorldSize_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    perGroupTokenNum_ = PER_GROUP_SIZE / sizeof(int16_t) / axisK_; // token num per group
    perGroupSizeAlign_ = Ceil(perGroupTokenNum_ * axisK_ * sizeof(int16_t), ALIGNED_LEN_256) * ALIGNED_LEN_256;
    maskBytesAlign_ = Ceil(Ceil(perGroupSizeAlign_ / sizeof(int16_t), BITS_PER_BYTE), UB_ALIGN) * UB_ALIGN;
    epWorldSizeAlign512_ = Ceil(epWorldSize_ * sizeof(int32_t), WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN;
    sendCntRankSizeAlign_ = Ceil(epWorldSize_ * sizeof(int32_t), ALIGNED_LEN_256) * ALIGNED_LEN_256;
    counterCnt_ = epWorldSizeAlign_ / sizeof(int32_t);
    counterAlign512_ = epWorldSizeAlign512_ / sizeof(int32_t);
    moeNumPerRankSize_ = moeExpertNumPerRank_ * sizeof(int32_t);
    moeNumPerRankAlign_ = Ceil(moeNumPerRankSize_, UB_ALIGN) * UB_ALIGN;
    expertPerRankAlign_ = moeNumPerRankAlign_ / sizeof(int32_t);
    moeExpertNumAlign_ = Ceil(moeExpertNum_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    moeNumPerRankAlign512_ = Ceil(moeNumPerRankSize_, WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN;
    moeExpertNumAlign512_ = Ceil(moeExpertNum_ * sizeof(int32_t), WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN;
    cntFlagWinSize_ = epWorldSize_ * WIN_ADDR_ALIGN;
    dstRankStateOffset_ = static_cast<uint64_t>(epRankId_) * WIN_ADDR_ALIGN;
    dstRankWinOffset_ = static_cast<uint64_t>(epRankId_) * axisMaxBS_ * perSlotBytes_; // 目标窗口地址偏移
    if (channelsPerRank_ == 0 || (epWorldSize_ > 0 && channelsPerRank_ > Mc2Aclnn::HCCL_MAX_RANK_SIZE / epWorldSize_)) {
        channelsPerRank_ = 1;
    }
    dstStateWinOffset_ = static_cast<uint64_t>(epRankId_) * dispatchNotifyCount_ * WIN_ADDR_ALIGN; // 目标状态地址偏移

    topkIdxGMTensor_.SetGlobalBuffer((__gm__ int32_t*)topkIdx);
    if constexpr (IsTopkWeights) {
        topkWeightsGMTensor_.SetGlobalBuffer((__gm__ float*)topkWeights);
    }
    if constexpr (Std::IsSame<XType, fp8_e5m2_t>::value || Std::IsSame<XType, fp8_e4m3fn_t>::value) {
        scalesGMTensor_.SetGlobalBuffer((__gm__ ScalesType*)scales);
        topkCopyAlign_ = Ceil(scalesBytes_, UB_ALIGN) * UB_ALIGN / sizeof(int32_t); // meta 槽内 topk 的 int32 偏移
    }
    if constexpr (IsCached) {
        cachedSlotIdxGMTensor_.SetGlobalBuffer((__gm__ int32_t*)cachedSlotIdx);
        maskBytesAlign_ = Ceil(Ceil(MAX_BATCH_SIZE / sizeof(int32_t), BITS_PER_BYTE), UB_ALIGN) * UB_ALIGN;
        sendSrcTokenIdxAddr_ = cachedSlotIdx;
        cachedStateAddr_ = workspaceGM;
    } else {
        sendSrcTokenIdxAddr_ = dstBufferSlotIdx;
    }
    numRecvPerRankGMTensor_.SetGlobalBuffer((__gm__ int32_t*)numRecvPerRank);
    numRecvPerExpertGMTensor_.SetGlobalBuffer((__gm__ int64_t*)numRecvPerExpert);
    dstSlotIdxGMTensor_.SetGlobalBuffer((__gm__ int32_t*)dstBufferSlotIdx);

    statusCopyParams_ = {static_cast<uint16_t>(epWorldSize_), 1U, static_cast<uint16_t>(STATE_STRIDE), 0U};
    clearStatusCopyParams_ = {static_cast<uint16_t>(epWorldSize_), 1U, 0U, static_cast<uint16_t>(STATE_STRIDE)};
    sendPerExpertParams_ = {1U, static_cast<uint16_t>(moeExpertNum_ * sizeof(int32_t)), 0U, 0U};
    sendPerRankParams_ = {1U, static_cast<uint16_t>(epWorldSize_ * sizeof(int32_t)), 0U, 0U};
    metaSlotCopyParams_ = {1U, static_cast<uint16_t>(metaSlotBytes_), 0U, 0U};
    topkCopyParams_ = {1U, static_cast<uint16_t>(axisK_ * TOPK_INFO_SIZE), 0U,
                       static_cast<uint16_t>((metaSlotBytes_ - kAlignSize_) / UB_ALIGN)};
    scalesCopyParams_ = {1U, static_cast<uint16_t>(scalesBytes_), 0U,
                         static_cast<uint16_t>((metaSlotBytes_ - topkCopyAlign_ * sizeof(int32_t)) / UB_ALIGN)};
    padParams_ = {true, 0, 0, 0};
    routePadParams_ = {false, 0U, 0U, 0U};

    scaleupCounterAddr_ = workspaceGM;
    sendCntWorkspaceAddr_ = workspaceGM + static_cast<uint64_t>(aivNum_) * epWorldSizeAlign512_;
    dstRankInfoAddr_ = workspaceGM + dstRankInfoOffset_;
    localCntStateWinAddr_ = GetWinAddrByRankId(mc2Context_, epRankId_, cntWinStateOffset_);
    localSlotStateWinAddr_ = GetWinAddrByRankId(mc2Context_, epRankId_, slotWinStateOffset_);
    payloadStashStateWinAddr_ = GetWinAddrByRankId(mc2Context_, epRankId_, payloadWinStateOffset_);
    localSlotWinAddr_ = GetWinAddrByRankId(mc2Context_, epRankId_, winDataOffset_) + dstRankWinOffset_;
    payloadStashWinAddr_ = GetWinAddrByRankId(mc2Context_, epRankId_, payloadStashWinOffset_);
    scaleupCounterGMTensor_.SetGlobalBuffer((__gm__ int32_t*)scaleupCounterAddr_);
    recvCounterGMTensor_.SetGlobalBuffer((__gm__ int32_t*)localCntStateWinAddr_);
    metaSlotGMTensor_.SetGlobalBuffer((__gm__ int32_t*)payloadStashWinAddr_);
    sendCntPerRankGMTensor_.SetGlobalBuffer((__gm__ int32_t*)sendCntWorkspaceAddr_);
    sendCntPerExpertGMTensor_.SetGlobalBuffer((__gm__ int32_t*)(sendCntWorkspaceAddr_ + epWorldSizeAlign512_));
    dstRankGMTensor_.SetGlobalBuffer((__gm__ int16_t*)dstRankInfoAddr_);
    diagWriter_.RunPosRecord(MOE_EP_DISPATCH_RUN_POS_INIT_DONE);
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::SplitToCore(uint32_t curSendCnt,
                                                                                 uint32_t curUseAivNum,
                                                                                 uint32_t& startId, uint32_t& endId,
                                                                                 uint32_t& sendNum)
{
    sendNum = curSendCnt / curUseAivNum;               // 每个aiv需要发送的数量
    uint32_t remainderNum = curSendCnt % curUseAivNum; // 余数
    startId = sendNum * aivId_;                        // 每个aiv发送时的起始rankid
    if (aivId_ < remainderNum) {                       // 前remainderRankNum个aiv需要多发1个卡的数据
        sendNum++;
        startId += aivId_;
    } else {
        startId += remainderNum;
    }
    endId = startId + sendNum;
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline bool MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::GetCoreAssignment(uint32_t& copyRankStart,
                                                                                       uint32_t& copyRankNum,
                                                                                       uint32_t& channelIndex,
                                                                                       uint32_t& coreIndexInGroup,
                                                                                       uint32_t& groupSize)
{
    uint32_t jettyCntPerRank = dispatchNotifyCount_ < channelsPerRank_ ? dispatchNotifyCount_ : channelsPerRank_;
    uint32_t maxJettyAivNum = epWorldSize_ * jettyCntPerRank;
    uint32_t activeAivNum = aivNum_ > maxJettyAivNum ? maxJettyAivNum : aivNum_;
    if (activeAivNum >= epWorldSize_) { // 多channel情况下，发送使用核数，必须整除 epWorldSize
        activeAivNum = activeAivNum / epWorldSize_ * epWorldSize_;
    }
    if (aivId_ >= activeAivNum) {
        return false;
    }

    bool splitRankChannels = channelsPerRank_ > 1 && activeAivNum >= epWorldSize_;
    coreIndexInGroup = 0;
    groupSize = 1;
    if (splitRankChannels) {
        groupSize = activeAivNum / epWorldSize_;
        uint32_t accumulated = 0;
        uint32_t targetRank = epWorldSize_;
        for (uint32_t rank = 0; rank < epWorldSize_; ++rank) {
            if (aivId_ < accumulated + groupSize) {
                targetRank = rank;
                coreIndexInGroup = aivId_ - accumulated;
                break;
            }
            accumulated += groupSize;
        }
        if (targetRank >= epWorldSize_ || groupSize == 0) {
            return false;
        }
        channelIndex = coreIndexInGroup;
        copyRankNum = 1;
        copyRankStart = targetRank;
    } else {
        channelIndex = 0;
        copyRankNum = rankNumPerCore_;
        copyRankStart = startRankId_;
    }
    return true;
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::BufferInit()
{
    tpipe_->InitBuffer(metaSlotQueue_, BUFFER_NUM, metaBatch_ * metaSlotBytes_);
    tpipe_->InitBuffer(metaScatterBuf_, MOE_EP_META_SCATTER_UB_BYTES);
    tpipe_->InitBuffer(topkIdsBuf_, 2 * perGroupSizeAlign_);
    tpipe_->InitBuffer(tempBuf_, perGroupSizeAlign_);
    tpipe_->InitBuffer(dstExpBuf_, 2 * perGroupSizeAlign_);
    tpipe_->InitBuffer(maskBuf_, 3 * maskBytesAlign_);
    tpipe_->InitBuffer(sendCntRankBuf_, sendCntRankSizeAlign_);
    tpipe_->InitBuffer(sendCntExpertBuf_, moeExpertNumAlign_);
    tpipe_->InitBuffer(numRecvPerExpertBuf_, 2 * moeNumPerRankAlign_);
    tpipe_->InitBuffer(numRecvPerRankBuf_, epWorldSizeAlign_);
    numRecvPerRankTensor_ = numRecvPerRankBuf_.Get<int32_t>();
    sendCntPerRankTensor_ = sendCntRankBuf_.Get<int32_t>();
    sendCntPerExpertTensor_ = sendCntExpertBuf_.Get<int32_t>();
    Duplicate<int32_t>(sendCntPerRankTensor_, 0, counterCnt_);
    Duplicate<int32_t>(sendCntPerExpertTensor_, 0, moeExpertNumAlign_ / sizeof(int32_t));
    ResetCounters();
    SyncAll<true>();
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::ResetCounters()
{
    if (aivId_ != aivNum_ - 1) { // 仅尾核做per-expert/rank 清0
        return;
    }
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopyPad(sendCntPerExpertGMTensor_, sendCntPerExpertTensor_, sendPerExpertParams_);
    DataCopy(sendCntPerRankGMTensor_, sendCntPerRankTensor_, counterCnt_);
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::CalSendCnt()
{
    SplitToCore(axisBS_, aivNum_, startTokenId_, endTokenId_, sendTokenNum_); // 按token数量分核
    if (startTokenId_ >= axisBS_) {
        return;
    }

    uint32_t groupCnt = Ceil(sendTokenNum_, perGroupTokenNum_);
    uint32_t calCnt = perGroupTokenNum_ * axisK_;
    LocalTensor<int32_t> topkIdsGroupTensor = topkIdsBuf_.Get<int32_t>();
    LocalTensor<int16_t> tempTensorInt16 = tempBuf_.Get<int16_t>();
    for (uint32_t group = 0; group < groupCnt; group++) {
        uint32_t groupOffset = group * perGroupTokenNum_;
        uint32_t tokenCnt = (group == groupCnt - 1) ? (sendTokenNum_ - groupOffset) : perGroupTokenNum_;
        if (group == groupCnt - 1) {
            calCnt = tokenCnt * axisK_;
        }
        uint32_t topkIdxOffset = (startTokenId_ + groupOffset) * axisK_;
        uint32_t tmpOffset = Ceil(calCnt * sizeof(int16_t), ALIGNED_LEN_256) * ALIGNED_LEN_256;
        uint32_t calCntAlign = tmpOffset / sizeof(int16_t);
        DataCopyExtParams topkIdsCntParams = {1U, static_cast<uint32_t>(calCnt * sizeof(int32_t)), 0U, 0U, 0U};
        dstRankTensor_ = dstExpBuf_.GetWithOffset<int16_t>(calCntAlign, 0);
        LocalTensor<int16_t> rawRankTensor = dstExpBuf_.GetWithOffset<int16_t>(calCntAlign, tmpOffset);
        if (group > 0) {
            SyncFunc<AscendC::HardEvent::V_MTE2>();
            SyncFunc<AscendC::HardEvent::MTE3_V>(); // Wait previous dstRank buffer.
        }
        DataCopyPad(topkIdsGroupTensor, topkIdxGMTensor_[topkIdxOffset], topkIdsCntParams,
                    routePadParams_); // copy topkId
        SyncFunc<AscendC::HardEvent::MTE2_V>();

        asc_vf_call<MoeEpDispatchVf::DedupAndCalCnt>(
            reinterpret_cast<__ubuf__ int32_t*>(topkIdsGroupTensor.GetPhyAddr()),
            reinterpret_cast<__ubuf__ int16_t*>(tempTensorInt16.GetPhyAddr()),
            reinterpret_cast<__ubuf__ int16_t*>(rawRankTensor.GetPhyAddr()),
            reinterpret_cast<__ubuf__ int16_t*>(dstRankTensor_.GetPhyAddr()),
            reinterpret_cast<__ubuf__ int32_t*>(sendCntPerExpertTensor_.GetPhyAddr()),
            reinterpret_cast<__ubuf__ int32_t*>(sendCntPerRankTensor_.GetPhyAddr()), calCnt, tokenCnt, axisK_,
            moeExpertNum_, epWorldSize_, moeExpertNumPerRank_);
        SyncFunc<AscendC::HardEvent::V_MTE3>();
        DataCopyExtParams dstRankCopyParams = {1U, static_cast<uint32_t>(calCnt * sizeof(int16_t)), 0U, 0U, 0U};
        DataCopyPad(dstRankGMTensor_[topkIdxOffset], dstRankTensor_, dstRankCopyParams);
    }
    // 拷入Workspace 对应计数区
    DataCopy(scaleupCounterGMTensor_[aivId_ * counterAlign512_], sendCntPerRankTensor_, counterCnt_);
    SetAtomicAdd<int32_t>();
    DataCopyPad(sendCntPerRankGMTensor_, sendCntPerRankTensor_, sendPerRankParams_);
    DataCopyPad(sendCntPerExpertGMTensor_, sendCntPerExpertTensor_, sendPerExpertParams_);
    SetAtomicNone();
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::Communication()
{
    // 通信初始化
    tpipe_->InitBuffer(hcommBuf_, HCOMM_INIT_SIZE);
    hcommTensor_ = hcommBuf_.Get<uint8_t>();
    hcomm_.Init(hcommTensor_, HCOMM_INIT_SIZE);
    SplitToCore(epWorldSize_, aivNum_, startRankId_, endRankId_, rankNumPerCore_); // 按卡分核
    coreAssigned_ = GetCoreAssignment(copyRankStart_, copyRankNum_, channelIndex_, coreIndexInGroup_, groupSize_);
    if (!coreAssigned_) {
        return;
    }

    uint64_t dstIndex = static_cast<uint64_t>(copyRankStart_) * dstSlotStride_;
    DataCopyExtParams sendCntCopyParams = {static_cast<uint16_t>(copyRankNum_), static_cast<uint32_t>(sizeof(int32_t)),
                                           0U, static_cast<int64_t>(axisBS_ * sizeof(int32_t)), 0U};
    LocalTensor<int32_t> tempTensor = topkIdsBuf_.GetWithOffset<int32_t>(copyRankNum_ * UB_STRIDE, 0);
    DataCopy(sendCntPerRankTensor_, sendCntPerRankGMTensor_, counterCnt_);
    SyncFunc<AscendC::HardEvent::MTE2_S>();

    for (uint32_t dstRankId = copyRankStart_; dstRankId < copyRankStart_ + copyRankNum_; ++dstRankId) {
        if (channelIndex_ != 0) {
            continue;
        }
        int32_t sendCnt = sendCntPerRankTensor_.GetValue(dstRankId);
        tempTensor.SetValue((dstRankId - copyRankStart_) * UB_STRIDE, sendCnt);
        uint64_t notifyVal = (static_cast<uint64_t>(sendCnt) << NOTIFY_VAL_SHIFT) | 1U;
        uint32_t srcStride = dstRankId * moeExpertNumPerRank_;
        // 计算目标窗口地址:
        GM_ADDR remoteStateAddr = GetWinAddrByRankId(mc2Context_, dstRankId, cntWinStateOffset_);
        GM_ADDR notifyAddr = remoteStateAddr + dstRankStateOffset_;
        GM_ADDR remoteCountAddr = remoteStateAddr + cntFlagWinSize_ + epRankId_ * moeNumPerRankAlign512_;
        GM_ADDR srcWorkspaceAddr = sendCntWorkspaceAddr_ + epWorldSizeAlign512_ + srcStride * sizeof(int32_t);
        if (dstRankId == copyRankStart_ + copyRankNum_ - 1) {
            SyncFunc<AscendC::HardEvent::S_MTE3>();
            DataCopyPad(dstSlotIdxGMTensor_[dstIndex], tempTensor, sendCntCopyParams);
        }

        if (dstRankId != epRankId_) { // 远端 使用URMA发送 count + state
            uint64_t commHandle = GetCommHandle(mc2Context_, dstRankId, channelIndex_);
            hcomm_.WriteWithNotifyNbi<true, PIPE_S, PIPE_MTE3, DATA_CFG>(commHandle, remoteCountAddr, srcWorkspaceAddr,
                                                                         moeNumPerRankSize_, notifyAddr, notifyVal);
            continue;
        }

        // 本端
        GlobalTensor<int32_t> countGMTensor;
        GlobalTensor<uint64_t> notifyGMTensor;
        countGMTensor.SetGlobalBuffer((__gm__ int32_t*)remoteCountAddr);
        notifyGMTensor.SetGlobalBuffer((__gm__ uint64_t*)notifyAddr);
        LocalTensor<uint64_t> notifyLocalTensor = tempBuf_.GetWithOffset<uint64_t>(INT64_UB_STRIDE, 0);
        LocalTensor<int32_t> cntPerExpertTensor = tempBuf_.GetWithOffset<int32_t>(expertPerRankAlign_, UB_ALIGN);
        DataCopyParams expertCntCopyParams = {1U, static_cast<uint16_t>(moeNumPerRankSize_), 0U, 0U};
        DataCopyPad(cntPerExpertTensor, sendCntPerExpertGMTensor_[srcStride], expertCntCopyParams, padParams_);
        notifyLocalTensor.SetValue(0, notifyVal);
        SyncFunc<AscendC::HardEvent::MTE2_MTE3>();
        DataCopyPad(countGMTensor, cntPerExpertTensor, expertCntCopyParams);
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        PipeBarrier<PIPE_MTE3>(); // perExpert 写完再写notifyVal
        DataCopy(notifyGMTensor, notifyLocalTensor, INT64_UB_STRIDE);
    }
    SyncFunc<AscendC::HardEvent::MTE3_MTE2>(); // tempBuf_ 复用
    SyncFunc<AscendC::HardEvent::MTE3_V>();
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::GetRecvCount()
{
    if (aivId_ != aivNum_ - 1) { // 最后一个核处理recvCount
        return;
    }

    uint32_t mask = 1;
    int32_t sumOfFlag = -1;
    int32_t commpareFlag = static_cast<int32_t>(epWorldSize_);
    LocalTensor<int32_t> recvCounterTensor = tempBuf_.GetWithOffset<int32_t>(epWorldSize_ * UB_STRIDE, 0);
    LocalTensor<float> tempFp32 = topkIdsBuf_.GetWithOffset<float>(epWorldSize_ * UB_STRIDE, 0);
    LocalTensor<float> recvCounterTensorFp32 = recvCounterTensor.template ReinterpretCast<float>();
    LocalTensor<float> numRecvPerRankTensorFp32 = numRecvPerRankTensor_.template ReinterpretCast<float>();
    // buf 复用
    SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
    SyncFunc<AscendC::HardEvent::S_V>();
    SyncFunc<AscendC::HardEvent::MTE3_V>();
    while (sumOfFlag != commpareFlag) { // 状态位check
        DataCopy(recvCounterTensor, recvCounterGMTensor_, statusCopyParams_);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        ReduceSum(numRecvPerRankTensorFp32, recvCounterTensorFp32, tempFp32, mask, epWorldSize_, 1);
        SyncFunc<AscendC::HardEvent::V_S>();
        sumOfFlag = numRecvPerRankTensor_.GetValue(0);
    }
    SetRecvNumPerExpert();                // 计算本卡上各专家接收的token总数
    SetRecvNumPerRank(recvCounterTensor); // 计算本端接收来自各卡的token总数
    // status clear
    Duplicate<int32_t>(recvCounterTensor, 0, epWorldSize_ * UB_STRIDE);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(recvCounterGMTensor_, recvCounterTensor, clearStatusCopyParams_);
    diagWriter_.RunPosRecord(MOE_EP_DISPATCH_RUN_POS_COUNT_READY);
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::SetRecvNumPerExpert()
{
    uint32_t recvNumAlign = epWorldSize_ * expertPerRankAlign_;
    GlobalTensor<int32_t> localExpertRecvGMTensor;
    localExpertRecvGMTensor.SetGlobalBuffer((__gm__ int32_t*)(localCntStateWinAddr_ + cntFlagWinSize_));
    numRecvPerExpertTensor_ = numRecvPerExpertBuf_.Get<int64_t>();
    LocalTensor<int64_t> tempRecvTensorInt64 = dstExpBuf_.Get<int64_t>();
    LocalTensor<int64_t> recvCntTensor = topkIdsBuf_.GetWithOffset<int64_t>(INT64_UB_STRIDE, 0);
    LocalTensor<int32_t> recvTensorInt32 = topkIdsBuf_.GetWithOffset<int32_t>(recvNumAlign, UB_ALIGN);
    LocalTensor<int64_t> sharedTmp = recvTensorInt32.template ReinterpretCast<int64_t>();
    LocalTensor<uint8_t> sharedTmpInt8 = recvTensorInt32.template ReinterpretCast<uint8_t>();

    const uint32_t shape[] = {epWorldSize_, expertPerRankAlign_};
    DataCopyParams inRecvCntParams = {static_cast<uint16_t>(epWorldSize_), static_cast<uint16_t>(moeNumPerRankAlign_),
                                      static_cast<uint16_t>(moeNumPerRankAlign512_ - moeNumPerRankAlign_), 0U};
    DataCopyParams recvPerExpertParams = {1U, static_cast<uint16_t>(moeExpertNumPerRank_ * sizeof(int64_t)), 0U, 0U};
    DataCopyPad(recvTensorInt32, localExpertRecvGMTensor, inRecvCntParams, padParams_);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    Cast(tempRecvTensorInt64, recvTensorInt32, RoundMode::CAST_NONE, recvNumAlign);
    ReduceSum<int64_t, AscendC::Pattern::Reduce::RA, true>(numRecvPerExpertTensor_, tempRecvTensorInt64, sharedTmpInt8,
                                                           shape, true);

    if constexpr (DoCpuSync) { // 计算actualA，并写入host pin
        GlobalTensor<int64_t> hostPinnedCounterTensor;
        hostPinnedCounterTensor.SetGlobalBuffer((__gm__ int64_t*)hostPinnedCounterAddrGM_);
        ReduceSum(recvCntTensor, numRecvPerExpertTensor_, sharedTmp, moeExpertNumPerRank_);
        SyncFunc<AscendC::HardEvent::V_MTE3>();
        DataCopy(hostPinnedCounterTensor, recvCntTensor, INT64_UB_STRIDE);
    }

    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopyPad(numRecvPerExpertGMTensor_, numRecvPerExpertTensor_, recvPerExpertParams);
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::SetRecvNumPerRank(
    LocalTensor<int32_t> recvTmpTensor)
{
    LocalTensor<uint32_t> gatherTmpTensor = topkIdsBuf_.GetWithOffset<uint32_t>(UB_STRIDE, epWorldSize_ * UB_ALIGN);
    SyncFunc<AscendC::HardEvent::V_S>();
    gatherTmpTensor.SetValue(0, 2); // 设置掩码，取源操作数每个datablock中的第2个元素
    uint32_t mask = 2;              // 源操作数每个datablock只需要处理两个元素
    uint64_t rsvdCnt = 0;
    GatherMaskParams recvMaskParams = {1, static_cast<uint16_t>(epWorldSize_), 1, 0};
    DataCopyParams recvPerRankParams = {1U, static_cast<uint16_t>(epWorldSize_ * sizeof(int32_t)), 0U, 0U};
    SyncFunc<AscendC::HardEvent::S_V>();
    GatherMask(numRecvPerRankTensor_, recvTmpTensor, gatherTmpTensor, true, mask, recvMaskParams, rsvdCnt);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopyPad(numRecvPerRankGMTensor_, numRecvPerRankTensor_, recvPerRankParams);
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::GetSlotStartNum()
{
    slotIdxPerRankTensor_ = sendCntExpertBuf_.GetWithOffset<int32_t>(counterCnt_, 0);
    Duplicate<int32_t>(slotIdxPerRankTensor_, 0, epWorldSize_);
    if (aivId_ == 0) { // 0核起始id 均为0
        return;
    }
    uint32_t groupCnt = Ceil(aivId_ * epWorldSizeAlign_, perGroupSizeAlign_ * 2);
    uint32_t copyNumPerGroup = perGroupSizeAlign_ * 2 / epWorldSizeAlign_;
    LocalTensor<int32_t> counterTmpTensor = topkIdsBuf_.Get<int32_t>();
    LocalTensor<int32_t> hitTmpTensor = sendCntRankBuf_.Get<int32_t>();
    LocalTensor<uint8_t> sharedTmpTensor = tempBuf_.Get<uint8_t>();
    DataCopyParams counterCopyParams = {1U, static_cast<uint16_t>(epWorldSizeAlign_),
                                        static_cast<uint16_t>(epWorldSizeAlign512_ - epWorldSizeAlign_), 0U};
    for (uint32_t i = 0; i < groupCnt; i++) {
        uint32_t copyNum = (i == groupCnt - 1) ? (aivId_ - copyNumPerGroup * i) : copyNumPerGroup;
        uint32_t gmOffset = i * copyNumPerGroup * epWorldSizeAlign512_ / sizeof(int32_t);
        counterCopyParams.blockCount = static_cast<uint16_t>(copyNum);
        DataCopyPad(counterTmpTensor, scaleupCounterGMTensor_[gmOffset], counterCopyParams, padParams_);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        const uint32_t shape[] = {copyNum, counterCnt_};
        ReduceSum<int32_t, AscendC::Pattern::Reduce::RA, true>(hitTmpTensor, counterTmpTensor, sharedTmpTensor, shape,
                                                               true);
        Add(slotIdxPerRankTensor_, slotIdxPerRankTensor_, hitTmpTensor, epWorldSize_);
        SyncFunc<AscendC::HardEvent::V_MTE2>();
    }
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::SetSrcTokenIdx(
    uint32_t topkOffset, uint32_t calCnt, uint32_t calCntAlign, uint32_t& sendLocalCnt,
    LocalTensor<int32_t>& cntLocalTensor)
{
    // 非 cache：按对端卡分组提取命中项
    uint64_t rsvdCnt = 0;
    LocalTensor<int32_t> internalIndexTensor = dstExpBuf_.Get<int32_t>();
    LocalTensor<int32_t> tempLocalTensor = topkIdsBuf_.Get<int32_t>();
    LocalTensor<uint32_t> gatherMaskTensor = maskBuf_.Get<uint32_t>();
    LocalTensor<uint8_t> compareMaskTensor = maskBuf_.Get<uint8_t>();
    DataCopyParams srcTokenCopyParams = {1U, 0U, 0U, 0U};

    SyncFunc<AscendC::HardEvent::S_V>(); // 需要等待上一轮本端命中判断结束
    CreateVecIndex(internalIndexTensor, static_cast<int32_t>(topkOffset), calCnt); // 生成元素索引（全局）
    Duplicate<int32_t>(tempLocalTensor, static_cast<int32_t>(axisK_), calCnt);
    Div(internalIndexTensor, internalIndexTensor, tempLocalTensor, calCnt);

    for (uint32_t rankId = 0; rankId < epWorldSize_; ++rankId) {
        // 调整处理顺序，先处理其他对端卡，本端卡最后处理，后续处理token可复用
        uint32_t dstRankId = rankId >= epRankId_ ? (rankId + 1) : rankId;
        dstRankId = (rankId == epWorldSize_ - 1) ? epRankId_ : dstRankId;
        CompareScalar(compareMaskTensor, dstRankIdTensor_, static_cast<int16_t>(dstRankId), AscendC::CMPMODE::EQ,
                      calCntAlign);
        if (rankId > 0) {
            SyncFunc<AscendC::HardEvent::MTE3_V>();
        }
        SyncFunc<AscendC::HardEvent::S_V>(); // 需要等待上一轮本端命中判断结束
        GatherMask(tempLocalTensor, internalIndexTensor, gatherMaskTensor, true, calCnt, {1, 1, 0, 0}, rsvdCnt);
        if (dstRankId == epRankId_) {
            sendLocalCnt = static_cast<uint32_t>(rsvdCnt);
        }
        if (rsvdCnt == 0) {
            continue;
        }

        SyncFunc<AscendC::HardEvent::V_S>();
        int32_t curSlot = cntLocalTensor.GetValue(dstRankId);
        int32_t curCnt = static_cast<int32_t>(rsvdCnt);
        uint64_t dstIndex = static_cast<uint64_t>(dstRankId) * dstSlotStride_ + 1 + curSlot;
        srcTokenCopyParams.blockLen = static_cast<uint32_t>(curCnt * sizeof(int32_t));
        SyncFunc<AscendC::HardEvent::V_MTE3>();
        DataCopyPad(dstSlotIdxGMTensor_[dstIndex], tempLocalTensor, srcTokenCopyParams);
        cntLocalTensor.SetValue(dstRankId, curSlot + curCnt);
        rsvdCnt = 0;
    }
    SyncFunc<AscendC::HardEvent::MTE3_V>(); // 下一轮等待这一轮做完
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::InitMetaScatter()
{
    uint32_t scatterBytes = MOE_EP_META_BATCH_MAX * sizeof(int32_t);
    uint32_t metaSetByteOffset = (topkCopyAlign_ + 2U * axisKAlign_) * sizeof(int32_t);
    rankValueTensor_ = metaScatterBuf_.GetWithOffset<int32_t>(MOE_EP_META_BATCH_MAX, 0U);
    tokenValueTensor_ = metaScatterBuf_.GetWithOffset<int32_t>(MOE_EP_META_BATCH_MAX, scatterBytes);
    rankOffsetTensor_ = metaScatterBuf_.GetWithOffset<uint32_t>(MOE_EP_META_BATCH_MAX, 2U * scatterBytes);
    tokenOffsetTensor_ = metaScatterBuf_.GetWithOffset<uint32_t>(MOE_EP_META_BATCH_MAX, 3U * scatterBytes);
    LocalTensor<int32_t> rankOffsetValueTensor = rankOffsetTensor_.template ReinterpretCast<int32_t>();
    LocalTensor<int32_t> tokenOffsetValueTensor = tokenOffsetTensor_.template ReinterpretCast<int32_t>();

    Duplicate<int32_t>(rankValueTensor_, static_cast<int32_t>(epRankId_), metaBatch_);
    CreateVecIndex<int32_t>(rankOffsetValueTensor, static_cast<int32_t>(0), metaBatch_);
    Muls<int32_t>(rankOffsetValueTensor, rankOffsetValueTensor, static_cast<int32_t>(metaSlotBytes_), metaBatch_);
    Adds<int32_t>(rankOffsetValueTensor, rankOffsetValueTensor, static_cast<int32_t>(metaSetByteOffset), metaBatch_);
    Adds<int32_t>(tokenOffsetValueTensor, rankOffsetValueTensor, static_cast<int32_t>(sizeof(int32_t)), metaBatch_);
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::ScatterMetaRouteInfo(uint32_t batchStartId,
                                                                                          uint32_t batchCnt)
{
    CreateVecIndex<int32_t>(tokenValueTensor_, static_cast<int32_t>(batchStartId), batchCnt);
    Scatter(metaLocalTensor_, rankValueTensor_, rankOffsetTensor_, 0U, batchCnt);
    Scatter(metaLocalTensor_, tokenValueTensor_, tokenOffsetTensor_, 0U, batchCnt);
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::CopyInMetaInfo(uint32_t startTokenId,
                                                                                    uint32_t batchCnt)
{
    LocalTensor<int32_t> metaCopyInTensor = metaSlotQueue_.AllocTensor<int32_t>();
    uint32_t topkOffset = startTokenId * axisK_;
    topkCopyParams_.blockCount = static_cast<uint16_t>(batchCnt);

    if constexpr (Std::IsSame<XType, fp8_e5m2_t>::value || Std::IsSame<XType, fp8_e4m3fn_t>::value) {
        scalesCopyParams_.blockCount = static_cast<uint16_t>(batchCnt);
        DataCopyPad(metaCopyInTensor.template ReinterpretCast<uint8_t>(),
                    scalesGMTensor_.template ReinterpretCast<uint8_t>()[startTokenId * scalesBytes_], scalesCopyParams_,
                    padParams_);
    }
    DataCopyPad(metaCopyInTensor[topkCopyAlign_], topkIdxGMTensor_[topkOffset], topkCopyParams_, padParams_);
    if constexpr (IsTopkWeights) {
        DataCopyPad(metaCopyInTensor[topkCopyAlign_ + axisKAlign_].template ReinterpretCast<float>(),
                    topkWeightsGMTensor_[topkOffset], topkCopyParams_, padParams_);
    }
    metaSlotQueue_.EnQue(metaCopyInTensor);
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::ProcessMetaSlot(uint32_t startId,
                                                                                     uint32_t tokenCnt,
                                                                                     uint32_t sendLocalCnt,
                                                                                     uint32_t& curLocalSlot)
{
    // meta 流水写 stash + 本端命中内联写本地窗 meta 区
    // srcTokenTensor: 非cache=SetSrcTokenIdx gather的本端token表(topkIdsBuf_), cache=cached表(topkIdsBuf_)
    if (tokenCnt == 0U) {
        return;
    }
    uint32_t localIdx = 0U;
    uint32_t tokenEndId = startId + tokenCnt;
    uint32_t firstBatchCnt = tokenCnt < metaBatch_ ? tokenCnt : metaBatch_;
    LocalTensor<int32_t> srcTokenTensor = topkIdsBuf_.Get<int32_t>();
    CopyInMetaInfo(startId, firstBatchCnt);

    for (uint32_t batchStartId = startId; batchStartId < tokenEndId; batchStartId += metaBatch_) {
        uint32_t batchCnt = tokenEndId - batchStartId;
        batchCnt = batchCnt < metaBatch_ ? batchCnt : metaBatch_;
        uint32_t batchEndId = batchStartId + batchCnt;
        metaLocalTensor_ = metaSlotQueue_.DeQue<int32_t>();
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        ScatterMetaRouteInfo(batchStartId, batchCnt);
        if (batchEndId < tokenEndId) {
            uint32_t nextBatchCnt = tokenEndId - batchEndId;
            nextBatchCnt = nextBatchCnt < metaBatch_ ? nextBatchCnt : metaBatch_;
            CopyInMetaInfo(batchEndId, nextBatchCnt);
        }

        SyncFunc<AscendC::HardEvent::V_MTE3>();
        DataCopyExtParams stashCopyParams = {1U, static_cast<uint32_t>(batchCnt * metaSlotBytes_), 0U, 0U, 0U};
        DataCopyPad(metaSlotGMTensor_[static_cast<uint64_t>(batchStartId) * metaSlotStride_], metaLocalTensor_,
                    stashCopyParams);

        for (; localIdx < sendLocalCnt; ++localIdx) { // 本地tokenId递增，只遍历当前batch内的本地命中项。
            uint32_t indexInBatch = static_cast<uint32_t>(srcTokenTensor.GetValue(localIdx)) - batchStartId;
            if (indexInBatch >= batchCnt) {
                break;
            }
            localMetaSlotGMTensor_.SetGlobalBuffer((
                __gm__ int32_t*)(localSlotWinAddr_ + static_cast<uint64_t>(curLocalSlot) * perSlotBytes_ + tokenSize_));
            DataCopyPad(localMetaSlotGMTensor_, metaLocalTensor_[indexInBatch * metaSlotStride_], metaSlotCopyParams_);
            curLocalSlot++;
        }
        metaSlotQueue_.FreeTensor<int32_t>(metaLocalTensor_);
    }
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::SendPhase()
{
    if (startTokenId_ >= axisBS_) {
        return;
    }
    GetSlotStartNum(); // 计算起始slot id
    InitMetaScatter();

    uint32_t groupCnt = Ceil(sendTokenNum_, perGroupTokenNum_);
    uint32_t tokenCnt = perGroupTokenNum_;
    uint32_t strideAlign = ALIGNED_LEN_256 / sizeof(int16_t);
    dstRankIdTensor_ = tempBuf_.Get<int16_t>();
    LocalTensor<int32_t> cntLocalTensor = numRecvPerRankBuf_.Get<int32_t>();
    DataCopyPadExtParams<int16_t> dstRankCopyPadParams{false, 0U, 0U, 0U};
    SyncFunc<AscendC::HardEvent::V_S>();
    uint32_t curLocalSlot = static_cast<uint32_t>(slotIdxPerRankTensor_.GetValue(epRankId_));
    uint32_t sendLocalCnt = 0;
    Copy<int32_t>(cntLocalTensor, slotIdxPerRankTensor_, counterCnt_);

    for (uint32_t group = 0; group < groupCnt; group++) {
        if (group == groupCnt - 1) {
            tokenCnt = sendTokenNum_ - group * perGroupTokenNum_;
        }
        uint32_t calCnt = tokenCnt * axisK_;
        uint32_t startId = startTokenId_ + group * perGroupTokenNum_;
        uint32_t topkOffset = startId * axisK_;
        uint32_t calCntAlign = Ceil(calCnt, strideAlign) * strideAlign;
        DataCopyExtParams dstRankCopyParams = {1U, static_cast<uint32_t>(calCnt * sizeof(int16_t)), 0U, 0U, 0U};
        DataCopyPad(dstRankIdTensor_, dstRankGMTensor_[topkOffset], dstRankCopyParams, dstRankCopyPadParams);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        SetSrcTokenIdx(topkOffset, calCnt, calCntAlign, sendLocalCnt, cntLocalTensor);
        SyncFunc<AscendC::HardEvent::V_S>();
        ProcessMetaSlot(startId, tokenCnt, sendLocalCnt, curLocalSlot);
        if (group < groupCnt - 1) {
            SyncFunc<AscendC::HardEvent::V_MTE2>();
        }
    }
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::GetLocalSlotStart(
    uint32_t sendCount, uint32_t maxBatchCnt, uint32_t alignStride, uint32_t& curLocalSlot,
    DataCopyExtParams& srcTokenIdxCopyParams)
{
    if (aivId_ == 0) {
        return;
    }

    uint64_t localRowStart = static_cast<uint64_t>(epRankId_) * dstSlotStride_ + 1;
    LocalTensor<int32_t> srcTokenTensor = topkIdsBuf_.Get<int32_t>();
    LocalTensor<uint32_t> gatherMaskTensor = maskBuf_.Get<uint32_t>();
    LocalTensor<uint8_t> ltMaskTensor = maskBuf_.Get<uint8_t>();
    uint32_t scanCount = startTokenId_ < sendCount ? startTokenId_ : sendCount;
    uint32_t groupCnt = Ceil(scanCount, maxBatchCnt);

    for (uint32_t group = 0; group < groupCnt; group++) {
        uint32_t startId = group * maxBatchCnt;
        uint32_t tokenCnt = (group == groupCnt - 1) ? (scanCount - startId) : maxBatchCnt;
        uint32_t tokenCntAlign = Ceil(tokenCnt, alignStride) * alignStride;
        uint64_t rsvdCnt = 0;
        srcTokenIdxCopyParams.blockLen = static_cast<uint32_t>(tokenCnt * sizeof(int32_t));
        DataCopyPad(srcTokenTensor, cachedSlotIdxGMTensor_[localRowStart + startId], srcTokenIdxCopyParams,
                    routePadParams_);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        CompareScalar(ltMaskTensor, srcTokenTensor, static_cast<int32_t>(startTokenId_), AscendC::CMPMODE::LT,
                      tokenCntAlign);
        GatherMask(srcTokenTensor, srcTokenTensor, gatherMaskTensor, true, tokenCnt, {1, 1, 0, 0}, rsvdCnt);
        if (rsvdCnt > 0) {
            curLocalSlot += static_cast<int32_t>(rsvdCnt);
        }
        SyncFunc<AscendC::HardEvent::V_MTE2>();
    }
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::SendPhaseCached()
{
    // cache 模式：直接以 cached_dst_slot_idx 输入表驱动发送（本端提取 + x/meta 写入）
    SplitToCore(axisBS_, aivNum_, startTokenId_, endTokenId_, sendTokenNum_); // 按token数量分核
    if (startTokenId_ >= axisBS_) {
        return;
    }

    uint32_t localSlotStart = 0;
    uint32_t maxBatchCnt = MAX_BATCH_SIZE / sizeof(int32_t);
    uint32_t alignStride = ALIGNED_LEN_256 / sizeof(int32_t);
    uint32_t groupCnt = Ceil(sendTokenNum_, maxBatchCnt);
    DataCopyExtParams srcTokenIdxCopyParams = {1U, 0U, 0U, 0U, 0U};
    LocalTensor<int32_t> srcTokenTensor = topkIdsBuf_.Get<int32_t>();
    LocalTensor<int32_t> sendCntTensor = maskBuf_.GetWithOffset<int32_t>(UB_STRIDE, 0);
    DataCopyParams sendCntHeaderCopyParams = {1U, static_cast<uint16_t>(sizeof(int32_t)), 0U, 0U};
    DataCopyPad(sendCntTensor, cachedSlotIdxGMTensor_[static_cast<uint64_t>(epRankId_) * dstSlotStride_],
                sendCntHeaderCopyParams, padParams_);
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    uint32_t sendCount = static_cast<uint32_t>(sendCntTensor.GetValue(0));
    InitMetaScatter();
    SyncFunc<AscendC::HardEvent::S_V>();
    GetLocalSlotStart(sendCount, maxBatchCnt, alignStride, localSlotStart, srcTokenIdxCopyParams);

    for (uint32_t group = 0; group < groupCnt; group++) {
        uint32_t startId = startTokenId_ + group * maxBatchCnt;
        uint32_t tokenCnt = (group == groupCnt - 1) ? (endTokenId_ - startId) : maxBatchCnt;
        uint32_t validCnt = (sendCount > localSlotStart) ? (sendCount - localSlotStart) : 0U;
        uint32_t segCnt = tokenCnt < validCnt ? tokenCnt : validCnt;
        if (segCnt > 0) {
            uint64_t srcTokenOffset = static_cast<uint64_t>(epRankId_) * dstSlotStride_ + 1 + localSlotStart;
            srcTokenIdxCopyParams.blockLen = static_cast<uint32_t>(segCnt * sizeof(int32_t));
            DataCopyPad(srcTokenTensor, cachedSlotIdxGMTensor_[srcTokenOffset], srcTokenIdxCopyParams, routePadParams_);
        }
        SyncFunc<AscendC::HardEvent::MTE2_S>();
        ProcessMetaSlot(startId, tokenCnt, segCnt, localSlotStart);
        SyncFunc<AscendC::HardEvent::S_MTE2>();
    }
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::SetLocalStatusAndBufferInitCached()
{
    tpipe_->InitBuffer(metaSlotQueue_, BUFFER_NUM, metaBatch_ * metaSlotBytes_);
    tpipe_->InitBuffer(metaScatterBuf_, MOE_EP_META_SCATTER_UB_BYTES);
    tpipe_->InitBuffer(topkIdsBuf_, MAX_BATCH_SIZE);
    tpipe_->InitBuffer(maskBuf_, 2 * maskBytesAlign_);
    tpipe_->InitBuffer(hcommBuf_, HCOMM_INIT_SIZE); // 通信初始化
    hcommTensor_ = hcommBuf_.Get<uint8_t>();
    hcomm_.Init(hcommTensor_, HCOMM_INIT_SIZE);

    if (aivId_ == 0) {
        GM_ADDR localStateAddr = GetWinAddrByRankId(mc2Context_, epRankId_, cntWinStateOffset_) + dstRankStateOffset_;
        LocalTensor<int32_t> statusTensor = maskBuf_.GetWithOffset<int32_t>(UB_STRIDE, 0);
        GlobalTensor<int32_t> notifyGMTensor;
        GlobalTensor<int32_t> statusGMTensor;
        notifyGMTensor.SetGlobalBuffer((__gm__ int32_t*)localStateAddr);
        statusGMTensor.SetGlobalBuffer((__gm__ int32_t*)cachedStateAddr_);
        Duplicate<int32_t>(statusTensor, 1, UB_STRIDE);
        SyncFunc<AscendC::HardEvent::V_MTE3>();
        DataCopy(statusGMTensor, statusTensor, UB_STRIDE);
        DataCopy(notifyGMTensor, statusTensor, UB_STRIDE);
    }
    SyncAll<true>();
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::CommunicationCached()
{
    SplitToCore(epWorldSize_, aivNum_, startRankId_, endRankId_, rankNumPerCore_); // 按卡分核
    coreAssigned_ = GetCoreAssignment(copyRankStart_, copyRankNum_, channelIndex_, coreIndexInGroup_, groupSize_);
    if (!coreAssigned_) {
        return;
    }

    for (uint32_t dstRankId = copyRankStart_; dstRankId < copyRankStart_ + copyRankNum_; ++dstRankId) {
        if ((channelIndex_ != 0) || (dstRankId == epRankId_)) {
            continue;
        }
        uint64_t commHandle = GetCommHandle(mc2Context_, dstRankId, channelIndex_);
        GM_ADDR notifyAddr = GetWinAddrByRankId(mc2Context_, dstRankId, cntWinStateOffset_) + dstRankStateOffset_;
        hcomm_.WriteNbi<true, PIPE_S, PIPE_MTE3, DATA_CFG>(commHandle, notifyAddr, cachedStateAddr_, WIN_ADDR_ALIGN);
    }
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::WaitStatusCached()
{
    if (aivId_ != aivNum_ - 1) {
        return;
    }

    uint32_t tmpOffset = epWorldSize_ * UB_ALIGN;
    uint32_t mask = 1;
    int32_t sumOfFlag = -1;
    int32_t commpareFlag = static_cast<int32_t>(epWorldSize_);
    LocalTensor<int32_t> waitStatusTensor = topkIdsBuf_.GetWithOffset<int32_t>(epWorldSize_ * UB_STRIDE, 0);
    LocalTensor<int32_t> sumStatusTensor = topkIdsBuf_.GetWithOffset<int32_t>(UB_STRIDE, tmpOffset);
    LocalTensor<float> tempFp32 = topkIdsBuf_.GetWithOffset<float>(epWorldSize_ * UB_STRIDE, tmpOffset + UB_ALIGN);
    LocalTensor<float> waitStatusTensorFp32 = waitStatusTensor.template ReinterpretCast<float>();
    LocalTensor<float> sumStatusTensorFp32 = sumStatusTensor.template ReinterpretCast<float>();

    while (sumOfFlag != commpareFlag) { // 状态位check
        DataCopy(waitStatusTensor, recvCounterGMTensor_, statusCopyParams_);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        ReduceSum(sumStatusTensorFp32, waitStatusTensorFp32, tempFp32, mask, epWorldSize_, 1);
        SyncFunc<AscendC::HardEvent::V_S>();
        sumOfFlag = sumStatusTensor.GetValue(0);
    }
    // status clear
    Duplicate<int32_t>(waitStatusTensor, 0, epWorldSize_ * UB_STRIDE);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(recvCounterGMTensor_, waitStatusTensor, clearStatusCopyParams_);
    diagWriter_.RunPosRecord(MOE_EP_DISPATCH_RUN_POS_COUNT_READY);
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::SetStatus()
{
    if (aivId_ >= groupSize_) {
        return;
    }

    int32_t notifyVal = dispatchNotifyCount_ / groupSize_;
    if (aivId_ < (dispatchNotifyCount_ % groupSize_)) {
        notifyVal++;
    }
    LocalTensor<int32_t> statusTensor = maskBuf_.Get<int32_t>();
    GlobalTensor<int32_t> srcStatusGMTensor;
    srcStatusGMTensor.SetGlobalBuffer((__gm__ int32_t*)(payloadStashStateWinAddr_ + aivId_ * WIN_ADDR_ALIGN));
    Duplicate<int32_t>(statusTensor, notifyVal, UB_STRIDE);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(srcStatusGMTensor, statusTensor, UB_STRIDE);
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::CommBufferInit()
{
    PipeBarrier<PIPE_ALL>();
    tpipe_->Reset();
    // 通信初始化
    uint32_t hcommWriteWqeCnt = Ceil(HCOMM_SQE_BYTES + SGE_PER_SQE * HCOMM_SGE_BYTES, HCOMM_WQE_BYTES);
    hcommBatchBufferBytes_ = HCOMM_BATCH_CAPACITY * hcommWriteWqeCnt * HCOMM_WQE_BYTES;
    uint32_t hcommBatchSize = hcommBatchBufferBytes_ / sizeof(uint8_t);
    tpipe_->InitBuffer(hcommBuf_, HCOMM_INIT_SIZE + hcommBatchBufferBytes_);
    hcommTensor_ = hcommBuf_.GetWithOffset<uint8_t>(HCOMM_INIT_SIZE, 0);
    hcommBatchTensor_ = hcommBuf_.GetWithOffset<uint8_t>(hcommBatchSize, HCOMM_INIT_SIZE);
    hcomm_.Init(hcommTensor_, HCOMM_INIT_SIZE);
    Duplicate<uint8_t>(hcommBatchTensor_, 0, hcommBatchSize);

    // ub buffer init
    tpipe_->InitBuffer(sendCntRankBuf_, epWorldSize_ * UB_ALIGN);
    tpipe_->InitBuffer(tempBuf_, MAX_BATCH_SIZE);
    tpipe_->InitBuffer(maskBuf_, UB_ALIGN);
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::WriteToRemoteWindow()
{
    CommBufferInit();
    if (!coreAssigned_) {
        return;
    }

    payloadStashStateWinAddr_ += channelIndex_ * WIN_ADDR_ALIGN;
    LocalTensor<int32_t> sendNumPerRankTensor = sendCntRankBuf_.Get<int32_t>();
    LocalTensor<int32_t> statusTensor = maskBuf_.Get<int32_t>();
    if constexpr (!IsCached) {
        DataCopy(sendNumPerRankTensor, sendCntPerRankGMTensor_, counterCnt_);
    } else {
        uint64_t gmOffset = static_cast<uint64_t>(copyRankStart_) * dstSlotStride_;
        DataCopyExtParams cntCopyParams = {static_cast<uint16_t>(copyRankNum_), static_cast<uint32_t>(sizeof(int32_t)),
                                           static_cast<int64_t>(axisBS_ * sizeof(int32_t)), 0U, 0U};
        DataCopyPad(sendNumPerRankTensor, cachedSlotIdxGMTensor_[gmOffset], cntCopyParams, routePadParams_);
    }
    SyncFunc<AscendC::HardEvent::MTE2_S>();

    for (uint32_t dstRankId = copyRankStart_; dstRankId < copyRankStart_ + copyRankNum_; ++dstRankId) {
        GM_ADDR notifyAddr = GetWinAddrByRankId(mc2Context_, dstRankId, slotWinStateOffset_) + dstStateWinOffset_;
        notifyAddr += channelIndex_ * WIN_ADDR_ALIGN;
        if (unlikely(dstRankId == epRankId_)) {
            GlobalTensor<int32_t> srcStatusGMTensor;
            GlobalTensor<int32_t> dstStatusGMTensor;
            srcStatusGMTensor.SetGlobalBuffer((__gm__ int32_t*)payloadStashStateWinAddr_);
            dstStatusGMTensor.SetGlobalBuffer((__gm__ int32_t*)notifyAddr);
            DataCopy(statusTensor, srcStatusGMTensor, UB_STRIDE);
            SyncFunc<AscendC::HardEvent::MTE2_MTE3>();
            DataCopy(dstStatusGMTensor, statusTensor, UB_STRIDE);
            continue;
        }

        uint32_t sendTokenNum = 0;
        if constexpr (!IsCached) {
            sendTokenNum = sendNumPerRankTensor.GetValue(dstRankId);
        } else {
            sendTokenNum = sendNumPerRankTensor.GetValue((dstRankId - copyRankStart_) * UB_STRIDE);
        }
        uint64_t commHandle = GetCommHandle(mc2Context_, dstRankId, channelIndex_);
        uint32_t cntPerChannel = Ceil(sendTokenNum, groupSize_);
        uint32_t tokenStart = coreIndexInGroup_ * cntPerChannel;
        if (sendTokenNum == 0 || tokenStart >= sendTokenNum) {
            hcomm_.WriteNbi<true, PIPE_S, PIPE_MTE3, DATA_CFG>(commHandle, notifyAddr, payloadStashStateWinAddr_,
                                                               WIN_ADDR_ALIGN);
            continue;
        }
        uint32_t tokenNum = (tokenStart + cntPerChannel > sendTokenNum) ? (sendTokenNum - tokenStart) : cntPerChannel;
        SendTokenByChannel(tokenStart, tokenNum, dstRankId, commHandle, notifyAddr);
    }
    DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(topkIdxGMTensor_);
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::SendTokenByChannel(
    uint32_t tokenStart, uint32_t tokenNum, uint32_t dstRankId, uint64_t commHandle, GM_ADDR notifyAddr)
{
    uint32_t processed = 0;
    uint32_t commCnt = 0;
    uint32_t maxBatchCnt = MAX_BATCH_SIZE / sizeof(int32_t);
    BufDesc localDescs[SGE_PER_SQE] = {};
    for (uint32_t i = 0; i < SGE_PER_SQE; i += SGE_PER_SLOT) { // srcLen 预填
        localDescs[i].len = tokenSize_;
        localDescs[i + 1].len = metaSlotBytes_;
    }
    DataCopyExtParams srcTokenCopyParams = {1U, 0U, 0U, 0U, 0U};
    LocalTensor<int32_t> srcTokenIdxTensor = tempBuf_.Get<int32_t>();
    GlobalTensor<int32_t> srcTokenIdxGMTensor;
    srcTokenIdxGMTensor.SetGlobalBuffer(
        (__gm__ int32_t*)(sendSrcTokenIdxAddr_ +
                          (static_cast<uint64_t>(dstRankId) * dstSlotStride_ + 1) * sizeof(int32_t)));
    GM_ADDR remoteWinBaseAddr = GetWinAddrByRankId(mc2Context_, dstRankId, winDataOffset_) + dstRankWinOffset_ +
                                static_cast<uint64_t>(tokenStart) * perSlotBytes_;
    SyncFunc<AscendC::HardEvent::V_S>(); // hcommBatchTensor_ init

    auto batchHandle = hcomm_.MakeBatchHandle(commHandle, hcommBatchTensor_, hcommBatchBufferBytes_, remoteWinBaseAddr);
    GM_ADDR remoteWinAddr = remoteWinBaseAddr;
    while (processed < tokenNum) {
        uint32_t batchCnt = (tokenNum - processed > maxBatchCnt) ? maxBatchCnt : tokenNum - processed;
        srcTokenCopyParams.blockLen = static_cast<uint32_t>(batchCnt * sizeof(int32_t));
        DataCopyPad(srcTokenIdxTensor, srcTokenIdxGMTensor[tokenStart + processed], srcTokenCopyParams,
                    routePadParams_);
        SyncFunc<AscendC::HardEvent::MTE2_S>();
        uint32_t batchGroup = Ceil(batchCnt, SLOT_NUM_PER_SQE);
        uint32_t srcIdx = 0;
        uint32_t slotNum = SLOT_NUM_PER_SQE;
        uint32_t sgePerSqe = SGE_PER_SQE;
        for (uint32_t i = 0; i < batchGroup; i++) {
            if (i == batchGroup - 1) {
                slotNum = batchCnt - i * SLOT_NUM_PER_SQE;
                sgePerSqe = slotNum * SGE_PER_SLOT;
            }
            for (uint32_t j = 0; j < sgePerSqe; j += SGE_PER_SLOT) {
                uint64_t srcTokenId = static_cast<uint64_t>(srcTokenIdxTensor.GetValue(srcIdx));
                GM_ADDR metaStashAddr = payloadStashWinAddr_ + srcTokenId * metaSlotBytes_;
                localDescs[j].addr = xAddr_ + srcTokenId * tokenSize_;
                localDescs[j + 1].addr = metaStashAddr;
                srcIdx++;
            }
            // 最后一批 token 的末笔写生成 CQE，epilogue 末尾等待发送完成
            if (processed + batchCnt == tokenNum && i + 1U == batchGroup) {
                hcomm_.WriteNbi<LAST_DATA_CFG>(batchHandle, remoteWinAddr, localDescs, sgePerSqe);
            } else {
                hcomm_.WriteNbi<DATA_CFG>(batchHandle, remoteWinAddr, localDescs, sgePerSqe);
            }
            commCnt++;
            if (commCnt == HCOMM_BATCH_CAPACITY) {
                hcomm_.BatchCommit(batchHandle);
                commCnt = 0;
            }
            remoteWinAddr += slotNum * perSlotBytes_;
        }
        processed += batchCnt;
        if (processed < tokenNum) {
            SyncFunc<AscendC::HardEvent::S_MTE2>();
        }
    }
    localDescs[0].addr = payloadStashStateWinAddr_;
    localDescs[0].len = WIN_ADDR_ALIGN;
    hcomm_.WriteNbi<NOTIFY_CFG>(batchHandle, notifyAddr, localDescs, 1);
    hcomm_.BatchCommit(batchHandle);
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::Process()
{
    if ASCEND_IS_AIV { // 全aiv处理
        if constexpr (!IsCached) {
            BufferInit();
            CalSendCnt();
            SyncAll<true>();
            Communication();
            SendPhase();
            SetStatus();
            GetRecvCount();
        } else {
            SetLocalStatusAndBufferInitCached();
            CommunicationCached();
            SendPhaseCached();
            SetStatus();
            WaitStatusCached();
        }
        SyncAll<true>();
        WriteToRemoteWindow();
        diagWriter_.RunPosRecord(MOE_EP_DISPATCH_RUN_POS_URMA_REQUESTS_ISSUE_DONE);
    }
}

#endif

} // namespace MoeEpDispatchImpl

#endif // MOE_EP_DISPATCH_H
