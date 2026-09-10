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
constexpr uint32_t HISTOGRAM_SCOPE_SIZE = 256U; // 直方图uint8, 统计范围0-255
constexpr uint8_t BUFFER_NUM = 2;
constexpr uint32_t NOTIFY_VAL_SHIFT = 32U;       // notify 高32位为 sendCnt, 低32位为 state(=1)
constexpr uint32_t MAX_BATCH_SIZE = 160 * 1024U; // srcTokenIdx 160KB
constexpr uint32_t HCOMM_BATCH_CAPACITY = 256U;
constexpr uint32_t HCOMM_WRITE_TASK_BYTES = 64U;
constexpr uint32_t HCOMM_BATCH_BUFFER_BYTES = 2 * HCOMM_BATCH_CAPACITY * HCOMM_WRITE_TASK_BYTES;
constexpr struct UrmaWqeEntry DATA_CFG = {.odr = 5, .fence = 1, .se = 0, .cqe = 0, .inlineEn = 0};
constexpr struct UrmaWqeEntry NOTIFY_CFG = {.odr = 6, .fence = 1, .se = 0, .cqe = 0, .inlineEn = 0};

template <TemplateMoeEpDispatchTypeClass>
class MoeEpDispatch {
public:
    __aicore__ inline MoeEpDispatch(){};
    __aicore__ inline void Init(GM_ADDR context, GM_ADDR x, GM_ADDR topkIdx, GM_ADDR topkWeights, GM_ADDR scales,
                                GM_ADDR cachedSlotIdx, GM_ADDR numRecvPerRank, GM_ADDR numRecvPerExpert,
                                GM_ADDR dstBufferSlotIdx, GM_ADDR workspaceGM, GM_ADDR tilingGM, TPipe *pipe,
                                const MoeEpDispatchTilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void BufferInit();
    __aicore__ inline void ResetCounters();
    __aicore__ inline void CalSendCntHistograms(LocalTensor<int16_t> dstTensor, LocalTensor<int16_t> &tempTensor,
                                                LocalTensor<uint32_t> sendCntTensor, uint32_t calCnt,
                                                uint32_t scopeSize);
    __aicore__ inline void DedupAndSendDirect(LocalTensor<int16_t> expertIdsTensor, uint32_t calCnt,
                                              uint32_t calCntAlign, uint32_t tmpOffset);
    __aicore__ inline void CalSendCntPerRank(uint32_t calCnt, uint32_t calCntAlign, uint32_t tmpOffset);
    __aicore__ inline void CalSendCntPerExpert(LocalTensor<int16_t> expertIdsTensor, uint32_t calCnt,
                                               uint32_t calCntAlign, uint32_t tmpOffset);
    __aicore__ inline void CalSendCnt();
    __aicore__ inline void Communication();
    __aicore__ inline void GetRecvCount();
    __aicore__ inline void SetRecvNumPerExpert();
    __aicore__ inline void SetRecvNumPerRank(LocalTensor<int32_t> recvTmpTensor);
    __aicore__ inline void WriteToRemoteWindow();
    __aicore__ inline void CommBufferInit();
    __aicore__ inline void SendTokenByChannel(uint32_t tokenStart, uint32_t tokenNum, uint32_t dstRankId,
                                              uint64_t commHandle, GM_ADDR notifyAddr);
    __aicore__ inline void SetSrcTokenIdxAndDstSlot(uint32_t topkOffset, uint32_t calCnt, uint32_t calCntAlign,
                                                    uint32_t &sendLocalCnt, LocalTensor<int32_t> &cntLocalTensor);
    __aicore__ inline void ProcessMetaSlot(uint32_t startId, uint32_t tokenCnt, uint32_t sendLocalCnt,
                                           uint32_t &topkOffset, uint32_t &curLocalSlot);
    __aicore__ inline void SendPhase();
    __aicore__ inline void LocalCopyPhase(uint32_t slotStart, uint32_t slotCnt);
    __aicore__ inline void GetSlotStartNum();
    __aicore__ inline void CopyInMetaInfo(uint32_t tokenId, uint32_t topkOffset);
    __aicore__ inline void SetStatus();
    __aicore__ inline void SplitToCore(uint32_t curSendCnt, uint32_t curUseAivNum, uint32_t &startId, uint32_t &endId,
                                       uint32_t &sendNum);
    __aicore__ inline bool GetCoreAssignment(uint32_t &copyRankStart, uint32_t &copyRankNum, uint32_t &channelIndex,
                                             uint32_t &coreIndexInGroup, uint32_t &groupSize);

    TPipe *tpipe_{nullptr};
    __gm__ Mc2Aclnn::MoeCommContext *mc2Context_{nullptr};
    AscendC::Hcomm<COMM_PROTOCOL_UBC_CTP> hcomm_; // 通信上下文
    MoeEpExceptionDump::MoeEpCoreDiagWriter diagWriter_;

    GlobalTensor<XType> xGMTensor_;
    GlobalTensor<int32_t> topkIdxGMTensor_;
    GlobalTensor<float> topkWeightsGMTensor_;
    GlobalTensor<int32_t> dstSlotIdxGMTensor_;
    GlobalTensor<int32_t> numRecvPerRankGMTensor_;
    GlobalTensor<int64_t> numRecvPerExpertGMTensor_;
    GlobalTensor<int32_t> cachedSlotIdxGMTensor_;
    GlobalTensor<int32_t> scaleupCounterGMTensor_;
    GlobalTensor<int32_t> recvCounterGMTensor_;
    GlobalTensor<int32_t> sendCntPerRankGMTensor_;
    GlobalTensor<int32_t> sendCntPerExpertGMTensor_;
    GlobalTensor<int32_t> srcTokenIdxGMTensor_;
    GlobalTensor<int16_t> dstRankGMTensor_;
    GlobalTensor<int32_t> metaSlotGMTensor_;
    GlobalTensor<int32_t> localMetaSlotGMTensor_;
    GlobalTensor<ScalesType> scalesGMTensor_;

    LocalTensor<int32_t> dstSlotIdxTensor_;
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

    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> metaSlotQueue_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> tokenSlotQueue_;
    TBuf<> topkIdsBuf_;
    TBuf<> tempBuf_;
    TBuf<> dstExpBuf_;
    TBuf<> maskBuf_;
    TBuf<> hcommBuf_; // 通信
    TBuf<> sendCntRankBuf_;
    TBuf<> sendCntExpertBuf_;
    TBuf<> numRecvPerRankBuf_;
    TBuf<> numRecvPerExpertBuf_;

    GM_ADDR workspaceGM_{nullptr};
    GM_ADDR hostPinnedCounterAddrGM_{nullptr};
    GM_ADDR xAddr_{nullptr};
    GM_ADDR scaleupCounterAddr_{nullptr};
    GM_ADDR sendCntWorkspaceAddr_{nullptr};
    GM_ADDR dstRankInfoAddr_{nullptr};
    GM_ADDR srcTokenTableAddr_{nullptr};
    GM_ADDR localCntStateWinAddr_{nullptr};
    GM_ADDR localSlotStateWinAddr_{nullptr};
    GM_ADDR localSlotWinAddr_{nullptr};
    GM_ADDR payloadStashWinAddr_{nullptr};
    GM_ADDR payloadStashStateWinAddr_{nullptr};

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
    uint32_t hAlignSize_{0};
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
    uint32_t metaSlotBytesAlign512_{0};
    uint32_t topkCopyAlign_{0};
    uint32_t srcTokenListBytes_{0};
    uint32_t metaSlotStride32_{0};
    uint32_t tokenCntAlign_{0};
    uint64_t dstRankInfoOffset_{0};
    uint64_t srcTokenTableOffset_{0};
    uint64_t cntWinStateOffset_{0};
    uint64_t slotWinStateOffset_{0};
    uint64_t payloadWinStateOffset_{0};
    uint64_t winDataOffset_{0};
    uint64_t payloadStashWinOffset_{0};
    uint64_t dstRankWinOffset_{0};
    uint64_t dstStateWinOffset_{0};
    bool coreAssigned_{false};

    DataCopyParams statusCopyParams_;
    DataCopyParams clearStatusCopyParams_;
    DataCopyParams sendPerExpertParams_;
    DataCopyParams sendPerRankParams_;
    DataCopyParams metaSlotCopyParams_; // stash 元数据槽整槽拷贝参数
    DataCopyParams xCopyParams_;
    DataCopyParams topkCopyParams_;
    DataCopyParams scalesCopyParams_;
    DataCopyPadParams padParams_;
};

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::Init(
    GM_ADDR context, GM_ADDR x, GM_ADDR topkIdx, GM_ADDR topkWeights, GM_ADDR scales, GM_ADDR cachedSlotIdx,
    GM_ADDR numRecvPerRank, GM_ADDR numRecvPerExpert, GM_ADDR dstBufferSlotIdx, GM_ADDR workspaceGM, GM_ADDR tilingGM,
    TPipe *pipe, const MoeEpDispatchTilingData *tilingData)
{
    tpipe_ = pipe;
    aivId_ = GetBlockIdx();
    workspaceGM_ = workspaceGM;
    mc2Context_ = reinterpret_cast<__gm__ Mc2Aclnn::MoeCommContext *>(context);
    epRankId_ = mc2Context_->epRankId;
    channelsPerRank_ = mc2Context_->channelsPerRank;
    constexpr size_t metadataOffset =
        offsetof(MoeEpDispatchTilingData, moeEpDispatchInfo) + offsetof(MoeEpDispatchInfo, dumpMetadata);
    MoeEpExceptionDump::WriteMetadata(context, tilingGM + metadataOffset);
    diagWriter_.Init(context, MOE_EP_CORE_DIAG_DISPATCH, tpipe_);

    const auto &info = tilingData->moeEpDispatchInfo;
    xAddr_ = x;
    metaSlotBytes_ = info.metaSlotBytes;
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
    srcTokenListBytes_ = info.workspace.srcTokenListBytes;
    dstRankInfoOffset_ = info.workspace.dstRankInfoOffset;
    srcTokenTableOffset_ = info.workspace.srcTokenTableOffset;
    cntWinStateOffset_ = info.window.cntWinStateOffset;
    slotWinStateOffset_ = info.window.slotWinStateOffset;
    payloadWinStateOffset_ = info.window.payloadWinStateOffset;
    winDataOffset_ = info.window.winDataOffset;
    payloadStashWinOffset_ = info.window.payloadStashWinOffset;
    dispatchNotifyCount_ = info.dispatchNotifyCount;
    hostPinnedCounterAddrGM_ = reinterpret_cast<GM_ADDR>(info.hostPinnedCounterAddr);

    hAlignSize_ = Ceil(axisH_ * sizeof(XType), UB_ALIGN) * UB_ALIGN;
    kAlignSize_ = Ceil(axisK_ * TOPK_INFO_SIZE, UB_ALIGN) * UB_ALIGN;
    axisKAlign_ = kAlignSize_ / TOPK_INFO_SIZE;
    metaSlotBytesAlign512_ = Ceil(metaSlotBytes_, WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN; // stash 紧凑步长
    metaSlotStride32_ = metaSlotBytesAlign512_ / sizeof(int32_t);
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
    dstRankWinOffset_ = static_cast<uint64_t>(epRankId_) * axisMaxBS_ * perSlotBytes_; // 目标窗口地址偏移
    tokenCntAlign_ = srcTokenListBytes_ / sizeof(int32_t);
    if (channelsPerRank_ == 0 || (epWorldSize_ > 0 && channelsPerRank_ > Mc2Aclnn::HCCL_MAX_RANK_SIZE / epWorldSize_)) {
        channelsPerRank_ = 1;
    }
    dstStateWinOffset_ = static_cast<uint64_t>(epRankId_) * dispatchNotifyCount_ * WIN_ADDR_ALIGN; // 目标状态地址偏移

    xGMTensor_.SetGlobalBuffer((__gm__ XType *)x);
    topkIdxGMTensor_.SetGlobalBuffer((__gm__ int32_t *)topkIdx);
    if constexpr (IsTopkWeights) {
        topkWeightsGMTensor_.SetGlobalBuffer((__gm__ float *)topkWeights);
    }
    if constexpr (Std::IsSame<XType, fp8_e5m2_t>::value || Std::IsSame<XType, fp8_e4m3fn_t>::value) {
        scalesGMTensor_.SetGlobalBuffer((__gm__ ScalesType *)scales);
        topkCopyAlign_ = Ceil(scalesBytes_, UB_ALIGN) * UB_ALIGN / sizeof(int32_t); // meta 槽内 topk 的 int32 偏移
    }
    if constexpr (IsCached) {
        cachedSlotIdxGMTensor_.SetGlobalBuffer((__gm__ int32_t *)cachedSlotIdx);
    }
    numRecvPerRankGMTensor_.SetGlobalBuffer((__gm__ int32_t *)numRecvPerRank);
    numRecvPerExpertGMTensor_.SetGlobalBuffer((__gm__ int64_t *)numRecvPerExpert);
    dstSlotIdxGMTensor_.SetGlobalBuffer((__gm__ int32_t *)dstBufferSlotIdx);

    statusCopyParams_ = {static_cast<uint16_t>(epWorldSize_), 1U, static_cast<uint16_t>(STATE_STRIDE), 0U};
    clearStatusCopyParams_ = {static_cast<uint16_t>(epWorldSize_), 1U, 0U, static_cast<uint16_t>(STATE_STRIDE)};
    sendPerExpertParams_ = {1U, static_cast<uint16_t>(moeExpertNum_ * sizeof(int32_t)), 0U, 0U};
    sendPerRankParams_ = {1U, static_cast<uint16_t>(epWorldSize_ * sizeof(int32_t)), 0U, 0U};
    metaSlotCopyParams_ = {1U, static_cast<uint16_t>(metaSlotBytes_), 0U, 0U};
    xCopyParams_ = {1U, static_cast<uint16_t>(axisH_ * sizeof(XType)), 0U, 0U};
    topkCopyParams_ = {1U, static_cast<uint16_t>(axisK_ * TOPK_INFO_SIZE), 0U, 0U};
    scalesCopyParams_ = {1U, static_cast<uint16_t>(scalesBytes_), 0U, 0U};
    padParams_ = {true, 0, 0, 0};

    scaleupCounterAddr_ = workspaceGM;
    sendCntWorkspaceAddr_ = workspaceGM + static_cast<uint64_t>(aivNum_) * epWorldSizeAlign512_;
    dstRankInfoAddr_ = workspaceGM + dstRankInfoOffset_;
    srcTokenTableAddr_ = workspaceGM + srcTokenTableOffset_;
    localCntStateWinAddr_ = GetWinAddrByRankId(mc2Context_, epRankId_, cntWinStateOffset_);
    localSlotStateWinAddr_ = GetWinAddrByRankId(mc2Context_, epRankId_, slotWinStateOffset_);
    payloadStashStateWinAddr_ = GetWinAddrByRankId(mc2Context_, epRankId_, payloadWinStateOffset_);
    localSlotWinAddr_ = GetWinAddrByRankId(mc2Context_, epRankId_, winDataOffset_) + dstRankWinOffset_;
    payloadStashWinAddr_ = GetWinAddrByRankId(mc2Context_, epRankId_, payloadStashWinOffset_);
    scaleupCounterGMTensor_.SetGlobalBuffer((__gm__ int32_t *)scaleupCounterAddr_);
    recvCounterGMTensor_.SetGlobalBuffer((__gm__ int32_t *)localCntStateWinAddr_);
    metaSlotGMTensor_.SetGlobalBuffer((__gm__ int32_t *)payloadStashWinAddr_);
    sendCntPerRankGMTensor_.SetGlobalBuffer((__gm__ int32_t *)sendCntWorkspaceAddr_);
    sendCntPerExpertGMTensor_.SetGlobalBuffer((__gm__ int32_t *)(sendCntWorkspaceAddr_ + epWorldSizeAlign512_));
    dstRankGMTensor_.SetGlobalBuffer((__gm__ int16_t *)dstRankInfoAddr_);
    srcTokenIdxGMTensor_.SetGlobalBuffer((__gm__ int32_t *)srcTokenTableAddr_);
    diagWriter_.RunPosRecord(MOE_EP_DISPATCH_RUN_POS_INIT_DONE);
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::SplitToCore(uint32_t curSendCnt,
                                                                                 uint32_t curUseAivNum,
                                                                                 uint32_t &startId, uint32_t &endId,
                                                                                 uint32_t &sendNum)
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
__aicore__ inline bool MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::GetCoreAssignment(uint32_t &copyRankStart,
                                                                                       uint32_t &copyRankNum,
                                                                                       uint32_t &channelIndex,
                                                                                       uint32_t &coreIndexInGroup,
                                                                                       uint32_t &groupSize)
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
    tpipe_->InitBuffer(metaSlotQueue_, BUFFER_NUM, metaSlotBytes_);
    tpipe_->InitBuffer(tokenSlotQueue_, BUFFER_NUM, hAlignSize_);
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
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::CalSendCntHistograms(
    LocalTensor<int16_t> dstTensor, LocalTensor<int16_t> &tempTensor, LocalTensor<uint32_t> sendCntTensor,
    uint32_t calCnt, uint32_t scopeSize)
{
    uint64_t rsvdCnt = 0;
    LocalTensor<uint16_t> gatherMaskTensor = maskBuf_.Get<uint16_t>();
    LocalTensor<uint8_t> histoSrcTensor = sendCntTensor.template ReinterpretCast<uint8_t>();
    LocalTensor<uint16_t> histoDstTensor = tempTensor.template ReinterpretCast<uint16_t>();
    GatherMask(tempTensor, dstTensor, gatherMaskTensor, true, calCnt, {1, 1, 0, 0}, rsvdCnt);
    SyncFunc<AscendC::HardEvent::V_S>();
    if (rsvdCnt == 0) {
        Duplicate<uint32_t>(sendCntTensor, 0, scopeSize);
        return;
    }
    Cast(histoSrcTensor, tempTensor, RoundMode::CAST_NONE, rsvdCnt);
    GetExpertFreq(histoDstTensor, histoSrcTensor, rsvdCnt);
    Cast(sendCntTensor, histoDstTensor, RoundMode::CAST_NONE, scopeSize);
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::DedupAndSendDirect(
    LocalTensor<int16_t> expertIdsTensor, uint32_t calCnt, uint32_t calCntAlign, uint32_t tmpOffset)
{
    uint32_t maskCnt = Ceil(calCntAlign, BITS_PER_BYTE);
    LocalTensor<int16_t> internalIndexTensor = dstExpBuf_.GetWithOffset<int16_t>(calCntAlign, tmpOffset);
    LocalTensor<int16_t> tempTensorInt16 = tempBuf_.Get<int16_t>();
    LocalTensor<int16_t> subTensorInt16 = topkIdsBuf_.GetWithOffset<int16_t>(calCntAlign, 0);
    LocalTensor<int32_t> gatherIdxTensor = topkIdsBuf_.Get<int32_t>();
    LocalTensor<uint8_t> selectMaskTensor = maskBuf_.GetWithOffset<uint8_t>(maskBytesAlign_, 0);
    LocalTensor<uint8_t> equalMaskTensor = maskBuf_.GetWithOffset<uint8_t>(maskBytesAlign_, maskBytesAlign_);
    LocalTensor<uint8_t> compareMaskTensor = maskBuf_.GetWithOffset<uint8_t>(maskBytesAlign_, 2 * maskBytesAlign_);

    Duplicate<int16_t>(internalIndexTensor, static_cast<int16_t>(moeExpertNumPerRank_), calCnt);
    SyncFunc<AscendC::HardEvent::MTE3_V>(); // 等待上一轮 MTE3 结束
    Div(dstRankTensor_, expertIdsTensor, internalIndexTensor, calCnt);

    if constexpr (IsCached) { // cache下复用 dstSlotIdx
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        CompareScalar(selectMaskTensor, dstSlotIdxTensor_, static_cast<int32_t>(0), AscendC::CMPMODE::GE, calCntAlign);
        return;
    }

    CompareScalar(selectMaskTensor, expertIdsTensor, static_cast<int16_t>(0), AscendC::CMPMODE::GE, calCntAlign);
    Select(dstRankTensor_, selectMaskTensor, dstRankTensor_, static_cast<int16_t>(-1),
           AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, calCnt); // 筛选无效expert id，避免数据污染
    Not(selectMaskTensor, selectMaskTensor, maskCnt);

    Duplicate<int16_t>(subTensorInt16, static_cast<int16_t>(axisK_), calCnt);
    CreateVecIndex(internalIndexTensor, static_cast<int16_t>(0), calCnt);
    Div(tempTensorInt16, internalIndexTensor, subTensorInt16, calCnt);
    Muls(tempTensorInt16, tempTensorInt16, static_cast<int16_t>(axisK_), calCnt);
    Sub(internalIndexTensor, internalIndexTensor, tempTensorInt16, calCnt);
    Muls(tempTensorInt16, tempTensorInt16, static_cast<int16_t>(sizeof(int16_t)), calCnt); // gather offset
    Cast(gatherIdxTensor, tempTensorInt16, RoundMode::CAST_NONE, calCnt);

    for (uint32_t k = 0; k < axisK_; k++) {
        Gather(tempTensorInt16, dstRankTensor_, gatherIdxTensor.template ReinterpretCast<uint32_t>(),
               static_cast<uint32_t>(0), calCnt);
        Compare(equalMaskTensor, dstRankTensor_, tempTensorInt16, AscendC::CMPMODE::EQ, calCntAlign);
        CompareScalar(compareMaskTensor, internalIndexTensor, static_cast<int16_t>(k), AscendC::CMPMODE::GT,
                      calCntAlign);
        And(compareMaskTensor, equalMaskTensor, compareMaskTensor, maskCnt);
        Or(selectMaskTensor, selectMaskTensor, compareMaskTensor, maskCnt);
        Adds(gatherIdxTensor, gatherIdxTensor, static_cast<int32_t>(sizeof(int16_t)), calCnt);
    }
    Not(selectMaskTensor, selectMaskTensor, maskCnt);
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::CalSendCntPerRank(uint32_t calCnt,
                                                                                       uint32_t calCntAlign,
                                                                                       uint32_t tmpOffset)
{
    LocalTensor<int16_t> tempTensor = topkIdsBuf_.GetWithOffset<int16_t>(calCntAlign, 0);
    LocalTensor<uint32_t> sendCntTensor = dstExpBuf_.GetWithOffset<uint32_t>(tmpOffset / sizeof(uint32_t), tmpOffset);
    LocalTensor<uint8_t> compareMaskTensor = maskBuf_.GetWithOffset<uint8_t>(maskBytesAlign_, 0);
    Select(dstRankTensor_, compareMaskTensor, dstRankTensor_, static_cast<int16_t>(-1),
           AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, calCnt); // 掩码直接使用

    if (epWorldSize_ <= HISTOGRAM_SCOPE_SIZE) {
        CompareScalar(compareMaskTensor, dstRankTensor_, static_cast<int16_t>(0), AscendC::CMPMODE::GE, calCntAlign);
        CalSendCntHistograms(dstRankTensor_, tempTensor, sendCntTensor, calCnt, epWorldSize_);
        Add(sendCntPerRankTensor_, sendCntPerRankTensor_, sendCntTensor.template ReinterpretCast<int32_t>(),
            static_cast<int32_t>(epWorldSize_));
        return;
    }

    uint32_t groupNum = Ceil(epWorldSize_, HISTOGRAM_SCOPE_SIZE);
    uint32_t maskCnt = Ceil(calCntAlign, BITS_PER_BYTE);
    LocalTensor<int16_t> curGroupTensor = topkIdsBuf_.GetWithOffset<int16_t>(calCntAlign, tmpOffset);
    LocalTensor<uint8_t> ltMaskTensor = maskBuf_.GetWithOffset<uint8_t>(maskBytesAlign_, maskBytesAlign_);
    for (uint32_t group = 0; group < groupNum; group++) {
        uint32_t baseId = group * HISTOGRAM_SCOPE_SIZE;
        uint32_t curGroupSize = (group == groupNum - 1) ? (epWorldSize_ - baseId) : HISTOGRAM_SCOPE_SIZE;
        Subs(curGroupTensor, dstRankTensor_, static_cast<int16_t>(baseId), calCnt);
        CompareScalar(compareMaskTensor, curGroupTensor, static_cast<int16_t>(0), AscendC::CMPMODE::GE, calCntAlign);
        CompareScalar(ltMaskTensor, curGroupTensor, static_cast<int16_t>(curGroupSize - 1), AscendC::CMPMODE::LE,
                      calCntAlign);
        And(compareMaskTensor, compareMaskTensor, ltMaskTensor, maskCnt);
        CalSendCntHistograms(curGroupTensor, tempTensor, sendCntTensor, calCnt, curGroupSize);
        Add(sendCntPerRankTensor_[baseId], sendCntPerRankTensor_[baseId],
            sendCntTensor.template ReinterpretCast<int32_t>(), static_cast<int32_t>(curGroupSize));
    }
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::CalSendCntPerExpert(
    LocalTensor<int16_t> expertIdsTensor, uint32_t calCnt, uint32_t calCntAlign, uint32_t tmpOffset)
{
    LocalTensor<int16_t> tempTensor = topkIdsBuf_.GetWithOffset<int16_t>(calCntAlign, 0);
    LocalTensor<uint32_t> sendCntTensor = dstExpBuf_.GetWithOffset<uint32_t>(tmpOffset / sizeof(uint32_t), tmpOffset);
    LocalTensor<uint8_t> compareMaskTensor = maskBuf_.GetWithOffset<uint8_t>(maskBytesAlign_, 0);
    if (moeExpertNum_ <= HISTOGRAM_SCOPE_SIZE) {
        CompareScalar(compareMaskTensor, expertIdsTensor, static_cast<int16_t>(0), AscendC::CMPMODE::GE, calCntAlign);
        CalSendCntHistograms(expertIdsTensor, tempTensor, sendCntTensor, calCnt, moeExpertNum_);
        Add(sendCntPerExpertTensor_, sendCntPerExpertTensor_, sendCntTensor.template ReinterpretCast<int32_t>(),
            static_cast<int32_t>(moeExpertNum_));
        return;
    }

    uint32_t groupNum = Ceil(moeExpertNum_, HISTOGRAM_SCOPE_SIZE);
    uint32_t maskCnt = Ceil(calCntAlign, BITS_PER_BYTE);
    LocalTensor<int16_t> curGroupTensor = topkIdsBuf_.GetWithOffset<int16_t>(calCntAlign, tmpOffset);
    LocalTensor<uint8_t> ltMaskTensor = maskBuf_.GetWithOffset<uint8_t>(maskBytesAlign_, maskBytesAlign_);
    for (uint32_t group = 0; group < groupNum; group++) {
        uint32_t baseId = group * HISTOGRAM_SCOPE_SIZE;
        uint32_t curGroupSize = (group == groupNum - 1) ? (moeExpertNum_ - baseId) : HISTOGRAM_SCOPE_SIZE;
        Subs(curGroupTensor, expertIdsTensor, static_cast<int16_t>(baseId), calCnt);
        CompareScalar(compareMaskTensor, curGroupTensor, static_cast<int16_t>(0), AscendC::CMPMODE::GE, calCntAlign);
        CompareScalar(ltMaskTensor, curGroupTensor, static_cast<int16_t>(curGroupSize - 1), AscendC::CMPMODE::LE,
                      calCntAlign);
        And(compareMaskTensor, compareMaskTensor, ltMaskTensor, maskCnt);
        CalSendCntHistograms(curGroupTensor, tempTensor, sendCntTensor, calCnt, curGroupSize);
        Add(sendCntPerExpertTensor_[baseId], sendCntPerExpertTensor_[baseId],
            sendCntTensor.template ReinterpretCast<int32_t>(), static_cast<int32_t>(curGroupSize));
    }
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
    DataCopyPadExtParams<int32_t> topkIdsCntCopyPadParams{false, 0U, 0U, 0U};

    for (uint32_t group = 0; group < groupCnt; group++) {
        uint32_t groupOffset = group * perGroupTokenNum_;
        if (group == groupCnt - 1) {
            calCnt = (sendTokenNum_ - groupOffset) * axisK_;
        }
        uint32_t topkIdxOffset = (startTokenId_ + groupOffset) * axisK_;
        uint32_t tmpOffset = Ceil(calCnt * sizeof(int16_t), ALIGNED_LEN_256) * ALIGNED_LEN_256;
        uint32_t calCntAlign = tmpOffset / sizeof(int16_t);
        DataCopyExtParams topkIdsCntParams = {1U, static_cast<uint32_t>(calCnt * sizeof(int32_t)), 0U, 0U, 0U};
        dstRankTensor_ = dstExpBuf_.GetWithOffset<int16_t>(calCntAlign, 0);
        if (group > 0) {
            SyncFunc<AscendC::HardEvent::V_MTE2>();
        }
        DataCopyPad(topkIdsGroupTensor, topkIdxGMTensor_[topkIdxOffset], topkIdsCntParams,
                    topkIdsCntCopyPadParams); // copy topkId
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        Cast(tempTensorInt16, topkIdsGroupTensor, RoundMode::CAST_NONE, calCnt);

        // perExpert 计算
        CalSendCntPerExpert(tempTensorInt16, calCnt, calCntAlign, tmpOffset);

        // 去重，计算 per-rank
        if constexpr (IsCached) { // cache下复用 dstSlotIdx
            dstSlotIdxTensor_ = topkIdsBuf_.Get<int32_t>();
            SyncFunc<AscendC::HardEvent::V_MTE2>();
            DataCopyPad(dstSlotIdxTensor_, cachedSlotIdxGMTensor_[topkIdxOffset], topkIdsCntParams,
                        topkIdsCntCopyPadParams); // copy slotId
        }
        DedupAndSendDirect(tempTensorInt16, calCnt, calCntAlign, tmpOffset); // 去重
        if constexpr (IsCached) {
            SyncFunc<AscendC::HardEvent::MTE2_MTE3>();
            DataCopyPad(dstSlotIdxGMTensor_[topkIdxOffset], dstSlotIdxTensor_, topkIdsCntParams);
            SyncFunc<AscendC::HardEvent::MTE3_V>();
        }
        CalSendCntPerRank(calCnt, calCntAlign, tmpOffset);
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

    DataCopy(sendCntPerRankTensor_, sendCntPerRankGMTensor_, counterCnt_);
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    for (uint32_t dstRankId = copyRankStart_; dstRankId < copyRankStart_ + copyRankNum_; ++dstRankId) {
        if (channelIndex_ != 0) {
            continue;
        }
        int32_t sendCnt = sendCntPerRankTensor_.GetValue(dstRankId);
        uint64_t notifyVal = (static_cast<uint64_t>(sendCnt) << NOTIFY_VAL_SHIFT) | 1U;
        uint32_t srcStride = dstRankId * moeExpertNumPerRank_;
        // 计算目标窗口地址:
        GM_ADDR remoteStateAddr = GetWinAddrByRankId(mc2Context_, dstRankId, cntWinStateOffset_);
        GM_ADDR notifyAddr = remoteStateAddr + epRankId_ * WIN_ADDR_ALIGN;
        GM_ADDR remoteCountAddr = remoteStateAddr + cntFlagWinSize_ + epRankId_ * moeNumPerRankAlign512_;
        GM_ADDR srcWorkspaceAddr = sendCntWorkspaceAddr_ + epWorldSizeAlign512_ + srcStride * sizeof(int32_t);

        if (dstRankId != epRankId_) { // 远端 使用URMA发送 count + state
            uint64_t commHandle = GetCommHandle(mc2Context_, dstRankId, channelIndex_);
            hcomm_.WriteWithNotifyNbi<true, PIPE_S, PIPE_MTE3, DATA_CFG>(commHandle, remoteCountAddr, srcWorkspaceAddr,
                                                                         moeNumPerRankSize_, notifyAddr, notifyVal);
            continue;
        }

        // 本端
        GlobalTensor<int32_t> countGMTensor;
        GlobalTensor<uint64_t> notifyGMTensor;
        countGMTensor.SetGlobalBuffer((__gm__ int32_t *)remoteCountAddr);
        notifyGMTensor.SetGlobalBuffer((__gm__ uint64_t *)notifyAddr);
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
    SyncFunc<AscendC::HardEvent::MTE3_V>(); // buf 复用
    while (sumOfFlag != commpareFlag) {     // 状态位check
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
    localExpertRecvGMTensor.SetGlobalBuffer((__gm__ int32_t *)(localCntStateWinAddr_ + cntFlagWinSize_));
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
        hostPinnedCounterTensor.SetGlobalBuffer((__gm__ int64_t *)hostPinnedCounterAddrGM_);
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
    SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
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
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::SetSrcTokenIdxAndDstSlot(
    uint32_t topkOffset, uint32_t calCnt, uint32_t calCntAlign, uint32_t &sendLocalCnt,
    LocalTensor<int32_t> &cntLocalTensor)
{
    uint64_t rsvdCnt = 0;
    LocalTensor<int16_t> internalIndexTensor = topkIdsBuf_.GetWithOffset<int16_t>(calCntAlign, 0);
    LocalTensor<int16_t> scatterTensor = topkIdsBuf_.GetWithOffset<int16_t>(calCntAlign, calCntAlign * sizeof(int16_t));
    LocalTensor<int32_t> dstSlotIdxTensor = topkIdsBuf_.Get<int32_t>();
    LocalTensor<int32_t> tempLocalTensor = dstExpBuf_.Get<int32_t>();
    LocalTensor<uint16_t> gatherMaskTensor = maskBuf_.Get<uint16_t>();
    LocalTensor<uint8_t> compareMaskTensor = maskBuf_.Get<uint8_t>();
    DataCopyParams srcTokenCopyParams = {1U, 0U, 0U, 0U};

    for (uint32_t rankId = 0; rankId < epWorldSize_; ++rankId) {
        // 调整处理顺序，先处理其他对端卡，本端卡最后处理，后续处理token可复用
        uint32_t dstRankId = rankId >= epRankId_ ? (rankId + 1) : rankId;
        dstRankId = (rankId == epWorldSize_ - 1) ? epRankId_ : dstRankId;
        CreateVecIndex(internalIndexTensor, static_cast<int16_t>(0), calCnt); // 生成元素索引
        CompareScalar(compareMaskTensor, dstRankIdTensor_, static_cast<int16_t>(dstRankId), AscendC::CMPMODE::EQ,
                      calCntAlign);
        if (rankId > 0) {
            SyncFunc<AscendC::HardEvent::MTE3_V>();
        }
        SyncFunc<AscendC::HardEvent::S_V>(); // 需要等待上一轮本端命中判断结束
        GatherMask(internalIndexTensor, internalIndexTensor, gatherMaskTensor, true, calCnt, {1, 1, 0, 0}, rsvdCnt);
        if (dstRankId == epRankId_) {
            sendLocalCnt = static_cast<uint32_t>(rsvdCnt);
        }
        if (rsvdCnt == 0) {
            continue;
        }

        SyncFunc<AscendC::HardEvent::V_S>();
        int32_t curSlot = cntLocalTensor.GetValue(dstRankId);
        int32_t curCnt = static_cast<int32_t>(rsvdCnt);
        if constexpr (!IsCached) {
            // slot 加 epWorldSize_ 偏移：散射值 >= epWorldSize_，避免与后续卡匹配错误
            CreateVecIndex(scatterTensor, static_cast<int16_t>(curSlot + epWorldSize_), curCnt);
            Cast(tempLocalTensor, internalIndexTensor, RoundMode::CAST_NONE, curCnt);
            Muls(tempLocalTensor, tempLocalTensor, static_cast<int32_t>(sizeof(int16_t)), curCnt);
            Scatter(dstRankIdTensor_, scatterTensor, tempLocalTensor.template ReinterpretCast<uint32_t>(),
                    static_cast<uint32_t>(0), curCnt);
        }
        Duplicate<int16_t>(scatterTensor, static_cast<int16_t>(axisK_), curCnt);
        Div(internalIndexTensor, internalIndexTensor, scatterTensor, curCnt);
        Cast(tempLocalTensor, internalIndexTensor, RoundMode::CAST_NONE, curCnt);
        Adds(tempLocalTensor, tempLocalTensor, static_cast<int32_t>(topkOffset / axisK_), curCnt);
        uint32_t dstIndex = dstRankId * tokenCntAlign_ + curSlot;
        srcTokenCopyParams.blockLen = static_cast<uint32_t>(curCnt * sizeof(int32_t));
        SyncFunc<AscendC::HardEvent::V_MTE3>();
        DataCopyPad(srcTokenIdxGMTensor_[dstIndex], tempLocalTensor, srcTokenCopyParams);
        cntLocalTensor.SetValue(dstRankId, curSlot + curCnt);
        rsvdCnt = 0;
    }
    if constexpr (!IsCached) {
        // 去偏移，有效位还原为 slot，未散射位(重复/无效)保持 -1
        CompareScalar(compareMaskTensor, dstRankIdTensor_, static_cast<int16_t>(0), AscendC::CMPMODE::GE, calCntAlign);
        Subs(dstRankIdTensor_, dstRankIdTensor_, static_cast<int16_t>(epWorldSize_), calCntAlign);
        Select(dstRankIdTensor_, compareMaskTensor, dstRankIdTensor_, static_cast<int16_t>(-1),
               AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, calCntAlign);
        Cast(dstSlotIdxTensor, dstRankIdTensor_, RoundMode::CAST_NONE, calCnt);
        SyncFunc<AscendC::HardEvent::V_MTE3>();
        DataCopyExtParams dstSlotCopyParams = {1U, static_cast<uint32_t>(calCnt * sizeof(int32_t)), 0U, 0U, 0U};
        DataCopyPad(dstSlotIdxGMTensor_[topkOffset], dstSlotIdxTensor, dstSlotCopyParams);
    }
    SyncFunc<AscendC::HardEvent::MTE3_V>(); // 下一轮等待这一轮做完
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::CopyInMetaInfo(uint32_t tokenId,
                                                                                    uint32_t topkOffset)
{
    LocalTensor<int32_t> metaCopyInTensor = metaSlotQueue_.AllocTensor<int32_t>();
    if constexpr (Std::IsSame<XType, fp8_e5m2_t>::value || Std::IsSame<XType, fp8_e4m3fn_t>::value) {
        DataCopyPad(metaCopyInTensor.template ReinterpretCast<uint8_t>(),
                    scalesGMTensor_.template ReinterpretCast<uint8_t>()[tokenId * scalesBytes_], scalesCopyParams_,
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
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::ProcessMetaSlot(
    uint32_t startId, uint32_t tokenCnt, uint32_t sendLocalCnt, uint32_t &topkOffset, uint32_t &curLocalSlot)
{
    // meta 流水写 stash + 本端命中内联写本地窗 meta 区
    LocalTensor<int32_t> srcTokenTensor = dstExpBuf_.Get<int32_t>();
    uint32_t endId = startId + tokenCnt;
    uint32_t localIdx = 0;
    uint32_t metaSetStride = topkCopyAlign_ + 2 * axisKAlign_; // meta 槽内 srcRank/tokenIdx 的下标
    CopyInMetaInfo(startId, topkOffset);
    for (uint32_t tokenId = startId; tokenId < endId; ++tokenId) {
        metaLocalTensor_ = metaSlotQueue_.DeQue<int32_t>();
        SyncFunc<AscendC::HardEvent::MTE2_S>();
        metaLocalTensor_.SetValue(metaSetStride, epRankId_);
        metaLocalTensor_.SetValue(metaSetStride + 1, tokenId);
        if (tokenId + 1 < endId) {
            topkOffset += axisK_;
            CopyInMetaInfo(tokenId + 1, topkOffset);
        }
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        DataCopyPad(metaSlotGMTensor_[tokenId * metaSlotStride32_], metaLocalTensor_, metaSlotCopyParams_);
        if ((localIdx < sendLocalCnt) &&
            (static_cast<uint32_t>(srcTokenTensor.GetValue(localIdx)) == tokenId)) { // 本端命中判断
            localMetaSlotGMTensor_.SetGlobalBuffer(
                (__gm__ int32_t *)(localSlotWinAddr_ + static_cast<uint64_t>(curLocalSlot) * perSlotBytes_ +
                                   hAlignSize_));
            DataCopyPad(localMetaSlotGMTensor_, metaLocalTensor_, metaSlotCopyParams_);
            localIdx++;
            curLocalSlot++;
        }
        metaSlotQueue_.FreeTensor<int32_t>(metaLocalTensor_);
    }
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::LocalCopyPhase(uint32_t slotStart,
                                                                                    uint32_t slotCnt)
{
    uint32_t localIdx = 0;
    uint32_t gmOffset = 0;
    uint32_t perSlotStride = perSlotBytes_ / sizeof(XType);
    LocalTensor<int32_t> srcTokenTensor = dstExpBuf_.Get<int32_t>();
    GlobalTensor<XType> localTokenSlotGMTensor;
    localTokenSlotGMTensor.SetGlobalBuffer(
        (__gm__ XType *)(localSlotWinAddr_ + static_cast<uint64_t>(slotStart) * perSlotBytes_));

    while (localIdx < slotCnt) {
        uint32_t tokenId = static_cast<uint32_t>(srcTokenTensor.GetValue(localIdx));
        LocalTensor<XType> tokenTensor = tokenSlotQueue_.AllocTensor<XType>();
        DataCopyPad(tokenTensor, xGMTensor_[tokenId * axisH_], xCopyParams_, padParams_);
        tokenSlotQueue_.EnQue(tokenTensor);
        tokenTensor = tokenSlotQueue_.DeQue<XType>();
        DataCopyPad(localTokenSlotGMTensor[gmOffset], tokenTensor, xCopyParams_);
        tokenSlotQueue_.FreeTensor<XType>(tokenTensor);
        localIdx++;
        gmOffset += perSlotStride;
    }
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::SendPhase()
{
    if (startTokenId_ >= axisBS_) {
        return;
    }
    GetSlotStartNum(); // 计算起始slot id 并落盘 slotStart 供发送侧读取

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
        SetSrcTokenIdxAndDstSlot(topkOffset, calCnt, calCntAlign, sendLocalCnt, cntLocalTensor);
        SyncFunc<AscendC::HardEvent::V_S>();
        LocalCopyPhase(curLocalSlot, sendLocalCnt); // 本端命中段独立拷 x
        ProcessMetaSlot(startId, tokenCnt, sendLocalCnt, topkOffset, curLocalSlot);
        if (group < groupCnt - 1) {
            SyncFunc<AscendC::HardEvent::V_MTE2>();
        }
    }
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
    srcStatusGMTensor.SetGlobalBuffer((__gm__ int32_t *)(payloadStashStateWinAddr_ + aivId_ * WIN_ADDR_ALIGN));
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
    uint32_t hcommBatchSize = HCOMM_BATCH_BUFFER_BYTES / sizeof(uint8_t);
    tpipe_->InitBuffer(hcommBuf_, HCOMM_INIT_SIZE + HCOMM_BATCH_BUFFER_BYTES);
    hcommTensor_ = hcommBuf_.GetWithOffset<uint8_t>(HCOMM_INIT_SIZE, 0);
    hcommBatchTensor_ = hcommBuf_.GetWithOffset<uint8_t>(hcommBatchSize, HCOMM_INIT_SIZE);
    hcomm_.Init(hcommTensor_, HCOMM_INIT_SIZE);
    Duplicate<uint8_t>(hcommBatchTensor_, 0, hcommBatchSize);

    // ub buffer init
    tpipe_->InitBuffer(sendCntRankBuf_, sendCntRankSizeAlign_);
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
    DataCopy(sendNumPerRankTensor, sendCntPerRankGMTensor_, counterCnt_);
    SyncFunc<AscendC::HardEvent::MTE2_S>();

    for (uint32_t dstRankId = copyRankStart_; dstRankId < copyRankStart_ + copyRankNum_; ++dstRankId) {
        uint32_t sendTokenNum = sendNumPerRankTensor.GetValue(dstRankId);
        GM_ADDR notifyAddr = GetWinAddrByRankId(mc2Context_, dstRankId, slotWinStateOffset_) + dstStateWinOffset_;
        notifyAddr += channelIndex_ * WIN_ADDR_ALIGN;
        if (unlikely(dstRankId == epRankId_)) {
            GlobalTensor<int32_t> srcStatusGMTensor;
            GlobalTensor<int32_t> dstStatusGMTensor;
            srcStatusGMTensor.SetGlobalBuffer((__gm__ int32_t *)payloadStashStateWinAddr_);
            dstStatusGMTensor.SetGlobalBuffer((__gm__ int32_t *)notifyAddr);
            DataCopy(statusTensor, srcStatusGMTensor, UB_STRIDE);
            SyncFunc<AscendC::HardEvent::MTE2_MTE3>();
            DataCopy(dstStatusGMTensor, statusTensor, UB_STRIDE);
            continue;
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
    uint32_t tokenSize = axisH_ * sizeof(XType);
    DataCopyExtParams srcTokenCopyParams = {1U, 0U, 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> srcTokenCopyPadParams{false, 0U, 0U, 0U};
    LocalTensor<int32_t> srcTokenIdxTensor = tempBuf_.Get<int32_t>();
    GlobalTensor<int32_t> srcTokenIdxGMTensor;
    srcTokenIdxGMTensor.SetGlobalBuffer(
        (__gm__ int32_t *)(srcTokenTableAddr_ + static_cast<uint64_t>(dstRankId) * srcTokenListBytes_));
    GM_ADDR remoteWinBaseAddr = GetWinAddrByRankId(mc2Context_, dstRankId, winDataOffset_) + dstRankWinOffset_ +
                                static_cast<uint64_t>(tokenStart) * perSlotBytes_;
    SyncFunc<AscendC::HardEvent::V_S>(); // hcommBatchTensor_ init
    auto batchHandle =
        hcomm_.MakeBatchHandle(commHandle, hcommBatchTensor_, HCOMM_BATCH_BUFFER_BYTES, remoteWinBaseAddr);
    GM_ADDR remoteWinAddr = remoteWinBaseAddr;
    while (processed < tokenNum) {
        uint32_t batchCnt = (tokenNum - processed > maxBatchCnt) ? maxBatchCnt : tokenNum - processed;
        srcTokenCopyParams.blockLen = static_cast<uint32_t>(batchCnt * sizeof(int32_t));
        DataCopyPad(srcTokenIdxTensor, srcTokenIdxGMTensor[tokenStart + processed], srcTokenCopyParams,
                    srcTokenCopyPadParams);
        SyncFunc<AscendC::HardEvent::MTE2_S>();
        for (uint32_t i = 0; i < batchCnt; i++) {
            uint64_t srcTokenId = static_cast<uint64_t>(srcTokenIdxTensor.GetValue(i));
            GM_ADDR metaStashAddr = payloadStashWinAddr_ + srcTokenId * metaSlotBytesAlign512_;
            hcomm_.WriteNbi<DATA_CFG>(batchHandle, remoteWinAddr, xAddr_ + srcTokenId * tokenSize, tokenSize);
            hcomm_.WriteNbi<DATA_CFG>(batchHandle, remoteWinAddr + hAlignSize_, metaStashAddr, metaSlotBytes_);
            commCnt++;
            if (commCnt == HCOMM_BATCH_CAPACITY) {
                hcomm_.BatchCommit(batchHandle);
                commCnt = 0;
            }
            processed++;
            remoteWinAddr += perSlotBytes_;
        }
        if (processed < tokenNum) {
            SyncFunc<AscendC::HardEvent::S_MTE2>();
        }
    }
    hcomm_.WriteNbi<NOTIFY_CFG>(batchHandle, notifyAddr, payloadStashStateWinAddr_, WIN_ADDR_ALIGN);
    hcomm_.BatchCommit(batchHandle);
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::Process()
{
    if ASCEND_IS_AIV { // 全aiv处理
        BufferInit();
        CalSendCnt();
        SyncAll<true>();
        Communication();
        GetRecvCount();
        SendPhase();
        SetStatus();
        SyncAll<true>();
        WriteToRemoteWindow();
        diagWriter_.RunPosRecord(MOE_EP_DISPATCH_RUN_POS_URMA_REQUESTS_ISSUE_DONE);
    }
}

#endif

} // namespace MoeEpDispatchImpl

#endif // MOE_EP_DISPATCH_H
