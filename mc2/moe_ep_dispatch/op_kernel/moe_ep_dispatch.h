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

#if (ASC_DEVKIT_VERSION_NUM >= 90200000) && (HCOMM_VERSION_NUM >= 90200000)
#define ENABLE_MOE_EP_KERNEL
#endif

#endif

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#include "kernel_tiling/kernel_tiling.h"
#if defined(ENABLE_MOE_EP_KERNEL)
#include "c_api/asc_simd.h"
#endif
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

constexpr uint32_t ALIGNED_LEN_256 = 256U;
constexpr uint32_t TOPK_INFO_SIZE = 4U;  // sizeof(int32_t)=sizeof(float)=4B
constexpr uint32_t UB_STRIDE = 8U;       // UB_ALIGN/sizeof(int32_t)=8
constexpr uint32_t INT64_UB_STRIDE = 4U; // UB_ALIGN/sizeof(int64_t)=4
constexpr uint32_t BITS_PER_BYTE = 8U;
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
constexpr uint32_t WRITE_INDEX_BATCH_TOKENS = HCOMM_BATCH_CAPACITY * SLOT_NUM_PER_SQE;
constexpr uint32_t REMOTE_INDEX_EVENT_COUNT = 3U; // 远端索引槽组使用的 MTE3->V 事件数
// SendPhase 的 MTE3->V 事件：0~2 对应远端索引槽组，3/4 对应 metadata，5 对应本端索引, 其他流水按各函数内的用途使用。
constexpr uint32_t SYNC_EVENT_ID0 = 0U;
constexpr uint32_t SYNC_EVENT_ID1 = 1U;
constexpr uint32_t SYNC_EVENT_ID2 = 2U;
constexpr uint32_t SYNC_EVENT_ID3 = 3U;
constexpr uint32_t SYNC_EVENT_ID4 = 4U;
constexpr uint32_t SYNC_EVENT_ID5 = 5U;
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
    __aicore__ inline void WaitRecvCnt(LocalTensor<int32_t>& recvTensor);
    __aicore__ inline void WriteToRemoteWindow();
    __aicore__ inline void CopyOutSendCnt(LocalTensor<int32_t> sendNumPerRankTensor);
    __aicore__ inline void SendTokenByChannel(uint32_t tokenStart, uint32_t tokenNum, uint32_t dstRankId,
                                              uint64_t commHandle, GM_ADDR notifyAddr);
    __aicore__ inline bool SetLocalSrcTokenIdx(uint32_t topkOffset, uint32_t calCnt, uint32_t calCntAlign,
                                               uint32_t& sendLocalCnt);
    __aicore__ inline void SetRemoteSrcTokenIdx(uint32_t tokenCnt, uint32_t calCnt, uint32_t calCntAlign,
                                                bool localUsed);
    __aicore__ inline void DrainRemoteIndexEvents();
    __aicore__ inline void ProcessMetaSlot(uint32_t startId, uint32_t tokenCnt, uint32_t sendLocalCnt,
                                           uint32_t& curLocalSlot);
    __aicore__ inline void SendPhase();
    __aicore__ inline void SendPhaseCached();
    __aicore__ inline void GetLocalSlotStart(uint32_t sendCount, uint32_t maxBatchCnt, uint32_t alignStride,
                                             uint32_t& curLocalSlot);
    __aicore__ inline void SetLocalStatusAndBufferInitCached();
    __aicore__ inline void CommunicationCached();
    __aicore__ inline void WaitStatusCached();
    __aicore__ inline void GetSlotStartNum();
    __aicore__ inline void CopyInMetaInfo(uint32_t startTokenId, uint32_t batchCnt, uint32_t bufferIdx,
                                          uint32_t bufferStride, uint32_t bufferOffset);
    __aicore__ inline void DrainMetaBuffers();
    __aicore__ inline void InitMetaScatter();
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
    LocalTensor<uint8_t> hcommTensor_;
    LocalTensor<uint8_t> hcommBatchTensor_;
    LocalTensor<int32_t> rankValueTensor_;
    LocalTensor<int32_t> tokenValueTensor_;
    LocalTensor<uint32_t> rankOffsetTensor_;
    LocalTensor<uint32_t> tokenOffsetTensor_;

    TBuf<> metaSlotBuf_;
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
    uint64_t dstSlotStrideBytes_{0};
    uint32_t tokenSize_{0};
    uint32_t topkSizeBytes_{0};
    uint32_t kAlignSize_{0};
    uint32_t axisKAlign_{0};
    uint32_t epWorldSizeBytes_{0};
    uint32_t epWorldSizeAlign_{0};
    uint32_t epWorldSizeAlign512_{0};
    uint32_t counterCnt_{0};
    uint32_t counterAlign512_{0};
    uint32_t perGroupSizeAlign_{0};
    uint32_t maskBytesAlign_{0};
    uint32_t sendCntRankSizeAlign_{0};
    uint32_t moeNumPerRankBytes_{0};
    uint32_t moeNumPerRankAlign_{0};
    uint32_t expertPerRankAlign_{0};
    uint32_t moeExpertNumBytes_{0};
    uint32_t moeExpertNumAlign_{0};
    uint32_t moeNumPerRankAlign512_{0};
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
    bool multiGroup_{false};
    bool metaBufferUsed_[BUFFER_NUM] = {};  // 该半 buffer 的 MTE3->V/MTE2 释放事件尚未被 wait 消费
    uint32_t remoteIndexPendingEvents_{0U}; // 尚未被 wait 消费的远端索引 MTE3->V 事件位图
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
    const auto& info = tilingData->moeEpDispatchInfo;
#if defined(ASCENDC_OOM) && ASCENDC_OOM == 1
    MoeEpDispatchBase::RegisterOomRanges<DoCpuSync>(mc2Context_, epRankId_, tilingGM, tilingData);
#endif
    constexpr size_t metadataOffset =
        offsetof(MoeEpDispatchTilingData, moeEpDispatchInfo) + offsetof(MoeEpDispatchInfo, dumpMetadata);
    MoeEpExceptionDump::WriteMetadata(context, tilingGM + metadataOffset);
    diagWriter_.Init(context, MOE_EP_CORE_DIAG_DISPATCH, tpipe_);

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
    topkSizeBytes_ = axisK_ * TOPK_INFO_SIZE;
    kAlignSize_ = Ceil(topkSizeBytes_, UB_ALIGN) * UB_ALIGN;
    axisKAlign_ = kAlignSize_ / TOPK_INFO_SIZE;
    metaSlotStride_ = metaSlotBytes_ / sizeof(int32_t);
    dstSlotStride_ = axisBS_ + 1;
    dstSlotStrideBytes_ = static_cast<uint64_t>(dstSlotStride_) * sizeof(int32_t);
    epWorldSizeBytes_ = epWorldSize_ * sizeof(int32_t);
    epWorldSizeAlign_ = Ceil(epWorldSizeBytes_, UB_ALIGN) * UB_ALIGN;
    perGroupTokenNum_ = PER_GROUP_SIZE / sizeof(int16_t) / axisK_; // token num per group
    perGroupSizeAlign_ = Ceil(perGroupTokenNum_ * axisK_ * sizeof(int16_t), ALIGNED_LEN_256) * ALIGNED_LEN_256;
    maskBytesAlign_ = Ceil(Ceil(perGroupSizeAlign_ / sizeof(int16_t), BITS_PER_BYTE), UB_ALIGN) * UB_ALIGN;
    epWorldSizeAlign512_ = Ceil(epWorldSizeBytes_, WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN;
    sendCntRankSizeAlign_ = Ceil(epWorldSizeBytes_, ALIGNED_LEN_256) * ALIGNED_LEN_256;
    counterCnt_ = epWorldSizeAlign_ / sizeof(int32_t);
    counterAlign512_ = epWorldSizeAlign512_ / sizeof(int32_t);
    moeNumPerRankBytes_ = moeExpertNumPerRank_ * sizeof(int32_t);
    moeNumPerRankAlign_ = Ceil(moeNumPerRankBytes_, UB_ALIGN) * UB_ALIGN;
    expertPerRankAlign_ = moeNumPerRankAlign_ / sizeof(int32_t);
    moeExpertNumBytes_ = moeExpertNum_ * sizeof(int32_t);
    moeExpertNumAlign_ = Ceil(moeExpertNumBytes_, UB_ALIGN) * UB_ALIGN;
    moeNumPerRankAlign512_ = Ceil(moeNumPerRankBytes_, WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN;
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

    scaleupCounterAddr_ = workspaceGM;
    sendCntWorkspaceAddr_ = workspaceGM + static_cast<uint64_t>(aivNum_) * epWorldSizeAlign512_;
    dstRankInfoAddr_ = workspaceGM + dstRankInfoOffset_;
    localCntStateWinAddr_ = GetWinAddrByRankId(mc2Context_, epRankId_, cntWinStateOffset_);
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
    sendNum = curSendCnt / curUseAivNum;               // 每个 AIV 分配的元素数
    uint32_t remainderNum = curSendCnt % curUseAivNum; // 余数
    startId = sendNum * aivId_;                        // 当前 AIV 的起始元素下标
    if (aivId_ < remainderNum) {                       // 前 remainderNum 个 AIV 多分配一个元素
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
    tpipe_->InitBuffer(metaSlotBuf_, BUFFER_NUM * metaBatch_ * metaSlotBytes_);
    tpipe_->InitBuffer(metaScatterBuf_, MOE_EP_META_SCATTER_UB_BYTES);
    tpipe_->InitBuffer(topkIdsBuf_, 2 * perGroupSizeAlign_);
    tpipe_->InitBuffer(tempBuf_, perGroupSizeAlign_);
    tpipe_->InitBuffer(dstExpBuf_, 2 * perGroupSizeAlign_);
    tpipe_->InitBuffer(maskBuf_, 3 * maskBytesAlign_);
    tpipe_->InitBuffer(sendCntRankBuf_, sendCntRankSizeAlign_);
    tpipe_->InitBuffer(sendCntExpertBuf_, moeExpertNumAlign_);
    tpipe_->InitBuffer(numRecvPerExpertBuf_, 2 * moeNumPerRankAlign_);
    tpipe_->InitBuffer(numRecvPerRankBuf_, epWorldSizeAlign_);
    tpipe_->InitBuffer(hcommBuf_, HCOMM_INIT_SIZE);
    dstRankTensor_ = tempBuf_.Get<int16_t>();
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
    event_t eventId0 = static_cast<event_t>(SYNC_EVENT_ID0);
    asc_sync_notify(PIPE_V, PIPE_MTE3, eventId0);
    asc_sync_wait(PIPE_V, PIPE_MTE3, eventId0);
    asc_copy_ub2gm_align((__gm__ uint8_t*)sendCntPerExpertGMTensor_.GetPhyAddr(),
                         (__ubuf__ uint8_t*)sendCntPerExpertTensor_.GetPhyAddr(), 1U, moeExpertNumBytes_,
                         asc_store_l2_cache_mode::NORMAL_FIRST_VICTIM, 0U, 0U);
    asc_copy_ub2gm_align((__gm__ uint8_t*)sendCntPerRankGMTensor_.GetPhyAddr(),
                         (__ubuf__ uint8_t*)sendCntPerRankTensor_.GetPhyAddr(), epWorldSizeAlign_);
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::CalSendCnt()
{
    SplitToCore(axisBS_, aivNum_, startTokenId_, endTokenId_, sendTokenNum_); // 按token数量分核
    if (startTokenId_ >= axisBS_) {
        return;
    }

    uint32_t groupCnt = Ceil(sendTokenNum_, perGroupTokenNum_);
    multiGroup_ = groupCnt > 1U;
    uint32_t calCnt = perGroupTokenNum_ * axisK_;
    LocalTensor<int32_t> topkIdsGroupTensor = topkIdsBuf_.Get<int32_t>();
    event_t eventId0 = static_cast<event_t>(SYNC_EVENT_ID0);
    event_t eventId4 = static_cast<event_t>(SYNC_EVENT_ID4);
    for (uint32_t group = 0; group < groupCnt; group++) {
        uint32_t groupOffset = group * perGroupTokenNum_;
        uint32_t tokenCnt = (group == groupCnt - 1) ? (sendTokenNum_ - groupOffset) : perGroupTokenNum_;
        if (group == groupCnt - 1) {
            calCnt = tokenCnt * axisK_;
        }
        uint32_t topkIdxOffset = (startTokenId_ + groupOffset) * axisK_;
        uint32_t tmpOffset = Ceil(calCnt * sizeof(int16_t), ALIGNED_LEN_256) * ALIGNED_LEN_256;
        uint32_t calCntAlign = tmpOffset / sizeof(int16_t);
        LocalTensor<int16_t> tempTensorInt16 = dstExpBuf_.GetWithOffset<int16_t>(calCntAlign, 0);
        LocalTensor<int16_t> rawRankTensor = dstExpBuf_.GetWithOffset<int16_t>(calCntAlign, perGroupSizeAlign_);
        asc_copy_gm2ub_align((__ubuf__ uint32_t*)topkIdsGroupTensor.GetPhyAddr(),
                             (__gm__ uint32_t*)topkIdxGMTensor_[topkIdxOffset].GetPhyAddr(), 1U,
                             calCnt * sizeof(int32_t), 0U, 0U, false, asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM, 0U,
                             0U); // copy topkId
        asc_sync_notify(PIPE_MTE2, PIPE_V, eventId0);
        asc_sync_wait(PIPE_MTE2, PIPE_V, eventId0);

        if (group > 0) {
            asc_sync_wait(PIPE_MTE3, PIPE_V, eventId4);
        }
        asc_vf_call<MoeEpDispatchVf::DedupAndCalCnt>(
            reinterpret_cast<__ubuf__ int32_t*>(topkIdsGroupTensor.GetPhyAddr()),
            reinterpret_cast<__ubuf__ int16_t*>(tempTensorInt16.GetPhyAddr()),
            reinterpret_cast<__ubuf__ int16_t*>(rawRankTensor.GetPhyAddr()),
            reinterpret_cast<__ubuf__ int16_t*>(dstRankTensor_.GetPhyAddr()),
            reinterpret_cast<__ubuf__ int32_t*>(sendCntPerExpertTensor_.GetPhyAddr()),
            reinterpret_cast<__ubuf__ int32_t*>(sendCntPerRankTensor_.GetPhyAddr()), calCnt, tokenCnt, axisK_,
            moeExpertNum_, epWorldSize_, moeExpertNumPerRank_);

        asc_sync_notify(PIPE_V, PIPE_MTE3, eventId4);
        asc_sync_wait(PIPE_V, PIPE_MTE3, eventId4);
        if (multiGroup_) {
            asc_copy_ub2gm_align((__gm__ uint8_t*)dstRankGMTensor_[topkIdxOffset].GetPhyAddr(),
                                 (__ubuf__ uint8_t*)dstRankTensor_.GetPhyAddr(), 1U, calCnt * sizeof(int16_t),
                                 asc_store_l2_cache_mode::NORMAL_FIRST_VICTIM, 0U, 0U);
        }
        if (group + 1U < groupCnt) {
            asc_sync_notify(PIPE_V, PIPE_MTE2, eventId0);
            asc_sync_wait(PIPE_V, PIPE_MTE2, eventId0);
            asc_sync_notify(PIPE_MTE3, PIPE_V, eventId4);
        }
    }
    // 将本核计数写出到 workspace，并累加全局 rank/expert 计数。
    asc_copy_ub2gm_align((__gm__ uint8_t*)scaleupCounterGMTensor_[aivId_ * counterAlign512_].GetPhyAddr(),
                         (__ubuf__ uint8_t*)sendCntPerRankTensor_.GetPhyAddr(), epWorldSizeAlign_);
    asc_set_atomic_add_int32();
    asc_copy_ub2gm_align((__gm__ int32_t*)sendCntPerRankGMTensor_.GetPhyAddr(),
                         (__ubuf__ int32_t*)sendCntPerRankTensor_.GetPhyAddr(), 1U, epWorldSizeBytes_,
                         asc_store_l2_cache_mode::NORMAL_FIRST_VICTIM, 0U, 0U);
    asc_copy_ub2gm_align((__gm__ int32_t*)sendCntPerExpertGMTensor_.GetPhyAddr(),
                         (__ubuf__ int32_t*)sendCntPerExpertTensor_.GetPhyAddr(), 1U, moeExpertNumBytes_,
                         asc_store_l2_cache_mode::NORMAL_FIRST_VICTIM, 0U, 0U);
    asc_disable_dma_atomic();
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::Communication()
{
    // 通信初始化
    hcommTensor_ = hcommBuf_.Get<uint8_t>();
    hcomm_.Init(hcommTensor_, HCOMM_INIT_SIZE);
    SplitToCore(epWorldSize_, aivNum_, startRankId_, endRankId_, rankNumPerCore_); // 按卡分核
    coreAssigned_ = GetCoreAssignment(copyRankStart_, copyRankNum_, channelIndex_, coreIndexInGroup_, groupSize_);
    if (!coreAssigned_ || channelIndex_ != 0) {
        return;
    }

    event_t eventId1 = static_cast<event_t>(SYNC_EVENT_ID1);
    event_t eventId2 = static_cast<event_t>(SYNC_EVENT_ID2);
    asc_copy_gm2ub_align((__ubuf__ uint8_t*)sendCntPerRankTensor_.GetPhyAddr(),
                         (__gm__ uint8_t*)sendCntPerRankGMTensor_.GetPhyAddr(), epWorldSizeAlign_);
    asc_sync_notify(PIPE_MTE2, PIPE_S, eventId1);
    asc_sync_wait(PIPE_MTE2, PIPE_S, eventId1);

    for (uint32_t dstRankId = copyRankStart_; dstRankId < copyRankStart_ + copyRankNum_; ++dstRankId) {
        int32_t sendCnt = sendCntPerRankTensor_.GetValue(dstRankId);
        uint64_t notifyVal = (static_cast<uint64_t>(sendCnt) << NOTIFY_VAL_SHIFT) | 1U;
        uint32_t srcStride = dstRankId * moeExpertNumPerRank_;
        // 计算目标窗口地址:
        GM_ADDR remoteStateAddr = GetWinAddrByRankId(mc2Context_, dstRankId, cntWinStateOffset_);
        GM_ADDR notifyAddr = remoteStateAddr + dstRankStateOffset_;
        GM_ADDR remoteCountAddr = remoteStateAddr + cntFlagWinSize_ + epRankId_ * moeNumPerRankAlign512_;
        GM_ADDR srcWorkspaceAddr = sendCntWorkspaceAddr_ + epWorldSizeAlign512_ + srcStride * sizeof(int32_t);
        if (dstRankId != epRankId_) { // 远端 使用URMA发送 count + state
            uint64_t commHandle = GetCommHandle(mc2Context_, dstRankId, channelIndex_);
            hcomm_.WriteWithNotifyNbi<true, PIPE_S, PIPE_MTE3, DATA_CFG>(commHandle, remoteCountAddr, srcWorkspaceAddr,
                                                                         moeNumPerRankBytes_, notifyAddr, notifyVal);
            continue;
        }

        // 本端
        GlobalTensor<int32_t> countGMTensor;
        GlobalTensor<uint64_t> notifyGMTensor;
        countGMTensor.SetGlobalBuffer((__gm__ int32_t*)remoteCountAddr);
        notifyGMTensor.SetGlobalBuffer((__gm__ uint64_t*)notifyAddr);
        LocalTensor<uint64_t> notifyLocalTensor = numRecvPerExpertBuf_.GetWithOffset<uint64_t>(INT64_UB_STRIDE, 0);
        LocalTensor<int32_t> cntPerExpertTensor =
            numRecvPerExpertBuf_.GetWithOffset<int32_t>(expertPerRankAlign_, UB_ALIGN);
        asc_set_copy_pad_val(static_cast<uint8_t>(0));
        asc_copy_gm2ub_align((__ubuf__ uint8_t*)cntPerExpertTensor.GetPhyAddr(),
                             (__gm__ uint8_t*)sendCntPerExpertGMTensor_[srcStride].GetPhyAddr(), 1U,
                             moeNumPerRankBytes_, 0U, 0U, true, asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM, 0U, 0U);
        notifyLocalTensor.SetValue(0, notifyVal);
        asc_sync_notify(PIPE_S, PIPE_MTE3, eventId1);
        asc_sync_notify(PIPE_MTE2, PIPE_MTE3, eventId2);
        asc_sync_wait(PIPE_MTE2, PIPE_MTE3, eventId2);
        asc_copy_ub2gm_align((__gm__ uint8_t*)countGMTensor.GetPhyAddr(),
                             (__ubuf__ uint8_t*)cntPerExpertTensor.GetPhyAddr(), 1U, moeNumPerRankBytes_,
                             asc_store_l2_cache_mode::NORMAL_FIRST_VICTIM, 0U, 0U);
        asc_sync_wait(PIPE_S, PIPE_MTE3, eventId1);
        asc_sync_pipe(PIPE_MTE3); // perExpert 写完再写notifyVal
        asc_copy_ub2gm_align((__gm__ uint8_t*)notifyGMTensor.GetPhyAddr(),
                             (__ubuf__ uint8_t*)notifyLocalTensor.GetPhyAddr(), UB_ALIGN);
        if (aivId_ == aivNum_ - 1) { // numRecvPerExpertBuf_ reuse
            asc_sync_notify(PIPE_MTE3, PIPE_V, eventId1);
            asc_sync_wait(PIPE_MTE3, PIPE_V, eventId1);
        }
    }
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::GetRecvCount()
{
    if (aivId_ != aivNum_ - 1) { // 最后一个核处理recvCount
        return;
    }

    uint32_t recvNumAlign = epWorldSize_ * expertPerRankAlign_;
    event_t eventId0 = static_cast<event_t>(SYNC_EVENT_ID0);
    event_t eventId1 = static_cast<event_t>(SYNC_EVENT_ID1);
    GlobalTensor<int32_t> localExpertRecvGMTensor;
    localExpertRecvGMTensor.SetGlobalBuffer((__gm__ int32_t*)(localCntStateWinAddr_ + cntFlagWinSize_));
    numRecvPerExpertTensor_ = numRecvPerExpertBuf_.Get<int64_t>();
    LocalTensor<int64_t> recvCntTensor = dstExpBuf_.GetWithOffset<int64_t>(INT64_UB_STRIDE, 0);
    LocalTensor<int32_t> recvTensorInt32 = dstExpBuf_.GetWithOffset<int32_t>(recvNumAlign, UB_ALIGN);
    LocalTensor<int32_t> recvCounterTensor = tempBuf_.GetWithOffset<int32_t>(epWorldSize_ * UB_STRIDE, 0);
    WaitRecvCnt(recvCounterTensor); // check wait count
    // The polling reduction used dstExpBuf_; release it before prefetching expert counts.
    asc_sync_notify(PIPE_V, PIPE_MTE2, eventId0);
    asc_sync_wait(PIPE_V, PIPE_MTE2, eventId0);
    asc_set_copy_pad_val(static_cast<uint8_t>(0));
    asc_copy_gm2ub_align((__ubuf__ uint8_t*)recvTensorInt32.GetPhyAddr(),
                         (__gm__ uint8_t*)localExpertRecvGMTensor.GetPhyAddr(), epWorldSize_, moeNumPerRankAlign_, 0U,
                         0U, true, asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM, moeNumPerRankAlign512_,
                         moeNumPerRankAlign_);
    asc_sync_notify(PIPE_MTE2, PIPE_V, eventId0);

    asc_vf_call<MoeEpDispatchVf::ExtractRecvCountPerRank>(
        reinterpret_cast<__ubuf__ int32_t*>(recvCounterTensor.GetPhyAddr()),
        reinterpret_cast<__ubuf__ int32_t*>(numRecvPerRankTensor_.GetPhyAddr()), epWorldSize_);
    asc_sync_notify(PIPE_V, PIPE_MTE3, eventId1);
    asc_sync_wait(PIPE_V, PIPE_MTE3, eventId1);
    asc_copy_ub2gm_align((__gm__ uint8_t*)numRecvPerRankGMTensor_.GetPhyAddr(),
                         (__ubuf__ uint8_t*)numRecvPerRankTensor_.GetPhyAddr(), 1U, epWorldSizeBytes_,
                         asc_store_l2_cache_mode::NORMAL_FIRST_VICTIM, 0U, 0U);

    asc_sync_wait(PIPE_MTE2, PIPE_V, eventId0);
    asc_vf_call<MoeEpDispatchVf::CalRecvCountPerExpert<DoCpuSync>>(
        reinterpret_cast<__ubuf__ int32_t*>(recvTensorInt32.GetPhyAddr()),
        reinterpret_cast<__ubuf__ int64_t*>(numRecvPerExpertTensor_.GetPhyAddr()),
        reinterpret_cast<__ubuf__ int64_t*>(recvCntTensor.GetPhyAddr()), epWorldSize_, moeExpertNumPerRank_,
        expertPerRankAlign_);
    asc_sync_notify(PIPE_V, PIPE_MTE3, eventId0);
    asc_sync_wait(PIPE_V, PIPE_MTE3, eventId0);

    if constexpr (DoCpuSync) {
        GlobalTensor<int64_t> hostPinnedCounterTensor;
        hostPinnedCounterTensor.SetGlobalBuffer((__gm__ int64_t*)hostPinnedCounterAddrGM_);
        asc_copy_ub2gm_align((__gm__ uint8_t*)hostPinnedCounterTensor.GetPhyAddr(),
                             (__ubuf__ uint8_t*)recvCntTensor.GetPhyAddr(), UB_ALIGN);
    }
    asc_copy_ub2gm_align((__gm__ uint8_t*)numRecvPerExpertGMTensor_.GetPhyAddr(),
                         (__ubuf__ uint8_t*)numRecvPerExpertTensor_.GetPhyAddr(), 1U,
                         moeExpertNumPerRank_ * sizeof(int64_t), asc_store_l2_cache_mode::NORMAL_FIRST_VICTIM, 0U, 0U);
    asc_copy_ub2gm_align((__gm__ uint8_t*)recvCounterGMTensor_.GetPhyAddr(),
                         (__ubuf__ uint8_t*)recvCounterTensor.GetPhyAddr(), epWorldSize_, UB_ALIGN,
                         asc_store_l2_cache_mode::NORMAL_FIRST_VICTIM, WIN_ADDR_ALIGN, UB_ALIGN);
    diagWriter_.RunPosRecord(MOE_EP_DISPATCH_RUN_POS_COUNT_READY);
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::WaitRecvCnt(LocalTensor<int32_t>& recvTensor)
{
    uint32_t mask = 1;
    int32_t sumOfFlag = -1;
    int32_t compareFlag = static_cast<int32_t>(epWorldSize_);
    event_t eventId0 = static_cast<event_t>(SYNC_EVENT_ID0);
    LocalTensor<float> tempFp32 = dstExpBuf_.GetWithOffset<float>(epWorldSize_ * UB_STRIDE, 0);
    LocalTensor<float> recvTensorFp32 = recvTensor.template ReinterpretCast<float>();
    LocalTensor<float> numRecvPerRankTensorFp32 = numRecvPerRankTensor_.template ReinterpretCast<float>();
    asc_sync_notify(PIPE_V, PIPE_MTE2, eventId0);
    asc_sync_wait(PIPE_V, PIPE_MTE2, eventId0); // buf 复用
    while (sumOfFlag != compareFlag) {          // 状态位check
        asc_copy_gm2ub_align((__ubuf__ uint8_t*)recvTensor.GetPhyAddr(),
                             (__gm__ uint8_t*)recvCounterGMTensor_.GetPhyAddr(), epWorldSize_, UB_ALIGN, 0U, 0U, false,
                             asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM, WIN_ADDR_ALIGN, UB_ALIGN);
        asc_sync_notify(PIPE_MTE2, PIPE_V, eventId0);
        asc_sync_wait(PIPE_MTE2, PIPE_V, eventId0);
        ReduceSum(numRecvPerRankTensorFp32, recvTensorFp32, tempFp32, mask, epWorldSize_, 1);
        asc_sync_notify(PIPE_V, PIPE_S, eventId0);
        asc_sync_wait(PIPE_V, PIPE_S, eventId0);
        sumOfFlag = numRecvPerRankTensor_.GetValue(0);
    }
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::GetSlotStartNum()
{
    slotIdxPerRankTensor_ = sendCntExpertBuf_.GetWithOffset<int32_t>(counterCnt_, 0);
    if (aivId_ == 0) { // 0核起始id 均为0
        Duplicate<int32_t>(slotIdxPerRankTensor_, 0, epWorldSize_);
        return;
    }
    event_t eventId2 = static_cast<event_t>(SYNC_EVENT_ID2);
    if (aivId_ == 1U) { // 首个计数行即当前核的前缀；SendPhase 通过后续 V->S 同步后由标量读取。
        asc_copy_gm2ub_align((__ubuf__ uint8_t*)slotIdxPerRankTensor_.GetPhyAddr(),
                             (__gm__ uint8_t*)scaleupCounterGMTensor_.GetPhyAddr(), epWorldSizeAlign_);
        asc_sync_notify(PIPE_MTE2, PIPE_V, eventId2);
        asc_sync_wait(PIPE_MTE2, PIPE_V, eventId2);
        return;
    }

    uint32_t copyNumPerGroup = perGroupSizeAlign_ * 2 / epWorldSizeAlign_;
    uint32_t groupCnt = Ceil(aivId_, copyNumPerGroup);
    uint32_t firstCopyNum = aivId_ < copyNumPerGroup ? aivId_ : copyNumPerGroup;
    event_t eventId0 = static_cast<event_t>(SYNC_EVENT_ID0);
    LocalTensor<int32_t> counterTmpTensor = topkIdsBuf_.Get<int32_t>();
    LocalTensor<int32_t> hitTmpTensor = sendCntRankBuf_.Get<int32_t>();
    LocalTensor<uint8_t> sharedTmpTensor = dstExpBuf_.Get<uint8_t>();
    asc_set_copy_pad_val(static_cast<uint8_t>(0));
    asc_copy_gm2ub_align((__ubuf__ uint8_t*)counterTmpTensor.GetPhyAddr(),
                         (__gm__ uint8_t*)scaleupCounterGMTensor_.GetPhyAddr(), firstCopyNum, epWorldSizeAlign_, 0U, 0U,
                         true, asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM, epWorldSizeAlign512_, epWorldSizeAlign_);
    asc_sync_notify(PIPE_MTE2, PIPE_V, eventId2);
    asc_sync_wait(PIPE_MTE2, PIPE_V, eventId2);
    const uint32_t firstShape[] = {firstCopyNum, counterCnt_};
    ReduceSum<int32_t, AscendC::Pattern::Reduce::RA, true>(slotIdxPerRankTensor_, counterTmpTensor, sharedTmpTensor,
                                                           firstShape, true);
    if (groupCnt > 1U) {
        asc_sync_notify(PIPE_V, PIPE_MTE2, eventId0);
    }

    for (uint32_t i = 1U; i < groupCnt; i++) {
        asc_sync_wait(PIPE_V, PIPE_MTE2, eventId0); // Only the reduction reads counterTmpTensor.
        uint32_t copyNum = (i == groupCnt - 1) ? (aivId_ - copyNumPerGroup * i) : copyNumPerGroup;
        uint32_t gmOffset = i * copyNumPerGroup * epWorldSizeAlign512_ / sizeof(int32_t);
        asc_set_copy_pad_val(static_cast<uint8_t>(0));
        asc_copy_gm2ub_align((__ubuf__ uint8_t*)counterTmpTensor.GetPhyAddr(),
                             (__gm__ uint8_t*)scaleupCounterGMTensor_[gmOffset].GetPhyAddr(), copyNum,
                             epWorldSizeAlign_, 0U, 0U, true, asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM,
                             epWorldSizeAlign512_, epWorldSizeAlign_);
        asc_sync_notify(PIPE_MTE2, PIPE_V, eventId2);
        asc_sync_wait(PIPE_MTE2, PIPE_V, eventId2);
        const uint32_t shape[] = {copyNum, counterCnt_};
        ReduceSum<int32_t, AscendC::Pattern::Reduce::RA, true>(hitTmpTensor, counterTmpTensor, sharedTmpTensor, shape,
                                                               true);
        if (i + 1U < groupCnt) {
            asc_sync_notify(PIPE_V, PIPE_MTE2, eventId0);
        }
        Add(slotIdxPerRankTensor_, slotIdxPerRankTensor_, hitTmpTensor, epWorldSize_);
    }
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline bool MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::SetLocalSrcTokenIdx(uint32_t topkOffset,
                                                                                         uint32_t calCnt,
                                                                                         uint32_t calCntAlign,
                                                                                         uint32_t& sendLocalCnt)
{
    uint64_t rsvdCnt = 0;
    event_t eventId0 = static_cast<event_t>(SYNC_EVENT_ID0);
    event_t eventId2 = static_cast<event_t>(SYNC_EVENT_ID2);
    LocalTensor<int32_t> internalIndexTensor = dstExpBuf_.Get<int32_t>();
    LocalTensor<int32_t> localIndexTensor = topkIdsBuf_.Get<int32_t>();
    LocalTensor<uint32_t> gatherMaskTensor = maskBuf_.Get<uint32_t>();
    LocalTensor<uint8_t> compareMaskTensor = maskBuf_.Get<uint8_t>();

    CreateVecIndex(internalIndexTensor, static_cast<int32_t>(topkOffset), calCnt);
    Duplicate<int32_t>(localIndexTensor, static_cast<int32_t>(axisK_), calCnt);
    Div(internalIndexTensor, internalIndexTensor, localIndexTensor, calCnt);

    if (multiGroup_) {
        asc_copy_gm2ub_align((__ubuf__ uint16_t*)dstRankTensor_.GetPhyAddr(),
                             (__gm__ uint16_t*)dstRankGMTensor_[topkOffset].GetPhyAddr(), 1U, calCnt * sizeof(int16_t),
                             0U, 0U, false, asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM, 0U, 0U);
        asc_sync_notify(PIPE_MTE2, PIPE_V, eventId0);
        asc_sync_wait(PIPE_MTE2, PIPE_V, eventId0);
    }
    CompareScalar(compareMaskTensor, dstRankTensor_, static_cast<int16_t>(epRankId_), AscendC::CMPMODE::EQ,
                  calCntAlign);
    GatherMask(localIndexTensor, internalIndexTensor, gatherMaskTensor, true, calCnt, {1, 1, 0, 0}, rsvdCnt);
    asc_sync_notify(PIPE_V, PIPE_S, eventId2);
    asc_sync_wait(PIPE_V, PIPE_S, eventId2); // Publish the local table to ProcessMetaSlot's scalar reader.
    sendLocalCnt = static_cast<uint32_t>(rsvdCnt);
    if (sendLocalCnt == 0) {
        return false;
    }
    asc_sync_notify(PIPE_V, PIPE_MTE3, eventId0);
    int32_t curSlot = slotIdxPerRankTensor_.GetValue(epRankId_);
    uint64_t dstIndex = static_cast<uint64_t>(epRankId_) * dstSlotStride_ + 1U + curSlot;
    asc_sync_wait(PIPE_V, PIPE_MTE3, eventId0);
    asc_copy_ub2gm_align((__gm__ uint8_t*)dstSlotIdxGMTensor_[dstIndex].GetPhyAddr(),
                         (__ubuf__ uint8_t*)localIndexTensor.GetPhyAddr(), 1U, sendLocalCnt * sizeof(int32_t),
                         asc_store_l2_cache_mode::NORMAL_FIRST_VICTIM, 0U, 0U);
    // This marker precedes metadata copies; zero-hit groups need no event or wait.
    asc_sync_notify(PIPE_MTE3, PIPE_V, static_cast<event_t>(SYNC_EVENT_ID5));
    slotIdxPerRankTensor_.SetValue(epRankId_, curSlot + static_cast<int32_t>(sendLocalCnt));
    return true;
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::SetRemoteSrcTokenIdx(uint32_t tokenCnt,
                                                                                          uint32_t calCnt,
                                                                                          uint32_t calCntAlign,
                                                                                          bool localUsed)
{
    uint32_t slotBytes = Ceil(tokenCnt * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    uint32_t slotStride = slotBytes / sizeof(int32_t);
    uint32_t slotBuffSize = 2U * perGroupSizeAlign_;
    uint32_t slotCount = slotBuffSize / slotBytes;
    // 多槽时 UB 索引槽 0 保留给本端，其他槽供远端轮转；单槽时需先等待本端消费者释放。
    uint32_t slotStartIndex = slotCount == 1 ? 0 : 1;
    uint32_t remoteSlotCount = slotCount - slotStartIndex;
    // 每个事件保护一组 UB 索引槽，组内搬出完成后才能覆盖；事件不与目标 rank 固定绑定。
    uint32_t slotsPerEvent = Ceil(remoteSlotCount, REMOTE_INDEX_EVENT_COUNT);
    uint32_t bufferIdx = 0U;
    uint64_t rsvdCnt = 0;
    LocalTensor<int32_t> internalIndexTensor = dstExpBuf_.Get<int32_t>();
    LocalTensor<uint32_t> gatherMaskTensor = maskBuf_.Get<uint32_t>();
    LocalTensor<uint8_t> compareMaskTensor = maskBuf_.Get<uint8_t>();
    event_t eventId1 = static_cast<event_t>(SYNC_EVENT_ID1);
    event_t eventId3 = static_cast<event_t>(SYNC_EVENT_ID3);
    event_t eventId4 = static_cast<event_t>(SYNC_EVENT_ID4);
    event_t eventId5 = static_cast<event_t>(SYNC_EVENT_ID5);

    if (localUsed && slotStartIndex == 0) {
        asc_sync_wait(PIPE_MTE3, PIPE_V, eventId5);
        asc_sync_notify(PIPE_S, PIPE_V, eventId3);
        asc_sync_wait(PIPE_S, PIPE_V, eventId3);
    }

    for (uint32_t rankId = 0; rankId < epWorldSize_; ++rankId) {
        if (rankId == epRankId_) { // 本端卡已经处理，跳过
            continue;
        }
        CompareScalar(compareMaskTensor, dstRankTensor_, static_cast<int16_t>(rankId), AscendC::CMPMODE::EQ,
                      calCntAlign);
        uint32_t eventIdx = bufferIdx / slotsPerEvent;
        event_t bufferEvent = static_cast<event_t>(eventIdx);
        uint32_t eventMask = 1U << eventIdx;
        if ((remoteIndexPendingEvents_ & eventMask) != 0U) {
            // Wait for this slot group before overwriting it; later groups may still be copying.
            asc_sync_wait(PIPE_MTE3, PIPE_V, bufferEvent);
            remoteIndexPendingEvents_ &= ~eventMask;
        }
        uint32_t slotBuffer = (slotStartIndex + bufferIdx) * slotBytes;
        LocalTensor<int32_t> tempLocalTensor = topkIdsBuf_.GetWithOffset<int32_t>(slotStride, slotBuffer);
        GatherMask(tempLocalTensor, internalIndexTensor, gatherMaskTensor, true, calCnt, {1, 1, 0, 0}, rsvdCnt);
        if (rsvdCnt == 0) {
            continue;
        }

        asc_sync_notify(PIPE_V, PIPE_MTE3, eventId4);
        int32_t curSlot = slotIdxPerRankTensor_.GetValue(rankId);
        asc_sync_notify(PIPE_V, PIPE_S, eventId1);
        asc_sync_wait(PIPE_V, PIPE_S, eventId1);
        int32_t curCnt = static_cast<int32_t>(rsvdCnt);
        uint64_t dstIndex = static_cast<uint64_t>(rankId) * dstSlotStride_ + 1 + curSlot;
        asc_sync_wait(PIPE_V, PIPE_MTE3, eventId4);
        asc_copy_ub2gm_align((__gm__ uint8_t*)dstSlotIdxGMTensor_[dstIndex].GetPhyAddr(),
                             (__ubuf__ uint8_t*)tempLocalTensor.GetPhyAddr(), 1U,
                             static_cast<uint32_t>(curCnt) * sizeof(int32_t),
                             asc_store_l2_cache_mode::NORMAL_FIRST_VICTIM, 0U, 0U);
        slotIdxPerRankTensor_.SetValue(rankId, curSlot + curCnt);
        rsvdCnt = 0;
        bufferIdx++;
        if (bufferIdx % slotsPerEvent == 0U || bufferIdx == remoteSlotCount) {
            asc_sync_notify(PIPE_MTE3, PIPE_V, bufferEvent);
            remoteIndexPendingEvents_ |= eventMask;
        }
        if (bufferIdx == remoteSlotCount) {
            bufferIdx = 0U;
        }
    }
    if (bufferIdx % slotsPerEvent != 0U) {
        // Publish the last partially filled group; zero-hit ranks did not advance the ring.
        uint32_t eventIdx = bufferIdx / slotsPerEvent;
        event_t bufferEvent = static_cast<event_t>(eventIdx);
        asc_sync_notify(PIPE_MTE3, PIPE_V, bufferEvent);
        remoteIndexPendingEvents_ |= 1U << eventIdx;
    }
    if (localUsed && slotCount > 1) {
        asc_sync_wait(PIPE_MTE3, PIPE_V, eventId5);
    }
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::DrainRemoteIndexEvents()
{
    for (uint32_t eventIdx = 0U; eventIdx < REMOTE_INDEX_EVENT_COUNT; ++eventIdx) {
        if ((remoteIndexPendingEvents_ & (1U << eventIdx)) == 0U) {
            continue;
        }
        asc_sync_wait(PIPE_MTE3, PIPE_V, static_cast<event_t>(eventIdx));
    }
    remoteIndexPendingEvents_ = 0U;
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
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::CopyInMetaInfo(
    uint32_t startTokenId, uint32_t batchCnt, uint32_t bufferIdx, uint32_t bufferStride, uint32_t bufferOffset)
{
    event_t eventId = static_cast<event_t>(SYNC_EVENT_ID3 + bufferIdx);
    uint32_t topkOffset = startTokenId * axisK_;
    LocalTensor<int32_t> metaCopyInTensor = metaSlotBuf_.GetWithOffset<int32_t>(bufferStride, bufferIdx * bufferOffset);
    if (metaBufferUsed_[bufferIdx]) {
        // 同一半 buffer 再次写入前，分别等待 MTE3 释放 V 和 MTE2 将要覆盖的区域。
        asc_sync_wait(PIPE_MTE3, PIPE_V, eventId);
        asc_sync_wait(PIPE_MTE3, PIPE_MTE2, eventId);
        metaBufferUsed_[bufferIdx] = false;
    }

    // V 写 rank/token，MTE2 写 scales/topk/weights，字段区域不重叠，可独立准备。
    CreateVecIndex<int32_t>(tokenValueTensor_, static_cast<int32_t>(startTokenId), batchCnt);
    Scatter(metaCopyInTensor, rankValueTensor_, rankOffsetTensor_, 0U, batchCnt);
    Scatter(metaCopyInTensor, tokenValueTensor_, tokenOffsetTensor_, 0U, batchCnt);

    if constexpr (Std::IsSame<XType, fp8_e5m2_t>::value || Std::IsSame<XType, fp8_e4m3fn_t>::value) {
        asc_set_copy_pad_val(static_cast<uint8_t>(0));
        asc_copy_gm2ub_align(
            (__ubuf__ uint8_t*)metaCopyInTensor.GetPhyAddr(),
            (__gm__ uint8_t*)scalesGMTensor_
                .template ReinterpretCast<uint8_t>()[static_cast<uint64_t>(startTokenId) * scalesBytes_]
                .GetPhyAddr(),
            batchCnt, scalesBytes_, 0U, 0U, true, asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM, scalesBytes_,
            metaSlotBytes_);
    }
    asc_set_copy_pad_val(static_cast<uint8_t>(0));
    asc_copy_gm2ub_align((__ubuf__ uint8_t*)metaCopyInTensor[topkCopyAlign_].GetPhyAddr(),
                         (__gm__ uint8_t*)topkIdxGMTensor_[topkOffset].GetPhyAddr(), batchCnt, topkSizeBytes_, 0U, 0U,
                         true, asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM, topkSizeBytes_, metaSlotBytes_);
    if constexpr (IsTopkWeights) {
        asc_set_copy_pad_val(static_cast<uint8_t>(0));
        asc_copy_gm2ub_align((__ubuf__ uint8_t*)metaCopyInTensor[topkCopyAlign_ + axisKAlign_].GetPhyAddr(),
                             (__gm__ uint8_t*)topkWeightsGMTensor_[topkOffset].GetPhyAddr(), batchCnt, topkSizeBytes_,
                             0U, 0U, true, asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM, topkSizeBytes_, metaSlotBytes_);
    }
    // 搬出 metadata 前，MTE3 需同时等待 MTE2 搬入字段和 V 生成字段就绪。
    asc_sync_notify(PIPE_MTE2, PIPE_MTE3, eventId);
    asc_sync_notify(PIPE_V, PIPE_MTE3, eventId);
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::DrainMetaBuffers()
{
    // wait 消费 metaSlotBuf_ 尚未回收的释放事件
    for (uint32_t bufferIdx = 0U; bufferIdx < BUFFER_NUM; ++bufferIdx) {
        if (!metaBufferUsed_[bufferIdx]) {
            continue;
        }
        event_t eventId = static_cast<event_t>(SYNC_EVENT_ID3 + bufferIdx);
        asc_sync_wait(PIPE_MTE3, PIPE_MTE2, eventId);
        asc_sync_wait(PIPE_MTE3, PIPE_V, eventId);
        metaBufferUsed_[bufferIdx] = false;
    }
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::ProcessMetaSlot(uint32_t startId,
                                                                                     uint32_t tokenCnt,
                                                                                     uint32_t sendLocalCnt,
                                                                                     uint32_t& curLocalSlot)
{
    // meta 流水写 stash + 本端命中内联写本地窗 meta 区
    if (tokenCnt == 0U) {
        return;
    }
    const uint32_t bufferStride = metaBatch_ * metaSlotStride_; // 每半 buffer 的 int32 元素数
    const uint32_t bufferOffset = metaBatch_ * metaSlotBytes_;  // 两半 buffer 间的字节偏移
    uint32_t bufferIdx = 0U;
    uint32_t localIdx = 0U;
    uint32_t tokenEndId = startId + tokenCnt;
    uint32_t firstBatchCnt = tokenCnt < metaBatch_ ? tokenCnt : metaBatch_;
    LocalTensor<int32_t> srcTokenTensor = topkIdsBuf_.Get<int32_t>();
    CopyInMetaInfo(startId, firstBatchCnt, bufferIdx, bufferStride, bufferOffset);
    if constexpr (IsCached) {
        asc_sync_wait(PIPE_MTE2, PIPE_S,
                      static_cast<event_t>(SYNC_EVENT_ID0)); // 已提交首批 metadata 准备，等待索引就绪
    }

    for (uint32_t batchStartId = startId; batchStartId < tokenEndId; batchStartId += metaBatch_) {
        uint32_t batchCnt = tokenEndId - batchStartId;
        batchCnt = batchCnt < metaBatch_ ? batchCnt : metaBatch_;
        uint32_t batchEndId = batchStartId + batchCnt;
        event_t eventId = static_cast<event_t>(SYNC_EVENT_ID3 + bufferIdx);
        metaLocalTensor_ = metaSlotBuf_.GetWithOffset<int32_t>(bufferStride, bufferIdx * bufferOffset);
        if (batchEndId < tokenEndId) {
            uint32_t nextBatchCnt = tokenEndId - batchEndId;
            nextBatchCnt = nextBatchCnt < metaBatch_ ? nextBatchCnt : metaBatch_;
            CopyInMetaInfo(batchEndId, nextBatchCnt, 1U - bufferIdx, bufferStride, bufferOffset);
        }

        asc_sync_wait(PIPE_V, PIPE_MTE3, eventId);
        asc_sync_wait(PIPE_MTE2, PIPE_MTE3, eventId);
        asc_copy_ub2gm_align(
            (__gm__ uint8_t*)metaSlotGMTensor_[static_cast<uint64_t>(batchStartId) * metaSlotStride_].GetPhyAddr(),
            (__ubuf__ uint8_t*)metaLocalTensor_.GetPhyAddr(), 1U, batchCnt * metaSlotBytes_,
            asc_store_l2_cache_mode::NORMAL_FIRST_VICTIM, 0U, 0U);

        for (; localIdx < sendLocalCnt; ++localIdx) { // 本地tokenId递增，只遍历当前batch内的本地命中项。
            uint32_t indexInBatch = static_cast<uint32_t>(srcTokenTensor.GetValue(localIdx)) - batchStartId;
            if (indexInBatch >= batchCnt) {
                break;
            }
            localMetaSlotGMTensor_.SetGlobalBuffer((
                __gm__ int32_t*)(localSlotWinAddr_ + static_cast<uint64_t>(curLocalSlot) * perSlotBytes_ + tokenSize_));
            asc_copy_ub2gm_align((__gm__ uint8_t*)localMetaSlotGMTensor_.GetPhyAddr(),
                                 (__ubuf__ uint8_t*)metaLocalTensor_[indexInBatch * metaSlotStride_].GetPhyAddr(),
                                 metaSlotBytes_);
            curLocalSlot++;
        }
        // Includes every local metadata copy, whose source address depends on the scalar index read.
        asc_sync_notify(PIPE_MTE3, PIPE_MTE2, eventId);
        asc_sync_notify(PIPE_MTE3, PIPE_V, eventId);
        metaBufferUsed_[bufferIdx] = true;
        bufferIdx = 1U - bufferIdx;
    }
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::SendPhase()
{
    if (startTokenId_ >= axisBS_) {
        return;
    }
    GetSlotStartNum(); // 计算起始slot id
    asc_sync_notify(PIPE_V, PIPE_S, static_cast<event_t>(SYNC_EVENT_ID0));
    InitMetaScatter();

    uint32_t groupCnt = Ceil(sendTokenNum_, perGroupTokenNum_);
    uint32_t tokenCnt = perGroupTokenNum_;
    uint32_t strideAlign = ALIGNED_LEN_256 / sizeof(int16_t);
    uint32_t sendLocalCnt = 0;
    event_t eventId2 = static_cast<event_t>(SYNC_EVENT_ID2);
    event_t eventId4 = static_cast<event_t>(SYNC_EVENT_ID4);

    asc_sync_wait(PIPE_V, PIPE_S, static_cast<event_t>(SYNC_EVENT_ID0));
    uint32_t curLocalSlot = static_cast<uint32_t>(slotIdxPerRankTensor_.GetValue(epRankId_));
    for (uint32_t group = 0; group < groupCnt; group++) {
        if (group == groupCnt - 1) {
            tokenCnt = sendTokenNum_ - group * perGroupTokenNum_;
        }
        uint32_t calCnt = tokenCnt * axisK_;
        uint32_t startId = startTokenId_ + group * perGroupTokenNum_;
        uint32_t topkOffset = startId * axisK_;
        uint32_t calCntAlign = Ceil(calCnt, strideAlign) * strideAlign;
        bool localUsed = SetLocalSrcTokenIdx(topkOffset, calCnt, calCntAlign, sendLocalCnt);
        ProcessMetaSlot(startId, tokenCnt, sendLocalCnt, curLocalSlot);
        if (group + 1U < groupCnt) {
            // Metadata consumes slot zero on S; release that reader before the next group's V overwrite.
            asc_sync_notify(PIPE_S, PIPE_V, eventId2);
        }
        SetRemoteSrcTokenIdx(tokenCnt, calCnt, calCntAlign, localUsed);
        if (group < groupCnt - 1) {
            asc_sync_notify(PIPE_V, PIPE_MTE2, eventId4);
            asc_sync_wait(PIPE_V, PIPE_MTE2, eventId4);
            DrainRemoteIndexEvents();
            asc_sync_wait(PIPE_S, PIPE_V, eventId2);
        }
    }
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::GetLocalSlotStart(uint32_t sendCount,
                                                                                       uint32_t maxBatchCnt,
                                                                                       uint32_t alignStride,
                                                                                       uint32_t& curLocalSlot)
{
    if (aivId_ == 0) {
        return;
    }

    uint64_t localRowStart = static_cast<uint64_t>(epRankId_) * dstSlotStride_ + 1;
    uint32_t scanCount = startTokenId_ < sendCount ? startTokenId_ : sendCount;
    uint32_t groupCnt = Ceil(scanCount, maxBatchCnt);
    LocalTensor<int32_t> srcTokenTensor = topkIdsBuf_.Get<int32_t>();
    LocalTensor<uint32_t> gatherMaskTensor = maskBuf_.Get<uint32_t>();
    LocalTensor<uint8_t> ltMaskTensor = maskBuf_.Get<uint8_t>();
    event_t eventId1 = static_cast<event_t>(SYNC_EVENT_ID1);
    event_t eventId2 = static_cast<event_t>(SYNC_EVENT_ID2);

    for (uint32_t group = 0; group < groupCnt; group++) {
        if (group > 0U) {
            asc_sync_wait(PIPE_V, PIPE_MTE2, eventId2);
        }
        uint32_t startId = group * maxBatchCnt;
        uint32_t tokenCnt = (group == groupCnt - 1) ? (scanCount - startId) : maxBatchCnt;
        uint32_t tokenCntAlign = Ceil(tokenCnt, alignStride) * alignStride;
        uint64_t rsvdCnt = 0;
        asc_copy_gm2ub_align((__ubuf__ uint32_t*)srcTokenTensor.GetPhyAddr(),
                             (__gm__ uint32_t*)cachedSlotIdxGMTensor_[localRowStart + startId].GetPhyAddr(), 1U,
                             tokenCnt * sizeof(int32_t), 0U, 0U, false, asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM, 0U,
                             0U);
        asc_sync_notify(PIPE_MTE2, PIPE_V, eventId1);
        asc_sync_wait(PIPE_MTE2, PIPE_V, eventId1);
        CompareScalar(ltMaskTensor, srcTokenTensor, static_cast<int32_t>(startTokenId_), AscendC::CMPMODE::LT,
                      tokenCntAlign);
        GatherMask(srcTokenTensor, srcTokenTensor, gatherMaskTensor, true, tokenCnt, {1, 1, 0, 0}, rsvdCnt);
        asc_sync_notify(PIPE_V, PIPE_MTE2, eventId2);
        if (rsvdCnt > 0) {
            curLocalSlot += static_cast<int32_t>(rsvdCnt);
        }
    }
    if (groupCnt > 0) {
        asc_sync_wait(PIPE_V, PIPE_MTE2, eventId2);
    }
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::SendPhaseCached()
{
    // cache 模式：从 cached_dst_slot_idx 提取本端索引，生成 metadata stash 并写出本端命中的 metadata。
    SplitToCore(axisBS_, aivNum_, startTokenId_, endTokenId_, sendTokenNum_); // 按token数量分核
    if (startTokenId_ >= axisBS_) {
        return;
    }
    InitMetaScatter();

    uint32_t localSlotStart = 0;
    uint32_t maxBatchCnt = MAX_BATCH_SIZE / sizeof(int32_t);
    uint32_t alignStride = ALIGNED_LEN_256 / sizeof(int32_t);
    uint32_t groupCnt = Ceil(sendTokenNum_, maxBatchCnt);
    bool indexReadPending = false;
    LocalTensor<int32_t> srcTokenTensor = topkIdsBuf_.Get<int32_t>();
    LocalTensor<int32_t> sendCntTensor = maskBuf_.GetWithOffset<int32_t>(UB_STRIDE, 0);
    event_t eventId0 = static_cast<event_t>(SYNC_EVENT_ID0);
    event_t eventId1 = static_cast<event_t>(SYNC_EVENT_ID1);

    asc_set_copy_pad_val(static_cast<uint8_t>(0));
    asc_copy_gm2ub_align(
        (__ubuf__ uint8_t*)sendCntTensor.GetPhyAddr(),
        (__gm__ uint8_t*)cachedSlotIdxGMTensor_[static_cast<uint64_t>(epRankId_) * dstSlotStride_].GetPhyAddr(), 1U,
        sizeof(int32_t), 0U, 0U, true, asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM, 0U, 0U);
    asc_sync_notify(PIPE_MTE2, PIPE_S, eventId0);
    asc_sync_wait(PIPE_MTE2, PIPE_S, eventId0);
    uint32_t sendCount = static_cast<uint32_t>(sendCntTensor.GetValue(0));
    asc_sync_notify(PIPE_S, PIPE_V, eventId0);
    asc_sync_wait(PIPE_S, PIPE_V, eventId0);
    GetLocalSlotStart(sendCount, maxBatchCnt, alignStride, localSlotStart);

    for (uint32_t group = 0; group < groupCnt; group++) {
        uint32_t startId = startTokenId_ + group * maxBatchCnt;
        uint32_t tokenCnt = (group == groupCnt - 1) ? (endTokenId_ - startId) : maxBatchCnt;
        uint32_t validCnt = (sendCount > localSlotStart) ? (sendCount - localSlotStart) : 0U;
        uint32_t segCnt = tokenCnt < validCnt ? tokenCnt : validCnt;
        if (segCnt > 0) {
            if (indexReadPending) {
                asc_sync_wait(PIPE_S, PIPE_MTE2, eventId1);
                indexReadPending = false;
            }
            uint64_t srcTokenOffset = static_cast<uint64_t>(epRankId_) * dstSlotStride_ + 1 + localSlotStart;
            asc_copy_gm2ub_align((__ubuf__ uint32_t*)srcTokenTensor.GetPhyAddr(),
                                 (__gm__ uint32_t*)cachedSlotIdxGMTensor_[srcTokenOffset].GetPhyAddr(), 1U,
                                 segCnt * sizeof(int32_t), 0U, 0U, false, asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM,
                                 0U, 0U);
        }
        // Published for every group so ProcessMetaSlot's matching wait never depends on segCnt.
        asc_sync_notify(PIPE_MTE2, PIPE_S, eventId0);
        ProcessMetaSlot(startId, tokenCnt, segCnt, localSlotStart);
        if (segCnt > 0) {
            asc_sync_notify(PIPE_S, PIPE_MTE2, eventId1);
            indexReadPending = true;
        }
    }
    if (indexReadPending) {
        asc_sync_wait(PIPE_S, PIPE_MTE2, eventId1);
    }
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::SetLocalStatusAndBufferInitCached()
{
    tpipe_->InitBuffer(metaSlotBuf_, BUFFER_NUM * metaBatch_ * metaSlotBytes_);
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
        event_t eventId1 = static_cast<event_t>(SYNC_EVENT_ID1);
        asc_sync_notify(PIPE_V, PIPE_MTE3, eventId1);
        asc_sync_wait(PIPE_V, PIPE_MTE3, eventId1);
        asc_copy_ub2gm_align((__gm__ uint8_t*)statusGMTensor.GetPhyAddr(), (__ubuf__ uint8_t*)statusTensor.GetPhyAddr(),
                             UB_ALIGN);
        asc_copy_ub2gm_align((__gm__ uint8_t*)notifyGMTensor.GetPhyAddr(), (__ubuf__ uint8_t*)statusTensor.GetPhyAddr(),
                             UB_ALIGN);
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
    int32_t compareFlag = static_cast<int32_t>(epWorldSize_);
    LocalTensor<int32_t> waitStatusTensor = topkIdsBuf_.GetWithOffset<int32_t>(epWorldSize_ * UB_STRIDE, 0);
    LocalTensor<int32_t> sumStatusTensor = topkIdsBuf_.GetWithOffset<int32_t>(UB_STRIDE, tmpOffset);
    LocalTensor<float> tempFp32 = topkIdsBuf_.GetWithOffset<float>(epWorldSize_ * UB_STRIDE, tmpOffset + UB_ALIGN);
    LocalTensor<float> waitStatusTensorFp32 = waitStatusTensor.template ReinterpretCast<float>();
    LocalTensor<float> sumStatusTensorFp32 = sumStatusTensor.template ReinterpretCast<float>();
    event_t eventId2 = static_cast<event_t>(SYNC_EVENT_ID2);

    while (sumOfFlag != compareFlag) { // 状态位check
        asc_copy_gm2ub_align((__ubuf__ uint8_t*)waitStatusTensor.GetPhyAddr(),
                             (__gm__ uint8_t*)recvCounterGMTensor_.GetPhyAddr(), epWorldSize_, UB_ALIGN, 0U, 0U, false,
                             asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM, WIN_ADDR_ALIGN, UB_ALIGN);
        asc_sync_notify(PIPE_MTE2, PIPE_V, eventId2);
        asc_sync_wait(PIPE_MTE2, PIPE_V, eventId2);
        ReduceSum(sumStatusTensorFp32, waitStatusTensorFp32, tempFp32, mask, epWorldSize_, 1);
        asc_sync_notify(PIPE_V, PIPE_S, eventId2);
        asc_sync_wait(PIPE_V, PIPE_S, eventId2);
        sumOfFlag = sumStatusTensor.GetValue(0);
    }
    // status clear
    Duplicate<int32_t>(waitStatusTensor, 0, epWorldSize_ * UB_STRIDE);
    asc_sync_notify(PIPE_V, PIPE_MTE3, eventId2);
    asc_sync_wait(PIPE_V, PIPE_MTE3, eventId2);
    asc_copy_ub2gm_align((__gm__ uint8_t*)recvCounterGMTensor_.GetPhyAddr(),
                         (__ubuf__ uint8_t*)waitStatusTensor.GetPhyAddr(), epWorldSize_, UB_ALIGN,
                         asc_store_l2_cache_mode::NORMAL_FIRST_VICTIM, WIN_ADDR_ALIGN, UB_ALIGN);
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
    event_t eventId5 = static_cast<event_t>(SYNC_EVENT_ID5);
    asc_sync_notify(PIPE_V, PIPE_MTE3, eventId5);
    asc_sync_wait(PIPE_V, PIPE_MTE3, eventId5);
    asc_copy_ub2gm_align((__gm__ uint8_t*)srcStatusGMTensor.GetPhyAddr(), (__ubuf__ uint8_t*)statusTensor.GetPhyAddr(),
                         UB_ALIGN);
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::CopyOutSendCnt(
    LocalTensor<int32_t> sendNumPerRankTensor)
{
    if (channelIndex_ != 0U) {
        return;
    }
    // Compact UB source: one int32 count per rank; GM rows retain the [count, indices...] layout.
    event_t eventId1 = static_cast<event_t>(SYNC_EVENT_ID1);
    asc_sync_notify(PIPE_MTE2, PIPE_MTE3, eventId1);
    asc_sync_wait(PIPE_MTE2, PIPE_MTE3, eventId1);
    asc_copy_ub2gm_align((__gm__ uint8_t*)dstSlotIdxGMTensor_.GetPhyAddr(),
                         (__ubuf__ uint8_t*)sendNumPerRankTensor.GetPhyAddr(), epWorldSize_, sizeof(int32_t),
                         asc_store_l2_cache_mode::NORMAL_FIRST_VICTIM, dstSlotStrideBytes_, sizeof(int32_t));
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::WriteToRemoteWindow()
{
    if (!coreAssigned_) {
        return;
    }

    event_t eventId4 = static_cast<event_t>(SYNC_EVENT_ID4);
    event_t eventId5 = static_cast<event_t>(SYNC_EVENT_ID5);
    bool countReadyPending = true;
    uint32_t hcommWriteWqeCnt = Ceil(HCOMM_SQE_BYTES + SGE_PER_SQE * HCOMM_SGE_BYTES, HCOMM_WQE_BYTES);
    hcommBatchBufferBytes_ = HCOMM_BATCH_CAPACITY * hcommWriteWqeCnt * HCOMM_WQE_BYTES;
    payloadStashStateWinAddr_ += channelIndex_ * WIN_ADDR_ALIGN;
    LocalTensor<int32_t> sendNumPerRankTensor = maskBuf_.GetWithOffset<int32_t>(counterCnt_, 0);
    hcommBatchTensor_ = topkIdsBuf_.GetWithOffset<uint8_t>(hcommBatchBufferBytes_, 0);
    Duplicate<uint8_t>(hcommBatchTensor_, 0, hcommBatchBufferBytes_);
    if constexpr (!IsCached) {
        asc_copy_gm2ub_align((__ubuf__ uint8_t*)sendNumPerRankTensor.GetPhyAddr(),
                             (__gm__ uint8_t*)sendCntPerRankGMTensor_.GetPhyAddr(), epWorldSizeAlign_);
    } else {
        uint64_t gmOffset = static_cast<uint64_t>(copyRankStart_) * dstSlotStride_;
        asc_copy_gm2ub_align((__ubuf__ uint32_t*)sendNumPerRankTensor.GetPhyAddr(),
                             (__gm__ uint32_t*)cachedSlotIdxGMTensor_[gmOffset].GetPhyAddr(), copyRankNum_,
                             sizeof(int32_t), 0U, 0U, false, asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM,
                             dstSlotStrideBytes_, sizeof(int32_t));
    }
    asc_sync_notify(PIPE_MTE2, PIPE_S, eventId4);

    for (uint32_t dstRankId = copyRankStart_; dstRankId < copyRankStart_ + copyRankNum_; ++dstRankId) {
        GM_ADDR notifyAddr = GetWinAddrByRankId(mc2Context_, dstRankId, slotWinStateOffset_) + dstStateWinOffset_;
        notifyAddr += channelIndex_ * WIN_ADDR_ALIGN;
        if (unlikely(dstRankId == epRankId_)) {
            if constexpr (!IsCached) {
                CopyOutSendCnt(sendNumPerRankTensor);
            }
            LocalTensor<int32_t> statusTensor = maskBuf_.GetWithOffset<int32_t>(UB_STRIDE, epWorldSizeAlign_);
            GlobalTensor<int32_t> srcStatusGMTensor;
            GlobalTensor<int32_t> dstStatusGMTensor;
            srcStatusGMTensor.SetGlobalBuffer((__gm__ int32_t*)payloadStashStateWinAddr_);
            dstStatusGMTensor.SetGlobalBuffer((__gm__ int32_t*)notifyAddr);
            asc_copy_gm2ub_align((__ubuf__ uint8_t*)statusTensor.GetPhyAddr(),
                                 (__gm__ uint8_t*)srcStatusGMTensor.GetPhyAddr(), UB_ALIGN);
            asc_sync_notify(PIPE_MTE2, PIPE_MTE3, eventId5);
            asc_sync_wait(PIPE_MTE2, PIPE_MTE3, eventId5);
            asc_copy_ub2gm_align((__gm__ uint8_t*)dstStatusGMTensor.GetPhyAddr(),
                                 (__ubuf__ uint8_t*)statusTensor.GetPhyAddr(), UB_ALIGN);
            continue;
        }

        uint64_t commHandle = GetCommHandle(mc2Context_, dstRankId, channelIndex_);
        uint32_t sendTokenNum = 0;
        if (countReadyPending) {
            asc_sync_wait(PIPE_MTE2, PIPE_S, eventId4);
            countReadyPending = false;
        }
        if constexpr (!IsCached) {
            sendTokenNum = sendNumPerRankTensor.GetValue(dstRankId);
        } else {
            sendTokenNum = sendNumPerRankTensor.GetValue(dstRankId - copyRankStart_);
        }
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
    if (countReadyPending) {
        asc_sync_wait(PIPE_MTE2, PIPE_S, eventId4);
    }
    asc_dcci_entire_out();
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::SendTokenByChannel(
    uint32_t tokenStart, uint32_t tokenNum, uint32_t dstRankId, uint64_t commHandle, GM_ADDR notifyAddr)
{
    uint32_t processed = 0;
    uint32_t commCnt = 0;
    uint32_t bufferIdx = 0U;
    event_t eventId0 = static_cast<event_t>(SYNC_EVENT_ID0);
    BufDesc localDescs[SGE_PER_SQE] = {};
    for (uint32_t i = 0; i < SGE_PER_SQE; i += SGE_PER_SLOT) { // srcLen 预填
        localDescs[i].len = tokenSize_;
        localDescs[i + 1].len = metaSlotBytes_;
    }
    // topkIdsBuf_ 前部存放 WQE batch，后部固定划分为两半索引 buffer，用于索引拷贝。
    LocalTensor<int32_t> srcTokenIdxTensor =
        topkIdsBuf_.GetWithOffset<int32_t>(BUFFER_NUM * WRITE_INDEX_BATCH_TOKENS, hcommBatchBufferBytes_);
    GlobalTensor<int32_t> srcTokenIdxGMTensor;
    srcTokenIdxGMTensor.SetGlobalBuffer((__gm__ int32_t*)(sendSrcTokenIdxAddr_ +
                                                          static_cast<uint64_t>(dstRankId) * dstSlotStrideBytes_ +
                                                          sizeof(int32_t)));
    GM_ADDR remoteWinBaseAddr = GetWinAddrByRankId(mc2Context_, dstRankId, winDataOffset_) + dstRankWinOffset_ +
                                static_cast<uint64_t>(tokenStart) * perSlotBytes_;
    uint32_t batchCnt = tokenNum < WRITE_INDEX_BATCH_TOKENS ? tokenNum : WRITE_INDEX_BATCH_TOKENS;
    uint32_t nextBatchStart = processed + batchCnt;
    uint32_t nextBatchCnt = tokenNum - nextBatchStart;
    asc_copy_gm2ub_align((__ubuf__ uint32_t*)srcTokenIdxTensor.GetPhyAddr(),
                         (__gm__ uint32_t*)srcTokenIdxGMTensor[tokenStart].GetPhyAddr(), 1U, batchCnt * sizeof(int32_t),
                         0U, 0U, false, asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM, 0U, 0U);
    asc_sync_notify(PIPE_MTE2, PIPE_S, eventId0);
    asc_sync_notify(PIPE_V, PIPE_S, eventId0);
    asc_sync_wait(PIPE_V, PIPE_S, eventId0); // hcommBatchTensor_ init

    auto batchHandle = hcomm_.MakeBatchHandle(commHandle, hcommBatchTensor_, hcommBatchBufferBytes_, remoteWinBaseAddr);
    GM_ADDR remoteWinAddr = remoteWinBaseAddr;
    while (processed < tokenNum) {
        uint32_t batchGroup = Ceil(batchCnt, SLOT_NUM_PER_SQE);
        uint32_t srcIdx = bufferIdx * WRITE_INDEX_BATCH_TOKENS;
        uint32_t slotNum = SLOT_NUM_PER_SQE;
        uint32_t sgePerSqe = SGE_PER_SQE;
        processed += batchCnt;
        if (nextBatchStart < tokenNum) {
            uint32_t nextBufferIdx = 1U - bufferIdx;
            nextBatchCnt = nextBatchCnt < WRITE_INDEX_BATCH_TOKENS ? nextBatchCnt : WRITE_INDEX_BATCH_TOKENS;
            asc_copy_gm2ub_align(
                (__ubuf__ uint32_t*)srcTokenIdxTensor[nextBufferIdx * WRITE_INDEX_BATCH_TOKENS].GetPhyAddr(),
                (__gm__ uint32_t*)srcTokenIdxGMTensor[tokenStart + nextBatchStart].GetPhyAddr(), 1U,
                nextBatchCnt * sizeof(int32_t), 0U, 0U, false, asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM, 0U, 0U);
            nextBatchStart = processed + nextBatchCnt;
            asc_sync_notify(PIPE_MTE2, PIPE_S, static_cast<event_t>(nextBufferIdx));
        }
        asc_sync_wait(PIPE_MTE2, PIPE_S, static_cast<event_t>(bufferIdx));
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
            if (processed == tokenNum && i + 1U == batchGroup) {
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
        batchCnt = nextBatchCnt;
        bufferIdx = 1 - bufferIdx;
        if (nextBatchStart < tokenNum) {
            asc_sync_notify(PIPE_S, PIPE_MTE2, static_cast<event_t>(SYNC_EVENT_ID2 + bufferIdx));
            asc_sync_wait(PIPE_S, PIPE_MTE2, static_cast<event_t>(SYNC_EVENT_ID2 + bufferIdx));
            nextBatchCnt = tokenNum - nextBatchStart;
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
            DrainRemoteIndexEvents();
        } else {
            SetLocalStatusAndBufferInitCached();
            CommunicationCached();
            SendPhaseCached();
            SetStatus();
            WaitStatusCached();
        }
        DrainMetaBuffers();
        SyncAll<true>();
        WriteToRemoteWindow();
        diagWriter_.RunPosRecord(MOE_EP_DISPATCH_RUN_POS_URMA_REQUESTS_ISSUE_DONE);
    }
}

#endif

} // namespace MoeEpDispatchImpl

#endif // MOE_EP_DISPATCH_H
