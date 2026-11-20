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
 * \file moe_ep_dispatch_clos_only.h
 * \brief Dispatch kernel implementation for Clos-only communication.
 */

#ifndef MOE_EP_DISPATCH_CLOS_ONLY_H
#define MOE_EP_DISPATCH_CLOS_ONLY_H

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
#include "../../common/op_kernel/moe_shared_jetty_context.h"

namespace MoeEpDispatchClosOnlyImpl {

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
constexpr uint32_t PER_GROUP_SIZE = 32 * 1024U;                   // 计算count 32KB per group
constexpr uint32_t NOTIFY_VAL_SHIFT = 32U;                        // notify 高32位为 sendCnt, 低32位为 state(=1)
constexpr uint32_t MAX_BATCH_SIZE = 160 * 1024U;                  // srcTokenIdx 160KB
constexpr uint32_t SGE_PER_SLOT = 2U;                             // 每 slot 两个 SGE: x 与 meta
constexpr uint32_t SGE_PER_SQE = 12U;                             // 单 SQE 的 SGE 数
constexpr uint32_t SLOT_NUM_PER_SQE = SGE_PER_SQE / SGE_PER_SLOT; // 单 SQE 的 slot 数
// A multiple of six tokens and one UB block: chunk boundaries do not add partial WQEs.
constexpr uint32_t SOURCE_INDEX_CHUNK_TOKENS = 1536U;
constexpr uint32_t SOURCE_INDEX_BUFFER_BYTES = SOURCE_INDEX_CHUNK_TOKENS * sizeof(int32_t);
constexpr uint32_t HCOMM_BATCH_CAPACITY = 128U; // UB capacity in full 12-SGE requests
constexpr uint32_t HCOMM_SGE_BYTES = 16U;
constexpr uint32_t HCOMM_SQE_BYTES = 48U;
constexpr uint32_t HCOMM_WQE_BYTES = 64U;
constexpr uint32_t HCOMM_MAX_WRITE_WQEBB =
    (HCOMM_SQE_BYTES + SGE_PER_SQE * HCOMM_SGE_BYTES + HCOMM_WQE_BYTES - 1U) / HCOMM_WQE_BYTES;
constexpr uint32_t HCOMM_BATCH_WQEBB_CAPACITY = HCOMM_BATCH_CAPACITY * HCOMM_MAX_WRITE_WQEBB;
// SQ capacity in WQEBBs, matching the original Combine template.
constexpr uint32_t HCOMM_SQ_CAPACITY = 32768U;
constexpr uint32_t HCOMM_COUNT_WQEBB = 2U; // Single-SGE WriteWithNotify includes a notify extension.
constexpr struct UrmaWqeEntry DATA_CFG = {.odr = 5, .fence = 1, .se = 0, .cqe = 0, .inlineEn = 0};
constexpr struct UrmaWqeEntry DATA_CQE_CFG = {.odr = 5, .fence = 1, .se = 0, .cqe = 1, .inlineEn = 0};
constexpr struct UrmaWqeEntry NOTIFY_CFG = {.odr = 6, .fence = 1, .se = 0, .cqe = 0, .inlineEn = 0};
constexpr struct UrmaWqeEntry NOTIFY_CQE_CFG = {.odr = 6, .fence = 1, .se = 0, .cqe = 1, .inlineEn = 0};

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
    struct TokenChunk {
        uint32_t rank;
        uint32_t start;
        uint32_t count;
        bool first;
        bool last;
    };
    __aicore__ inline void InitTilingAndRoles(const MoeEpDispatchInfo& info);
    __aicore__ inline void InitDerivedValues();
    __aicore__ inline void BindInputTensors(GM_ADDR topkIdx, GM_ADDR topkWeights, GM_ADDR scales, GM_ADDR cachedSlotIdx,
                                            GM_ADDR numRecvPerRank, GM_ADDR numRecvPerExpert, GM_ADDR dstBufferSlotIdx);
    __aicore__ inline void InitCopyParams();
    __aicore__ inline void BindWorkspaceAndWindow(GM_ADDR workspaceGM);
    __aicore__ inline void BufferInit();
    __aicore__ inline void BufferInitCached();
    __aicore__ inline void InitCountCommunication();
    __aicore__ inline void CommunicationCached();
    __aicore__ inline void ResetCounters();
    __aicore__ inline void Communication();
    __aicore__ inline void SendCountToPeer(uint32_t peer, GM_ADDR dst, GM_ADDR src, GM_ADDR notifyAddr,
                                           uint64_t notifyVal);
    __aicore__ inline void GetRecvCount();
    __aicore__ inline void SetRecvNumPerExpert();
    __aicore__ inline void SetRecvNumPerRank(LocalTensor<int32_t> recvTmpTensor);
    __aicore__ inline void WriteToRemoteWindow();
    __aicore__ inline void CommBufferInit();
    __aicore__ inline bool NextTokenChunk(uint32_t& peerOffset, uint32_t& start, uint32_t& remaining,
                                          TokenChunk& chunk);
    __aicore__ inline void PrefetchTokenIndices(const TokenChunk& chunk, uint32_t buffer, bool reuse);
    __aicore__ inline void SendTokenChunk(const TokenChunk& chunk, uint32_t buffer, BufDesc* localDescs);
    __aicore__ inline void FillPayloadDescs(const LocalTensor<int32_t>& indices, uint32_t offset, uint32_t slots,
                                            BufDesc* localDescs);
    __aicore__ inline void WriteTokenPayloads(const TokenChunk& chunk, uint32_t buffer, BufDesc* localDescs);
    __aicore__ inline void SendTokenByJetty();
    __aicore__ inline void WritePayload(GM_ADDR dst, BufDesc* localDescs, uint32_t sgeCount);
    __aicore__ inline void SendCompletion(uint32_t dstRankId, GM_ADDR notifyAddr, bool requestCqe);
    __aicore__ inline void InitSqBudget();
    __aicore__ inline void ReserveBatchSpace(uint32_t wqebbCount);
    __aicore__ inline void ReserveBatchCqe();
    __aicore__ inline void CommitBatch(bool force);
    __aicore__ inline void CalSendCntDirect();
    __aicore__ inline void GetSlotStartNumDirect();
    __aicore__ inline void PrepareSendDirect();
    __aicore__ inline void SetSrcTokenIdxDirect(uint32_t topkOffset, uint32_t calCnt, uint32_t calCntAlign,
                                                uint32_t& sendLocalCnt, LocalTensor<int32_t>& counts);
    __aicore__ inline void SplitToCore(uint32_t count, uint32_t cores, uint32_t& start, uint32_t& end, uint32_t& size);
    __aicore__ inline void InitMetaScatter();
    __aicore__ inline void ScatterMetaRouteInfo(uint32_t start, uint32_t count);
    __aicore__ inline void CopyInMetaInfo(uint32_t start, uint32_t count);
    __aicore__ inline void GetLocalSlotStart(uint32_t sendCount, uint32_t maxBatchCnt, uint32_t alignStride,
                                             uint32_t& curLocalSlot, DataCopyExtParams& srcTokenIdxCopyParams);
    __aicore__ inline void SendPhaseCached();
    __aicore__ inline void WaitStatusCached();
    __aicore__ inline void ProcessMetaSlot(uint32_t start, uint32_t count, uint32_t localCount, uint32_t& localSlot);

    __aicore__ inline uint64_t GetOwnedJettyHandle()
    {
        return mc2Context_->multiChannelHandles[jettyIndex_];
    }

    TPipe* tpipe_{nullptr};
    __gm__ Mc2Kernel::MoeSharedJettyContext* mc2Context_{nullptr};
    AscendC::Hcomm<COMM_PROTOCOL_UBC_CTP> hcomm_; // 通信上下文
    AscendC::BatchHandle<AscendC::MultiChannelHandle> payloadBatch_{};
    MoeEpExceptionDump::MoeEpCoreDiagWriter diagWriter_;

    GlobalTensor<int32_t> topkIdxGMTensor_;
    GlobalTensor<float> topkWeightsGMTensor_;
    GlobalTensor<int32_t> dstSlotIdxGMTensor_;
    GlobalTensor<int32_t> numRecvPerRankGMTensor_;
    GlobalTensor<int64_t> numRecvPerExpertGMTensor_;
    GlobalTensor<int32_t> cachedSlotIdxGMTensor_;
    GlobalTensor<int32_t> sendCounterGMTensor_;
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
    LocalTensor<int16_t> dstRankTensor_;
    LocalTensor<int16_t> dstRankIdTensor_;
    LocalTensor<uint8_t> hcommTensor_;
    LocalTensor<uint8_t> hcommBatchTensor_;

    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> metaSlotQueue_;
    TBuf<> metaScatterBuf_;
    LocalTensor<int32_t> rankValueTensor_;
    LocalTensor<int32_t> tokenValueTensor_;
    LocalTensor<uint32_t> rankOffsetTensor_;
    LocalTensor<uint32_t> tokenOffsetTensor_;
    LocalTensor<int32_t> slotIdxPerRankTensor_;
    uint32_t startTokenId_{0};
    uint32_t endTokenId_{0};
    uint32_t sendTokenNum_{0};
    uint32_t metaBatch_{1};
    DataCopyParams topkCopyParams_;
    DataCopyParams scalesCopyParams_;
    DataCopyPadExtParams<int32_t> routePadParams_{false, 0U, 0U, 0U};
    TBuf<> topkIdsBuf_;
    TBuf<> tempBuf_;
    TBuf<> dstExpBuf_;
    TBuf<> maskBuf_;
    TBuf<> hcommBuf_; // 通信
    TBuf<> sendCntRankBuf_;
    TBuf<> sendCntExpertBuf_;
    TBuf<> numRecvPerRankBuf_;
    TBuf<> numRecvPerExpertBuf_;

    GM_ADDR hostPinnedCounterAddrGM_{nullptr};
    GM_ADDR xAddr_{nullptr};
    GM_ADDR sendSrcTokenIdxAddr_{nullptr}; // Direct rows: [sendCount, srcTokenId...], output or cached input.
    GM_ADDR sendCounterAddr_{nullptr};
    GM_ADDR sendCntWorkspaceAddr_{nullptr};
    GM_ADDR dstRankInfoAddr_{nullptr};
    GM_ADDR localCntStateWinAddr_{nullptr};
    GM_ADDR localSlotWinAddr_{nullptr};
    GM_ADDR payloadStashWinAddr_{nullptr};
    GM_ADDR completionFlagSourceWinAddr_{nullptr};

    uint32_t axisBS_{0};
    uint32_t axisH_{0};
    uint32_t axisK_{0};
    uint32_t epWorldSize_{0};
    uint32_t moeExpertNumPerRank_{0};
    uint32_t axisMaxBS_{0};
    uint32_t scalesBytes_{0};
    uint32_t perSlotBytes_{0};
    uint32_t moeExpertNum_{0};
    uint32_t aivNum_{0};
    uint32_t epRankId_{0};
    uint32_t aivId_{0};
    uint32_t perGroupTokenNum_{0};
    uint32_t jettyIndex_{0};
    uint32_t lastRemoteRank_{0};
    event_t indexReadyEvents_[2];
    event_t indexReuseEvents_[2];
    uint32_t hcommBatchBufferBytes_{0};
    uint32_t pendingWqebb_{0};
    uint32_t preparedCqeCount_{0};
    uint32_t budgetJettyIndex_{Mc2Kernel::MOE_SHARED_JETTY_NUM};
    uint32_t sqPendingWqebb_{0};
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
    uint32_t topkCopyAlign_{0};
    uint32_t metaSlotStride_{0};
    uint32_t dstSlotStride_{0};
    uint64_t sendCntOffset_{0};
    uint64_t dstRankInfoOffset_{0};
    uint64_t cntWinStateOffset_{0};
    uint64_t slotWinStateOffset_{0};
    uint64_t winDataOffset_{0};
    uint64_t payloadStashWinOffset_{0};
    uint64_t completionFlagSourceWinOffset_{0};
    uint64_t dstRankWinOffset_{0};
    uint64_t dstStateWinOffset_{0};

    DataCopyParams statusCopyParams_;
    DataCopyParams clearStatusCopyParams_;
    DataCopyParams sendPerExpertParams_;
    DataCopyParams sendPerRankParams_;
    DataCopyParams metaSlotCopyParams_; // stash 元数据槽整槽拷贝参数
    DataCopyPadParams padParams_;
};

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::InitTilingAndRoles(const MoeEpDispatchInfo& info)
{
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
    sendCntOffset_ = info.workspace.sendCntOffset;
    dstRankInfoOffset_ = info.workspace.dstRankInfoOffset;
    cntWinStateOffset_ = info.window.cntWinStateOffset;
    slotWinStateOffset_ = info.window.slotWinStateOffset;
    winDataOffset_ = info.window.winDataOffset;
    payloadStashWinOffset_ = info.window.payloadStashWinOffset;
    completionFlagSourceWinOffset_ = info.window.completionFlagSourceWinOffset;
    dispatchNotifyCount_ = info.dispatchNotifyCount;
    hostPinnedCounterAddrGM_ = reinterpret_cast<GM_ADDR>(info.hostPinnedCounterAddr);
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::InitDerivedValues()
{
    tokenSize_ = axisH_ * sizeof(XType);
    kAlignSize_ = Ceil(axisK_ * TOPK_INFO_SIZE, UB_ALIGN) * UB_ALIGN;
    axisKAlign_ = kAlignSize_ / TOPK_INFO_SIZE;
    metaSlotStride_ = metaSlotBytes_ / sizeof(int32_t);
    dstSlotStride_ = axisBS_ + 1U;
    epWorldSizeAlign_ = Ceil(epWorldSize_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    perGroupTokenNum_ = PER_GROUP_SIZE / sizeof(int16_t) / axisK_;
    perGroupSizeAlign_ = Ceil(perGroupTokenNum_ * axisK_ * sizeof(int16_t), ALIGNED_LEN_256) * ALIGNED_LEN_256;
    const uint32_t maskElements = IsCached ? MAX_BATCH_SIZE / sizeof(int32_t) : perGroupSizeAlign_ / sizeof(int16_t);
    maskBytesAlign_ = Ceil(Ceil(maskElements, BITS_PER_BYTE), UB_ALIGN) * UB_ALIGN;
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
    dstRankWinOffset_ = static_cast<uint64_t>(epRankId_) * axisMaxBS_ * perSlotBytes_;
    dstStateWinOffset_ = static_cast<uint64_t>(epRankId_) * dispatchNotifyCount_ * WIN_ADDR_ALIGN;
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::BindInputTensors(
    GM_ADDR topkIdx, GM_ADDR topkWeights, GM_ADDR scales, GM_ADDR cachedSlotIdx, GM_ADDR numRecvPerRank,
    GM_ADDR numRecvPerExpert, GM_ADDR dstBufferSlotIdx)
{
    topkIdxGMTensor_.SetGlobalBuffer((__gm__ int32_t*)topkIdx);
    if constexpr (IsTopkWeights) {
        topkWeightsGMTensor_.SetGlobalBuffer((__gm__ float*)topkWeights);
    }
    if constexpr (Std::IsSame<XType, fp8_e5m2_t>::value || Std::IsSame<XType, fp8_e4m3fn_t>::value) {
        scalesGMTensor_.SetGlobalBuffer((__gm__ ScalesType*)scales);
        topkCopyAlign_ = Ceil(scalesBytes_, UB_ALIGN) * UB_ALIGN / sizeof(int32_t);
    }
    if constexpr (IsCached) {
        cachedSlotIdxGMTensor_.SetGlobalBuffer((__gm__ int32_t*)cachedSlotIdx);
        sendSrcTokenIdxAddr_ = cachedSlotIdx;
    } else {
        sendSrcTokenIdxAddr_ = dstBufferSlotIdx;
    }
    numRecvPerRankGMTensor_.SetGlobalBuffer((__gm__ int32_t*)numRecvPerRank);
    numRecvPerExpertGMTensor_.SetGlobalBuffer((__gm__ int64_t*)numRecvPerExpert);
    dstSlotIdxGMTensor_.SetGlobalBuffer((__gm__ int32_t*)dstBufferSlotIdx);
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::InitCopyParams()
{
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
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::BindWorkspaceAndWindow(GM_ADDR workspaceGM)
{
    sendCounterAddr_ = workspaceGM;
    sendCntWorkspaceAddr_ = workspaceGM + sendCntOffset_;
    dstRankInfoAddr_ = workspaceGM + dstRankInfoOffset_;
    localCntStateWinAddr_ = GetWinAddrByRankId(mc2Context_, epRankId_, cntWinStateOffset_);
    localSlotWinAddr_ = GetWinAddrByRankId(mc2Context_, epRankId_, winDataOffset_) + dstRankWinOffset_;
    payloadStashWinAddr_ = GetWinAddrByRankId(mc2Context_, epRankId_, payloadStashWinOffset_);
    completionFlagSourceWinAddr_ = GetWinAddrByRankId(mc2Context_, epRankId_, completionFlagSourceWinOffset_);
    sendCounterGMTensor_.SetGlobalBuffer((__gm__ int32_t*)sendCounterAddr_);
    recvCounterGMTensor_.SetGlobalBuffer((__gm__ int32_t*)localCntStateWinAddr_);
    metaSlotGMTensor_.SetGlobalBuffer((__gm__ int32_t*)payloadStashWinAddr_);
    sendCntPerRankGMTensor_.SetGlobalBuffer((__gm__ int32_t*)sendCntWorkspaceAddr_);
    sendCntPerExpertGMTensor_.SetGlobalBuffer((__gm__ int32_t*)(sendCntWorkspaceAddr_ + epWorldSizeAlign512_));
    dstRankGMTensor_.SetGlobalBuffer((__gm__ int16_t*)dstRankInfoAddr_);
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::Init(
    GM_ADDR context, GM_ADDR x, GM_ADDR topkIdx, GM_ADDR topkWeights, GM_ADDR scales, GM_ADDR cachedSlotIdx,
    GM_ADDR numRecvPerRank, GM_ADDR numRecvPerExpert, GM_ADDR dstBufferSlotIdx, GM_ADDR workspaceGM, GM_ADDR tilingGM,
    TPipe* pipe, const MoeEpDispatchTilingData* tilingData)
{
    tpipe_ = pipe;
    aivId_ = GetBlockIdx();
    mc2Context_ = reinterpret_cast<__gm__ Mc2Kernel::MoeSharedJettyContext*>(context);
    epRankId_ = mc2Context_->epRankId;
    const auto& info = tilingData->moeEpDispatchInfo;
#if defined(ASCENDC_OOM) && ASCENDC_OOM == 1
    MoeEpDispatchBase::RegisterOomRanges<DoCpuSync>(mc2Context_, epRankId_, tilingGM, tilingData);
#endif
    constexpr size_t metadataOffset =
        offsetof(MoeEpDispatchTilingData, moeEpDispatchInfo) + offsetof(MoeEpDispatchInfo, dumpMetadata);
    MoeEpExceptionDump::WriteMetadata<Mc2Kernel::MoeSharedJettyContext>(context, tilingGM + metadataOffset);
    diagWriter_.Init<Mc2Kernel::MoeSharedJettyContext>(context, MOE_EP_CORE_DIAG_DISPATCH, tpipe_);

    xAddr_ = x;
    InitTilingAndRoles(info);
    InitDerivedValues();
    BindInputTensors(topkIdx, topkWeights, scales, cachedSlotIdx, numRecvPerRank, numRecvPerExpert, dstBufferSlotIdx);
    InitCopyParams();
    BindWorkspaceAndWindow(workspaceGM);
    diagWriter_.RunPosRecord(MOE_EP_DISPATCH_RUN_POS_INIT_DONE);
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::BufferInit()
{
    if constexpr (!IsCached) {
        tpipe_->InitBuffer(metaSlotQueue_, 2U, metaBatch_ * metaSlotBytes_);
        tpipe_->InitBuffer(metaScatterBuf_, MOE_EP_META_SCATTER_UB_BYTES);
    }
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
    if (aivId_ == 0U) {
        LocalTensor<int32_t> flag = maskBuf_.Get<int32_t>();
        Duplicate<int32_t>(flag, 1, WIN_ADDR_ALIGN / sizeof(int32_t));
        GlobalTensor<int32_t> source;
        source.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(completionFlagSourceWinAddr_));
        SyncFunc<AscendC::HardEvent::V_MTE3>();
        DataCopy(source, flag, WIN_ADDR_ALIGN / sizeof(int32_t));
        SyncFunc<AscendC::HardEvent::MTE3_V>();
    }
    if constexpr (!IsCached) {
        if (aivId_ == aivNum_ - 1U) {
            SyncFunc<AscendC::HardEvent::V_MTE3>();
            DataCopyPad(sendCntPerExpertGMTensor_, sendCntPerExpertTensor_, sendPerExpertParams_);
            DataCopy(sendCntPerRankGMTensor_, sendCntPerRankTensor_, counterCnt_);
        }
    }
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::CalSendCntDirect()
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
    DataCopy(sendCounterGMTensor_[aivId_ * counterAlign512_], sendCntPerRankTensor_, counterCnt_);
    SetAtomicAdd<int32_t>();
    DataCopyPad(sendCntPerRankGMTensor_, sendCntPerRankTensor_, sendPerRankParams_);
    DataCopyPad(sendCntPerExpertGMTensor_, sendCntPerExpertTensor_, sendPerExpertParams_);
    SetAtomicNone();
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::GetSlotStartNumDirect()
{
    slotIdxPerRankTensor_ = sendCntExpertBuf_.GetWithOffset<int32_t>(counterCnt_, 0);
    Duplicate<int32_t>(slotIdxPerRankTensor_, 0, epWorldSize_);
    if (aivId_ == 0) { // 0核起始id 均为0
        return;
    }
    uint32_t copyNumPerGroup = perGroupSizeAlign_ * 2 / epWorldSizeAlign_;
    uint32_t groupCnt = Ceil(aivId_, copyNumPerGroup);
    LocalTensor<int32_t> counterTmpTensor = topkIdsBuf_.Get<int32_t>();
    LocalTensor<int32_t> hitTmpTensor = sendCntRankBuf_.Get<int32_t>();
    LocalTensor<uint8_t> sharedTmpTensor = tempBuf_.Get<uint8_t>();
    DataCopyParams counterCopyParams = {1U, static_cast<uint16_t>(epWorldSizeAlign_),
                                        static_cast<uint16_t>(epWorldSizeAlign512_ - epWorldSizeAlign_), 0U};
    for (uint32_t i = 0; i < groupCnt; i++) {
        uint32_t copyNum = (i == groupCnt - 1) ? (aivId_ - copyNumPerGroup * i) : copyNumPerGroup;
        uint32_t gmOffset = i * copyNumPerGroup * epWorldSizeAlign512_ / sizeof(int32_t);
        counterCopyParams.blockCount = static_cast<uint16_t>(copyNum);
        DataCopyPad(counterTmpTensor, sendCounterGMTensor_[gmOffset], counterCopyParams, padParams_);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        const uint32_t shape[] = {copyNum, counterCnt_};
        ReduceSum<int32_t, AscendC::Pattern::Reduce::RA, true>(hitTmpTensor, counterTmpTensor, sharedTmpTensor, shape,
                                                               true);
        Add(slotIdxPerRankTensor_, slotIdxPerRankTensor_, hitTmpTensor, epWorldSize_);
        SyncFunc<AscendC::HardEvent::V_MTE2>();
    }
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::PrepareSendDirect()
{
    if (startTokenId_ >= axisBS_) {
        return;
    }
    GetSlotStartNumDirect(); // 计算起始slot id
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
        SetSrcTokenIdxDirect(topkOffset, calCnt, calCntAlign, sendLocalCnt, cntLocalTensor);
        SyncFunc<AscendC::HardEvent::V_S>();
        ProcessMetaSlot(startId, tokenCnt, sendLocalCnt, curLocalSlot);
        if (group < groupCnt - 1) {
            SyncFunc<AscendC::HardEvent::V_MTE2>();
        }
    }
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
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::SetSrcTokenIdxDirect(
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
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::SendCountToPeer(uint32_t peer, GM_ADDR dst,
                                                                                     GM_ADDR src, GM_ADDR notifyAddr,
                                                                                     uint64_t notifyVal)
{
    uint32_t peerIndex = peer < epRankId_ ? peer : peer - 1;
    auto& handle = hcomm_.GetHandleRef(payloadBatch_, peerIndex, dst);
    ReserveBatchSpace(HCOMM_COUNT_WQEBB);
    if (sqPendingWqebb_ + HCOMM_COUNT_WQEBB + HCOMM_MAX_WRITE_WQEBB > HCOMM_SQ_CAPACITY) {
        ReserveBatchCqe();
        hcomm_.WriteWithNotifyNbi<DATA_CQE_CFG>(handle, dst, src, moeNumPerRankSize_, notifyAddr, notifyVal);
        ++preparedCqeCount_;
        pendingWqebb_ += HCOMM_COUNT_WQEBB;
        CommitBatch(true);
        hcomm_.Drain(handle);
        sqPendingWqebb_ = 0U;
        return;
    } else {
        hcomm_.WriteWithNotifyNbi<DATA_CFG>(handle, dst, src, moeNumPerRankSize_, notifyAddr, notifyVal);
        sqPendingWqebb_ += HCOMM_COUNT_WQEBB;
    }
    pendingWqebb_ += HCOMM_COUNT_WQEBB;
    CommitBatch(true);
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::Communication()
{
    InitCountCommunication();
    if (jettyIndex_ >= Mc2Kernel::MOE_SHARED_JETTY_NUM) {
        return;
    }

    DataCopy(sendCntPerRankTensor_, sendCntPerRankGMTensor_, counterCnt_);
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    const uint32_t countOwners = aivNum_ < Mc2Kernel::MOE_SHARED_JETTY_NUM ? aivNum_ : Mc2Kernel::MOE_SHARED_JETTY_NUM;
    for (uint32_t dstRankId = aivId_; dstRankId < epWorldSize_; dstRankId += countOwners) {
        int32_t sendCnt = sendCntPerRankTensor_.GetValue(dstRankId);
        // The first element of every direct row is its send count.
        LocalTensor<int32_t> header = topkIdsBuf_.Get<int32_t>();
        header.SetValue(0, sendCnt);
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        DataCopyExtParams headerCopy{1U, static_cast<uint32_t>(sizeof(int32_t)), 0U, 0U, 0U};
        DataCopyPad(dstSlotIdxGMTensor_[static_cast<uint64_t>(dstRankId) * dstSlotStride_], header, headerCopy);
        SyncFunc<AscendC::HardEvent::MTE3_S>();
        uint64_t notifyVal = (static_cast<uint64_t>(sendCnt) << NOTIFY_VAL_SHIFT) | 1U;
        uint32_t srcStride = dstRankId * moeExpertNumPerRank_;
        // 计算目标窗口地址:
        GM_ADDR remoteStateAddr = GetWinAddrByRankId(mc2Context_, dstRankId, cntWinStateOffset_);
        GM_ADDR notifyAddr = remoteStateAddr + epRankId_ * WIN_ADDR_ALIGN;
        GM_ADDR remoteCountAddr = remoteStateAddr + cntFlagWinSize_ + epRankId_ * moeNumPerRankAlign512_;
        GM_ADDR srcWorkspaceAddr = sendCntWorkspaceAddr_ + epWorldSizeAlign512_ + srcStride * sizeof(int32_t);

        if (dstRankId != epRankId_) { // 远端 使用URMA发送 count + state
            SendCountToPeer(dstRankId, remoteCountAddr, srcWorkspaceAddr, notifyAddr, notifyVal);
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
    SyncFunc<AscendC::HardEvent::MTE3_V>();
    SyncFunc<AscendC::HardEvent::MTE3_MTE2>(); // tempBuf_ 复用
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
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::CommBufferInit()
{
    PipeBarrier<PIPE_ALL>();
    tpipe_->Reset();
    // 通信初始化
    hcommBatchBufferBytes_ = HCOMM_BATCH_WQEBB_CAPACITY * HCOMM_WQE_BYTES;
    uint32_t hcommBatchSize = hcommBatchBufferBytes_ / sizeof(uint8_t);
    tpipe_->InitBuffer(hcommBuf_, HCOMM_INIT_SIZE + hcommBatchBufferBytes_);
    hcommTensor_ = hcommBuf_.GetWithOffset<uint8_t>(HCOMM_INIT_SIZE, 0);
    hcommBatchTensor_ = hcommBuf_.GetWithOffset<uint8_t>(hcommBatchSize, HCOMM_INIT_SIZE);
    hcomm_.Init(hcommTensor_, HCOMM_INIT_SIZE);
    Duplicate<uint8_t>(hcommBatchTensor_, 0, hcommBatchSize);
    SyncFunc<AscendC::HardEvent::V_S>();

    // ub buffer init
    tpipe_->InitBuffer(sendCntRankBuf_, IsCached ? epWorldSize_ * UB_ALIGN : sendCntRankSizeAlign_);
    tpipe_->InitBuffer(tempBuf_, 2U * SOURCE_INDEX_BUFFER_BYTES);
    tpipe_->InitBuffer(maskBuf_, UB_ALIGN);
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::WriteToRemoteWindow()
{
    CommBufferInit();
    LocalTensor<int32_t> counts = sendCntRankBuf_.Get<int32_t>();
    if constexpr (!IsCached) {
        DataCopy(counts, sendCntPerRankGMTensor_, counterCnt_);
    } else {
        DataCopyExtParams headers{static_cast<uint16_t>(epWorldSize_), static_cast<uint32_t>(sizeof(int32_t)),
                                  static_cast<uint32_t>(axisBS_ * sizeof(int32_t)), 0U, 0U};
        DataCopyPad(counts, cachedSlotIdxGMTensor_, headers, routePadParams_);
    }
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    // One owner per logical Jetty, including Jettys 56..63 on a 56-AIV device.
    for (jettyIndex_ = aivId_; jettyIndex_ < Mc2Kernel::MOE_SHARED_JETTY_NUM; jettyIndex_ += aivNum_) {
        payloadBatch_ = hcomm_.MakeBatchHandle(static_cast<MultiChannelHandle>(GetOwnedJettyHandle()),
                                               hcommBatchTensor_, hcommBatchBufferBytes_);
        pendingWqebb_ = 0U;
        preparedCqeCount_ = 0U;
        InitSqBudget();
        SendTokenByJetty();
        CommitBatch(true);
    }
    DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(topkIdxGMTensor_);
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline bool MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::NextTokenChunk(uint32_t& peerOffset,
                                                                                    uint32_t& start,
                                                                                    uint32_t& remaining,
                                                                                    TokenChunk& chunk)
{
    while (peerOffset < epWorldSize_) {
        const uint32_t rank = (jettyIndex_ + peerOffset) % epWorldSize_;
        const bool first = remaining == 0U;
        if (first) {
            LocalTensor<int32_t> counts = sendCntRankBuf_.Get<int32_t>();
            const uint32_t total = static_cast<uint32_t>(counts.GetValue(IsCached ? rank * UB_STRIDE : rank));
            const uint32_t perJetty = Ceil(total, Mc2Kernel::MOE_SHARED_JETTY_NUM);
            start = jettyIndex_ * perJetty;
            remaining = start < total ? total - start : 0U;
            remaining = remaining < perJetty ? remaining : perJetty;
            if (rank == epRankId_) {
                remaining = 0U;
            }
        }
        const uint32_t count = remaining < SOURCE_INDEX_CHUNK_TOKENS ? remaining : SOURCE_INDEX_CHUNK_TOKENS;
        chunk = {rank, start, count, first, count == remaining};
        start += count;
        remaining -= count;
        if (chunk.last) {
            ++peerOffset;
            start = 0U;
        }
        return true;
    }
    return false;
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::FillPayloadDescs(
    const LocalTensor<int32_t>& indices, uint32_t offset, uint32_t slots, BufDesc* localDescs)
{
    for (uint32_t slot = 0U; slot < slots; ++slot) {
        const uint64_t token = static_cast<uint64_t>(indices.GetValue(offset + slot));
        localDescs[slot * SGE_PER_SLOT].addr = xAddr_ + token * tokenSize_;
        localDescs[slot * SGE_PER_SLOT + 1U].addr = payloadStashWinAddr_ + token * metaSlotBytes_;
    }
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::WriteTokenPayloads(const TokenChunk& chunk,
                                                                                        uint32_t buffer,
                                                                                        BufDesc* localDescs)
{
    LocalTensor<int32_t> indices = tempBuf_.Get<int32_t>()[buffer * SOURCE_INDEX_CHUNK_TOKENS];
    GM_ADDR remote = GetWinAddrByRankId(mc2Context_, chunk.rank, winDataOffset_) + dstRankWinOffset_ +
                     static_cast<uint64_t>(chunk.start) * perSlotBytes_;
    uint32_t offset = 0U;
    const uint32_t fullTokens = chunk.count / SLOT_NUM_PER_SQE * SLOT_NUM_PER_SQE;
    while (offset < fullTokens) {
        ReserveBatchSpace(HCOMM_MAX_WRITE_WQEBB);
        const uint32_t batchCap = (HCOMM_BATCH_WQEBB_CAPACITY - pendingWqebb_) / HCOMM_MAX_WRITE_WQEBB;
        // Leave room for one CQE-bearing request before the SQ is exhausted.
        const uint32_t remainingSq = HCOMM_SQ_CAPACITY - sqPendingWqebb_;
        const uint32_t sqCap =
            remainingSq > HCOMM_MAX_WRITE_WQEBB ? (remainingSq - HCOMM_MAX_WRITE_WQEBB) / HCOMM_MAX_WRITE_WQEBB : 0U;
        uint32_t run = (fullTokens - offset) / SLOT_NUM_PER_SQE;
        run = run < batchCap ? run : batchCap;
        run = run < sqCap ? run : sqCap;
        if (run != 0U) {
            // These requests fit both budgets and need no per-WQE capacity checks or CQEs.
            for (uint32_t request = 0U; request < run; ++request) {
                FillPayloadDescs(indices, offset, SLOT_NUM_PER_SQE, localDescs);
                hcomm_.WriteNbi<DATA_CFG>(payloadBatch_.handle, remote, localDescs, SGE_PER_SQE);
                offset += SLOT_NUM_PER_SQE;
                remote += SLOT_NUM_PER_SQE * perSlotBytes_;
            }
            const uint32_t used = run * HCOMM_MAX_WRITE_WQEBB;
            pendingWqebb_ += used;
            sqPendingWqebb_ += used;
            CommitBatch(false);
        } else {
            // Retain the original CQE boundary handling.
            FillPayloadDescs(indices, offset, SLOT_NUM_PER_SQE, localDescs);
            WritePayload(remote, localDescs, SGE_PER_SQE);
            offset += SLOT_NUM_PER_SQE;
            remote += SLOT_NUM_PER_SQE * perSlotBytes_;
        }
    }
    if (offset < chunk.count) {
        const uint32_t slots = chunk.count - offset;
        FillPayloadDescs(indices, offset, slots, localDescs);
        WritePayload(remote, localDescs, slots * SGE_PER_SLOT);
    }
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::PrefetchTokenIndices(const TokenChunk& chunk,
                                                                                          uint32_t buffer, bool reuse)
{
    if (chunk.count == 0U || chunk.rank == epRankId_) {
        return;
    }
    if (reuse) {
        // Scalar has consumed the previous indices before MTE2 overwrites this buffer.
        SetFlag<HardEvent::S_MTE2>(indexReuseEvents_[buffer]);
        WaitFlag<HardEvent::S_MTE2>(indexReuseEvents_[buffer]);
    }
    LocalTensor<int32_t> indices = tempBuf_.Get<int32_t>()[buffer * SOURCE_INDEX_CHUNK_TOKENS];
    GlobalTensor<int32_t> source;
    source.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(
        sendSrcTokenIdxAddr_ + (static_cast<uint64_t>(chunk.rank) * dstSlotStride_ + 1U) * sizeof(int32_t)));
    DataCopyExtParams params{1U, static_cast<uint32_t>(chunk.count * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> pad{false, 0U, 0U, 0U};
    DataCopyPad(indices, source[chunk.start], params, pad);
    SetFlag<HardEvent::MTE2_S>(indexReadyEvents_[buffer]);
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::SendTokenChunk(const TokenChunk& chunk,
                                                                                    uint32_t buffer,
                                                                                    BufDesc* localDescs)
{
    GM_ADDR notifyAddr = GetWinAddrByRankId(mc2Context_, chunk.rank, slotWinStateOffset_) + dstStateWinOffset_ +
                         jettyIndex_ * WIN_ADDR_ALIGN;
    if (chunk.rank != epRankId_ && chunk.first) {
        const uint32_t peerIndex = chunk.rank < epRankId_ ? chunk.rank : chunk.rank - 1U;
        hcomm_.GetHandleRef(payloadBatch_, peerIndex, notifyAddr);
    }
    if (chunk.count != 0U && chunk.rank != epRankId_) {
        WriteTokenPayloads(chunk, buffer, localDescs);
    }
    if (chunk.last) {
        // Empty slices also publish exactly one completion per Jetty/peer.
        SendCompletion(chunk.rank, notifyAddr, chunk.rank == lastRemoteRank_);
        CommitBatch(true);
    }
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::SendTokenByJetty()
{
    BufDesc localDescs[SGE_PER_SQE] = {};
    for (uint32_t i = 0U; i < SGE_PER_SQE; i += SGE_PER_SLOT) {
        localDescs[i].len = tokenSize_;
        localDescs[i + 1U].len = metaSlotBytes_;
    }
    lastRemoteRank_ = (jettyIndex_ + epWorldSize_ - 1U) % epWorldSize_;
    if (lastRemoteRank_ == epRankId_) {
        lastRemoteRank_ = (lastRemoteRank_ + epWorldSize_ - 1U) % epWorldSize_;
    }
    for (uint32_t buffer = 0U; buffer < 2U; ++buffer) {
        indexReadyEvents_[buffer] = static_cast<event_t>(tpipe_->AllocEventID<HardEvent::MTE2_S>());
        indexReuseEvents_[buffer] = static_cast<event_t>(tpipe_->AllocEventID<HardEvent::S_MTE2>());
    }
    TokenChunk chunks[2] = {};
    bool valid[2] = {};
    uint32_t peerOffset = 0U;
    uint32_t start = 0U;
    uint32_t remaining = 0U;
    uint32_t current = 0U;
    valid[0] = NextTokenChunk(peerOffset, start, remaining, chunks[0]);
    if (valid[0]) {
        PrefetchTokenIndices(chunks[0], 0U, false);
        valid[1] = NextTokenChunk(peerOffset, start, remaining, chunks[1]);
        if (valid[1]) {
            PrefetchTokenIndices(chunks[1], 1U, false);
        }
    }
    while (valid[current]) {
        if (chunks[current].count != 0U && chunks[current].rank != epRankId_) {
            WaitFlag<HardEvent::MTE2_S>(indexReadyEvents_[current]);
        }
        SendTokenChunk(chunks[current], current, localDescs);
        // Prefetch k+2 after Scalar(k) is done; its MTE2 overlaps Scalar(k+1).
        valid[current] = NextTokenChunk(peerOffset, start, remaining, chunks[current]);
        if (valid[current]) {
            PrefetchTokenIndices(chunks[current], current, true);
        }
        current ^= 1U;
    }
    for (uint32_t buffer = 0U; buffer < 2U; ++buffer) {
        tpipe_->ReleaseEventID<HardEvent::MTE2_S>(indexReadyEvents_[buffer]);
        tpipe_->ReleaseEventID<HardEvent::S_MTE2>(indexReuseEvents_[buffer]);
    }
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::InitSqBudget()
{
    // Keep count-phase usage when the same Jetty is reopened for payload sends.
    if (budgetJettyIndex_ != jettyIndex_) {
        sqPendingWqebb_ = 0U;
        budgetJettyIndex_ = jettyIndex_;
    }
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::ReserveBatchSpace(uint32_t wqebbCount)
{
    if (pendingWqebb_ + wqebbCount > HCOMM_BATCH_WQEBB_CAPACITY) {
        CommitBatch(true);
    }
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::ReserveBatchCqe()
{
    // Older CQEs can be polled by BatchCommit; the current batch must fit in the CQ by itself.
    if (preparedCqeCount_ == Mc2Kernel::MOE_SHARED_JETTY_CQ_DEPTH) {
        CommitBatch(true);
    }
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::CommitBatch(bool force)
{
    if (pendingWqebb_ != 0U && (force || pendingWqebb_ >= HCOMM_BATCH_WQEBB_CAPACITY)) {
        hcomm_.BatchCommit(payloadBatch_);
        pendingWqebb_ = 0U;
        preparedCqeCount_ = 0U;
    }
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::WritePayload(GM_ADDR dst, BufDesc* localDescs,
                                                                                  uint32_t sgeCount)
{
    uint32_t wqebbCount = Ceil(HCOMM_SQE_BYTES + sgeCount * HCOMM_SGE_BYTES, HCOMM_WQE_BYTES);
    ReserveBatchSpace(wqebbCount);
    if (sqPendingWqebb_ + wqebbCount + HCOMM_MAX_WRITE_WQEBB > HCOMM_SQ_CAPACITY) {
        ReserveBatchCqe();
        hcomm_.WriteNbi<DATA_CQE_CFG>(payloadBatch_.handle, dst, localDescs, sgeCount);
        ++preparedCqeCount_;
        pendingWqebb_ += wqebbCount;
        CommitBatch(true);
        hcomm_.Drain(payloadBatch_.handle);
        sqPendingWqebb_ = 0U;
        return;
    } else {
        hcomm_.WriteNbi<DATA_CFG>(payloadBatch_.handle, dst, localDescs, sgeCount);
        sqPendingWqebb_ += wqebbCount;
    }
    pendingWqebb_ += wqebbCount;
    CommitBatch(false);
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::SendCompletion(uint32_t dstRankId,
                                                                                    GM_ADDR notifyAddr, bool requestCqe)
{
    if (dstRankId == epRankId_) {
        LocalTensor<int32_t> statusTensor = maskBuf_.Get<int32_t>();
        GlobalTensor<int32_t> source;
        GlobalTensor<int32_t> target;
        source.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(completionFlagSourceWinAddr_));
        target.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(notifyAddr));
        DataCopy(statusTensor, source, UB_STRIDE);
        SyncFunc<AscendC::HardEvent::MTE2_MTE3>();
        DataCopy(target, statusTensor, UB_STRIDE);
        SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
        return;
    }
    constexpr uint32_t flagWqebb = (HCOMM_SQE_BYTES + HCOMM_SGE_BYTES) / HCOMM_WQE_BYTES;
    ReserveBatchSpace(flagWqebb);
    const bool drainNow = sqPendingWqebb_ + flagWqebb + HCOMM_MAX_WRITE_WQEBB > HCOMM_SQ_CAPACITY;
    if (requestCqe || drainNow) {
        ReserveBatchCqe();
        hcomm_.WriteNbi<NOTIFY_CQE_CFG>(payloadBatch_.handle, notifyAddr, completionFlagSourceWinAddr_, WIN_ADDR_ALIGN);
        ++preparedCqeCount_;
    } else {
        hcomm_.WriteNbi<NOTIFY_CFG>(payloadBatch_.handle, notifyAddr, completionFlagSourceWinAddr_, WIN_ADDR_ALIGN);
    }
    sqPendingWqebb_ += flagWqebb;
    pendingWqebb_ += flagWqebb;
    CommitBatch(drainNow);
    if (drainNow) {
        hcomm_.Drain(payloadBatch_.handle);
        sqPendingWqebb_ = 0U;
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
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::InitCountCommunication()
{
    jettyIndex_ = aivId_;
    if (jettyIndex_ >= Mc2Kernel::MOE_SHARED_JETTY_NUM) {
        return;
    }
    tpipe_->InitBuffer(hcommBuf_, HCOMM_INIT_SIZE);
    hcommTensor_ = hcommBuf_.Get<uint8_t>();
    Duplicate<uint8_t>(hcommTensor_, 0, HCOMM_INIT_SIZE);
    SyncFunc<AscendC::HardEvent::V_S>();
    payloadBatch_ =
        hcomm_.MakeBatchHandle(static_cast<MultiChannelHandle>(GetOwnedJettyHandle()), hcommTensor_, HCOMM_INIT_SIZE);
    InitSqBudget();
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::BufferInitCached()
{
    tpipe_->InitBuffer(metaSlotQueue_, 2U, metaBatch_ * metaSlotBytes_);
    tpipe_->InitBuffer(metaScatterBuf_, MOE_EP_META_SCATTER_UB_BYTES);
    tpipe_->InitBuffer(topkIdsBuf_, MAX_BATCH_SIZE);
    tpipe_->InitBuffer(maskBuf_, 2U * maskBytesAlign_);
    ResetCounters();
    SyncAll<true>();
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::CommunicationCached()
{
    InitCountCommunication();
    if (jettyIndex_ >= Mc2Kernel::MOE_SHARED_JETTY_NUM) {
        return;
    }
    const uint32_t owners = aivNum_ < Mc2Kernel::MOE_SHARED_JETTY_NUM ? aivNum_ : Mc2Kernel::MOE_SHARED_JETTY_NUM;
    for (uint32_t peer = aivId_; peer < epWorldSize_; peer += owners) {
        GM_ADDR notifyAddr = GetWinAddrByRankId(mc2Context_, peer, cntWinStateOffset_) + epRankId_ * WIN_ADDR_ALIGN;
        if (peer != epRankId_) {
            const uint32_t peerIndex = peer < epRankId_ ? peer : peer - 1U;
            hcomm_.GetHandleRef(payloadBatch_, peerIndex, notifyAddr);
        }
        SendCompletion(peer, notifyAddr, true);
        CommitBatch(true);
    }
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::Process()
{
    if ASCEND_IS_AIV {
        if constexpr (!IsCached) {
            BufferInit();
            CalSendCntDirect();
            SyncAll<true>();
            Communication();
            PrepareSendDirect();
            GetRecvCount();
        } else {
            BufferInitCached();
            CommunicationCached();
            SendPhaseCached();
            WaitStatusCached();
        }
        diagWriter_.RunPosRecord(MOE_EP_DISPATCH_RUN_POS_COUNT_READY);
        SyncAll<true>();
        WriteToRemoteWindow();
        diagWriter_.RunPosRecord(MOE_EP_DISPATCH_RUN_POS_URMA_REQUESTS_ISSUE_DONE);
    }
}

#endif
} // namespace MoeEpDispatchClosOnlyImpl

#endif // MOE_EP_DISPATCH_CLOS_ONLY_H
