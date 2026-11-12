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
constexpr uint32_t PER_GROUP_SIZE = 32 * 1024U; // 计算count 32KB per group
constexpr uint32_t HISTOGRAM_SCOPE_SIZE = 256U; // 直方图uint8, 统计范围0-255
constexpr uint32_t META_BATCH_TOKENS = 8U;
constexpr uint32_t NOTIFY_VAL_SHIFT = 32U;                        // notify 高32位为 sendCnt, 低32位为 state(=1)
constexpr uint32_t MAX_BATCH_SIZE = 160 * 1024U;                  // srcTokenIdx 160KB
constexpr uint32_t SGE_PER_SLOT = 2U;                             // 每 slot 两个 SGE: x 与 meta
constexpr uint32_t SGE_PER_SQE = 12U;                             // 单 SQE 的 SGE 数
constexpr uint32_t SLOT_NUM_PER_SQE = SGE_PER_SQE / SGE_PER_SLOT; // 单 SQE 的 slot 数
// Two aligned halves reuse the original 160 KiB source-index allocation. Full chunks end on an SQE boundary.
constexpr uint32_t SOURCE_INDEX_BUFFER_BYTES = MAX_BATCH_SIZE / 2U;
constexpr uint32_t SOURCE_INDEX_CHUNK_TOKENS =
    SOURCE_INDEX_BUFFER_BYTES / sizeof(int32_t) / SLOT_NUM_PER_SQE * SLOT_NUM_PER_SQE;
constexpr uint32_t HCOMM_BATCH_CAPACITY = 128U; // UB capacity in full 12-SGE requests
constexpr uint32_t HCOMM_SGE_BYTES = 16U;
constexpr uint32_t HCOMM_SQE_BYTES = 48U;
constexpr uint32_t HCOMM_WQE_BYTES = 64U;
constexpr struct UrmaWqeEntry DATA_CFG = {.odr = 5, .fence = 1, .se = 0, .cqe = 0, .inlineEn = 0};
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
    __aicore__ inline void ResetCounters();
    __aicore__ inline void AccumulateSendCounts(LocalTensor<int16_t> ids, LocalTensor<int32_t> counts, uint32_t count,
                                                uint32_t scopeSize);
    __aicore__ inline void DedupAndSend(LocalTensor<int16_t> expertIdsTensor, uint32_t calCnt, uint32_t calCntAlign,
                                        uint32_t tmpOffset);
    __aicore__ inline void CalSendCntGroup(uint32_t tokenStart, uint32_t tokenCount);
    __aicore__ inline void BuildSendPrefixes();
    __aicore__ inline void ScanTileCounts(uint32_t peer, uint32_t firstTile, uint32_t rows);
    __aicore__ inline void ReduceExpertPartials();
    __aicore__ inline void PrepareMetadata(uint32_t tokenStart, uint32_t tokenCount);
    __aicore__ inline void LoadMetadataExtras(uint32_t tokenStart, uint32_t tokenCount);
    __aicore__ inline void CalSendCnt();
    __aicore__ inline void Communication();
    __aicore__ inline void SendCountToPeer(uint32_t peer, GM_ADDR dst, GM_ADDR src, GM_ADDR notifyAddr,
                                           uint64_t notifyVal);
    __aicore__ inline void GetRecvCount();
    __aicore__ inline void SetRecvNumPerExpert();
    __aicore__ inline void SetRecvNumPerRank(LocalTensor<int32_t> recvTmpTensor);
    __aicore__ inline void WriteToRemoteWindow();
    __aicore__ inline void LoadTileDescriptor(uint32_t tile);
    __aicore__ inline void PublishSendCompletions();
    __aicore__ inline void CommBufferInit();
    __aicore__ inline bool NextTokenChunk(uint32_t& peerOffset, uint32_t& start, uint32_t& remaining,
                                          TokenChunk& chunk);
    __aicore__ inline void PrefetchTokenIndices(const TokenChunk& chunk, uint32_t buffer, TEventID event);
    __aicore__ inline void SendTokenChunk(const TokenChunk& chunk, uint32_t buffer, TEventID event,
                                          BufDesc* localDescs);
    __aicore__ inline void WriteTokenPayloads(const TokenChunk& chunk, uint32_t buffer, BufDesc* localDescs);
    __aicore__ inline void SendTokenPipeline();
    __aicore__ inline void WritePayload(GM_ADDR dst, BufDesc* localDescs, uint32_t sgeCount);
    __aicore__ inline void SendCompletion(uint32_t dstRankId, GM_ADDR notifyAddr);
    __aicore__ inline void ReserveBatchSpace(uint32_t wqebbCount);
    __aicore__ inline void CommitBatch(bool force);
    __aicore__ inline void SetSrcTokenIdxAndDstSlot(uint32_t topkOffset, uint32_t calCnt, uint32_t calCntAlign,
                                                    uint32_t& sendLocalCnt, LocalTensor<int32_t>& cntLocalTensor);
    __aicore__ inline void StoreUncachedSlots(uint32_t topkOffset, uint32_t count, uint32_t aligned);
    __aicore__ inline void StoreSourceTokens(uint32_t topkOffset, uint32_t peer, uint32_t slot, uint32_t count,
                                             LocalTensor<int16_t> indices, LocalTensor<int16_t> divisor);
    __aicore__ inline void CopyLocalMetadata(uint32_t count, uint32_t firstSlot);
    __aicore__ inline void PrepareSendTiles();
    __aicore__ inline void PrepareTile(uint32_t tile);
    __aicore__ inline void PublishTileReady(uint32_t tile);
    __aicore__ inline void WaitTileReady(uint32_t tile);

    __aicore__ inline uint64_t GetOwnedJettyHandle()
    {
        return mc2Context_->multiChannelHandles[aivId_];
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
    GlobalTensor<int32_t> tilePrefixGMTensor_;
    GlobalTensor<int32_t> expertPartialGMTensor_;
    GlobalTensor<int32_t> tileReadyGMTensor_;
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
    LocalTensor<int16_t> dstRankTensor_;
    LocalTensor<int16_t> dstRankIdTensor_;
    LocalTensor<uint8_t> hcommTensor_;
    LocalTensor<uint8_t> hcommBatchTensor_;

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
    GM_ADDR sendCounterAddr_{nullptr};
    GM_ADDR sendCntWorkspaceAddr_{nullptr};
    GM_ADDR dstRankInfoAddr_{nullptr};
    GM_ADDR srcTokenTableAddr_{nullptr};
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
    uint32_t tileTokens_{0};
    uint32_t tileCount_{0};
    uint32_t perGroupTokenNum_{0};
    uint32_t sendCoreCount_{0};
    uint32_t prepareCoreCount_{0};
    uint32_t prepareCoreStart_{0};
    uint32_t prepareCoreId_{0};
    uint32_t hcommBatchBufferBytes_{0};
    uint32_t pendingWqebb_{0};
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
    uint32_t srcTokenListBytes_{0};
    uint32_t metaSlotStride_{0};
    uint32_t tokenCntAlign_{0};
    uint64_t sendCntOffset_{0};
    uint64_t tilePrefixOffset_{0};
    uint64_t expertPartialOffset_{0};
    uint64_t tileReadyOffset_{0};
    uint64_t dstRankInfoOffset_{0};
    uint64_t srcTokenTableOffset_{0};
    uint64_t cntWinStateOffset_{0};
    uint64_t slotWinStateOffset_{0};
    uint64_t winDataOffset_{0};
    uint64_t payloadStashWinOffset_{0};
    uint64_t completionFlagSourceWinOffset_{0};
    uint64_t dstRankWinOffset_{0};
    uint64_t dstStateWinOffset_{0};
    bool isSendCore_{false};
    bool isPrepareCore_{false};

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
    sendCoreCount_ = Mc2Kernel::MOE_SHARED_JETTY_NUM;
    prepareCoreStart_ = sendCoreCount_;
    prepareCoreCount_ = aivNum_ > prepareCoreStart_ ? aivNum_ - prepareCoreStart_ : 0U;
    isSendCore_ = aivId_ < sendCoreCount_;
    isPrepareCore_ = aivId_ >= prepareCoreStart_ && aivId_ < prepareCoreStart_ + prepareCoreCount_;
    prepareCoreId_ = isPrepareCore_ ? aivId_ - prepareCoreStart_ : 0U;
    tileTokens_ = info.workspace.tileTokens;
    tileCount_ = info.workspace.tileCount;
    sendCntOffset_ = info.workspace.sendCntOffset;
    tilePrefixOffset_ = info.workspace.tilePrefixOffset;
    expertPartialOffset_ = info.workspace.expertPartialOffset;
    tileReadyOffset_ = info.workspace.tileReadyOffset;
    srcTokenListBytes_ = info.workspace.srcTokenListBytes;
    dstRankInfoOffset_ = info.workspace.dstRankInfoOffset;
    srcTokenTableOffset_ = info.workspace.srcTokenTableOffset;
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
    epWorldSizeAlign_ = Ceil(epWorldSize_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    perGroupTokenNum_ = PER_GROUP_SIZE / sizeof(int16_t) / axisK_;
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
    dstRankWinOffset_ = static_cast<uint64_t>(epRankId_) * axisMaxBS_ * perSlotBytes_;
    tokenCntAlign_ = srcTokenListBytes_ / sizeof(int32_t);
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
    padParams_ = {true, 0, 0, 0};
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::BindWorkspaceAndWindow(GM_ADDR workspaceGM)
{
    sendCounterAddr_ = workspaceGM;
    sendCntWorkspaceAddr_ = workspaceGM + sendCntOffset_;
    dstRankInfoAddr_ = workspaceGM + dstRankInfoOffset_;
    srcTokenTableAddr_ = workspaceGM + srcTokenTableOffset_;
    localCntStateWinAddr_ = GetWinAddrByRankId(mc2Context_, epRankId_, cntWinStateOffset_);
    localSlotWinAddr_ = GetWinAddrByRankId(mc2Context_, epRankId_, winDataOffset_) + dstRankWinOffset_;
    payloadStashWinAddr_ = GetWinAddrByRankId(mc2Context_, epRankId_, payloadStashWinOffset_);
    completionFlagSourceWinAddr_ = GetWinAddrByRankId(mc2Context_, epRankId_, completionFlagSourceWinOffset_);
    sendCounterGMTensor_.SetGlobalBuffer((__gm__ int32_t*)sendCounterAddr_);
    tilePrefixGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(workspaceGM + tilePrefixOffset_));
    expertPartialGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(workspaceGM + expertPartialOffset_));
    tileReadyGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(workspaceGM + tileReadyOffset_));
    recvCounterGMTensor_.SetGlobalBuffer((__gm__ int32_t*)localCntStateWinAddr_);
    metaSlotGMTensor_.SetGlobalBuffer((__gm__ int32_t*)payloadStashWinAddr_);
    sendCntPerRankGMTensor_.SetGlobalBuffer((__gm__ int32_t*)sendCntWorkspaceAddr_);
    sendCntPerExpertGMTensor_.SetGlobalBuffer((__gm__ int32_t*)(sendCntWorkspaceAddr_ + epWorldSizeAlign512_));
    dstRankGMTensor_.SetGlobalBuffer((__gm__ int16_t*)dstRankInfoAddr_);
    srcTokenIdxGMTensor_.SetGlobalBuffer((__gm__ int32_t*)srcTokenTableAddr_);
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::Init(
    GM_ADDR context, GM_ADDR x, GM_ADDR topkIdx, GM_ADDR topkWeights, GM_ADDR scales, GM_ADDR cachedSlotIdx,
    GM_ADDR numRecvPerRank, GM_ADDR numRecvPerExpert, GM_ADDR dstBufferSlotIdx, GM_ADDR workspaceGM, GM_ADDR tilingGM,
    TPipe* pipe, const MoeEpDispatchTilingData* tilingData)
{
    tpipe_ = pipe;
    aivId_ = GetBlockIdx();
    workspaceGM_ = workspaceGM;
    mc2Context_ = reinterpret_cast<__gm__ Mc2Kernel::MoeSharedJettyContext*>(context);
    epRankId_ = mc2Context_->epRankId;
#if defined(ASCENDC_OOM) && ASCENDC_OOM == 1
    MoeEpDispatchBase::RegisterOomRanges<DoCpuSync>(mc2Context_, epRankId_, tilingGM, tilingData);
#endif
    constexpr size_t metadataOffset =
        offsetof(MoeEpDispatchTilingData, moeEpDispatchInfo) + offsetof(MoeEpDispatchInfo, dumpMetadata);
    MoeEpExceptionDump::WriteMetadata<Mc2Kernel::MoeSharedJettyContext>(context, tilingGM + metadataOffset);
    diagWriter_.Init<Mc2Kernel::MoeSharedJettyContext>(context, MOE_EP_CORE_DIAG_DISPATCH, tpipe_);

    const auto& info = tilingData->moeEpDispatchInfo;
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
    LocalTensor<int32_t> flag = maskBuf_.Get<int32_t>();
    if (aivId_ == 0U) {
        Duplicate<int32_t>(flag, 1, WIN_ADDR_ALIGN / sizeof(int32_t));
        GlobalTensor<int32_t> source;
        source.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(completionFlagSourceWinAddr_));
        SyncFunc<AscendC::HardEvent::V_MTE3>();
        DataCopy(source, flag, WIN_ADDR_ALIGN / sizeof(int32_t));
        SyncFunc<AscendC::HardEvent::MTE3_V>();
    }
    Duplicate<int32_t>(flag, 0, UB_STRIDE);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    for (uint32_t tile = aivId_; tile < tileCount_; tile += aivNum_) {
        DataCopy(tileReadyGMTensor_[static_cast<uint64_t>(tile) * UB_STRIDE], flag, UB_STRIDE);
    }
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::AccumulateSendCounts(LocalTensor<int16_t> ids,
                                                                                          LocalTensor<int32_t> counts,
                                                                                          uint32_t count,
                                                                                          uint32_t scopeSize)
{
    // TopK input has already been narrowed into tempBuf_; this scratch no longer aliases the input.
    LocalTensor<uint16_t> histogram = topkIdsBuf_.Get<uint16_t>();
    LocalTensor<int32_t> wide =
        topkIdsBuf_.GetWithOffset<int32_t>(HISTOGRAM_SCOPE_SIZE, HISTOGRAM_SCOPE_SIZE * sizeof(uint16_t));
    for (uint32_t base = 0U; base < scopeSize; base += HISTOGRAM_SCOPE_SIZE) {
        uint32_t bins = scopeSize - base < HISTOGRAM_SCOPE_SIZE ? scopeSize - base : HISTOGRAM_SCOPE_SIZE;
        GetRouteHistogram(histogram, ids, count, base, bins);
        Cast(wide.template ReinterpretCast<uint32_t>(), histogram, RoundMode::CAST_NONE, bins);
        Add(counts[base], counts[base], wide, bins);
        PipeBarrier<PIPE_V>();
    }
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::DedupAndSend(LocalTensor<int16_t> expertIdsTensor,
                                                                                  uint32_t calCnt, uint32_t calCntAlign,
                                                                                  uint32_t tmpOffset)
{
    uint32_t maskCnt = Ceil(calCntAlign, BITS_PER_BYTE);
    LocalTensor<int16_t> internalIndexTensor = dstExpBuf_.GetWithOffset<int16_t>(calCntAlign, tmpOffset);
    LocalTensor<int16_t> tempTensorInt16 = tempBuf_.Get<int16_t>();
    LocalTensor<int16_t> subTensorInt16 = topkIdsBuf_.GetWithOffset<int16_t>(calCntAlign, 0);
    LocalTensor<int32_t> gatherIdxTensor = topkIdsBuf_.Get<int32_t>();
    LocalTensor<uint8_t> selectMaskTensor = maskBuf_.GetWithOffset<uint8_t>(maskBytesAlign_, 0);
    LocalTensor<uint8_t> equalMaskTensor = maskBuf_.GetWithOffset<uint8_t>(maskBytesAlign_, maskBytesAlign_);
    LocalTensor<uint8_t> compareMaskTensor = maskBuf_.GetWithOffset<uint8_t>(maskBytesAlign_, 2 * maskBytesAlign_);

    // A smaller next group can place scratch inside the previous group's rank DMA source.
    SyncFunc<AscendC::HardEvent::MTE3_V>();
    Duplicate<int16_t>(internalIndexTensor, static_cast<int16_t>(moeExpertNumPerRank_), calCnt);
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

    // The last TopK column has no later element to suppress.
    for (uint32_t k = 0; k + 1U < axisK_; k++) {
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
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::CalSendCntGroup(uint32_t tokenStart,
                                                                                     uint32_t tokenCount)
{
    uint32_t calCnt = tokenCount * axisK_;
    uint32_t topkIdxOffset = tokenStart * axisK_;
    uint32_t tmpOffset = Ceil(calCnt * sizeof(int16_t), ALIGNED_LEN_256) * ALIGNED_LEN_256;
    uint32_t calCntAlign = tmpOffset / sizeof(int16_t);
    LocalTensor<int32_t> topkIdsGroupTensor = topkIdsBuf_.Get<int32_t>();
    LocalTensor<int16_t> tempTensorInt16 = tempBuf_.Get<int16_t>();
    DataCopyPadExtParams<int32_t> copyPadParams{false, 0U, 0U, 0U};
    DataCopyExtParams copyParams = {1U, static_cast<uint32_t>(calCnt * sizeof(int32_t)), 0U, 0U, 0U};
    dstRankTensor_ = dstExpBuf_.GetWithOffset<int16_t>(calCntAlign, 0);
    SyncFunc<AscendC::HardEvent::V_MTE2>();
    DataCopyPad(topkIdsGroupTensor, topkIdxGMTensor_[topkIdxOffset], copyParams, copyPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    PrepareMetadata(tokenStart, tokenCount);
    Cast(tempTensorInt16, topkIdsGroupTensor, RoundMode::CAST_NONE, calCnt);
    PipeBarrier<PIPE_V>();
    AccumulateSendCounts(tempTensorInt16, sendCntPerExpertTensor_, calCnt, moeExpertNum_);
    if constexpr (IsCached) {
        dstSlotIdxTensor_ = topkIdsBuf_.Get<int32_t>();
        SyncFunc<AscendC::HardEvent::V_MTE2>();
        DataCopyPad(dstSlotIdxTensor_, cachedSlotIdxGMTensor_[topkIdxOffset], copyParams, copyPadParams);
    }
    DedupAndSend(tempTensorInt16, calCnt, calCntAlign, tmpOffset);
    // Complete dedup before its UB scratch is reused by per-rank counting.
    SyncFunc<AscendC::HardEvent::V_S>();
    SyncFunc<AscendC::HardEvent::S_V>();
    if constexpr (IsCached) {
        SyncFunc<AscendC::HardEvent::MTE2_MTE3>();
        DataCopyPad(dstSlotIdxGMTensor_[topkIdxOffset], dstSlotIdxTensor_, copyParams);
        SyncFunc<AscendC::HardEvent::MTE3_V>();
    }
    LocalTensor<uint8_t> selected = maskBuf_.Get<uint8_t>();
    Select(dstRankTensor_, selected, dstRankTensor_, static_cast<int16_t>(-1),
           AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, calCnt);
    PipeBarrier<PIPE_V>();
    AccumulateSendCounts(dstRankTensor_, sendCntPerRankTensor_, calCnt, epWorldSize_);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopyExtParams dstRankCopyParams = {1U, static_cast<uint32_t>(calCnt * sizeof(int16_t)), 0U, 0U, 0U};
    DataCopyPad(dstRankGMTensor_[topkIdxOffset], dstRankTensor_, dstRankCopyParams);
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::CalSendCnt()
{
    for (uint32_t tile = aivId_; tile < tileCount_; tile += aivNum_) {
        uint32_t start = tile * tileTokens_;
        uint32_t count = axisBS_ - start < tileTokens_ ? axisBS_ - start : tileTokens_;
        Duplicate<int32_t>(sendCntPerRankTensor_, 0, counterCnt_);
        CalSendCntGroup(start, count);
        SyncFunc<AscendC::HardEvent::V_MTE3>();
        DataCopy(sendCounterGMTensor_[static_cast<uint64_t>(tile) * counterAlign512_], sendCntPerRankTensor_,
                 counterCnt_);
        SyncFunc<AscendC::HardEvent::MTE3_V>();
    }
    // Empty count owners also publish zeros: the reduction reads every physical AIV row.
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(expertPartialGMTensor_[static_cast<uint64_t>(aivId_) * moeExpertNumAlign512_ / sizeof(int32_t)],
             sendCntPerExpertTensor_, moeExpertNumAlign_ / sizeof(int32_t));
}
template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::ScanTileCounts(uint32_t peer, uint32_t firstTile,
                                                                                    uint32_t rows)
{
    SyncFunc<AscendC::HardEvent::MTE3_V>();
    LocalTensor<int32_t> input = topkIdsBuf_.Get<int32_t>();
    LocalTensor<int32_t> output = topkIdsBuf_.Get<int32_t>()[perGroupSizeAlign_ / sizeof(int32_t)];
    // DataCopyPad uses bytes for blockLen and GM stride, unlike DataCopy's 32-byte units.
    DataCopyParams read{static_cast<uint16_t>(rows), UB_ALIGN, static_cast<uint16_t>(epWorldSizeAlign512_ - UB_ALIGN),
                        0U};
    SyncFunc<AscendC::HardEvent::V_MTE2>();
    DataCopyPad(input, sendCounterGMTensor_[static_cast<uint64_t>(firstTile) * counterAlign512_ + peer], read,
                padParams_);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    // Inclusive scan along tile rows, with eight independent peer columns per data block.
    for (uint32_t step = 1U; step < rows; step *= 2U) {
        Adds(output, input, int32_t{0}, step * UB_STRIDE);
        Add(output[step * UB_STRIDE], input[step * UB_STRIDE], input, (rows - step) * UB_STRIDE);
        PipeBarrier<PIPE_V>();
        auto previous = input;
        input = output;
        output = previous;
    }
    BinaryRepeatParams broadcastCarry{1, 1, 0, 1, 1, 0};
    Add(output, input, sendCntPerRankTensor_, UB_STRIDE, static_cast<uint8_t>(rows), broadcastCarry);
    PipeBarrier<PIPE_V>();
    Copy(sendCntPerRankTensor_, output[(rows - 1U) * UB_STRIDE], UB_STRIDE);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopyParams write{static_cast<uint16_t>(rows), UB_ALIGN, 0U,
                         static_cast<uint16_t>(epWorldSizeAlign512_ - UB_ALIGN)};
    DataCopyPad(tilePrefixGMTensor_[static_cast<uint64_t>(firstTile + 1U) * counterAlign512_ + peer], output, write);
    SyncFunc<AscendC::HardEvent::MTE3_V>();
}
template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::BuildSendPrefixes()
{
    constexpr uint32_t rowsPerBatch = 128U; // Fits repeat-count and UB limits independently of BS/P.
    for (uint32_t peer = aivId_ * UB_STRIDE; peer < epWorldSize_; peer += aivNum_ * UB_STRIDE) {
        SyncFunc<AscendC::HardEvent::MTE3_V>();
        Duplicate<int32_t>(sendCntPerRankTensor_, 0, UB_STRIDE);
        SyncFunc<AscendC::HardEvent::V_MTE3>();
        DataCopy(tilePrefixGMTensor_[peer], sendCntPerRankTensor_, UB_STRIDE);
        for (uint32_t first = 0U; first < tileCount_; first += rowsPerBatch) {
            uint32_t rows = tileCount_ - first < rowsPerBatch ? tileCount_ - first : rowsPerBatch;
            ScanTileCounts(peer, first, rows);
        }
        SyncFunc<AscendC::HardEvent::V_MTE3>();
        DataCopy(sendCntPerRankGMTensor_[peer], sendCntPerRankTensor_, UB_STRIDE);
    }
}
template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::ReduceExpertPartials()
{
    LocalTensor<int32_t> input = topkIdsBuf_.Get<int32_t>();
    LocalTensor<uint8_t> scratch = tempBuf_.Get<uint8_t>();
    LocalTensor<int32_t> result = numRecvPerRankBuf_.Get<int32_t>();
    for (uint32_t expert = aivId_ * UB_STRIDE; expert < moeExpertNum_; expert += aivNum_ * UB_STRIDE) {
        SyncFunc<AscendC::HardEvent::MTE3_V>();
        SyncFunc<AscendC::HardEvent::V_MTE2>();
        DataCopyParams read{static_cast<uint16_t>(aivNum_), UB_ALIGN,
                            static_cast<uint16_t>(moeExpertNumAlign512_ - UB_ALIGN), 0U};
        DataCopyPad(input, expertPartialGMTensor_[expert], read, padParams_);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        const uint32_t shape[] = {aivNum_, UB_STRIDE};
        ReduceSum<int32_t, AscendC::Pattern::Reduce::RA, true>(result, input, scratch, shape, true);
        SyncFunc<AscendC::HardEvent::V_MTE3>();
        DataCopy(sendCntPerExpertGMTensor_[expert], result, UB_STRIDE);
    }
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::SendCountToPeer(uint32_t peer, GM_ADDR dst,
                                                                                     GM_ADDR src, GM_ADDR notifyAddr,
                                                                                     uint64_t notifyVal)
{
    auto multiHandle = static_cast<MultiChannelHandle>(GetOwnedJettyHandle());
    auto batch = hcomm_.MakeBatchHandle(multiHandle, hcommTensor_, HCOMM_INIT_SIZE);
    uint32_t peerIndex = peer < epRankId_ ? peer : peer - 1;
    auto& handle = hcomm_.GetHandleRef(batch, peerIndex, dst);
    hcomm_.WriteWithNotifyNbi<DATA_CFG>(handle, dst, src, moeNumPerRankSize_, notifyAddr, notifyVal);
    hcomm_.BatchCommit(batch);
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::Communication()
{
    if (!isSendCore_) {
        return;
    }
    tpipe_->InitBuffer(hcommBuf_, HCOMM_INIT_SIZE);
    hcommTensor_ = hcommBuf_.Get<uint8_t>();
    Duplicate<uint8_t>(hcommTensor_, 0, HCOMM_INIT_SIZE);
    SyncFunc<AscendC::HardEvent::V_S>();

    DataCopy(sendCntPerRankTensor_, sendCntPerRankGMTensor_, counterCnt_);
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    for (uint32_t dstRankId = aivId_; dstRankId < epWorldSize_; dstRankId += sendCoreCount_) {
        int32_t sendCnt = sendCntPerRankTensor_.GetValue(dstRankId);
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
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::SetSrcTokenIdxAndDstSlot(
    uint32_t topkOffset, uint32_t calCnt, uint32_t calCntAlign, uint32_t& sendLocalCnt,
    LocalTensor<int32_t>& cntLocalTensor)
{
    uint64_t rsvdCnt = 0;
    LocalTensor<int16_t> internalIndexTensor = topkIdsBuf_.GetWithOffset<int16_t>(calCntAlign, 0);
    LocalTensor<int16_t> scatterTensor = topkIdsBuf_.GetWithOffset<int16_t>(calCntAlign, calCntAlign * sizeof(int16_t));
    LocalTensor<int32_t> tempLocalTensor = dstExpBuf_.Get<int32_t>();
    LocalTensor<uint16_t> gatherMaskTensor = maskBuf_.Get<uint16_t>();
    LocalTensor<uint8_t> compareMaskTensor = maskBuf_.Get<uint8_t>();

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
        StoreSourceTokens(topkOffset, dstRankId, curSlot, curCnt, internalIndexTensor, scatterTensor);
        cntLocalTensor.SetValue(dstRankId, curSlot + curCnt);
        rsvdCnt = 0;
    }
    if constexpr (!IsCached) {
        StoreUncachedSlots(topkOffset, calCnt, calCntAlign);
    }
    SyncFunc<AscendC::HardEvent::MTE3_V>(); // 下一轮等待这一轮做完
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::StoreSourceTokens(uint32_t topkOffset,
                                                                                       uint32_t peer, uint32_t slot,
                                                                                       uint32_t count,
                                                                                       LocalTensor<int16_t> indices,
                                                                                       LocalTensor<int16_t> divisor)
{
    LocalTensor<int32_t> tokens = dstExpBuf_.Get<int32_t>();
    Duplicate<int16_t>(divisor, static_cast<int16_t>(axisK_), count);
    Div(indices, indices, divisor, count);
    Cast(tokens, indices, RoundMode::CAST_NONE, count);
    Adds(tokens, tokens, static_cast<int32_t>(topkOffset / axisK_), count);
    DataCopyParams copy{1U, static_cast<uint16_t>(count * sizeof(int32_t)), 0U, 0U};
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopyPad(srcTokenIdxGMTensor_[peer * tokenCntAlign_ + slot], tokens, copy);
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::StoreUncachedSlots(uint32_t topkOffset,
                                                                                        uint32_t count,
                                                                                        uint32_t aligned)
{
    LocalTensor<uint8_t> valid = maskBuf_.Get<uint8_t>();
    LocalTensor<int32_t> slots = topkIdsBuf_.Get<int32_t>();
    // Remove the peer-disambiguation offset; invalid and deduplicated entries remain -1.
    CompareScalar(valid, dstRankIdTensor_, static_cast<int16_t>(0), AscendC::CMPMODE::GE, aligned);
    Subs(dstRankIdTensor_, dstRankIdTensor_, static_cast<int16_t>(epWorldSize_), aligned);
    Select(dstRankIdTensor_, valid, dstRankIdTensor_, static_cast<int16_t>(-1),
           AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, aligned);
    Cast(slots, dstRankIdTensor_, RoundMode::CAST_NONE, count);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopyExtParams copy{1U, static_cast<uint32_t>(count * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPad(dstSlotIdxGMTensor_[topkOffset], slots, copy);
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::PrepareMetadata(uint32_t tokenStart,
                                                                                     uint32_t tokenCount)
{
    uint32_t count = tokenCount * axisK_;
    uint32_t aligned = Ceil(count * sizeof(int16_t), ALIGNED_LEN_256) * ALIGNED_LEN_256;
    LocalTensor<int16_t> index = tempBuf_.Get<int16_t>();
    LocalTensor<int16_t> token = tempBuf_.GetWithOffset<int16_t>(count, aligned);
    LocalTensor<int16_t> divisor = topkIdsBuf_.GetWithOffset<int16_t>(count, perGroupSizeAlign_);
    LocalTensor<int32_t> offsets = tempBuf_.GetWithOffset<int32_t>(count, PER_GROUP_SIZE / 2U);
    metaLocalTensor_ = dstExpBuf_.Get<int32_t>();
    SyncFunc<AscendC::HardEvent::MTE3_V>();
    CreateVecIndex(index, int16_t{0}, count);
    Duplicate<int16_t>(divisor, static_cast<int16_t>(axisK_), count);
    Div(token, index, divisor, count);
    Muls(token, token, static_cast<int16_t>(metaSlotStride_ - axisK_), count);
    Add(index, index, token, count);
    Muls(index, index, static_cast<int16_t>(sizeof(int32_t)), count);
    Cast(offsets, index, RoundMode::CAST_NONE, count);
    Scatter(metaLocalTensor_[topkCopyAlign_], topkIdsBuf_.Get<int32_t>(), offsets.template ReinterpretCast<uint32_t>(),
            0U, count);
    LoadMetadataExtras(tokenStart, tokenCount);
    SyncFunc<AscendC::HardEvent::V_S>();
    uint32_t header = topkCopyAlign_ + 2U * axisKAlign_;
    for (uint32_t i = 0U; i < tokenCount; ++i) {
        metaLocalTensor_.SetValue(i * metaSlotStride_ + header, epRankId_);
        metaLocalTensor_.SetValue(i * metaSlotStride_ + header + 1U, tokenStart + i);
    }
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopyExtParams copy{1U, tokenCount * metaSlotBytes_, 0U, 0U, 0U};
    DataCopyPad(metaSlotGMTensor_[static_cast<uint64_t>(tokenStart) * metaSlotStride_], metaLocalTensor_, copy);
    // Preserve raw TopK in topkIdsBuf_; the following histogram reuses all other scratch.
    SyncFunc<AscendC::HardEvent::MTE3_V>();
}
template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::LoadMetadataExtras(uint32_t tokenStart,
                                                                                        uint32_t tokenCount)
{
    if constexpr (IsTopkWeights) {
        DataCopyExtParams weights{static_cast<uint16_t>(tokenCount), axisK_ * TOPK_INFO_SIZE, 0U,
                                  (metaSlotBytes_ - kAlignSize_) / UB_ALIGN, 0U};
        DataCopyPadExtParams<float> pad{false, 0U, 0U, 0};
        DataCopyPad(metaLocalTensor_[topkCopyAlign_ + axisKAlign_].template ReinterpretCast<float>(),
                    topkWeightsGMTensor_[tokenStart * axisK_], weights, pad);
    }
    if constexpr (Std::IsSame<XType, fp8_e5m2_t>::value || Std::IsSame<XType, fp8_e4m3fn_t>::value) {
        DataCopyExtParams scales{static_cast<uint16_t>(tokenCount), scalesBytes_, 0U,
                                 static_cast<uint32_t>((metaSlotBytes_ - topkCopyAlign_ * sizeof(int32_t)) / UB_ALIGN),
                                 0U};
        DataCopyPadExtParams<uint8_t> pad{false, 0U, 0U, 0};
        DataCopyPad(metaLocalTensor_.template ReinterpretCast<uint8_t>(),
                    scalesGMTensor_.template ReinterpretCast<uint8_t>()[tokenStart * scalesBytes_], scales, pad);
    }
    SyncFunc<AscendC::HardEvent::MTE2_S>();
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::CopyLocalMetadata(uint32_t count,
                                                                                       uint32_t firstSlot)
{
    LocalTensor<int32_t> indices = dstExpBuf_.Get<int32_t>();
    LocalTensor<int32_t> metadata = topkIdsBuf_.Get<int32_t>();
    SyncFunc<AscendC::HardEvent::V_MTE2>();
    SyncFunc<AscendC::HardEvent::S_MTE2>();
    for (uint32_t first = 0U; first < count; first += META_BATCH_TOKENS) {
        uint32_t rows = count - first < META_BATCH_TOKENS ? count - first : META_BATCH_TOKENS;
        for (uint32_t i = 0U; i < rows; ++i) {
            uint32_t token = static_cast<uint32_t>(indices.GetValue(first + i));
            DataCopyPad(metadata[i * metaSlotStride_],
                        metaSlotGMTensor_[static_cast<uint64_t>(token) * metaSlotStride_], metaSlotCopyParams_,
                        padParams_);
        }
        SyncFunc<AscendC::HardEvent::MTE2_MTE3>();
        localMetaSlotGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(
            localSlotWinAddr_ + static_cast<uint64_t>(firstSlot + first) * perSlotBytes_ + tokenSize_));
        DataCopyExtParams copy{static_cast<uint16_t>(rows), metaSlotBytes_, 0U, perSlotBytes_ - metaSlotBytes_, 0U};
        DataCopyPad(localMetaSlotGMTensor_, metadata, copy);
        SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
    }
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::PrepareTile(uint32_t tile)
{
    uint32_t start = tile * tileTokens_;
    uint32_t tokens = axisBS_ - start < tileTokens_ ? axisBS_ - start : tileTokens_;
    uint32_t count = tokens * axisK_;
    uint32_t aligned = Ceil(count * sizeof(int16_t), ALIGNED_LEN_256) * ALIGNED_LEN_256 / sizeof(int16_t);
    LocalTensor<int32_t> slots = numRecvPerRankBuf_.Get<int32_t>();
    dstRankIdTensor_ = tempBuf_.Get<int16_t>();
    SyncFunc<AscendC::HardEvent::S_MTE2>();
    SyncFunc<AscendC::HardEvent::V_MTE2>();
    SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
    DataCopy(slots, tilePrefixGMTensor_[static_cast<uint64_t>(tile) * counterAlign512_], counterCnt_);
    DataCopyExtParams routes{1U, static_cast<uint32_t>(count * sizeof(int16_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int16_t> pad{false, 0U, 0U, 0};
    DataCopyPad(dstRankIdTensor_, dstRankGMTensor_[start * axisK_], routes, pad);
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    uint32_t localSlot = static_cast<uint32_t>(slots.GetValue(epRankId_));
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    uint32_t localCount = 0U;
    SetSrcTokenIdxAndDstSlot(start * axisK_, count, aligned, localCount, slots);
    SyncFunc<AscendC::HardEvent::V_S>();
    CopyLocalMetadata(localCount, localSlot);
    PublishTileReady(tile);
}
template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::PrepareSendTiles()
{
    if (!isPrepareCore_) {
        return;
    }
    for (uint32_t tile = prepareCoreId_; tile < tileCount_; tile += prepareCoreCount_) {
        PrepareTile(tile);
    }
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::PublishTileReady(uint32_t tile)
{
    LocalTensor<int32_t> ready = maskBuf_.Get<int32_t>();
    SyncFunc<AscendC::HardEvent::S_V>();
    Duplicate<int32_t>(ready, 1, UB_STRIDE);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    // A tile is immutable after publication; metadata and its entire peer index slice precede ready.
    PipeBarrier<PIPE_MTE3>();
    DataCopy(tileReadyGMTensor_[static_cast<uint64_t>(tile) * UB_STRIDE], ready, UB_STRIDE);
    SyncFunc<AscendC::HardEvent::MTE3_V>();
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::WaitTileReady(uint32_t tile)
{
    LocalTensor<int32_t> ready = maskBuf_.Get<int32_t>();
    int32_t value = 0;
    while (value != 1) {
        SyncFunc<AscendC::HardEvent::S_MTE2>();
        DataCopy(ready, tileReadyGMTensor_[static_cast<uint64_t>(tile) * UB_STRIDE], UB_STRIDE);
        SyncFunc<AscendC::HardEvent::MTE2_S>();
        value = ready.GetValue(0);
    }
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
    SyncFunc<AscendC::HardEvent::V_S>();

    // ub buffer init
    tpipe_->InitBuffer(sendCntRankBuf_, 2U * sendCntRankSizeAlign_);
    tpipe_->InitBuffer(tempBuf_, MAX_BATCH_SIZE);
    tpipe_->InitBuffer(maskBuf_, UB_ALIGN);
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::LoadTileDescriptor(uint32_t tile)
{
    WaitTileReady(tile);
    SyncFunc<AscendC::HardEvent::S_MTE2>();
    LocalTensor<int32_t> prefix = sendCntRankBuf_.Get<int32_t>();
    DataCopyParams rows{2U, static_cast<uint16_t>(epWorldSizeAlign_),
                        static_cast<uint16_t>(epWorldSizeAlign512_ - epWorldSizeAlign_), 0U};
    DataCopyPad(prefix, tilePrefixGMTensor_[static_cast<uint64_t>(tile) * counterAlign512_], rows, padParams_);
    SyncFunc<AscendC::HardEvent::MTE2_S>();
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::WriteToRemoteWindow()
{
    if (!isSendCore_) {
        return;
    }
    CommBufferInit();
    payloadBatch_ = hcomm_.MakeBatchHandle(static_cast<MultiChannelHandle>(GetOwnedJettyHandle()), hcommBatchTensor_,
                                           hcommBatchBufferBytes_);
    pendingWqebb_ = 0U;
    // Tile boundaries do not end a batch; ready tiles have immutable source metadata and indices.
    for (uint32_t tile = aivId_; tile < tileCount_; tile += sendCoreCount_) {
        LoadTileDescriptor(tile);
        SendTokenPipeline();
    }
    PublishSendCompletions();
    CommitBatch(true);
    DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(topkIdxGMTensor_);
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline bool MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::NextTokenChunk(uint32_t& peerOffset,
                                                                                    uint32_t& start,
                                                                                    uint32_t& remaining,
                                                                                    TokenChunk& chunk)
{
    if (peerOffset == epWorldSize_) {
        return false;
    }
    uint32_t rank = (aivId_ + peerOffset) % epWorldSize_;
    bool first = remaining == 0U;
    if (first && rank != epRankId_) {
        LocalTensor<int32_t> prefix = sendCntRankBuf_.Get<int32_t>();
        start = static_cast<uint32_t>(prefix.GetValue(rank));
        remaining = static_cast<uint32_t>(prefix.GetValue(counterCnt_ + rank)) - start;
    }
    uint32_t count = remaining < SOURCE_INDEX_CHUNK_TOKENS ? remaining : SOURCE_INDEX_CHUNK_TOKENS;
    chunk = {rank, start, count, first, count == remaining};
    start += count;
    remaining -= count;
    if (chunk.last) {
        ++peerOffset;
        start = 0U;
    }
    return true;
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::PrefetchTokenIndices(const TokenChunk& chunk,
                                                                                          uint32_t buffer,
                                                                                          TEventID event)
{
    if (chunk.count == 0U) {
        return;
    }
    // Reuse only after the previous scalar descriptor construction has finished reading this half.
    SyncFunc<AscendC::HardEvent::S_MTE2>();
    LocalTensor<int32_t> indices = tempBuf_.Get<int32_t>()[buffer * SOURCE_INDEX_BUFFER_BYTES / sizeof(int32_t)];
    GlobalTensor<int32_t> source;
    source.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t*>(srcTokenTableAddr_ + static_cast<uint64_t>(chunk.rank) * srcTokenListBytes_));
    DataCopyExtParams params{1U, static_cast<uint32_t>(chunk.count * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> pad{false, 0U, 0U, 0U};
    DataCopyPad(indices, source[chunk.start], params, pad);
    SetFlag<AscendC::HardEvent::MTE2_S>(event);
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::WriteTokenPayloads(const TokenChunk& chunk,
                                                                                        uint32_t buffer,
                                                                                        BufDesc* localDescs)
{
    LocalTensor<int32_t> indices = tempBuf_.Get<int32_t>()[buffer * SOURCE_INDEX_BUFFER_BYTES / sizeof(int32_t)];
    GM_ADDR remote = GetWinAddrByRankId(mc2Context_, chunk.rank, winDataOffset_) + dstRankWinOffset_ +
                     static_cast<uint64_t>(chunk.start) * perSlotBytes_;
    for (uint32_t offset = 0U; offset < chunk.count; offset += SLOT_NUM_PER_SQE) {
        uint32_t slots = chunk.count - offset;
        slots = slots < SLOT_NUM_PER_SQE ? slots : SLOT_NUM_PER_SQE;
        for (uint32_t slot = 0U; slot < slots; ++slot) {
            uint64_t token = static_cast<uint64_t>(indices.GetValue(offset + slot));
            localDescs[slot * SGE_PER_SLOT].addr = xAddr_ + token * tokenSize_;
            localDescs[slot * SGE_PER_SLOT + 1U].addr = payloadStashWinAddr_ + token * metaSlotBytes_;
        }
        WritePayload(remote, localDescs, slots * SGE_PER_SLOT);
        remote += slots * perSlotBytes_;
    }
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::SendTokenChunk(const TokenChunk& chunk,
                                                                                    uint32_t buffer, TEventID event,
                                                                                    BufDesc* localDescs)
{
    GM_ADDR notifyAddr =
        GetWinAddrByRankId(mc2Context_, chunk.rank, slotWinStateOffset_) + dstStateWinOffset_ + aivId_ * WIN_ADDR_ALIGN;
    if (chunk.rank != epRankId_) {
        if (chunk.first) {
            uint32_t peerIndex = chunk.rank < epRankId_ ? chunk.rank : chunk.rank - 1U;
            // Rebind only when consuming the first chunk of a peer, never during prefetch.
            hcomm_.GetHandleRef(payloadBatch_, peerIndex, notifyAddr);
        }
        if (chunk.count != 0U) {
            WaitFlag<AscendC::HardEvent::MTE2_S>(event);
            WriteTokenPayloads(chunk, buffer, localDescs);
        }
    }
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::SendTokenPipeline()
{
    TEventID ready[2] = {tpipe_->AllocEventID<AscendC::HardEvent::MTE2_S>(),
                         tpipe_->AllocEventID<AscendC::HardEvent::MTE2_S>()};
    BufDesc localDescs[SGE_PER_SQE] = {};
    for (uint32_t i = 0U; i < SGE_PER_SQE; i += SGE_PER_SLOT) {
        localDescs[i].len = tokenSize_;
        localDescs[i + 1U].len = metaSlotBytes_;
    }
    TokenChunk chunks[2];
    uint32_t peerOffset = 0U;
    uint32_t start = 0U;
    uint32_t remaining = 0U;
    uint32_t current = 0U;
    bool hasCurrent = NextTokenChunk(peerOffset, start, remaining, chunks[current]);
    if (hasCurrent) {
        PrefetchTokenIndices(chunks[current], current, ready[current]);
    }
    while (hasCurrent) {
        uint32_t next = current ^ 1U;
        bool hasNext = NextTokenChunk(peerOffset, start, remaining, chunks[next]);
        if (hasNext) {
            PrefetchTokenIndices(chunks[next], next, ready[next]);
        }
        SendTokenChunk(chunks[current], current, ready[current], localDescs);
        current = next;
        hasCurrent = hasNext;
    }
    tpipe_->ReleaseEventID<AscendC::HardEvent::MTE2_S>(ready[0]);
    tpipe_->ReleaseEventID<AscendC::HardEvent::MTE2_S>(ready[1]);
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::ReserveBatchSpace(uint32_t wqebbCount)
{
    uint32_t limit = hcommBatchBufferBytes_ / HCOMM_WQE_BYTES;
    if (pendingWqebb_ + wqebbCount > limit) {
        CommitBatch(true);
    }
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::CommitBatch(bool force)
{
    uint32_t limit = hcommBatchBufferBytes_ / HCOMM_WQE_BYTES;
    if (pendingWqebb_ != 0U && (force || pendingWqebb_ >= limit)) {
        hcomm_.BatchCommit(payloadBatch_);
        pendingWqebb_ = 0U;
    }
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::WritePayload(GM_ADDR dst, BufDesc* localDescs,
                                                                                  uint32_t sgeCount)
{
    uint32_t wqebbCount = Ceil(HCOMM_SQE_BYTES + sgeCount * HCOMM_SGE_BYTES, HCOMM_WQE_BYTES);
    ReserveBatchSpace(wqebbCount);
    hcomm_.WriteNbi<DATA_CFG>(payloadBatch_.handle, dst, localDescs, sgeCount);
    pendingWqebb_ += wqebbCount;
    CommitBatch(false);
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::SendCompletion(uint32_t dstRankId,
                                                                                    GM_ADDR notifyAddr)
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
    hcomm_.WriteNbi<NOTIFY_CFG>(payloadBatch_.handle, notifyAddr, completionFlagSourceWinAddr_, WIN_ADDR_ALIGN);
    pendingWqebb_ += flagWqebb;
    CommitBatch(false);
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::PublishSendCompletions()
{
    // Exactly one ordered flag per sender/peer, even for senders without any BS tile.
    for (uint32_t offset = 0U; offset < epWorldSize_; ++offset) {
        uint32_t peer = (aivId_ + offset) % epWorldSize_;
        GM_ADDR notify =
            GetWinAddrByRankId(mc2Context_, peer, slotWinStateOffset_) + dstStateWinOffset_ + aivId_ * WIN_ADDR_ALIGN;
        if (peer != epRankId_) {
            uint32_t peerIndex = peer < epRankId_ ? peer : peer - 1U;
            hcomm_.GetHandleRef(payloadBatch_, peerIndex, notify);
        }
        SendCompletion(peer, notify);
    }
}
template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::Process()
{
    if ASCEND_IS_AIV {
        BufferInit();
        CalSendCnt();
        SyncAll<true>();
        BuildSendPrefixes();
        ReduceExpertPartials();
        SyncAll<true>();
        diagWriter_.RunPosRecord(MOE_EP_DISPATCH_RUN_POS_COUNT_READY);
        Communication();
        PrepareSendTiles();
        GetRecvCount();
        WriteToRemoteWindow();
        diagWriter_.RunPosRecord(MOE_EP_DISPATCH_RUN_POS_URMA_REQUESTS_ISSUE_DONE);
    }
}

#endif
} // namespace MoeEpDispatchClosOnlyImpl

#endif // MOE_EP_DISPATCH_CLOS_ONLY_H
