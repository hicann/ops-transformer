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
 * \file moe_ep_dispatch_hybrid.h
 * \brief
 */

#ifndef MOE_EP_DISPATCH_HYBRID_H
#define MOE_EP_DISPATCH_HYBRID_H

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

#if __has_include("../common/moe_distribute_base.h")
#include "../common/moe_distribute_base.h"
#include "../common/mc2_kernel_utils.h"
#include "../common/mc2_moe_context.h"
#include "../common/moe_ep_exception_dump_writer.h"
#else
#include "../../common/op_kernel/moe_distribute_base.h"
#include "../../common/op_kernel/mc2_kernel_utils.h"
#include "../../common/op_kernel/mc2_moe_context.h"
#include "../../common/op_kernel/moe_ep_exception_dump_writer.h"
#endif

namespace MoeEpDispatchHybridImpl {

#if defined(ENABLE_MOE_EP_KERNEL)

#define TemplateMoeEpDispatchHybridTypeClass \
    typename XType, typename ScalesType, bool DoCpuSync, bool IsCached, bool IsTopkWeights, uint8_t NetworkMode
#define TemplateMoeEpDispatchHybridTypeFunc XType, ScalesType, DoCpuSync, IsCached, IsTopkWeights, NetworkMode

using namespace AscendC;
using namespace Mc2Kernel;

constexpr uint32_t UB_ALIGN = 32U;
constexpr uint32_t WIN_ADDR_ALIGN = 512U;
constexpr uint32_t ALIGNED_LEN_256 = 256U;
constexpr uint32_t TOPK_INFO_SIZE = 4U;  // sizeof(int32_t)=sizeof(float)=4B
constexpr uint32_t UB_STRIDE = 8U;       // UB_ALIGN/sizeof(int32_t)=8
constexpr uint32_t INT64_UB_STRIDE = 4U; // UB_ALIGN/sizeof(int64_t)=4
constexpr int32_t BITS_PER_BYTE = 8;
constexpr uint32_t HCOMM_INIT_SIZE = 512U;
constexpr uint32_t PER_GROUP_SIZE = 40 * 1024U; // token 分核后 40KB 足够，并为并存 counter 预留 UB
constexpr uint32_t EXPERT_NUM_PER_GROUP = 256U; // GetExpertFreq 单次最多256个bin
constexpr uint32_t DATA_BLOCK_NUM = 8U;         // 256B/32B=8
constexpr uint8_t BUFFER_NUM = 2;
constexpr int64_t STATUS_POLL_BACKOFF_CYCLES = 100; // ready 未就绪时的轮询退避周期
constexpr uint32_t STATUS_CLEAR_BATCH_RECORDS = 64U;
// status[0]发布总slot数，其余status只表示对应slot已到达。
constexpr uint32_t SCALEOUT_SLOT_READY = 1U;
// 高32位为nextSlot，低32位为slotsLeft；低32位为-1表示count尚未读到。
constexpr int64_t SOURCE_STATE_UNREAD = (static_cast<int64_t>(0) << 32) | 0xFFFFFFFFLL;
constexpr int32_t SOURCE_COUNT_UNREAD = -1;

template <TemplateMoeEpDispatchHybridTypeClass>
class MoeEpDispatchHybrid {
public:
    __aicore__ inline MoeEpDispatchHybrid(){};
    __aicore__ inline void Init(GM_ADDR context, GM_ADDR x, GM_ADDR topkIdx, GM_ADDR topkWeights, GM_ADDR scales,
                                GM_ADDR cachedSlotIdx, GM_ADDR cachedRouteCount, GM_ADDR cachedRouteDstScaleout,
                                GM_ADDR cachedRouteScaleoutSlot, GM_ADDR numRecvPerRank, GM_ADDR numRecvPerExpert,
                                GM_ADDR dstBufferSlotIdx, GM_ADDR routeCount, GM_ADDR routeDstScaleout,
                                GM_ADDR routeScaleoutSlot, GM_ADDR workspaceGM, GM_ADDR tilingGM, TPipe *pipe,
                                const MoeEpDispatchTilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void PrepareDispatchPayloads();
    __aicore__ inline void TransferDispatchPayloads();
    __aicore__ inline void InitTilingFields(const MoeEpDispatchTilingData *tilingData);
    __aicore__ inline void InitAlignmentFields(const MoeEpDispatchTilingData *tilingData);
    __aicore__ inline void InitGlobalTensors(GM_ADDR x, GM_ADDR topkIdx, GM_ADDR topkWeights, GM_ADDR scales,
                                             GM_ADDR cachedSlotIdx, GM_ADDR cachedRouteCount,
                                             GM_ADDR cachedRouteDstScaleout, GM_ADDR cachedRouteScaleoutSlot,
                                             GM_ADDR numRecvPerRank, GM_ADDR numRecvPerExpert, GM_ADDR dstBufferSlotIdx,
                                             GM_ADDR routeCount, GM_ADDR routeDstScaleout, GM_ADDR routeScaleoutSlot);
    __aicore__ inline void InitCopyParams();
    __aicore__ inline void InitGlobalAddresses(const MoeEpDispatchTilingData *tilingData);
    __aicore__ inline void CalSendCntPerRank(LocalTensor<int16_t> expertIdsTensor, uint32_t calCnt);
    __aicore__ inline void CalSendCntPerExpert(LocalTensor<int16_t> expertIdsTensor, uint32_t calCnt);
    __aicore__ inline void CalSendCntPerScaleout(LocalTensor<int16_t> rankIdsTensor, uint32_t calCnt,
                                                 LocalTensor<int32_t> scaleoutCounterTensor);
    __aicore__ inline void CalSendCnt();
    __aicore__ inline LocalTensor<int32_t> InitSendCountBuffers();
    __aicore__ inline void ResetSendCountWorkspace();
    __aicore__ inline void FlushSendCountToWorkspace(LocalTensor<int32_t> scaleoutCounterTensor);
    __aicore__ inline void ExchangeCount();
    __aicore__ inline void SendCountToRemoteRank(uint32_t dstRankId, uint64_t notifyValue);
    __aicore__ inline void CopyLocalCountToWindow(uint32_t dstRankId, uint64_t notifyVal);
    __aicore__ inline bool IsCommTargetRank(uint32_t dstRankId);
    __aicore__ inline void SendRemainingCountRange();
    __aicore__ inline void SendCommTargetCounts(uint32_t remoteScaleoutStart, uint32_t remoteScaleoutEnd,
                                                uint32_t localRankStart, uint32_t localRankEnd);
    __aicore__ inline void GetRecvCount();
    __aicore__ inline void SetRecvNumPerExpert();
    __aicore__ inline void WriteHostRecvTokenCount(LocalTensor<int64_t> recvPerExpertTensor);
    __aicore__ inline void SetRecvNumPerRank(LocalTensor<int32_t> recvTmpTensor);
    __aicore__ inline uint32_t DedupAndBuildSendEntries(uint32_t tokenId);
    __aicore__ inline void GetSlotStartNum(uint32_t tokenRangeIndex);
    __aicore__ inline void ReduceCounterRange(GlobalTensor<int32_t> counterGMTensor, LocalTensor<int32_t> dstTensor,
                                              uint32_t counterAlign512, uint32_t counterCnt, uint32_t counterValueCount,
                                              uint32_t counterCoreCount);
    __aicore__ inline uint32_t FindOrCreateRemoteRouteEntry(uint32_t tokenId, uint32_t dstScaleoutIndex,
                                                            uint32_t &routeEntryCount, bool &isNewRouteEntry);
    __aicore__ inline void WriteRouteTableToGM(uint32_t tokenId, uint32_t routeEntryCount);
    __aicore__ inline uint32_t CopyCachedRouteForToken(uint32_t tokenId);
    __aicore__ inline void AllocateScaleupSlotForExpert(uint32_t tokenId, uint32_t topkIndex, uint32_t dstRankId,
                                                        uint32_t &scaleupSlot);
    __aicore__ inline void UpdateScaleoutRouteForExpert(uint32_t tokenId, uint32_t dstRankId,
                                                        uint32_t &routeEntryCount);
    __aicore__ inline void WritePayloadStash(uint32_t tokenId);
    __aicore__ inline void WriteSendEntry(GM_ADDR sendEntryBaseAddr, uint32_t sendEntryIndex, uint32_t sourceSlotIndex,
                                          uint32_t destinationSlotIndex);
    __aicore__ inline void ReadSendEntry(GM_ADDR sendEntryAddr, uint32_t &sourceSlotIndex,
                                         uint32_t &destinationSlotIndex);
    __aicore__ inline void WriteScaleupSendEntry(uint32_t dstRankId, uint32_t tokenId, uint32_t scaleupSlot);
    __aicore__ inline void WriteScaleoutSendEntriesFromRoute(uint32_t tokenId, uint32_t routeEntryCount);
    __aicore__ inline void SplitRangeForCore(uint32_t itemCount, uint32_t coreCount, uint32_t coreIndex,
                                             uint32_t &itemStart, uint32_t &itemEnd);
    __aicore__ inline bool InitOwnedScaleupRankRange(uint32_t &destinationScaleupStart,
                                                     uint32_t &destinationScaleupEnd);
    __aicore__ inline void SendScaleoutPayloadsToProxy();
    __aicore__ inline void ReduceScaleoutCounterGroup(uint32_t counterGroup, LocalTensor<int32_t> counterSumTensor);
    __aicore__ inline void SendScaleoutPayloadToProxy(uint32_t remoteServerOrdinal, uint32_t dstScaleoutIndex,
                                                      uint32_t proxyRankId, uint32_t scaleoutSlotCount);
    __aicore__ inline void SendScaleoutPayloadRange(uint32_t tokenRangeIndex, uint32_t remoteServerOrdinal,
                                                    uint32_t dstScaleoutIndex, uint32_t scaleoutSlotCount,
                                                    uint64_t commHandle, GM_ADDR remoteScaleoutBase,
                                                    GM_ADDR remoteStatusBase);
    __aicore__ inline void PrepareFanoutDestinations(GM_ADDR scaleoutSlotAddr, uint32_t destinationScaleupStart,
                                                     uint32_t destinationScaleupEnd, bool waitForPreviousRouteRead);
    __aicore__ inline void SendFanoutPayloadForSlot(GM_ADDR scaleoutSlotAddr, uint32_t srcRankId,
                                                    uint32_t destinationScaleupStart, uint32_t destinationScaleupEnd);
    __aicore__ inline bool TryGetScaleoutSlotCount(uint32_t srcScaleoutIndex, uint32_t &count);
    __aicore__ inline bool TryScaleoutSlotReady(uint32_t srcScaleoutIndex, uint32_t scaleoutSlot);
    __aicore__ inline void GetSourceState(uint32_t srcOrdinal, int32_t &nextSlot, int32_t &slotsLeft);
    __aicore__ inline void SetSourceState(uint32_t srcOrdinal, int32_t nextSlot, int32_t slotsLeft);
    __aicore__ inline void ProcessOneScaleoutSlot(uint32_t srcScaleoutIndex, uint32_t scaleoutSlot,
                                                  uint32_t destinationScaleupStart, uint32_t destinationScaleupEnd,
                                                  bool &hasReadRouteInfo);
    __aicore__ inline void SendFanoutPayloads(uint32_t destinationScaleupStart, uint32_t destinationScaleupEnd);
    __aicore__ inline void SendScaleupPayloads();
    __aicore__ inline void SendPayloadsToScaleupRank(uint32_t dstRankId);
    __aicore__ inline void SendScaleupEntries(uint32_t dstRankId, uint64_t commHandle);
    __aicore__ inline void SendEntries(GM_ADDR sendEntryBaseAddr, uint32_t sendEntryCount, GM_ADDR sourceDataBaseAddr,
                                       uint32_t sourceSlotBytes, uint32_t srcRankId, uint32_t dstRankId,
                                       uint64_t commHandle);
    __aicore__ inline void SendFanoutPayloadToRank(GM_ADDR sourcePayloadAddr, uint32_t srcRankId, uint32_t dstRankId,
                                                   uint32_t destinationSlot);
    __aicore__ inline void CopyPayloadToLocal(GM_ADDR sourcePayloadAddr, GM_ADDR destinationPayloadAddr);
    __aicore__ inline void PublishFinalReadyStatuses(uint32_t dstRankId);
    __aicore__ inline void WriteFinalReadyStatusToRank(uint32_t srcRankId, uint32_t dstRankId);
    __aicore__ inline void ClearScaleoutReceiveStatus(GM_ADDR currentStatusAddr, uint32_t statusRecordCount);
    __aicore__ inline void ClearReceivedScaleoutStatuses();
    __aicore__ inline void InitPreparePayloadsBuffers();
    __aicore__ inline void InitRouteStateTensors(uint32_t rankSlotOffset, uint32_t routeStateOffset,
                                                 uint32_t routeCountOffset, uint32_t sendEntryStartOffset);
    __aicore__ inline void InitRouteStateForToken(uint32_t tokenId);
    __aicore__ inline void PrepareTokenPayload(uint32_t tokenId, uint32_t topkOffset);
    __aicore__ inline uint32_t ProcessCachedTokenRoutes(uint32_t tokenId, uint32_t topkOffset);
    __aicore__ inline void PreparePayloads();
    __aicore__ inline void InitPayloadBuildTokenRanges(uint32_t &tokenRangeIndexStart, uint32_t &tokenRangeIndexEnd);
    __aicore__ inline void SplitToCore(uint32_t itemCount, uint32_t coreCount, uint32_t &itemStart, uint32_t &itemEnd,
                                       uint32_t &itemNum);

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
    GlobalTensor<int32_t> cachedRouteCountGMTensor_;
    GlobalTensor<int32_t> cachedRouteDstScaleoutGMTensor_;
    GlobalTensor<int32_t> cachedRouteScaleoutSlotGMTensor_;
    GlobalTensor<int32_t> routeCountGMTensor_;
    GlobalTensor<int32_t> routeDstScaleoutGMTensor_;
    GlobalTensor<int32_t> routeScaleoutSlotGMTensor_;
    GlobalTensor<int32_t> scaleupCounterGMTensor_;
    GlobalTensor<int32_t> scaleoutCounterGMTensor_;
    GlobalTensor<int32_t> recvCounterGMTensor_;
    GlobalTensor<int32_t> sendCntGMTensor_;
    GlobalTensor<ScalesType> scalesGMTensor_;

    LocalTensor<XType> xLocalTensor_;
    LocalTensor<XType> tokenSlotTensor_;
    LocalTensor<int32_t> topkIdxTensor_;
    LocalTensor<int32_t> dstSlotIdxTensor_;
    LocalTensor<int32_t> metaLocalTensor_;
    LocalTensor<int32_t> numRecvPerRankTensor_;
    LocalTensor<int64_t> numRecvPerExpertTensor_;
    LocalTensor<int32_t> sendCntPerRankTensor_;
    LocalTensor<int32_t> sendCntPerExpertTensor_;
    LocalTensor<uint8_t> hcommTensor_;
    // PreparePayloads 使用的 UB tensor 切片，在 PreparePayloads 初始化时从 numRecvPerRankBuf_ 切出
    LocalTensor<int32_t> slotIdxPerRankTensor_;
    LocalTensor<int32_t> counterSumTensor_; // prefix/scaleout 归约结果，发送阶段与 rank slot 映射复用 UB
    LocalTensor<int32_t> sendRankSlotTensor_;
    LocalTensor<int32_t> routeScaleoutIndexTensor_;
    LocalTensor<int32_t> routeScaleoutSlotTensor_;
    LocalTensor<int32_t> sendRouteIndexByScaleoutTensor_; // 发送阶段记录 scaleout 到 route entry 的映射
    LocalTensor<int32_t> routeCountTensor_; // route 表项数的固定 UB 搬出块，避免 GM DCache 伪共享
    LocalTensor<int32_t> scaleupSendEntryStartTensor_;  // 当前 count 分片的 scaleup 发送记录起始 slot
    LocalTensor<int32_t> scaleoutSendEntryStartTensor_; // 当前 count 分片的 scaleout 发送记录起始 slot
    LocalTensor<int64_t> sourceStateTensor_; // fanout source 调度状态：高32位nextSlot，低32位slotsLeft

    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> perSlotQueue_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> dstSlotQueue_;
    TBuf<> topkIdsBuf_;
    TBuf<> tempBuf_;
    TBuf<> dstExpBuf_;
    TBuf<> hcommBuf_; // 通信
    TBuf<> numRecvBuf_;
    TBuf<> numRecvPerRankBuf_;
    TBuf<> numRecvPerExpertBuf_;
    TBuf<> recvCntBuf_;
    TBuf<> recvTempBuf_;
    TBuf<> sendEntryWriteBuf_;
    TBuf<> sendEntryReadBuf_;
    TBuf<> routeInfoBuf_;
    TBuf<> sourceStateBuf_;

    GM_ADDR workspaceGM_{nullptr};
    GM_ADDR hostPinnedCounterAddrGM_{nullptr};
    GM_ADDR scaleupCounterAddr_{nullptr};
    GM_ADDR sendCntWorkspaceAddr_{nullptr};
    GM_ADDR scaleoutSendEntryAddr_{nullptr};
    GM_ADDR scaleoutCounterAddr_{nullptr};
    GM_ADDR scaleupSendEntryAddr_{nullptr};
    GM_ADDR payloadStashWinAddr_{nullptr};

    uint32_t axisBS_{0};
    uint32_t axisH_{0};
    uint32_t axisK_{0};
    uint32_t epWorldSize_{0};
    uint32_t moeExpertNumPerRank_{0};
    uint32_t axisMaxBS_{0};
    uint32_t scalesBytes_{0};
    uint32_t perSlotBytes_{0};
    uint32_t scaleoutSlotBytes_{0};
    uint32_t moeExpertNum_{0};
    uint32_t serverNum_{0};
    uint32_t remoteServerCount_{0};
    uint32_t numScaleoutSendAiv_{0};
    uint32_t numNodeLocalAiv_{0};
    uint32_t aivNum_{0};
    uint32_t epRankId_{0};
    uint32_t aivId_{0};
    uint32_t rankNumPerServer_{0};
    uint32_t currentServerIndex_{0};
    uint32_t serverStartRank_{0};
    uint32_t serverEndRank_{0};
    uint32_t startTokenId_{0};
    uint32_t endTokenId_{0};
    uint32_t sendTokenNum_{0};
    uint32_t startRankId_{0};
    uint32_t endRankId_{0};
    uint32_t rankNumPerCore_{0};
    uint32_t hAlignSize_{0};
    uint32_t kAlignSize_{0};
    uint32_t axisKAlign_{0};
    uint32_t currentTokenRangeIndex_{0};
    uint64_t sendEntryTokenRangeBytes_{0};
    uint32_t serverNumAlign_{0};    // server 数对齐到 UB_ALIGN
    uint32_t perGroupSizeAlign_{0}; // 每组 token 数 * axisK * sizeof(int16_t) 对齐到 256
    uint32_t perGroupTokenNum_{0};  // 每组 token 数
    uint32_t epWorldSizeAlign_{0};
    uint32_t epWorldSizeAlign512_{0};
    uint32_t moeNumPerRankAlign_{0};
    uint32_t moeExpertNumAlign_{0};
    uint32_t moeNumPerRankAlign512_{0};
    uint32_t cntPerRankSizeAlign512_{0};
    uint32_t counterCnt_{0};
    uint32_t counterAlign512_{0};
    uint32_t scaleoutCounterCnt_{0};
    uint32_t scaleoutCounterAlign512_{0};
    uint32_t metaOffset_{0};
    uint64_t cntWinStateOffset_{0};
    uint64_t slotWinStateOffset_{0};
    uint64_t winDataOffset_{0};
    uint64_t scaleoutRecvDataOffset_{0};
    uint64_t scaleoutRecvStatusOffset_{0};
    uint32_t dispatchNotifyCount_{1};
    bool sendEntryWritePending_{false};

    DataCopyParams statusCopyParams_;
    DataCopyParams clearStatusCopyParams_;
    DataCopyParams topkCopyParams_;
    DataCopyParams xCopyParams_;             // PreparePayloads 复制单 token x payload
    DataCopyParams scalesCopyParams_;        // PreparePayloads 复制 fp8 scales payload
    DataCopyParams fanoutPayloadCopyParams_; // fanout 搬运完整 slot payload
    DataCopyExtParams routeCountCopyParams_;
    DataCopyExtParams routeCopyParams_;
    DataCopyExtParams sendEntryCopyParams_;
    DataCopyPadParams padParams_;
    DataCopyPadExtParams<int32_t> routeCopyPadParams_{true, 0U, 0U, 0};
    DataCopyPadExtParams<uint32_t> sendEntryCopyPadParams_{true, 0U, 0U, 0U};
};

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::Init(
    GM_ADDR context, GM_ADDR x, GM_ADDR topkIdx, GM_ADDR topkWeights, GM_ADDR scales, GM_ADDR cachedSlotIdx,
    GM_ADDR cachedRouteCount, GM_ADDR cachedRouteDstScaleout, GM_ADDR cachedRouteScaleoutSlot, GM_ADDR numRecvPerRank,
    GM_ADDR numRecvPerExpert, GM_ADDR dstBufferSlotIdx, GM_ADDR routeCount, GM_ADDR routeDstScaleout,
    GM_ADDR routeScaleoutSlot, GM_ADDR workspaceGM, GM_ADDR tilingGM, TPipe *pipe,
    const MoeEpDispatchTilingData *tilingData)
{
    tpipe_ = pipe;
    aivId_ = GetBlockIdx();
    workspaceGM_ = workspaceGM;
    mc2Context_ = (__gm__ Mc2Aclnn::MoeCommContext *)context;
    epRankId_ = mc2Context_->epRankId;
    constexpr size_t metadataOffset =
        offsetof(MoeEpDispatchTilingData, moeEpDispatchInfo) + offsetof(MoeEpDispatchInfo, dumpMetadata);
    MoeEpExceptionDump::WriteMetadata(context, tilingGM + metadataOffset);
    diagWriter_.Init(context, MOE_EP_CORE_DIAG_DISPATCH, tpipe_);

    InitTilingFields(tilingData);
    InitGlobalTensors(x, topkIdx, topkWeights, scales, cachedSlotIdx, cachedRouteCount, cachedRouteDstScaleout,
                      cachedRouteScaleoutSlot, numRecvPerRank, numRecvPerExpert, dstBufferSlotIdx, routeCount,
                      routeDstScaleout, routeScaleoutSlot);
    InitCopyParams();
    InitGlobalAddresses(tilingData);
    diagWriter_.RunPosRecord(MOE_EP_DISPATCH_RUN_POS_INIT_DONE);
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::InitTilingFields(
    const MoeEpDispatchTilingData *tilingData)
{
    const auto &info = tilingData->moeEpDispatchInfo;
    axisBS_ = info.cfg.numTokens;
    axisH_ = info.cfg.hidden;
    axisK_ = info.cfg.topK;
    epWorldSize_ = info.cfg.epWorldSize;
    moeExpertNumPerRank_ = info.cfg.numLocalExperts;
    axisMaxBS_ = info.cfg.numMaxTokensPerRank;
    scalesBytes_ = info.scalesBytes;
    perSlotBytes_ = info.perSlotBytes;
    scaleoutSlotBytes_ = info.window.scaleoutSlotAlignedBytes;
    serverNum_ = info.hybrid.serverNum;
    remoteServerCount_ = serverNum_ - 1U;
    numScaleoutSendAiv_ = info.hybrid.scaleoutAivNum;
    numNodeLocalAiv_ = info.hybrid.scaleupAivNum;
    sendEntryTokenRangeBytes_ = info.workspace.sendEntryTokenRangeBytes;
    aivNum_ = info.aivNum;
    rankNumPerServer_ = info.hybrid.rankNumPerServer;
    currentServerIndex_ = MoeEpDispatchBase::GetCurrentServerIndex(epRankId_, rankNumPerServer_);
    serverStartRank_ = MoeEpDispatchBase::GetServerStartRank(epRankId_, rankNumPerServer_);
    serverEndRank_ = MoeEpDispatchBase::GetServerEndRank(epRankId_, rankNumPerServer_, epWorldSize_);
    cntWinStateOffset_ = info.window.cntWinStateOffset;
    slotWinStateOffset_ = info.window.slotWinStateOffset;
    winDataOffset_ = info.window.winDataOffset;
    scaleoutRecvDataOffset_ = info.window.scaleoutRecvDataOffset;
    scaleoutRecvStatusOffset_ = info.window.scaleoutRecvStatusOffset;
    dispatchNotifyCount_ = info.dispatchNotifyCount;
    hostPinnedCounterAddrGM_ = (GM_ADDR)info.hostPinnedCounterAddr;
    moeExpertNum_ = moeExpertNumPerRank_ * epWorldSize_;

    InitAlignmentFields(tilingData);
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::InitAlignmentFields(
    const MoeEpDispatchTilingData *tilingData)
{
    const auto &info = tilingData->moeEpDispatchInfo;
    hAlignSize_ = Ceil(axisH_ * sizeof(XType), UB_ALIGN) * UB_ALIGN;
    kAlignSize_ = Ceil(axisK_ * TOPK_INFO_SIZE, UB_ALIGN) * UB_ALIGN;
    axisKAlign_ = kAlignSize_ / TOPK_INFO_SIZE;
    serverNumAlign_ = Ceil(serverNum_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN / sizeof(int32_t);
    metaOffset_ = hAlignSize_;
    epWorldSizeAlign_ = Ceil(epWorldSize_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    counterCnt_ = epWorldSizeAlign_ / sizeof(int32_t);
    scaleoutCounterCnt_ = Ceil(serverNum_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN / sizeof(int32_t);
    // GM workspace perRankCount 按 512B/rank 对齐, 与 direct 一致
    epWorldSizeAlign512_ = Ceil(epWorldSize_ * sizeof(int32_t), WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN;
    counterAlign512_ = epWorldSizeAlign512_ / sizeof(int32_t);
    scaleoutCounterAlign512_ = Ceil(serverNum_ * sizeof(int32_t), WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN / sizeof(int32_t);
    // GM workspace perRankCount: 每 rank 512B (state + dstRankRecvNum), 与 direct 一致
    cntPerRankSizeAlign512_ = epWorldSize_ * WIN_ADDR_ALIGN;
    moeNumPerRankAlign_ = Ceil(moeExpertNumPerRank_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    moeExpertNumAlign_ = Ceil(moeExpertNum_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    moeNumPerRankAlign512_ = Ceil(moeExpertNumPerRank_ * sizeof(int32_t), WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN;
    perGroupTokenNum_ = PER_GROUP_SIZE / sizeof(int16_t) / axisK_;
    perGroupSizeAlign_ = Ceil(perGroupTokenNum_ * axisK_ * sizeof(int16_t), ALIGNED_LEN_256) * ALIGNED_LEN_256;
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::InitGlobalTensors(
    GM_ADDR x, GM_ADDR topkIdx, GM_ADDR topkWeights, GM_ADDR scales, GM_ADDR cachedSlotIdx, GM_ADDR cachedRouteCount,
    GM_ADDR cachedRouteDstScaleout, GM_ADDR cachedRouteScaleoutSlot, GM_ADDR numRecvPerRank, GM_ADDR numRecvPerExpert,
    GM_ADDR dstBufferSlotIdx, GM_ADDR routeCount, GM_ADDR routeDstScaleout, GM_ADDR routeScaleoutSlot)
{
    xGMTensor_.SetGlobalBuffer((__gm__ XType *)x);
    topkIdxGMTensor_.SetGlobalBuffer((__gm__ int32_t *)topkIdx);
    if constexpr (IsTopkWeights) {
        topkWeightsGMTensor_.SetGlobalBuffer((__gm__ float *)topkWeights);
    }
    if constexpr (Std::IsSame<XType, fp8_e5m2_t>::value || Std::IsSame<XType, fp8_e4m3fn_t>::value) {
        scalesGMTensor_.SetGlobalBuffer((__gm__ ScalesType *)scales);
        metaOffset_ += Ceil(scalesBytes_, UB_ALIGN) * UB_ALIGN;
    }
    if constexpr (IsCached) {
        cachedSlotIdxGMTensor_.SetGlobalBuffer((__gm__ int32_t *)cachedSlotIdx);
        cachedRouteCountGMTensor_.SetGlobalBuffer((__gm__ int32_t *)cachedRouteCount);
        cachedRouteDstScaleoutGMTensor_.SetGlobalBuffer((__gm__ int32_t *)cachedRouteDstScaleout);
        cachedRouteScaleoutSlotGMTensor_.SetGlobalBuffer((__gm__ int32_t *)cachedRouteScaleoutSlot);
    }
    numRecvPerRankGMTensor_.SetGlobalBuffer((__gm__ int32_t *)numRecvPerRank);
    numRecvPerExpertGMTensor_.SetGlobalBuffer((__gm__ int64_t *)numRecvPerExpert);
    dstSlotIdxGMTensor_.SetGlobalBuffer((__gm__ int32_t *)dstBufferSlotIdx);
    routeCountGMTensor_.SetGlobalBuffer((__gm__ int32_t *)routeCount);
    routeDstScaleoutGMTensor_.SetGlobalBuffer((__gm__ int32_t *)routeDstScaleout);
    routeScaleoutSlotGMTensor_.SetGlobalBuffer((__gm__ int32_t *)routeScaleoutSlot);
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::InitCopyParams()
{
    tpipe_->InitBuffer(numRecvPerRankBuf_, epWorldSizeAlign_);
    numRecvPerRankTensor_ = numRecvPerRankBuf_.Get<int32_t>();

    statusCopyParams_ = {static_cast<uint16_t>(epWorldSize_), 1U,
                         static_cast<uint16_t>((WIN_ADDR_ALIGN - UB_ALIGN) / UB_ALIGN), 0U};
    clearStatusCopyParams_ = {static_cast<uint16_t>(epWorldSize_), 1U, 0U,
                              static_cast<uint16_t>((WIN_ADDR_ALIGN - UB_ALIGN) / UB_ALIGN)};
    topkCopyParams_ = {1U, static_cast<uint16_t>(axisK_ * TOPK_INFO_SIZE), 0U, 0U};
    xCopyParams_ = {1U, static_cast<uint16_t>(axisH_ * sizeof(XType)), 0U, 0U};
    scalesCopyParams_ = {1U, static_cast<uint16_t>(scalesBytes_), 0U, 0U};
    fanoutPayloadCopyParams_ = {1U, static_cast<uint16_t>(perSlotBytes_), 0U, 0U};
    routeCountCopyParams_ = {1U, static_cast<uint32_t>(sizeof(int32_t)), 0U, 0U, 0U};
    routeCopyParams_ = {1U, static_cast<uint32_t>(axisK_ * sizeof(int32_t)), 0U, 0U, 0U};
    sendEntryCopyParams_ = {1U, MOE_EP_SEND_ENTRY_BYTES, 0U, 0U, 0U};
    padParams_ = {true, 0, 0, 0};
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::InitGlobalAddresses(
    const MoeEpDispatchTilingData *tilingData)
{
    const auto &info = tilingData->moeEpDispatchInfo;
    scaleupCounterAddr_ = workspaceGM_;
    sendCntWorkspaceAddr_ = scaleupCounterAddr_ + aivNum_ * epWorldSizeAlign512_;
    scaleoutCounterAddr_ = workspaceGM_ + info.workspace.routeWorkspaceOffset;
    scaleoutSendEntryAddr_ = workspaceGM_ + info.workspace.scaleoutSendEntryOffset;
    scaleupSendEntryAddr_ = workspaceGM_ + info.workspace.scaleupSendEntryOffset;
    payloadStashWinAddr_ =
        MoeEpDispatchBase::GetWindowAddrByRankId(mc2Context_, epRankId_, info.window.payloadStashWinOffset);
    GM_ADDR localCntStateWinAddr = MoeEpDispatchBase::GetWindowAddrByRankId(mc2Context_, epRankId_, cntWinStateOffset_);
    scaleupCounterGMTensor_.SetGlobalBuffer((__gm__ int32_t *)scaleupCounterAddr_);
    scaleoutCounterGMTensor_.SetGlobalBuffer((__gm__ int32_t *)scaleoutCounterAddr_);
    recvCounterGMTensor_.SetGlobalBuffer((__gm__ int32_t *)localCntStateWinAddr);
    sendCntGMTensor_.SetGlobalBuffer((__gm__ int32_t *)sendCntWorkspaceAddr_);
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::SplitToCore(
    uint32_t itemCount, uint32_t coreCount, uint32_t &itemStart, uint32_t &itemEnd, uint32_t &itemNum)
{
    SplitRangeForCore(itemCount, coreCount, aivId_, itemStart, itemEnd);
    itemNum = itemEnd - itemStart;
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::InitPayloadBuildTokenRanges(
    uint32_t &tokenRangeIndexStart, uint32_t &tokenRangeIndexEnd)
{
    if (aivNum_ <= 1U) {
        tokenRangeIndexStart = 0U;
        tokenRangeIndexEnd = 1U;
        return;
    }
    if (aivId_ == 0U) {
        tokenRangeIndexStart = 0U;
        tokenRangeIndexEnd = 0U;
        return;
    }

    // AIV0等待接收count，其余AIV接管全部原token范围，保持slot前缀与count行一一对应。
    uint32_t payloadWorkerCount = aivNum_ - 1U;
    uint32_t payloadWorkerIndex = aivId_ - 1U;
    uint32_t tokenRangeCountPerWorker = aivNum_ / payloadWorkerCount;
    uint32_t remainingTokenRangeCount = aivNum_ % payloadWorkerCount;
    tokenRangeIndexStart = tokenRangeCountPerWorker * payloadWorkerIndex;
    if (payloadWorkerIndex < remainingTokenRangeCount) {
        tokenRangeIndexStart += payloadWorkerIndex;
        tokenRangeIndexEnd = tokenRangeIndexStart + tokenRangeCountPerWorker + 1U;
    } else {
        tokenRangeIndexStart += remainingTokenRangeCount;
        tokenRangeIndexEnd = tokenRangeIndexStart + tokenRangeCountPerWorker;
    }
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::CalSendCntPerRank(
    LocalTensor<int16_t> expertIdsTensor, uint32_t calCnt)
{
    uint32_t tmpOffset = Ceil(calCnt * sizeof(int16_t), ALIGNED_LEN_256) * ALIGNED_LEN_256;
    uint32_t tokenCnt = calCnt / axisK_;
    uint32_t calCntAlign = tmpOffset / sizeof(int16_t);
    uint32_t tokenCntAlign = Ceil(tokenCnt * sizeof(int16_t), ALIGNED_LEN_256) * ALIGNED_LEN_256 / sizeof(int16_t);
    uint32_t mask = tokenCnt;
    uint32_t shape[2] = {tokenCnt, axisK_};
    LocalTensor<int16_t> dstTensorInt16 = dstExpBuf_.Get<int16_t>();
    LocalTensor<int16_t> tempTensorInt16 = topkIdsBuf_.GetWithOffset<int16_t>(calCntAlign, 0);
    LocalTensor<uint16_t> maskTensorInt16 = topkIdsBuf_.GetWithOffset<uint16_t>(calCntAlign, tmpOffset);
    LocalTensor<uint8_t> gatherMaskTensorInt8 = maskTensorInt16.template ReinterpretCast<uint8_t>();

    Duplicate<int16_t>(dstTensorInt16, static_cast<int16_t>(moeExpertNumPerRank_), calCnt);
    Div(tempTensorInt16, expertIdsTensor, dstTensorInt16, calCnt);
    // 筛选无效expert id 消除影响
    CompareScalar(gatherMaskTensorInt8, expertIdsTensor, static_cast<int16_t>(0), AscendC::CMPMODE::GE, calCntAlign);
    Select(dstTensorInt16, gatherMaskTensorInt8, tempTensorInt16, static_cast<int16_t>(-1),
           AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, calCnt);

    for (uint32_t dstRankId = 0U; dstRankId < epWorldSize_; dstRankId++) {
        // 筛选出发送到目标卡的token
        uint64_t rsvdCnt = 0;
        Subs(expertIdsTensor, dstTensorInt16, static_cast<int16_t>(dstRankId), calCnt);
        Abs(tempTensorInt16, expertIdsTensor, calCnt);
        ReduceMin<int16_t, Pattern::Reduce::AR, true>(expertIdsTensor, tempTensorInt16, shape, false); // 0为目标
        Duplicate<uint16_t>(maskTensorInt16, 0, tokenCntAlign); // GatherMask前清0
        CompareScalar(gatherMaskTensorInt8, expertIdsTensor, static_cast<int16_t>(0), AscendC::CMPMODE::EQ,
                      tokenCntAlign);
        GatherMask(tempTensorInt16, expertIdsTensor, maskTensorInt16, true, mask, {1, 1, 0, 0}, rsvdCnt);
        if (rsvdCnt == 0U) {
            continue;
        }
        SyncFunc<AscendC::HardEvent::V_S>();
        uint32_t offset = dstRankId * UB_STRIDE + 1U;
        int32_t curRankCnt = sendCntPerRankTensor_.GetValue(offset) + static_cast<int32_t>(rsvdCnt);
        sendCntPerRankTensor_.SetValue(offset, curRankCnt);
    }
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::CalSendCntPerExpert(
    LocalTensor<int16_t> expertIdsTensor, uint32_t calCnt)
{
    // 使用 GetExpertFreq 硬件向量化统计 expert 频率，替代逐 expert scalar 循环
    uint32_t tmpOffset = Ceil(calCnt * sizeof(int16_t), ALIGNED_LEN_256) * ALIGNED_LEN_256;
    uint32_t calCntAlign = tmpOffset / sizeof(int16_t);
    LocalTensor<int16_t> tempTensorInt16 = topkIdsBuf_.GetWithOffset<int16_t>(calCntAlign, tmpOffset);
    LocalTensor<uint16_t> gatherMaskTensorU16 = topkIdsBuf_.Get<uint16_t>();
    LocalTensor<uint16_t> gatherMaskU16GE = topkIdsBuf_.GetWithOffset<uint16_t>(calCntAlign, 0);
    LocalTensor<uint16_t> gatherMaskU16LT = topkIdsBuf_.GetWithOffset<uint16_t>(calCntAlign, tmpOffset);
    LocalTensor<uint8_t> gatherMaskU8GE = gatherMaskU16GE.ReinterpretCast<uint8_t>();
    LocalTensor<uint8_t> gatherMaskU8LT = gatherMaskU16LT.ReinterpretCast<uint8_t>();
    LocalTensor<int16_t> dstTensorInt16 = dstExpBuf_.Get<int16_t>();
    LocalTensor<uint8_t> dstTensorU8 = dstExpBuf_.Get<uint8_t>();
    LocalTensor<uint32_t> freqTensorU32 = dstExpBuf_.Get<uint32_t>();
    uint32_t maskU16Cnt = Ceil(calCntAlign, BITS_PER_BYTE * sizeof(uint16_t));

    uint32_t groupNum = Ceil(moeExpertNum_, EXPERT_NUM_PER_GROUP);
    for (uint32_t group = 0; group < groupNum; group++) {
        uint32_t baseExpertId = group * EXPERT_NUM_PER_GROUP;
        uint32_t curGroupSize = (baseExpertId + EXPERT_NUM_PER_GROUP > moeExpertNum_) ? (moeExpertNum_ - baseExpertId) :
                                                                                        EXPERT_NUM_PER_GROUP;
        uint64_t rsvdCnt = 0;
        Subs(dstTensorInt16, expertIdsTensor, static_cast<int16_t>(baseExpertId), calCnt);
        Duplicate<uint16_t>(gatherMaskTensorU16, 0, calCntAlign * 2);
        CompareScalar(gatherMaskU8GE, dstTensorInt16, static_cast<int16_t>(0), AscendC::CMPMODE::GE, calCntAlign);
        CompareScalar(gatherMaskU8LT, dstTensorInt16, static_cast<int16_t>(curGroupSize - 1), AscendC::CMPMODE::LE,
                      calCntAlign);
        And(gatherMaskU16GE, gatherMaskU16GE, gatherMaskU16LT, maskU16Cnt);
        GatherMask(tempTensorInt16, dstTensorInt16, gatherMaskU16GE, true, calCnt, {1, 1, 0, 0}, rsvdCnt);
        SyncFunc<AscendC::HardEvent::V_S>();
        if (rsvdCnt == 0) {
            continue;
        }

        Cast(dstTensorU8, tempTensorInt16, RoundMode::CAST_NONE, rsvdCnt);
        MoeEpDispatchBase::GetExpertFreq(gatherMaskTensorU16, dstTensorU8, rsvdCnt);
        Cast(freqTensorU32, gatherMaskTensorU16, RoundMode::CAST_NONE, curGroupSize);
        Add(sendCntPerExpertTensor_[baseExpertId], sendCntPerExpertTensor_[baseExpertId],
            freqTensorU32.ReinterpretCast<int32_t>(), static_cast<int32_t>(curGroupSize));
    }
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::CalSendCntPerScaleout(
    LocalTensor<int16_t> rankIdsTensor, uint32_t calCnt, LocalTensor<int32_t> scaleoutCounterTensor)
{
    if (serverNum_ == 0U) {
        return;
    }

    uint32_t tmpOffset = Ceil(calCnt * sizeof(int16_t), ALIGNED_LEN_256) * ALIGNED_LEN_256;
    uint32_t tokenCnt = calCnt / axisK_;
    uint32_t calCntAlign = tmpOffset / sizeof(int16_t);
    uint32_t tokenCntAlign = Ceil(tokenCnt * sizeof(int16_t), ALIGNED_LEN_256) * ALIGNED_LEN_256 / sizeof(int16_t);
    uint32_t mask = tokenCnt;
    uint32_t shape[2] = {tokenCnt, axisK_};
    LocalTensor<int16_t> scaleoutIdsTensor = tempBuf_.Get<int16_t>();
    LocalTensor<int16_t> tempTensorInt16 = topkIdsBuf_.GetWithOffset<int16_t>(calCntAlign, 0);
    LocalTensor<uint16_t> maskTensorInt16 = topkIdsBuf_.GetWithOffset<uint16_t>(calCntAlign, tmpOffset);
    LocalTensor<uint8_t> gatherMaskTensorInt8 = maskTensorInt16.template ReinterpretCast<uint8_t>();

    Duplicate<int16_t>(scaleoutIdsTensor, static_cast<int16_t>(rankNumPerServer_), calCnt);
    Div(tempTensorInt16, rankIdsTensor, scaleoutIdsTensor, calCnt);
    CompareScalar(gatherMaskTensorInt8, rankIdsTensor, static_cast<int16_t>(0), AscendC::CMPMODE::GE, calCntAlign);
    Select(scaleoutIdsTensor, gatherMaskTensorInt8, tempTensorInt16, static_cast<int16_t>(-1),
           AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, calCnt);

    for (uint32_t dstScaleoutIndex = 0U; dstScaleoutIndex < serverNum_; dstScaleoutIndex++) {
        if (dstScaleoutIndex == currentServerIndex_) {
            continue;
        }
        uint64_t rsvdCnt = 0;
        Subs(rankIdsTensor, scaleoutIdsTensor, static_cast<int16_t>(dstScaleoutIndex), calCnt);
        Abs(tempTensorInt16, rankIdsTensor, calCnt);
        ReduceMin<int16_t, Pattern::Reduce::AR, true>(rankIdsTensor, tempTensorInt16, shape, false);
        Duplicate<uint16_t>(maskTensorInt16, 0, tokenCntAlign);
        CompareScalar(gatherMaskTensorInt8, rankIdsTensor, static_cast<int16_t>(0), AscendC::CMPMODE::EQ,
                      tokenCntAlign);
        GatherMask(tempTensorInt16, rankIdsTensor, maskTensorInt16, true, mask, {1, 1, 0, 0}, rsvdCnt);
        if (rsvdCnt == 0U) {
            continue;
        }
        SyncFunc<AscendC::HardEvent::V_S>();
        int32_t scaleoutCount = scaleoutCounterTensor.GetValue(dstScaleoutIndex) + static_cast<int32_t>(rsvdCnt);
        scaleoutCounterTensor.SetValue(dstScaleoutIndex, scaleoutCount);
    }
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline LocalTensor<int32_t> MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::InitSendCountBuffers()
{
    uint32_t sendCntRankSizeAlign = Ceil(epWorldSize_ * UB_STRIDE * sizeof(int32_t), ALIGNED_LEN_256) * ALIGNED_LEN_256;
    uint32_t scaleoutCounterBytes = scaleoutCounterCnt_ * sizeof(int32_t);
    uint32_t sendCntBufferBytes = sendCntRankSizeAlign + moeExpertNumAlign_ + scaleoutCounterBytes;
    tpipe_->InitBuffer(topkIdsBuf_, 2 * perGroupSizeAlign_);
    tpipe_->InitBuffer(tempBuf_, perGroupSizeAlign_);
    tpipe_->InitBuffer(dstExpBuf_, perGroupSizeAlign_);
    tpipe_->InitBuffer(numRecvBuf_, sendCntBufferBytes);
    sendCntPerRankTensor_ = numRecvBuf_.GetWithOffset<int32_t>(sendCntRankSizeAlign / sizeof(int32_t), 0U);
    sendCntPerExpertTensor_ =
        numRecvBuf_.GetWithOffset<int32_t>(moeExpertNumAlign_ / sizeof(int32_t), sendCntRankSizeAlign);
    LocalTensor<int32_t> scaleoutCounterTensor =
        numRecvBuf_.GetWithOffset<int32_t>(scaleoutCounterCnt_, sendCntRankSizeAlign + moeExpertNumAlign_);
    Duplicate<int32_t>(sendCntPerRankTensor_, 0, sendCntRankSizeAlign / sizeof(int32_t));
    Duplicate<int32_t>(sendCntPerExpertTensor_, 0, moeExpertNumAlign_ / sizeof(int32_t));
    Duplicate<int32_t>(scaleoutCounterTensor, 0, scaleoutCounterCnt_);
    return scaleoutCounterTensor;
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::ResetSendCountWorkspace()
{
    if (aivId_ == aivNum_ - 1U) {
        uint32_t expertCntOffset = cntPerRankSizeAlign512_ / sizeof(int32_t);
        DataCopyParams expertCntCopyParams = {1U, static_cast<uint16_t>(moeExpertNum_ * sizeof(int32_t)), 0U, 0U};
        SyncFunc<AscendC::HardEvent::V_MTE3>();
        DataCopy(sendCntGMTensor_, sendCntPerRankTensor_, clearStatusCopyParams_);
        DataCopyPad(sendCntGMTensor_[expertCntOffset], sendCntPerExpertTensor_, expertCntCopyParams);
        PipeBarrier<PIPE_MTE3>(); // workspace 清零完成后，其他核才能开始 AtomicAdd
    }
    SyncAll<true>();
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::FlushSendCountToWorkspace(
    LocalTensor<int32_t> scaleoutCounterTensor)
{
    LocalTensor<uint32_t> gatherIndexTensor = topkIdsBuf_.GetWithOffset<uint32_t>(UB_STRIDE, epWorldSize_ * UB_ALIGN);
    SyncFunc<AscendC::HardEvent::V_S>(); // 前序 Vector 计算复用 topkIdsBuf_，完成后才能写 gather index
    gatherIndexTensor.SetValue(0, 2U);   // 源操作数每个datablock取下标为1的元素
    uint32_t compactMask = 2U;
    uint64_t reservedCount = 0UL;
    GatherMaskParams gatherMaskParams = {1, static_cast<uint16_t>(epWorldSize_), 1, 0};
    DataCopyParams sendPerRankParams = {static_cast<uint16_t>(epWorldSize_), static_cast<uint16_t>(UB_ALIGN), 0U,
                                        static_cast<uint16_t>(WIN_ADDR_ALIGN - UB_ALIGN)};
    DataCopyParams sendPerExpertParams = {1U, static_cast<uint16_t>(moeExpertNum_ * sizeof(int32_t)), 0U, 0U};
    SyncFunc<AscendC::HardEvent::S_V>();
    GatherMask(numRecvPerRankTensor_, sendCntPerRankTensor_, gatherIndexTensor, true, compactMask, gatherMaskParams,
               reservedCount);

    SyncFunc<AscendC::HardEvent::S_MTE3>();
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    SetAtomicAdd<int32_t>();
    DataCopyPad(sendCntGMTensor_, sendCntPerRankTensor_, sendPerRankParams);
    DataCopyPad(sendCntGMTensor_[cntPerRankSizeAlign512_ / sizeof(int32_t)], sendCntPerExpertTensor_,
                sendPerExpertParams);
    SetAtomicNone();
    DataCopy(scaleupCounterGMTensor_[aivId_ * counterAlign512_], numRecvPerRankTensor_, counterCnt_);
    if (serverNum_ > 0U) {
        DataCopy(scaleoutCounterGMTensor_[aivId_ * scaleoutCounterAlign512_], scaleoutCounterTensor,
                 scaleoutCounterCnt_);
    }
    PipeBarrier<PIPE_MTE3>(); // count 和 route counter 发布完成后，其他核才可读取
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::CalSendCnt()
{
    SplitToCore(axisBS_, aivNum_, startTokenId_, endTokenId_, sendTokenNum_);
    uint32_t groupCnt = Ceil(sendTokenNum_, perGroupTokenNum_);
    uint32_t calCnt = perGroupTokenNum_ * axisK_;
    LocalTensor<int32_t> scaleoutCounterTensor = InitSendCountBuffers();
    DataCopyPadExtParams<int32_t> topkIdsCntCopyPadParams{false, 0U, 0U, 0U};

    LocalTensor<int32_t> topkIdsGroupTensor = topkIdsBuf_.Get<int32_t>();
    LocalTensor<int16_t> tempTensorInt16 = tempBuf_.Get<int16_t>();

    ResetSendCountWorkspace();
    if (aivId_ == 0U) {                            // 状态位仅做1次累加
        uint64_t mask[2] = {0x101010101010101, 0}; // 一次性操作256字节，也是8个datablock，每8个数将首个设置为1
        Duplicate<int32_t>(sendCntPerRankTensor_, 1, mask, Ceil(epWorldSize_, DATA_BLOCK_NUM), 1, DATA_BLOCK_NUM);
    }

    for (uint32_t group = 0; group < groupCnt; group++) {
        if (group == groupCnt - 1) {
            calCnt = (sendTokenNum_ - group * perGroupTokenNum_) * axisK_;
        }
        if (group > 0) {
            SyncFunc<AscendC::HardEvent::V_MTE2>();
        }
        uint32_t topkIdxOffset = (startTokenId_ + group * perGroupTokenNum_) * axisK_;
        DataCopyExtParams topkIdsCntParams = {1U, static_cast<uint32_t>(calCnt * sizeof(int32_t)), 0U, 0U, 0U};
        DataCopyPad(topkIdsGroupTensor, topkIdxGMTensor_[topkIdxOffset], topkIdsCntParams,
                    topkIdsCntCopyPadParams); // copy topkId
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        Cast(tempTensorInt16, topkIdsGroupTensor, RoundMode::CAST_NONE, calCnt);

        // 每核只扫描自己的 token 分片，并在同一批 topk 上生成 count 和两级 route counter。
        CalSendCntPerExpert(tempTensorInt16, calCnt);
        CalSendCntPerRank(tempTensorInt16, calCnt);
        CalSendCntPerScaleout(dstExpBuf_.Get<int16_t>(), calCnt, scaleoutCounterTensor);
    }

    FlushSendCountToWorkspace(scaleoutCounterTensor);
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::SendCountToRemoteRank(
    uint32_t dstRankId, uint64_t notifyValue)
{
    // 计算目标窗口地址:
    GM_ADDR remoteStateAddr = MoeEpDispatchBase::GetWindowAddrByRankId(mc2Context_, dstRankId, cntWinStateOffset_);
    GM_ADDR notifyAddr = remoteStateAddr + epRankId_ * WIN_ADDR_ALIGN;
    GM_ADDR remoteCountAddr = remoteStateAddr + epWorldSize_ * WIN_ADDR_ALIGN + epRankId_ * moeNumPerRankAlign512_;
    GM_ADDR srcWorkspaceAddr =
        sendCntWorkspaceAddr_ + cntPerRankSizeAlign512_ + dstRankId * moeExpertNumPerRank_ * sizeof(int32_t);
    uint64_t commHandle = MoeEpDispatchBase::GetCommHandle(mc2Context_, dstRankId);

    // notifyValue 的低 32 位为 ready，高 32 位为该目标 rank 的 token count。
    hcomm_.WriteWithNotifyNbi(commHandle, remoteCountAddr, srcWorkspaceAddr,
                              static_cast<uint64_t>(moeExpertNumPerRank_ * sizeof(int32_t)), notifyAddr, notifyValue);
    hcomm_.Drain(commHandle);
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::CopyLocalCountToWindow(
    uint32_t dstRankId, uint64_t notifyVal)
{
    // 本端
    // 计算目标窗口地址:
    GM_ADDR remoteStateAddr = MoeEpDispatchBase::GetWindowAddrByRankId(mc2Context_, dstRankId, cntWinStateOffset_);
    GM_ADDR notifyAddr = remoteStateAddr + epRankId_ * WIN_ADDR_ALIGN;
    GM_ADDR remoteCountAddr = remoteStateAddr + epWorldSize_ * WIN_ADDR_ALIGN + epRankId_ * moeNumPerRankAlign512_;
    GlobalTensor<int32_t> countGMTensor;
    GlobalTensor<uint64_t> notifyGMTensor;
    countGMTensor.SetGlobalBuffer((__gm__ int32_t *)remoteCountAddr);
    notifyGMTensor.SetGlobalBuffer((__gm__ uint64_t *)notifyAddr);
    LocalTensor<int32_t> cntPerExpertTensor = tempBuf_.Get<int32_t>();
    DataCopyParams expertCntCopyParams = {1U, static_cast<uint16_t>(moeExpertNumPerRank_ * sizeof(int32_t)), 0U, 0U};
    uint32_t srcOffset = cntPerRankSizeAlign512_ / sizeof(int32_t) + dstRankId * moeExpertNumPerRank_;
    DataCopyPad(cntPerExpertTensor, sendCntGMTensor_[srcOffset], expertCntCopyParams, padParams_);
    SyncFunc<AscendC::HardEvent::MTE2_MTE3>();
    DataCopyPad(countGMTensor, cntPerExpertTensor, expertCntCopyParams);
    SyncFunc<AscendC::HardEvent::MTE3_S>(); // perExpert 写完再写notifyVal
    notifyGMTensor.SetValue(0, notifyVal);
    DataCacheCleanAndInvalid<uint64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(notifyGMTensor);
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline bool MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::IsCommTargetRank(uint32_t dstRankId)
{
    if (dstRankId >= serverStartRank_ && dstRankId < serverEndRank_) {
        return true;
    }
    return rankNumPerServer_ > 0U && dstRankId % rankNumPerServer_ == epRankId_ % rankNumPerServer_;
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::SendRemainingCountRange()
{
    GlobalTensor<uint64_t> numSendGMTensorInt64;
    numSendGMTensorInt64.SetGlobalBuffer((__gm__ uint64_t *)(sendCntWorkspaceAddr_ + startRankId_ * WIN_ADDR_ALIGN));
    LocalTensor<uint64_t> sendCntPerRankInt64 = dstExpBuf_.Get<uint64_t>();
    // GM 512B/rank 散开读, UB 每 rank 32B (UB_ALIGN), 只取前 8B (state+count)
    DataCopyParams cntCopyParams = {static_cast<uint16_t>(rankNumPerCore_), static_cast<uint16_t>(UB_ALIGN),
                                    static_cast<uint16_t>(WIN_ADDR_ALIGN - UB_ALIGN), 0U};
    DataCopyPad(sendCntPerRankInt64, numSendGMTensorInt64, cntCopyParams, padParams_);
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    for (uint32_t dstRankId = startRankId_; dstRankId < endRankId_; dstRankId++) {
        if (IsCommTargetRank(dstRankId)) {
            continue;
        }
        uint32_t notifyStride = (dstRankId - startRankId_) * INT64_UB_STRIDE;
        SendCountToRemoteRank(dstRankId, sendCntPerRankInt64.GetValue(notifyStride));
    }
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::SendCommTargetCounts(
    uint32_t remoteScaleoutStart, uint32_t remoteScaleoutEnd, uint32_t localRankStart, uint32_t localRankEnd)
{
    uint32_t currentScaleupIndex = rankNumPerServer_ == 0U ? epRankId_ : epRankId_ % rankNumPerServer_;
    for (uint32_t remoteScaleoutOrdinal = remoteScaleoutStart; remoteScaleoutOrdinal < remoteScaleoutEnd;
         remoteScaleoutOrdinal++) {
        uint32_t dstScaleoutIndex =
            remoteScaleoutOrdinal < currentServerIndex_ ? remoteScaleoutOrdinal : remoteScaleoutOrdinal + 1U;
        uint32_t proxyRankId = dstScaleoutIndex * rankNumPerServer_ + currentScaleupIndex;
        if (proxyRankId >= epWorldSize_) {
            continue;
        }
        __gm__ uint64_t *notifyValueAddr =
            (__gm__ uint64_t *)(sendCntWorkspaceAddr_ + static_cast<uint64_t>(proxyRankId) * WIN_ADDR_ALIGN);
        SendCountToRemoteRank(proxyRankId, static_cast<uint64_t>(ReadGmByPassDCache(notifyValueAddr)));
    }
    for (uint32_t localRankIndex = localRankStart; localRankIndex < localRankEnd; localRankIndex++) {
        uint32_t dstRankId = serverStartRank_ + localRankIndex;
        __gm__ uint64_t *notifyValueAddr =
            (__gm__ uint64_t *)(sendCntWorkspaceAddr_ + static_cast<uint64_t>(dstRankId) * WIN_ADDR_ALIGN);
        uint64_t notifyValue = static_cast<uint64_t>(ReadGmByPassDCache(notifyValueAddr));
        if (dstRankId != epRankId_) {
            SendCountToRemoteRank(dstRankId, notifyValue);
        } else {
            CopyLocalCountToWindow(dstRankId, notifyValue);
        }
    }
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::ExchangeCount()
{
    // 计数发送规则：
    // 1. 需要发数据的卡（本节点卡、跨节点代理卡）由负责发数据的核一并发送计数，
    //    保证每个通道每轮只由一个核使用；
    // 2. 其余卡按核数均匀分片发送。
    SplitToCore(epWorldSize_, aivNum_, startRankId_, endRankId_, rankNumPerCore_);
    bool hasRemainingCountTarget = false;
    for (uint32_t dstRankId = startRankId_; dstRankId < endRankId_; dstRankId++) {
        if (!IsCommTargetRank(dstRankId)) {
            hasRemainingCountTarget = true;
            break;
        }
    }
    uint32_t remoteScaleoutStart = 0U;
    uint32_t remoteScaleoutEnd = 0U;
    if (aivId_ < numScaleoutSendAiv_) {
        SplitRangeForCore(remoteServerCount_, numScaleoutSendAiv_, aivId_, remoteScaleoutStart, remoteScaleoutEnd);
    }
    uint32_t localRankStart = 0U;
    uint32_t localRankEnd = 0U;
    bool ownsLocalRank = InitOwnedScaleupRankRange(localRankStart, localRankEnd);
    if (aivNum_ == 1U && aivId_ == 0U) {
        localRankStart = 0U;
        localRankEnd = serverEndRank_ - serverStartRank_;
        ownsLocalRank = localRankStart < localRankEnd;
    }
    if (!hasRemainingCountTarget && remoteScaleoutStart >= remoteScaleoutEnd && !ownsLocalRank) {
        return;
    }

    // 通信初始化
    tpipe_->InitBuffer(hcommBuf_, HCOMM_INIT_SIZE);
    hcommTensor_ = hcommBuf_.Get<uint8_t>();
    hcomm_.Init(hcommTensor_, HCOMM_INIT_SIZE);

    if (hasRemainingCountTarget) {
        SendRemainingCountRange();
    }
    SendCommTargetCounts(remoteScaleoutStart, remoteScaleoutEnd, localRankStart, localRankEnd);
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::GetRecvCount()
{
    if (aivId_ != 0U) {
        return;
    }

    uint32_t serverRankCount = epWorldSize_;
    uint32_t mask = 1U;
    int32_t sumOfFlag = -1;
    int32_t compareFlag = static_cast<int32_t>(serverRankCount);
    LocalTensor<int32_t> recvCounterTensor = topkIdsBuf_.GetWithOffset<int32_t>(serverRankCount * UB_STRIDE, 0);
    LocalTensor<float> tempFp32 =
        topkIdsBuf_.GetWithOffset<float>(serverRankCount * UB_STRIDE, serverRankCount * UB_ALIGN);
    LocalTensor<float> recvCounterTensorFp32 = recvCounterTensor.template ReinterpretCast<float>();
    LocalTensor<float> numRecvPerRankTensorFp32 = numRecvPerRankTensor_.template ReinterpretCast<float>();
    SyncFunc<AscendC::HardEvent::MTE3_V>(); // 等待本核上轮计数器清0(buf 复用)
    while (sumOfFlag != compareFlag) {      // 状态位check
        DataCopy(recvCounterTensor, recvCounterGMTensor_, statusCopyParams_);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        ReduceSum(numRecvPerRankTensorFp32, recvCounterTensorFp32, tempFp32, mask, serverRankCount, 1);
        SyncFunc<AscendC::HardEvent::V_S>();
        sumOfFlag = numRecvPerRankTensor_.GetValue(0);
    }
    SetRecvNumPerExpert();                // 计算本卡上各专家接收的token总数
    SetRecvNumPerRank(recvCounterTensor); // 计算本端接收来自各卡的token总数
    // status clear
    Duplicate<int32_t>(recvCounterTensor, 0, serverRankCount * UB_STRIDE);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(recvCounterGMTensor_, recvCounterTensor, clearStatusCopyParams_);
    diagWriter_.RunPosRecord(MOE_EP_DISPATCH_RUN_POS_COUNT_READY);
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::SetRecvNumPerExpert()
{
    uint32_t recvExpertSizeAlign = moeNumPerRankAlign_;
    uint32_t recvExpertAlign = recvExpertSizeAlign / sizeof(int32_t);
    uint32_t serverRankCount = epWorldSize_;
    tpipe_->InitBuffer(recvCntBuf_, serverRankCount * recvExpertSizeAlign);
    tpipe_->InitBuffer(numRecvPerExpertBuf_, 2 * moeNumPerRankAlign_);
    tpipe_->InitBuffer(recvTempBuf_, UB_ALIGN);
    numRecvPerExpertTensor_ = numRecvPerExpertBuf_.Get<int64_t>();
    LocalTensor<int32_t> recvTensorInt32 = recvCntBuf_.Get<int32_t>();
    LocalTensor<int64_t> rankRecvTensorInt64 = dstExpBuf_.Get<int64_t>();
    LocalTensor<uint8_t> sharedTmpInt8 = recvTensorInt32.template ReinterpretCast<uint8_t>();

    const uint32_t recvExpertShape[] = {serverRankCount, recvExpertAlign};
    DataCopyParams inRecvCntParams = {static_cast<uint16_t>(serverRankCount),
                                      static_cast<uint16_t>(recvExpertSizeAlign),
                                      static_cast<uint16_t>(moeNumPerRankAlign512_ - recvExpertSizeAlign), 0U};
    DataCopyParams recvPerExpertParams = {1U, static_cast<uint16_t>(moeExpertNumPerRank_ * sizeof(int64_t)), 0U, 0U};
    uint32_t recvExpertOffset = epWorldSize_ * WIN_ADDR_ALIGN / sizeof(int32_t);
    DataCopyPad(recvTensorInt32, recvCounterGMTensor_[recvExpertOffset], inRecvCntParams, padParams_);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    uint32_t recvExpertElementCount = serverRankCount * recvExpertAlign;
    if (recvExpertElementCount * sizeof(int64_t) <= PER_GROUP_SIZE) {
        Cast(rankRecvTensorInt64, recvTensorInt32, RoundMode::CAST_NONE, recvExpertElementCount);
        ReduceSum<int64_t, AscendC::Pattern::Reduce::RA, true>(numRecvPerExpertTensor_, rankRecvTensorInt64,
                                                               sharedTmpInt8, recvExpertShape, true);
    } else {
        Duplicate<int64_t>(numRecvPerExpertTensor_, 0, recvExpertAlign);
        // 大world_size下逐rank Cast并累加，避免完整int64矩阵超过dstExpBuf_。
        for (uint32_t srcRankId = 0U; srcRankId < serverRankCount; srcRankId++) {
            Cast(rankRecvTensorInt64, recvTensorInt32[srcRankId * recvExpertAlign], RoundMode::CAST_NONE,
                 recvExpertAlign);
            Add(numRecvPerExpertTensor_, numRecvPerExpertTensor_, rankRecvTensorInt64, recvExpertAlign);
        }
    }

    if constexpr (DoCpuSync) { // 计算actualA，并写入host pin
        WriteHostRecvTokenCount(numRecvPerExpertTensor_);
    }

    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopyPad(numRecvPerExpertGMTensor_, numRecvPerExpertTensor_, recvPerExpertParams);
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::WriteHostRecvTokenCount(
    LocalTensor<int64_t> recvPerExpertTensor)
{
    LocalTensor<int64_t> recvCountTensor = recvTempBuf_.Get<int64_t>();
    LocalTensor<int64_t> reduceWorkspaceTensor = recvCntBuf_.Get<int64_t>();
    GlobalTensor<int64_t> hostPinnedCounterTensor;
    hostPinnedCounterTensor.SetGlobalBuffer((__gm__ int64_t *)hostPinnedCounterAddrGM_);
    ReduceSum(recvCountTensor, recvPerExpertTensor, reduceWorkspaceTensor, moeExpertNumPerRank_);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(hostPinnedCounterTensor, recvCountTensor, UB_ALIGN / sizeof(int64_t));
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::SetRecvNumPerRank(
    LocalTensor<int32_t> recvTmpTensor)
{
    LocalTensor<uint32_t> gatherIndexTensor = topkIdsBuf_.GetWithOffset<uint32_t>(UB_STRIDE, epWorldSize_ * UB_ALIGN);
    gatherIndexTensor.SetValue(0, 2U); // 源操作数每个datablock取下标为1的元素
    uint32_t compactMask = 2U;
    uint64_t reservedCount = 0UL;
    GatherMaskParams recvMaskParams = {1, static_cast<uint16_t>(epWorldSize_), 1, 0};
    DataCopyParams recvPerRankParams = {1U, static_cast<uint16_t>(epWorldSize_ * sizeof(int32_t)), 0U, 0U};
    SyncFunc<AscendC::HardEvent::S_V>();
    GatherMask(numRecvPerRankTensor_, recvTmpTensor, gatherIndexTensor, true, compactMask, recvMaskParams,
               reservedCount);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopyPad(numRecvPerRankGMTensor_, numRecvPerRankTensor_, recvPerRankParams);
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::ReduceCounterRange(
    GlobalTensor<int32_t> counterGMTensor, LocalTensor<int32_t> dstTensor, uint32_t counterAlign512,
    uint32_t counterCnt, uint32_t counterValueCount, uint32_t counterCoreCount)
{
    LocalTensor<int32_t> counterTmpTensor = topkIdsBuf_.Get<int32_t>();
    uint32_t counterBytes = counterCnt * sizeof(int32_t);
    uint32_t counterAlignBytes = counterAlign512 * sizeof(int32_t);
    uint32_t copyNumPerGroup = perGroupSizeAlign_ * 2 / counterBytes;
    if (copyNumPerGroup == 0U) {
        copyNumPerGroup = 1U;
    }
    uint32_t groupCnt = Ceil(counterCoreCount, copyNumPerGroup);
    for (uint32_t groupIndex = 0U; groupIndex < groupCnt; groupIndex++) {
        uint32_t copyNum =
            (groupIndex == groupCnt - 1U) ? (counterCoreCount - copyNumPerGroup * groupIndex) : copyNumPerGroup;
        uint32_t gmOffset = groupIndex * copyNumPerGroup * counterAlign512;
        DataCopyParams counterCopyParams = {static_cast<uint16_t>(copyNum), static_cast<uint16_t>(counterBytes),
                                            static_cast<uint16_t>(counterAlignBytes - counterBytes), 0U};
        DataCopyPad(counterTmpTensor, counterGMTensor[gmOffset], counterCopyParams, padParams_);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        const uint32_t counterShape[] = {copyNum, counterCnt};
        ReduceSum<int32_t, AscendC::Pattern::Reduce::RA, true>(counterSumTensor_, counterTmpTensor, counterShape,
                                                               false);
        Add(dstTensor, dstTensor, counterSumTensor_, counterValueCount);
        if (groupIndex + 1U < groupCnt) {
            SyncFunc<AscendC::HardEvent::V_MTE2>();
        }
    }
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::GetSlotStartNum(
    uint32_t tokenRangeIndex)
{
    // 预聚合当前token范围之前的counter，保持接管范围后的slot编号与count行一致。
    Duplicate<int32_t>(slotIdxPerRankTensor_, 0, epWorldSize_);
    SyncFunc<AscendC::HardEvent::MTE3_MTE2>(); // 各核 counter 写入完成后再由 MTE2 读取前缀
    ReduceCounterRange(scaleupCounterGMTensor_, slotIdxPerRankTensor_, counterAlign512_, counterCnt_, epWorldSize_,
                       tokenRangeIndex);

    // scaleout slot 起始偏移预聚合后存在 slotIdxPerRankTensor_ 后半段
    LocalTensor<int32_t> scaleoutSlotIdxTensor =
        numRecvPerRankBuf_.GetWithOffset<int32_t>(serverNumAlign_, epWorldSizeAlign_);
    if (serverNum_ > 0U) {
        if (tokenRangeIndex > 0U) {
            SyncFunc<AscendC::HardEvent::V_MTE2>();
        }
        Duplicate<int32_t>(scaleoutSlotIdxTensor, 0, serverNumAlign_);
        ReduceCounterRange(scaleoutCounterGMTensor_, scaleoutSlotIdxTensor, scaleoutCounterAlign512_,
                           scaleoutCounterCnt_, serverNum_, tokenRangeIndex);
    }
    Adds(scaleupSendEntryStartTensor_, slotIdxPerRankTensor_, 0, epWorldSize_);
    Adds(scaleoutSendEntryStartTensor_, scaleoutSlotIdxTensor, 0, serverNum_);
    SyncFunc<AscendC::HardEvent::V_S>();
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline uint32_t MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::FindOrCreateRemoteRouteEntry(
    uint32_t tokenId, uint32_t dstScaleoutIndex, uint32_t &routeEntryCount, bool &isNewRouteEntry)
{
    // 直接索引 O(1) 查找：sendRouteIndexByScaleoutTensor_[dstScaleoutIndex] 存已建表项下标
    int32_t existingIdx = sendRouteIndexByScaleoutTensor_.GetValue(dstScaleoutIndex);
    if (existingIdx >= 0) {
        isNewRouteEntry = false;
        return static_cast<uint32_t>(existingIdx);
    }

    isNewRouteEntry = true;
    uint32_t routeEntryIndex = routeEntryCount;
    ASCENDC_ASSERT(routeEntryIndex < axisK_, {
        KERNEL_LOG(KERNEL_ERROR, "remote route entry count (%u) exceeds axisK (%u), token %u dstScaleoutIndex %u",
                   routeEntryIndex, axisK_, tokenId, dstScaleoutIndex);
    });
    routeEntryCount++;

    routeScaleoutIndexTensor_.SetValue(routeEntryIndex, static_cast<int32_t>(dstScaleoutIndex));
    sendRouteIndexByScaleoutTensor_.SetValue(dstScaleoutIndex, static_cast<int32_t>(routeEntryIndex));
    return routeEntryIndex;
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::WriteRouteTableToGM(
    uint32_t tokenId, uint32_t routeEntryCount)
{
    // token 处理结束后批量写回 GM，避免循环内逐元素 GM scalar 写
    if (tokenId > startTokenId_) {
        SyncFunc<AscendC::HardEvent::MTE3_S>(); // 上一 token 的 route count 搬出后再复用固定 UB
    }
    routeCountTensor_.SetValue(0, static_cast<int32_t>(routeEntryCount));
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopyPad(routeCountGMTensor_[tokenId], routeCountTensor_, routeCountCopyParams_);
    DataCopyPad(routeDstScaleoutGMTensor_[tokenId * axisK_], routeScaleoutIndexTensor_, routeCopyParams_);
    DataCopyPad(routeScaleoutSlotGMTensor_[tokenId * axisK_], routeScaleoutSlotTensor_, routeCopyParams_);
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline uint32_t MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::CopyCachedRouteForToken(
    uint32_t tokenId)
{
    // cache 模式也返回本次输出，保持和 dst slot cache 路径一致的 handle 契约。
    if (tokenId > startTokenId_) {
        SyncFunc<AscendC::HardEvent::MTE3_MTE2>(); // 上一轮 route 输出读 UB，本轮 cache route 重新搬入 UB
        SyncFunc<AscendC::HardEvent::S_MTE2>();    // 上一轮 scaleout workspace 写入前读取 route index/slot
    }
    DataCopyPad(routeCountTensor_, cachedRouteCountGMTensor_[tokenId], routeCountCopyParams_, routeCopyPadParams_);
    DataCopyPad(routeScaleoutIndexTensor_, cachedRouteDstScaleoutGMTensor_[tokenId * axisK_], routeCopyParams_,
                routeCopyPadParams_);
    DataCopyPad(routeScaleoutSlotTensor_, cachedRouteScaleoutSlotGMTensor_[tokenId * axisK_], routeCopyParams_,
                routeCopyPadParams_);
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    uint32_t routeEntryCount = static_cast<uint32_t>(routeCountTensor_.GetValue(0));
    SyncFunc<AscendC::HardEvent::MTE2_MTE3>();
    WriteRouteTableToGM(tokenId, routeEntryCount);
    return routeEntryCount;
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::AllocateScaleupSlotForExpert(
    uint32_t tokenId, uint32_t topkIndex, uint32_t dstRankId, uint32_t &scaleupSlot)
{
    int32_t rankSlot = sendRankSlotTensor_.GetValue(dstRankId);
    if (rankSlot >= 0) {
        // 同一 rank 已分配过 slot，直接复用
        scaleupSlot = static_cast<uint32_t>(rankSlot);
        dstSlotIdxTensor_.SetValue(topkIndex, rankSlot);
        return;
    }

    // 本地递增分配 slot，无 AtomicAdd
    scaleupSlot = static_cast<uint32_t>(slotIdxPerRankTensor_.GetValue(dstRankId));
    dstSlotIdxTensor_.SetValue(topkIndex, static_cast<int32_t>(scaleupSlot));
    sendRankSlotTensor_.SetValue(dstRankId, static_cast<int32_t>(scaleupSlot));
    slotIdxPerRankTensor_.SetValue(dstRankId, static_cast<int32_t>(scaleupSlot + 1U));
    if (dstRankId >= serverStartRank_ && dstRankId < serverEndRank_) {
        WriteScaleupSendEntry(dstRankId, tokenId, scaleupSlot);
    }
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::UpdateScaleoutRouteForExpert(
    uint32_t tokenId, uint32_t dstRankId, uint32_t &routeEntryCount)
{
    uint32_t dstScaleoutIndex = rankNumPerServer_ == 0U ? 0U : dstRankId / rankNumPerServer_;
    bool isNewRouteEntry = false;
    uint32_t routeEntryIndex =
        FindOrCreateRemoteRouteEntry(tokenId, dstScaleoutIndex, routeEntryCount, isNewRouteEntry);
    // 新建表项时分配 scaleout slot（本地递增）
    if (isNewRouteEntry) {
        uint32_t scaleoutSlotOffset = epWorldSizeAlign_ / sizeof(int32_t) + dstScaleoutIndex;
        uint32_t scaleoutSlot = static_cast<uint32_t>(slotIdxPerRankTensor_.GetValue(scaleoutSlotOffset));
        routeScaleoutSlotTensor_.SetValue(routeEntryIndex, static_cast<int32_t>(scaleoutSlot));
        slotIdxPerRankTensor_.SetValue(scaleoutSlotOffset, static_cast<int32_t>(scaleoutSlot + 1U));
    }
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline uint32_t MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::DedupAndBuildSendEntries(
    uint32_t tokenId)
{
    uint32_t routeEntryCount = 0U;
    for (uint32_t topkIndex = 0; topkIndex < axisK_; topkIndex++) {
        int32_t expertId = topkIdxTensor_.GetValue(topkIndex);
        if (expertId < 0) {
            dstSlotIdxTensor_.SetValue(topkIndex, -1);
            continue;
        }

        uint32_t dstRankId = static_cast<uint32_t>(expertId) / moeExpertNumPerRank_;
        uint32_t scaleupSlot = 0U;
        AllocateScaleupSlotForExpert(tokenId, topkIndex, dstRankId, scaleupSlot);

        if (dstRankId >= serverStartRank_ && dstRankId < serverEndRank_) {
            continue;
        }

        UpdateScaleoutRouteForExpert(tokenId, dstRankId, routeEntryCount);
    }
    // 批量写回 route 表到 GM
    WriteRouteTableToGM(tokenId, routeEntryCount);
    return routeEntryCount;
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::WritePayloadStash(uint32_t tokenId)
{
    GM_ADDR payloadStashAddr = payloadStashWinAddr_ + static_cast<uint64_t>(tokenId) * scaleoutSlotBytes_;
    GlobalTensor<XType> payloadStashTensor;
    payloadStashTensor.SetGlobalBuffer((__gm__ XType *)payloadStashAddr);
    DataCopyPad(payloadStashTensor, tokenSlotTensor_, fanoutPayloadCopyParams_);

    GlobalTensor<int32_t> destinationSlotStashTensor;
    destinationSlotStashTensor.SetGlobalBuffer((__gm__ int32_t *)(payloadStashAddr + perSlotBytes_));
    DataCopyPad(destinationSlotStashTensor, dstSlotIdxTensor_, routeCopyParams_);
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::WriteSendEntry(
    GM_ADDR sendEntryBaseAddr, uint32_t sendEntryIndex, uint32_t sourceSlotIndex, uint32_t destinationSlotIndex)
{
    LocalTensor<uint32_t> sendEntryTensor = sendEntryWriteBuf_.Get<uint32_t>();
    if (sendEntryWritePending_) {
        SyncFunc<AscendC::HardEvent::MTE3_S>();
    }
    sendEntryTensor.SetValue(0, sourceSlotIndex);
    sendEntryTensor.SetValue(1, destinationSlotIndex);
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    GlobalTensor<uint32_t> sendEntryGMTensor;
    sendEntryGMTensor.SetGlobalBuffer(
        (__gm__ uint32_t *)(sendEntryBaseAddr + static_cast<uint64_t>(sendEntryIndex) * MOE_EP_SEND_ENTRY_BYTES));
    DataCopyPad(sendEntryGMTensor, sendEntryTensor, sendEntryCopyParams_);
    sendEntryWritePending_ = true;
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::ReadSendEntry(
    GM_ADDR sendEntryAddr, uint32_t &sourceSlotIndex, uint32_t &destinationSlotIndex)
{
    GlobalTensor<uint32_t> sendEntryGMTensor;
    sendEntryGMTensor.SetGlobalBuffer((__gm__ uint32_t *)sendEntryAddr);
    LocalTensor<uint32_t> sendEntryTensor = sendEntryReadBuf_.Get<uint32_t>();
    SyncFunc<AscendC::HardEvent::S_MTE2>();
    DataCopyPad(sendEntryTensor, sendEntryGMTensor, sendEntryCopyParams_, sendEntryCopyPadParams_);
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    sourceSlotIndex = sendEntryTensor.GetValue(0);
    destinationSlotIndex = sendEntryTensor.GetValue(1);
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::WriteScaleupSendEntry(
    uint32_t dstRankId, uint32_t tokenId, uint32_t scaleupSlot)
{
    uint32_t localRankIndex = dstRankId - serverStartRank_;
    uint32_t localSendEntryIndex =
        scaleupSlot - static_cast<uint32_t>(scaleupSendEntryStartTensor_.GetValue(dstRankId));
    GM_ADDR sendEntryBaseAddr =
        scaleupSendEntryAddr_ +
        (static_cast<uint64_t>(localRankIndex) * aivNum_ + currentTokenRangeIndex_) * sendEntryTokenRangeBytes_;
    WriteSendEntry(sendEntryBaseAddr, localSendEntryIndex, tokenId, scaleupSlot);
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::WriteScaleoutSendEntriesFromRoute(
    uint32_t tokenId, uint32_t routeEntryCount)
{
    for (uint32_t routeEntryIndex = 0U; routeEntryIndex < routeEntryCount; routeEntryIndex++) {
        uint32_t dstScaleoutIndex = static_cast<uint32_t>(routeScaleoutIndexTensor_.GetValue(routeEntryIndex));
        uint32_t scaleoutSlot = static_cast<uint32_t>(routeScaleoutSlotTensor_.GetValue(routeEntryIndex));
        uint32_t remoteServerOrdinal =
            dstScaleoutIndex < currentServerIndex_ ? dstScaleoutIndex : dstScaleoutIndex - 1U;
        uint32_t localSendEntryIndex =
            scaleoutSlot - static_cast<uint32_t>(scaleoutSendEntryStartTensor_.GetValue(dstScaleoutIndex));
        GM_ADDR sendEntryBaseAddr =
            scaleoutSendEntryAddr_ + (static_cast<uint64_t>(remoteServerOrdinal) * aivNum_ + currentTokenRangeIndex_) *
                                         sendEntryTokenRangeBytes_;
        WriteSendEntry(sendEntryBaseAddr, localSendEntryIndex, tokenId, scaleoutSlot);
    }
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::SplitRangeForCore(
    uint32_t itemCount, uint32_t coreCount, uint32_t coreIndex, uint32_t &itemStart, uint32_t &itemEnd)
{
    uint32_t itemCountPerCore = itemCount / coreCount;
    uint32_t remainderItemCount = itemCount % coreCount;
    itemStart = itemCountPerCore * coreIndex;
    if (coreIndex < remainderItemCount) {
        itemStart += coreIndex;
        itemEnd = itemStart + itemCountPerCore + 1U;
    } else {
        itemStart += remainderItemCount;
        itemEnd = itemStart + itemCountPerCore;
    }
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline bool MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::InitOwnedScaleupRankRange(
    uint32_t &destinationScaleupStart, uint32_t &destinationScaleupEnd)
{
    if (numNodeLocalAiv_ == 0U || aivId_ < numScaleoutSendAiv_ || aivId_ >= numScaleoutSendAiv_ + numNodeLocalAiv_) {
        return false;
    }
    uint32_t currentServerRankCount = serverEndRank_ - serverStartRank_;
    SplitRangeForCore(currentServerRankCount, numNodeLocalAiv_, aivId_ - numScaleoutSendAiv_, destinationScaleupStart,
                      destinationScaleupEnd);
    return destinationScaleupStart < destinationScaleupEnd;
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::ReduceScaleoutCounterGroup(
    uint32_t counterGroup, LocalTensor<int32_t> counterSumTensor)
{
    // 每次只归约 owner 需要的 8 个相邻 server counter，避免每个 owner 重复扫描全部 server 列。
    LocalTensor<int32_t> counterRowsTensor = topkIdsBuf_.Get<int32_t>();
    DataCopyParams counterCopyParams = {static_cast<uint16_t>(aivNum_), 1U,
                                        static_cast<uint16_t>(scaleoutCounterAlign512_ / UB_STRIDE - 1U), 0U};
    const uint32_t counterShape[] = {aivNum_, UB_STRIDE};
    uint32_t counterOffset = counterGroup * UB_STRIDE;
    DataCopy(counterRowsTensor, scaleoutCounterGMTensor_[counterOffset], counterCopyParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    ReduceSum<int32_t, AscendC::Pattern::Reduce::RA, true>(counterSumTensor, counterRowsTensor, counterShape, false);
    SyncFunc<AscendC::HardEvent::V_S>();
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::SendScaleoutPayloadRange(
    uint32_t tokenRangeIndex, uint32_t remoteServerOrdinal, uint32_t dstScaleoutIndex, uint32_t scaleoutSlotCount,
    uint64_t commHandle, GM_ADDR remoteScaleoutBase, GM_ADDR remoteStatusBase)
{
    uint64_t remoteServerDataOffset = static_cast<uint64_t>(currentServerIndex_) * axisMaxBS_ * scaleoutSlotBytes_;
    uint64_t remoteServerStatusOffset = static_cast<uint64_t>(currentServerIndex_) * axisMaxBS_ * WIN_ADDR_ALIGN;
    __gm__ int32_t *rangeCountAddr =
        (__gm__ int32_t *)(scaleoutCounterAddr_ +
                           (static_cast<uint64_t>(tokenRangeIndex) * scaleoutCounterAlign512_ + dstScaleoutIndex) *
                               sizeof(int32_t));
    uint32_t rangeSendEntryCount = static_cast<uint32_t>(ReadGmByPassDCache(rangeCountAddr));
    GM_ADDR sendEntryBaseAddr =
        scaleoutSendEntryAddr_ +
        (static_cast<uint64_t>(remoteServerOrdinal) * aivNum_ + tokenRangeIndex) * sendEntryTokenRangeBytes_;
    for (uint32_t sendEntryIndex = 0U; sendEntryIndex < rangeSendEntryCount; sendEntryIndex++) {
        uint32_t sourceTokenId = 0U;
        uint32_t destinationScaleoutSlot = 0U;
        ReadSendEntry(sendEntryBaseAddr + static_cast<uint64_t>(sendEntryIndex) * MOE_EP_SEND_ENTRY_BYTES,
                      sourceTokenId, destinationScaleoutSlot);
        GM_ADDR localScaleoutAddr = payloadStashWinAddr_ + static_cast<uint64_t>(sourceTokenId) * scaleoutSlotBytes_;
        GM_ADDR remoteScaleoutAddr = remoteScaleoutBase + remoteServerDataOffset +
                                     static_cast<uint64_t>(destinationScaleoutSlot) * scaleoutSlotBytes_;
        GM_ADDR remoteNotifyAddr = remoteStatusBase + remoteServerStatusOffset +
                                   static_cast<uint64_t>(destinationScaleoutSlot) * WIN_ADDR_ALIGN;
        uint32_t notifyValue = destinationScaleoutSlot == 0U ? scaleoutSlotCount + 1U : SCALEOUT_SLOT_READY;
        hcomm_.WriteWithNotifyNbi(commHandle, remoteScaleoutAddr, localScaleoutAddr, scaleoutSlotBytes_,
                                  remoteNotifyAddr, notifyValue);
    }
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::SendScaleoutPayloadToProxy(
    uint32_t remoteServerOrdinal, uint32_t dstScaleoutIndex, uint32_t proxyRankId, uint32_t scaleoutSlotCount)
{
    uint64_t commHandle = MoeEpDispatchBase::GetCommHandle(mc2Context_, proxyRankId);
    GM_ADDR remoteScaleoutBase =
        MoeEpDispatchBase::GetWindowAddrByRankId(mc2Context_, proxyRankId, scaleoutRecvDataOffset_);
    GM_ADDR remoteStatusBase =
        MoeEpDispatchBase::GetWindowAddrByRankId(mc2Context_, proxyRankId, scaleoutRecvStatusOffset_);
    if (scaleoutSlotCount == 0U) {
        uint64_t remoteServerDataOffset = static_cast<uint64_t>(currentServerIndex_) * axisMaxBS_ * scaleoutSlotBytes_;
        uint64_t remoteServerStatusOffset = static_cast<uint64_t>(currentServerIndex_) * axisMaxBS_ * WIN_ADDR_ALIGN;
        GM_ADDR emptyPayloadSourceAddr = sendCntWorkspaceAddr_ + proxyRankId * WIN_ADDR_ALIGN;
        hcomm_.WriteWithNotifyNbi(commHandle, remoteScaleoutBase + remoteServerDataOffset, emptyPayloadSourceAddr,
                                  sizeof(uint64_t), remoteStatusBase + remoteServerStatusOffset, 1U);
        hcomm_.Drain(commHandle);
        return;
    }
    for (uint32_t tokenRangeIndex = 0U; tokenRangeIndex < aivNum_; tokenRangeIndex++) {
        SendScaleoutPayloadRange(tokenRangeIndex, remoteServerOrdinal, dstScaleoutIndex, scaleoutSlotCount, commHandle,
                                 remoteScaleoutBase, remoteStatusBase);
    }
    hcomm_.Drain(commHandle);
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::SendScaleoutPayloadsToProxy()
{
    if (remoteServerCount_ == 0U || numScaleoutSendAiv_ == 0U) {
        return;
    }
    LocalTensor<int32_t> scaleoutCounterSumTensor = counterSumTensor_;
    uint32_t counterGroupCount = Ceil(serverNum_, UB_STRIDE);
    uint32_t currentCounterGroup = counterGroupCount;
    uint32_t remoteScaleoutStart = 0U;
    uint32_t remoteScaleoutEnd = 0U;
    SplitRangeForCore(serverNum_ - 1U, numScaleoutSendAiv_, aivId_, remoteScaleoutStart, remoteScaleoutEnd);
    uint32_t currentScaleupIndex = rankNumPerServer_ == 0U ? epRankId_ : epRankId_ % rankNumPerServer_;
    for (uint32_t remoteScaleoutOrdinal = remoteScaleoutStart; remoteScaleoutOrdinal < remoteScaleoutEnd;
         remoteScaleoutOrdinal++) {
        uint32_t dstScaleoutIndex =
            remoteScaleoutOrdinal < currentServerIndex_ ? remoteScaleoutOrdinal : remoteScaleoutOrdinal + 1U;
        uint32_t proxyRankId = dstScaleoutIndex * rankNumPerServer_ + currentScaleupIndex;
        if (proxyRankId >= epWorldSize_) {
            continue;
        }

        uint32_t counterGroup = dstScaleoutIndex / UB_STRIDE;
        if (counterGroup != currentCounterGroup) {
            if (currentCounterGroup < counterGroupCount) {
                SyncFunc<AscendC::HardEvent::S_V>();
            }
            ReduceScaleoutCounterGroup(counterGroup, scaleoutCounterSumTensor);
            currentCounterGroup = counterGroup;
        }
        uint32_t scaleoutSlotCount =
            static_cast<uint32_t>(scaleoutCounterSumTensor.GetValue(dstScaleoutIndex % UB_STRIDE));
        SendScaleoutPayloadToProxy(remoteScaleoutOrdinal, dstScaleoutIndex, proxyRankId, scaleoutSlotCount);
    }
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::PrepareFanoutDestinations(
    GM_ADDR scaleoutSlotAddr, uint32_t destinationScaleupStart, uint32_t destinationScaleupEnd,
    bool waitForPreviousRouteRead)
{
    LocalTensor<int32_t> expertIdsTensor = routeInfoBuf_.Get<int32_t>();
    LocalTensor<int32_t> destinationSlotTensor = routeInfoBuf_.GetWithOffset<int32_t>(axisKAlign_, kAlignSize_);
    GlobalTensor<int32_t> topkGMTensor;
    GlobalTensor<int32_t> destinationSlotGMTensor;
    topkGMTensor.SetGlobalBuffer((__gm__ int32_t *)(scaleoutSlotAddr + metaOffset_));
    destinationSlotGMTensor.SetGlobalBuffer((__gm__ int32_t *)(scaleoutSlotAddr + perSlotBytes_));

    if (waitForPreviousRouteRead) {
        SyncFunc<AscendC::HardEvent::S_V>();
        SyncFunc<AscendC::HardEvent::S_MTE2>();
    }
    Duplicate<int32_t>(sendRankSlotTensor_, -1, rankNumPerServer_);
    DataCopyPad(expertIdsTensor, topkGMTensor, routeCopyParams_, routeCopyPadParams_);
    DataCopyPad(destinationSlotTensor, destinationSlotGMTensor, routeCopyParams_, routeCopyPadParams_);
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    SyncFunc<AscendC::HardEvent::V_S>();

    for (uint32_t topkIndex = 0U; topkIndex < axisK_; topkIndex++) {
        int32_t expertId = expertIdsTensor.GetValue(topkIndex);
        int32_t destinationSlot = destinationSlotTensor.GetValue(topkIndex);
        if (expertId < 0 || destinationSlot < 0) {
            continue;
        }
        uint32_t dstRankId = static_cast<uint32_t>(expertId) / moeExpertNumPerRank_;
        if (dstRankId < serverStartRank_ || dstRankId >= serverEndRank_) {
            continue;
        }
        uint32_t localRankIndex = dstRankId - serverStartRank_;
        if (localRankIndex < destinationScaleupStart || localRankIndex >= destinationScaleupEnd ||
            sendRankSlotTensor_.GetValue(localRankIndex) >= 0) {
            continue;
        }
        sendRankSlotTensor_.SetValue(localRankIndex, destinationSlot);
    }
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::SendFanoutPayloadToRank(
    GM_ADDR sourcePayloadAddr, uint32_t srcRankId, uint32_t dstRankId, uint32_t destinationSlot)
{
    GM_ADDR destinationPayloadAddr = MoeEpDispatchBase::GetWindowAddrByRankId(mc2Context_, dstRankId, winDataOffset_) +
                                     (static_cast<uint64_t>(srcRankId) * axisMaxBS_ + destinationSlot) * perSlotBytes_;
    if (dstRankId == epRankId_) {
        CopyPayloadToLocal(sourcePayloadAddr, destinationPayloadAddr);
        return;
    }

    uint64_t commHandle = MoeEpDispatchBase::GetCommHandle(mc2Context_, dstRankId);
    GM_ADDR unusedNotifyAddr = MoeEpDispatchBase::GetWindowAddrByRankId(mc2Context_, dstRankId, slotWinStateOffset_) +
                               static_cast<uint64_t>(srcRankId) * dispatchNotifyCount_ * WIN_ADDR_ALIGN + UB_ALIGN;
    hcomm_.WriteWithNotifyNbi(commHandle, destinationPayloadAddr, sourcePayloadAddr, perSlotBytes_, unusedNotifyAddr,
                              1U);
    hcomm_.Drain(commHandle);
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::SendFanoutPayloadForSlot(
    GM_ADDR scaleoutSlotAddr, uint32_t srcRankId, uint32_t destinationScaleupStart, uint32_t destinationScaleupEnd)
{
    for (uint32_t localRankIndex = destinationScaleupStart; localRankIndex < destinationScaleupEnd; localRankIndex++) {
        int32_t destinationSlot = sendRankSlotTensor_.GetValue(localRankIndex);
        if (destinationSlot < 0) {
            continue;
        }
        SendFanoutPayloadToRank(scaleoutSlotAddr, srcRankId, serverStartRank_ + localRankIndex,
                                static_cast<uint32_t>(destinationSlot));
    }
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline bool MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::TryGetScaleoutSlotCount(
    uint32_t srcScaleoutIndex, uint32_t &count)
{
    GM_ADDR statusAddr = MoeEpDispatchBase::GetWindowAddrByRankId(mc2Context_, epRankId_, scaleoutRecvStatusOffset_) +
                         static_cast<uint64_t>(srcScaleoutIndex) * axisMaxBS_ * WIN_ADDR_ALIGN;
    uint32_t statusValue = static_cast<uint32_t>(ReadGmByPassDCache((__gm__ int32_t *)statusAddr));
    if (statusValue == 0U) {
        return false;
    }
    count = statusValue - 1U;
    return true;
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline bool MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::TryScaleoutSlotReady(
    uint32_t srcScaleoutIndex, uint32_t scaleoutSlot)
{
    GM_ADDR statusAddr = MoeEpDispatchBase::GetWindowAddrByRankId(mc2Context_, epRankId_, scaleoutRecvStatusOffset_) +
                         (static_cast<uint64_t>(srcScaleoutIndex) * axisMaxBS_ + scaleoutSlot) * WIN_ADDR_ALIGN;
    return static_cast<uint32_t>(ReadGmByPassDCache((__gm__ int32_t *)statusAddr)) == SCALEOUT_SLOT_READY;
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::GetSourceState(uint32_t srcOrdinal,
                                                                                                int32_t &nextSlot,
                                                                                                int32_t &slotsLeft)
{
    int64_t state = sourceStateTensor_.GetValue(srcOrdinal);
    nextSlot = static_cast<int32_t>(state >> 32);
    slotsLeft = static_cast<int32_t>(state & 0xFFFFFFFFLL);
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::SetSourceState(uint32_t srcOrdinal,
                                                                                                int32_t nextSlot,
                                                                                                int32_t slotsLeft)
{
    int64_t state = (static_cast<int64_t>(nextSlot) << 32) | static_cast<uint32_t>(slotsLeft);
    sourceStateTensor_.SetValue(srcOrdinal, state);
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::ProcessOneScaleoutSlot(
    uint32_t srcScaleoutIndex, uint32_t scaleoutSlot, uint32_t destinationScaleupStart, uint32_t destinationScaleupEnd,
    bool &hasReadRouteInfo)
{
    GM_ADDR localScaleoutDataBase =
        MoeEpDispatchBase::GetWindowAddrByRankId(mc2Context_, epRankId_, scaleoutRecvDataOffset_);
    GM_ADDR scaleoutSlotAddr =
        localScaleoutDataBase +
        (static_cast<uint64_t>(srcScaleoutIndex) * axisMaxBS_ + scaleoutSlot) * scaleoutSlotBytes_;
    PrepareFanoutDestinations(scaleoutSlotAddr, destinationScaleupStart, destinationScaleupEnd, hasReadRouteInfo);
    uint32_t currentScaleupIndex = rankNumPerServer_ == 0U ? epRankId_ : epRankId_ % rankNumPerServer_;
    uint32_t srcRankId = srcScaleoutIndex * rankNumPerServer_ + currentScaleupIndex;
    SendFanoutPayloadForSlot(scaleoutSlotAddr, srcRankId, destinationScaleupStart, destinationScaleupEnd);
    hasReadRouteInfo = true;
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::SendFanoutPayloads(
    uint32_t destinationScaleupStart, uint32_t destinationScaleupEnd)
{
    bool hasReadRouteInfo = false;
    Duplicate<int64_t>(sourceStateTensor_, SOURCE_STATE_UNREAD, remoteServerCount_);
    SyncFunc<AscendC::HardEvent::V_S>();
    uint32_t completedSourceCount = 0U;
    while (completedSourceCount != remoteServerCount_) {
        bool madeProgress = false;
        for (uint32_t sourceServerOrdinal = 0U; sourceServerOrdinal < remoteServerCount_; sourceServerOrdinal++) {
            int32_t nextSlot = 0;
            int32_t slotsLeft = 0;
            GetSourceState(sourceServerOrdinal, nextSlot, slotsLeft);
            if (slotsLeft == 0) {
                continue;
            }
            uint32_t srcScaleoutIndex =
                sourceServerOrdinal < currentServerIndex_ ? sourceServerOrdinal : sourceServerOrdinal + 1U;
            if (slotsLeft == SOURCE_COUNT_UNREAD) {
                uint32_t count = 0U;
                if (!TryGetScaleoutSlotCount(srcScaleoutIndex, count)) {
                    continue;
                }
                slotsLeft = static_cast<int32_t>(count);
                nextSlot = 0;
                if (slotsLeft == 0) {
                    SetSourceState(sourceServerOrdinal, nextSlot, slotsLeft);
                    completedSourceCount++;
                    madeProgress = true;
                    continue;
                }
                madeProgress = true;
            }
            if (nextSlot > 0 && !TryScaleoutSlotReady(srcScaleoutIndex, static_cast<uint32_t>(nextSlot))) {
                continue;
            }
            ProcessOneScaleoutSlot(srcScaleoutIndex, static_cast<uint32_t>(nextSlot), destinationScaleupStart,
                                   destinationScaleupEnd, hasReadRouteInfo);
            nextSlot++;
            slotsLeft--;
            SetSourceState(sourceServerOrdinal, nextSlot, slotsLeft);
            madeProgress = true;
            if (slotsLeft == 0) {
                completedSourceCount++;
            }
        }
        if (!madeProgress) {
            int64_t backoffStartCycle = GetSystemCycle();
            while (GetSystemCycle() - backoffStartCycle < STATUS_POLL_BACKOFF_CYCLES) {
            }
        }
    }
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::CopyPayloadToLocal(
    GM_ADDR sourcePayloadAddr, GM_ADDR destinationPayloadAddr)
{
    GlobalTensor<XType> sourcePayloadTensor;
    GlobalTensor<XType> destinationPayloadTensor;
    sourcePayloadTensor.SetGlobalBuffer((__gm__ XType *)sourcePayloadAddr);
    destinationPayloadTensor.SetGlobalBuffer((__gm__ XType *)destinationPayloadAddr);
    LocalTensor<XType> payloadTensor = perSlotQueue_.AllocTensor<XType>();
    DataCopyPad(payloadTensor, sourcePayloadTensor, fanoutPayloadCopyParams_, padParams_);
    perSlotQueue_.EnQue(payloadTensor);
    LocalTensor<XType> payloadOutTensor = perSlotQueue_.DeQue<XType>();
    DataCopyPad(destinationPayloadTensor, payloadOutTensor, fanoutPayloadCopyParams_);
    perSlotQueue_.FreeTensor<XType>(payloadOutTensor);
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::SendEntries(
    GM_ADDR sendEntryBaseAddr, uint32_t sendEntryCount, GM_ADDR sourceDataBaseAddr, uint32_t sourceSlotBytes,
    uint32_t srcRankId, uint32_t dstRankId, uint64_t commHandle)
{
    GM_ADDR destinationDataBase = MoeEpDispatchBase::GetWindowAddrByRankId(mc2Context_, dstRankId, winDataOffset_) +
                                  static_cast<uint64_t>(srcRankId) * axisMaxBS_ * perSlotBytes_;
    GM_ADDR unusedNotifyAddr = MoeEpDispatchBase::GetWindowAddrByRankId(mc2Context_, dstRankId, slotWinStateOffset_) +
                               static_cast<uint64_t>(srcRankId) * dispatchNotifyCount_ * WIN_ADDR_ALIGN + UB_ALIGN;
    for (uint32_t sendEntryIndex = 0U; sendEntryIndex < sendEntryCount; sendEntryIndex++) {
        uint32_t sourceSlotIndex = 0U;
        uint32_t destinationSlotIndex = 0U;
        ReadSendEntry(sendEntryBaseAddr + static_cast<uint64_t>(sendEntryIndex) * MOE_EP_SEND_ENTRY_BYTES,
                      sourceSlotIndex, destinationSlotIndex);
        GM_ADDR sourcePayloadAddr = sourceDataBaseAddr + static_cast<uint64_t>(sourceSlotIndex) * sourceSlotBytes;
        GM_ADDR destinationPayloadAddr =
            destinationDataBase + static_cast<uint64_t>(destinationSlotIndex) * perSlotBytes_;
        if (dstRankId == epRankId_) {
            CopyPayloadToLocal(sourcePayloadAddr, destinationPayloadAddr);
        } else {
            hcomm_.WriteWithNotifyNbi(commHandle, destinationPayloadAddr, sourcePayloadAddr, perSlotBytes_,
                                      unusedNotifyAddr, 1U);
            hcomm_.Drain(commHandle);
        }
    }
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::SendScaleupEntries(uint32_t dstRankId,
                                                                                                    uint64_t commHandle)
{
    uint32_t localRankIndex = dstRankId - serverStartRank_;
    for (uint32_t tokenRangeIndex = 0U; tokenRangeIndex < aivNum_; tokenRangeIndex++) {
        __gm__ int32_t *rangeCountAddr =
            (__gm__ int32_t *)(scaleupCounterAddr_ +
                               (static_cast<uint64_t>(tokenRangeIndex) * counterAlign512_ + dstRankId) *
                                   sizeof(int32_t));
        uint32_t sendEntryCount = static_cast<uint32_t>(ReadGmByPassDCache(rangeCountAddr));
        GM_ADDR sendEntryBaseAddr =
            scaleupSendEntryAddr_ +
            (static_cast<uint64_t>(localRankIndex) * aivNum_ + tokenRangeIndex) * sendEntryTokenRangeBytes_;
        SendEntries(sendEntryBaseAddr, sendEntryCount, payloadStashWinAddr_, scaleoutSlotBytes_, epRankId_, dstRankId,
                    commHandle);
    }
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::PublishFinalReadyStatuses(
    uint32_t dstRankId)
{
    WriteFinalReadyStatusToRank(epRankId_, dstRankId);
    uint32_t currentScaleupIndex = rankNumPerServer_ == 0U ? epRankId_ : epRankId_ % rankNumPerServer_;
    for (uint32_t sourceServerOrdinal = 0U; sourceServerOrdinal < remoteServerCount_; sourceServerOrdinal++) {
        uint32_t srcScaleoutIndex =
            sourceServerOrdinal < currentServerIndex_ ? sourceServerOrdinal : sourceServerOrdinal + 1U;
        uint32_t srcRankId = srcScaleoutIndex * rankNumPerServer_ + currentScaleupIndex;
        WriteFinalReadyStatusToRank(srcRankId, dstRankId);
    }
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::SendPayloadsToScaleupRank(
    uint32_t dstRankId)
{
    uint64_t commHandle = dstRankId == epRankId_ ? 0UL : MoeEpDispatchBase::GetCommHandle(mc2Context_, dstRankId);
    SendScaleupEntries(dstRankId, commHandle);
    if (dstRankId == epRankId_) {
        PipeBarrier<PIPE_MTE3>();
    }
    PublishFinalReadyStatuses(dstRankId);
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::SendScaleupPayloads()
{
    uint32_t destinationScaleupStart = 0U;
    uint32_t destinationScaleupEnd = 0U;
    if (numNodeLocalAiv_ == 0U) {
        if (aivNum_ != 1U || aivId_ != 0U) {
            return;
        }
        destinationScaleupEnd = serverEndRank_ - serverStartRank_;
    } else if (!InitOwnedScaleupRankRange(destinationScaleupStart, destinationScaleupEnd)) {
        return;
    }

    SendFanoutPayloads(destinationScaleupStart, destinationScaleupEnd);
    for (uint32_t localRankIndex = destinationScaleupStart; localRankIndex < destinationScaleupEnd; localRankIndex++) {
        SendPayloadsToScaleupRank(serverStartRank_ + localRankIndex);
    }
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::WriteFinalReadyStatusToRank(
    uint32_t srcRankId, uint32_t dstRankId)
{
    GM_ADDR notifyBase = MoeEpDispatchBase::GetWindowAddrByRankId(mc2Context_, dstRankId, slotWinStateOffset_) +
                         static_cast<uint64_t>(srcRankId) * dispatchNotifyCount_ * WIN_ADDR_ALIGN;
    if (dstRankId == epRankId_) {
        SyncFunc<AscendC::HardEvent::MTE3_S>();
        for (uint32_t notifyIndex = 0U; notifyIndex < dispatchNotifyCount_; notifyIndex++) {
            GlobalTensor<int32_t> statusTensor;
            statusTensor.SetGlobalBuffer(
                (__gm__ int32_t *)(notifyBase + static_cast<uint64_t>(notifyIndex) * WIN_ADDR_ALIGN));
            statusTensor.SetValue(0, 1);
            DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(statusTensor);
        }
        return;
    }

    uint64_t commHandle = MoeEpDispatchBase::GetCommHandle(mc2Context_, dstRankId);
    GM_ADDR readySourceAddr = sendCntWorkspaceAddr_ + dstRankId * WIN_ADDR_ALIGN;
    for (uint32_t notifyIndex = 0U; notifyIndex < dispatchNotifyCount_; notifyIndex++) {
        GM_ADDR notifyAddr = notifyBase + static_cast<uint64_t>(notifyIndex) * WIN_ADDR_ALIGN;
        GM_ADDR unusedNotifyAddr = notifyAddr + UB_ALIGN;
        hcomm_.WriteWithNotifyNbi(commHandle, notifyAddr, readySourceAddr, sizeof(uint64_t), unusedNotifyAddr, 1U);
    }
    hcomm_.Drain(commHandle);
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::ClearScaleoutReceiveStatus(
    GM_ADDR currentStatusAddr, uint32_t statusRecordCount)
{
    uint32_t clearBatchCapacity = STATUS_CLEAR_BATCH_RECORDS;
    uint32_t initializedRecordCount = statusRecordCount < clearBatchCapacity ? statusRecordCount : clearBatchCapacity;
    LocalTensor<int32_t> clearStatusTensor = topkIdsBuf_.Get<int32_t>();
    SyncFunc<AscendC::HardEvent::S_V>();
    Duplicate<int32_t>(clearStatusTensor, 0, initializedRecordCount * UB_STRIDE);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    uint32_t clearedRecordCount = 0U;
    while (clearedRecordCount < statusRecordCount) {
        uint32_t remainingRecordCount = statusRecordCount - clearedRecordCount;
        uint32_t currentRecordCount =
            remainingRecordCount < clearBatchCapacity ? remainingRecordCount : clearBatchCapacity;
        GlobalTensor<int32_t> currentStatusTensor;
        currentStatusTensor.SetGlobalBuffer(
            (__gm__ int32_t *)(currentStatusAddr + static_cast<uint64_t>(clearedRecordCount) * WIN_ADDR_ALIGN));
        DataCopyParams clearStatusParams = {static_cast<uint16_t>(currentRecordCount), 1U, 0U,
                                            static_cast<uint16_t>((WIN_ADDR_ALIGN - UB_ALIGN) / UB_ALIGN)};
        DataCopy(currentStatusTensor, clearStatusTensor, clearStatusParams);
        clearedRecordCount += currentRecordCount;
    }
    PipeBarrier<PIPE_MTE3>();
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::ClearReceivedScaleoutStatuses()
{
    uint32_t sourceScaleoutStart = 0U;
    uint32_t sourceScaleoutEnd = 0U;
    uint32_t sourceScaleoutCountPerCore = 0U;
    SplitToCore(serverNum_, aivNum_, sourceScaleoutStart, sourceScaleoutEnd, sourceScaleoutCountPerCore);
    GM_ADDR localScaleoutStatusBase =
        MoeEpDispatchBase::GetWindowAddrByRankId(mc2Context_, epRankId_, scaleoutRecvStatusOffset_);
    for (uint32_t srcScaleoutIndex = sourceScaleoutStart; srcScaleoutIndex < sourceScaleoutEnd; srcScaleoutIndex++) {
        if (srcScaleoutIndex == currentServerIndex_) {
            continue;
        }
        GM_ADDR currentStatusAddr =
            localScaleoutStatusBase + static_cast<uint64_t>(srcScaleoutIndex) * axisMaxBS_ * WIN_ADDR_ALIGN;
        uint32_t publishedStatus = static_cast<uint32_t>(ReadGmByPassDCache((__gm__ int32_t *)currentStatusAddr));
        uint32_t receivedScaleoutSlotCount = publishedStatus - 1U;
        uint32_t statusRecordCount = receivedScaleoutSlotCount == 0U ? 1U : receivedScaleoutSlotCount;
        ClearScaleoutReceiveStatus(currentStatusAddr, statusRecordCount);
    }
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::InitRouteStateTensors(
    uint32_t rankSlotOffset, uint32_t routeStateOffset, uint32_t routeCountOffset, uint32_t sendEntryStartOffset)
{
    // 从 numRecvPerRankBuf_ 切片出各 UB tensor，存为成员变量供后续函数使用
    uint32_t routeStateBytes = Ceil(axisK_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    uint32_t routeScaleoutSlotOffset = routeStateOffset + routeStateBytes;
    uint32_t routeScaleoutStateOffset = routeScaleoutSlotOffset + routeStateBytes;
    slotIdxPerRankTensor_ = numRecvPerRankBuf_.Get<int32_t>();
    counterSumTensor_ = numRecvPerRankBuf_.GetWithOffset<int32_t>(counterCnt_, rankSlotOffset);
    sendRankSlotTensor_ = numRecvPerRankBuf_.GetWithOffset<int32_t>(epWorldSize_, rankSlotOffset);
    routeScaleoutIndexTensor_ = numRecvPerRankBuf_.GetWithOffset<int32_t>(axisKAlign_, routeStateOffset);
    routeScaleoutSlotTensor_ = numRecvPerRankBuf_.GetWithOffset<int32_t>(axisKAlign_, routeScaleoutSlotOffset);
    sendRouteIndexByScaleoutTensor_ =
        numRecvPerRankBuf_.GetWithOffset<int32_t>(serverNumAlign_, routeScaleoutStateOffset);
    routeCountTensor_ = numRecvPerRankBuf_.GetWithOffset<int32_t>(UB_STRIDE, routeCountOffset);
    scaleupSendEntryStartTensor_ = numRecvPerRankBuf_.GetWithOffset<int32_t>(counterCnt_, sendEntryStartOffset);
    scaleoutSendEntryStartTensor_ =
        numRecvPerRankBuf_.GetWithOffset<int32_t>(serverNumAlign_, sendEntryStartOffset + epWorldSizeAlign_);
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::InitPreparePayloadsBuffers()
{
    tpipe_->InitBuffer(perSlotQueue_, BUFFER_NUM, perSlotBytes_);
    tpipe_->InitBuffer(dstSlotQueue_, 1, kAlignSize_);
    tpipe_->InitBuffer(sendEntryWriteBuf_, UB_ALIGN);
    tpipe_->InitBuffer(topkIdsBuf_, 2 * perGroupSizeAlign_); // GetSlotStartNum 需要复用
    // UB布局：slot前缀 + prefix归约/rank去重复用区 + route scaleout信息。
    uint32_t routeStateBytes = Ceil(axisK_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    uint32_t scaleoutIdxBytes = Ceil(serverNum_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    uint32_t slotIndexBytes = epWorldSizeAlign_ + scaleoutIdxBytes;
    uint32_t rankSlotBytes = epWorldSizeAlign_;
    uint32_t routeCountBytes = UB_ALIGN;
    // prefix归约结果和token内rank去重映射生命周期不重叠，复用同一切片。
    uint32_t routeStateOffset = slotIndexBytes + rankSlotBytes;
    uint32_t routeCountOffset = routeStateOffset + routeStateBytes * 2U + scaleoutIdxBytes;
    uint32_t sendEntryStartOffset = routeCountOffset + routeCountBytes;
    uint32_t sendBufSize = sendEntryStartOffset + epWorldSizeAlign_ + scaleoutIdxBytes;
    tpipe_->InitBuffer(numRecvPerRankBuf_, sendBufSize);
    InitRouteStateTensors(slotIndexBytes, routeStateOffset, routeCountOffset, sendEntryStartOffset);
    sendEntryWritePending_ = false;
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::InitRouteStateForToken(
    uint32_t tokenId)
{
    if (tokenId > startTokenId_) {
        SyncFunc<AscendC::HardEvent::S_V>();
    }
    if constexpr (!IsCached) {
        if (tokenId > startTokenId_) {
            SyncFunc<AscendC::HardEvent::MTE3_V>(); // 非 cache 下一轮用 V 重新初始化 route UB
        }
    }
    Duplicate<int32_t>(sendRankSlotTensor_, -1, epWorldSize_);
    if constexpr (!IsCached) {
        Duplicate<int32_t>(routeScaleoutIndexTensor_, -1, axisKAlign_);
        Duplicate<int32_t>(routeScaleoutSlotTensor_, -1, axisKAlign_);
        Duplicate<int32_t>(sendRouteIndexByScaleoutTensor_, -1, serverNumAlign_);
    }
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::PrepareTokenPayload(
    uint32_t tokenId, uint32_t topkOffset)
{
    xLocalTensor_ = perSlotQueue_.AllocTensor<XType>();
    metaLocalTensor_ = xLocalTensor_[metaOffset_ / sizeof(XType)].template ReinterpretCast<int32_t>();
    DataCopyPad(xLocalTensor_, xGMTensor_[tokenId * axisH_], xCopyParams_, padParams_);
    if constexpr (Std::IsSame<XType, fp8_e5m2_t>::value || Std::IsSame<XType, fp8_e4m3fn_t>::value) {
        DataCopyPad(xLocalTensor_[hAlignSize_ / sizeof(XType)].template ReinterpretCast<ScalesType>(),
                    scalesGMTensor_[tokenId * scalesBytes_ / sizeof(ScalesType)], scalesCopyParams_, padParams_);
    }
    DataCopyPad(metaLocalTensor_, topkIdxGMTensor_[topkOffset], topkCopyParams_, padParams_);
    if constexpr (IsTopkWeights) {
        DataCopyPad(metaLocalTensor_[axisKAlign_].template ReinterpretCast<float>(), topkWeightsGMTensor_[topkOffset],
                    topkCopyParams_, padParams_);
    }
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    metaLocalTensor_.SetValue(2 * axisKAlign_, epRankId_);
    metaLocalTensor_.SetValue(2 * axisKAlign_ + 1, tokenId);
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    perSlotQueue_.EnQue(xLocalTensor_);
    tokenSlotTensor_ = perSlotQueue_.DeQue<XType>();
    topkIdxTensor_ = tokenSlotTensor_[metaOffset_ / sizeof(XType)].template ReinterpretCast<int32_t>();
    dstSlotIdxTensor_ = dstSlotQueue_.AllocTensor<int32_t>();
    SyncFunc<AscendC::HardEvent::V_S>();
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline uint32_t MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::ProcessCachedTokenRoutes(
    uint32_t tokenId, uint32_t topkOffset)
{
    DataCopyPad(dstSlotIdxTensor_, cachedSlotIdxGMTensor_[topkOffset], topkCopyParams_, padParams_);
    uint32_t routeEntryCount = CopyCachedRouteForToken(tokenId);
    dstSlotQueue_.EnQue(dstSlotIdxTensor_);
    dstSlotIdxTensor_ = dstSlotQueue_.DeQue<int32_t>();
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    for (uint32_t topkIndex = 0; topkIndex < axisK_; topkIndex++) {
        int32_t slot = dstSlotIdxTensor_.GetValue(topkIndex);
        if (slot == -1) {
            continue;
        }
        int32_t expertId = topkIdxTensor_.GetValue(topkIndex);
        uint32_t dstRankId = expertId / moeExpertNumPerRank_;
        if (dstRankId < serverStartRank_ || dstRankId >= serverEndRank_) {
            continue;
        }
        if (sendRankSlotTensor_.GetValue(dstRankId) == slot) {
            continue;
        }
        WriteScaleupSendEntry(dstRankId, tokenId, static_cast<uint32_t>(slot));
        sendRankSlotTensor_.SetValue(dstRankId, slot);
    }
    return routeEntryCount;
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::PreparePayloads()
{
    InitPreparePayloadsBuffers();

    uint32_t tokenRangeIndexStart = 0U;
    uint32_t tokenRangeIndexEnd = 0U;
    InitPayloadBuildTokenRanges(tokenRangeIndexStart, tokenRangeIndexEnd);
    bool hasBuiltPayload = false;
    for (uint32_t tokenRangeIndex = tokenRangeIndexStart; tokenRangeIndex < tokenRangeIndexEnd; tokenRangeIndex++) {
        SplitRangeForCore(axisBS_, aivNum_, tokenRangeIndex, startTokenId_, endTokenId_);
        sendTokenNum_ = endTokenId_ - startTokenId_;
        currentTokenRangeIndex_ = tokenRangeIndex;
        GetSlotStartNum(tokenRangeIndex);

        for (uint32_t tokenId = startTokenId_; tokenId < endTokenId_; ++tokenId) {
            uint32_t topkOffset = tokenId * axisK_;
            InitRouteStateForToken(tokenId);
            PrepareTokenPayload(tokenId, topkOffset);

            uint32_t routeEntryCount = 0U;
            if constexpr (!IsCached) {
                routeEntryCount = DedupAndBuildSendEntries(tokenId);
                SyncFunc<AscendC::HardEvent::S_MTE3>();
                dstSlotQueue_.EnQue(dstSlotIdxTensor_);
                dstSlotIdxTensor_ = dstSlotQueue_.DeQue<int32_t>();
            } else {
                routeEntryCount = ProcessCachedTokenRoutes(tokenId, topkOffset);
            }
            WritePayloadStash(tokenId);
            WriteScaleoutSendEntriesFromRoute(tokenId, routeEntryCount);
            DataCopyPad(dstSlotIdxGMTensor_[topkOffset], dstSlotIdxTensor_, topkCopyParams_);
            perSlotQueue_.FreeTensor<XType>(tokenSlotTensor_);
            dstSlotQueue_.FreeTensor<int32_t>(dstSlotIdxTensor_);
            hasBuiltPayload = true;
        }
        if (tokenRangeIndex + 1U < tokenRangeIndexEnd && sendTokenNum_ > 0U) {
            PipeBarrier<PIPE_ALL>(); // 下一token范围复用route UB前，完成当前范围的GM搬出。
        }
    }
    if (hasBuiltPayload) {
        PipeBarrier<PIPE_ALL>(); // PreparePayloads 写 GM 后，SyncAll 后会继续消费这些 GM 输出
    }
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::PrepareDispatchPayloads()
{
    CalSendCnt();
    SyncAll<true>();

    ExchangeCount();
    SyncAll<true>(); // 本Rank的count全部发出后，AIV0等待远端count，其余AIV开始本地构建。
    if (aivId_ == 0U) {
        GetRecvCount();
    }
    PipeBarrier<PIPE_ALL>();
    tpipe_->Reset();
    if (aivId_ != 0U || aivNum_ == 1U) {
        PreparePayloads();
    }
    SyncAll<true>();

    PipeBarrier<PIPE_ALL>();
    tpipe_->Reset();
    uint32_t transferScratchBytes =
        (aivNum_ > STATUS_CLEAR_BATCH_RECORDS ? aivNum_ : STATUS_CLEAR_BATCH_RECORDS) * UB_ALIGN;
    tpipe_->InitBuffer(perSlotQueue_, BUFFER_NUM, perSlotBytes_);
    tpipe_->InitBuffer(hcommBuf_, HCOMM_INIT_SIZE);
    tpipe_->InitBuffer(sendEntryReadBuf_, UB_ALIGN);
    tpipe_->InitBuffer(routeInfoBuf_, 2U * kAlignSize_);
    tpipe_->InitBuffer(numRecvPerRankBuf_, Ceil(rankNumPerServer_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN);
    tpipe_->InitBuffer(topkIdsBuf_, transferScratchBytes);
    tpipe_->InitBuffer(recvTempBuf_, UB_ALIGN);
    uint32_t sourceStateBytes = Ceil(remoteServerCount_ * sizeof(int64_t), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(sourceStateBuf_, sourceStateBytes);
    sourceStateTensor_ = sourceStateBuf_.Get<int64_t>();
    hcommTensor_ = hcommBuf_.Get<uint8_t>();
    hcomm_.Init(hcommTensor_, HCOMM_INIT_SIZE);
    sendRankSlotTensor_ = numRecvPerRankBuf_.Get<int32_t>();
    counterSumTensor_ = recvTempBuf_.Get<int32_t>();
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::TransferDispatchPayloads()
{
    if (aivNum_ == 1U) {
        SendScaleoutPayloadsToProxy();
        SendScaleupPayloads();
    } else if (aivId_ < numScaleoutSendAiv_) {
        SendScaleoutPayloadsToProxy();
    } else if (aivId_ < numScaleoutSendAiv_ + numNodeLocalAiv_) {
        // 节点内 owner 在 scaleout slot 到达后直接转发，不等待全部跨超发送结束。
        SendScaleupPayloads();
    }
    SyncAll<true>(); // 所有节点内发送完成后才能清理本轮Scaleout接收状态。
    ClearReceivedScaleoutStatuses();
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::Process()
{
    if ASCEND_IS_AIV { // 全aiv处理
        PrepareDispatchPayloads();
        TransferDispatchPayloads();
        diagWriter_.RunPosRecord(MOE_EP_DISPATCH_RUN_POS_URMA_REQUESTS_ISSUE_DONE);
    }
}

#endif

} // namespace MoeEpDispatchHybridImpl

#endif // MOE_EP_DISPATCH_HYBRID_H
