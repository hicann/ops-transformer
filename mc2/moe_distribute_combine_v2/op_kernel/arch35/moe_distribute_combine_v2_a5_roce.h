/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 **/

/*!
 * \file moe_distribute_combine_v2_a5_roce.h
 * \brief
 */
#ifndef MOE_DISTRIBUTE_COMBINE_V2_A5_ROCE_H
#define MOE_DISTRIBUTE_COMBINE_V2_A5_ROCE_H

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
// This RoCE implementation requires the Ascend 950 Hcomm APIs.
#include <cstdint>
#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "adv_api/reduce/sum.h"
#include "adv_api/hcomm/hcomm.h"
#include "utils/std/algorithm.h"
#include "../moe_distribute_combine_v2_tiling.h"
#include "../moe_distribute_combine_v2_quant.h"
#include "../../../common/op_kernel/mc2_moe_context.h"
#include "../../../moe_distribute_dispatch_v2/op_kernel/moe_distribute_v2_base.h"

namespace MoeDistributeCombineV2RoceImpl {
using namespace AscendC;
using namespace Mc2Kernel;
using namespace Mc2Aclnn;
using namespace MoeDistributeV2Base;

#define RoceCombineTypeClass \
    typename ExpandXType, typename XType, typename ExpandIdxType, uint8_t QuantMode, bool HasAddRmsNorm
#define RoceCombineTypeFunc ExpandXType, XType, ExpandIdxType, QuantMode, HasAddRmsNorm

template <RoceCombineTypeClass>
class MoeDistributeCombineV2Roce {
public:
    __aicore__ inline void Init(GM_ADDR mc2Context, GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR expandIdx,
                                GM_ADDR epSendCount, GM_ADDR residualX, GM_ADDR gamma, GM_ADDR expertScales,
                                GM_ADDR expandScales, GM_ADDR xActiveMask, GM_ADDR sharedExpertX, GM_ADDR elasticInfo,
                                GM_ADDR oriX, GM_ADDR constExpertAlpha1, GM_ADDR constExpertAlpha2,
                                GM_ADDR constExpertV, GM_ADDR performanceInfo, GM_ADDR yOut, GM_ADDR rstdOut,
                                GM_ADDR XOut, GM_ADDR workspaceGM, TPipe* pipe,
                                const MoeDistributeCombineV2Info* combineV2Info);
    __aicore__ inline void Process();

private:
    __aicore__ inline GM_ADDR GetWinAddrByRankId(const int32_t rankId);
    __aicore__ inline GM_ADDR GetWinStateAddrByRankId(const int32_t rankId);
    __aicore__ inline uint32_t GetSharedExpertRank(uint32_t sourceRank, uint32_t sharedId);
    __aicore__ inline uint32_t GetLocalSharedSlotMask(uint32_t sourceRank);
    __aicore__ inline GM_ADDR GetInterServerReceiveAddr(uint32_t computeServer, uint32_t token, uint32_t topk);
    __aicore__ inline void InitTilingAttrs(const MoeDistributeCombineV2Info* combineV2Info);
    __aicore__ inline void InitCommContext(GM_ADDR mc2Context);
    __aicore__ inline void AddRmsNormAddCompute(uint32_t tokenIndex, uint32_t tokenOffset, uint32_t numCol,
                                                LocalTensor<float>& x1TmpFloatLocal,
                                                LocalTensor<float>& x2TmpFloatLocal,
                                                LocalTensor<float>& addOutTmpFloatLocal,
                                                const DataCopyExtParams& copyExtParams,
                                                const DataCopyPadExtParams<XType>& copyPadExtParams);
    __aicore__ inline void AddRmsNormRmsNormCompute(uint32_t tokenIndex, uint32_t tokenOffset, uint32_t numCol,
                                                    LocalTensor<float>& xFp32, LocalTensor<float>& sqx,
                                                    LocalTensor<XType>& gammaLocal,
                                                    const DataCopyExtParams& copyExtParams);
    __aicore__ inline void InitScatterLayout();
    __aicore__ inline void InitInterServerLayout();
    __aicore__ inline uint32_t InitCommQuantBuffers();
    __aicore__ inline uint32_t InitScatterBuffers();
    __aicore__ inline void LoadScatterBatch(uint32_t rowStart, uint32_t rowCount, LocalTensor<int32_t>& rows,
                                            LocalTensor<float>& weights, LocalTensor<float>& broadcastWeights);
    __aicore__ inline void CopyScatterInput(uint32_t row, uint32_t bufferIndex);
    __aicore__ inline void PackScatterToken(uint32_t bufferIndex, LocalTensor<float>& broadcastWeights,
                                            uint32_t weightIndex);
    __aicore__ inline void SendScatterToken(uint32_t bufferIndex, GM_ADDR destination);
    __aicore__ inline void FinishScatterBatch(uint32_t tokenCount, GM_ADDR pendingAddress,
                                              LocalTensor<float>& broadcastWeights, uint32_t pendingWeightIndex);
    __aicore__ inline void SendScatterBatch(uint32_t rowStart, uint32_t rowCount, LocalTensor<int32_t>& rows,
                                            LocalTensor<float>& broadcastWeights);
    __aicore__ inline void ScatterTokenToLocal();
    __aicore__ inline void InitReceiveFlagBuffers(uint32_t blockCount);
    __aicore__ inline void InitReceiveBuffers(uint32_t blockCount);
    __aicore__ inline void InitGatherBuffers();
    __aicore__ inline bool IsReceiveSlotReady(GM_ADDR slotAddress);
    template <bool IsIntraServer>
    __aicore__ inline void CopyReceiveInput(GM_ADDR slotAddress, uint32_t bufferIndex);
    template <bool IsIntraServer>
    __aicore__ inline void AccumulateReceiveInput(uint32_t bufferIndex, float expertScale = 1.0f);
    __aicore__ inline void LoadGatherExpectedMasks(uint32_t mapCount);
    __aicore__ inline uint32_t BuildGatherReceivePlan(uint32_t mapCount);
    __aicore__ inline GM_ADDR GetGatherTokenAddr(uint32_t mapIndex);
    __aicore__ inline bool PollGatherToken(uint32_t mapIndex);
    __aicore__ inline void AccumulateGatherToken(uint32_t mapIndex);
    __aicore__ inline void WriteGatherResult(uint32_t sourceServer, uint32_t token, uint32_t bufferIndex);
    __aicore__ inline void ForwardTokenPacket(uint64_t slotOffsetBytes, uint32_t bufferIndex);
    __aicore__ inline bool ForwardReadyTokenRoutes(uint32_t mapIndex, uint32_t& forwardedCount);
    __aicore__ inline void GatherReadyTokens(uint32_t pendingCount);
    __aicore__ inline bool SendReadyRoutes(BatchHandle<ChannelHandle>& batch, GM_ADDR remoteReceive, uint32_t serverId,
                                           uint32_t token, uint32_t& batchCount);
    __aicore__ inline void SendReadyTokens(BatchHandle<ChannelHandle>& batch, GM_ADDR remoteReceive, uint32_t serverId,
                                           uint32_t pendingCount);
    __aicore__ inline void ScatterTokenToServer(uint32_t serverId);
    __aicore__ inline void ScatterTokensToRemote();
    __aicore__ inline void GatherTokenFromLocal();
    __aicore__ inline void InitFinalQuantBuffers();
    __aicore__ inline void InitLocalExpertBuffers(uint32_t tokenCount);
    __aicore__ inline void InitFinalBuffers(uint32_t tokenCount);
    __aicore__ inline void LoadFinalRoutes(uint32_t tokenCount, LocalTensor<ExpandIdxType>& experts,
                                           LocalTensor<bool>& activeMask);
    __aicore__ inline uint32_t IncludeFinalReceiveSlot(uint32_t token, uint32_t slot, uint32_t server,
                                                       LocalTensor<int32_t>& lastTokenByServer);
    __aicore__ inline uint32_t BuildTokenReceiveMask(uint32_t token, LocalTensor<bool>& activeMask,
                                                     LocalTensor<int32_t>& lastTokenByServer, bool& needsOutput);
    __aicore__ inline void BuildFinalServerIds(uint32_t routeCount);
    __aicore__ inline uint32_t BuildFinalReceivePlan(uint32_t tokenCount);
    __aicore__ inline GM_ADDR GetFinalReceiveAddr(uint32_t token, uint32_t topk);
    __aicore__ inline bool PollFinalToken(uint32_t token);
    __aicore__ inline void ReduceFinalReceiveSlots(uint32_t token, uint32_t remainingMask);
    __aicore__ inline LocalTensor<float> GetFinalFloatScratch();
    template <typename InputType>
    __aicore__ inline void LoadFinalVector(GlobalTensor<InputType>& source, uint64_t row,
                                           LocalTensor<float>& destination);
    __aicore__ inline float CalConstExpertAlpha(GlobalTensor<ExpandXType>& parameters, uint32_t constId);
    __aicore__ inline void ProcessConstantExpert(uint32_t constId, float weight);
    __aicore__ inline void AddLocalExpertContribution(uint32_t token, uint32_t topk);
    __aicore__ inline void ReduceFinalToken(uint32_t token);
    __aicore__ inline void WriteFinalOutput(uint32_t token, uint32_t bufferIndex);
    __aicore__ inline void CombineReadyTokens(uint32_t pendingCount);
    __aicore__ inline void ReceiveAndCombine();

    TPipe* tpipe_{nullptr};
    GlobalTensor<ExpandXType> expandXGM_;
    GlobalTensor<ExpandXType> oriXGM_;
    GlobalTensor<ExpandXType> constExpertAlpha1GM_;
    GlobalTensor<ExpandXType> constExpertAlpha2GM_;
    GlobalTensor<ExpandXType> constExpertVGM_;
    GlobalTensor<ExpandIdxType> expertIdsGM_;
    GlobalTensor<ExpandIdxType> expandIdxGM_;
    GlobalTensor<int32_t> epSendCountGM_;
    GlobalTensor<int32_t> elasticInfoGM_;
    GlobalTensor<int32_t> performanceInfoGM_;
    GlobalTensor<uint16_t> serverCommInfoGM_;
    GlobalTensor<float> expertScalesGM_;
    GlobalTensor<float> expandScalesGM_;
    GlobalTensor<bool> xActiveMaskGM_;
    GlobalTensor<XType> gammaGM_;
    GlobalTensor<XType> sharedExpertXGM_;
    GlobalTensor<XType> expandOutGlobal_;
    GlobalTensor<XType> residualXGM_;
    GlobalTensor<XType> yOutGlobal_;
    GlobalTensor<float> rstdOutGlobal_;
    TBuf<> scatterInputBuf_;
    TBuf<> scatterPackedBuf_;
    TBuf<> commQuantResultBuf_;
    TBuf<> receiveInputBuf_;
    TBuf<> receiveSumBuf_;
    TBuf<> receiveResultBuf_;
    TBuf<> receiveDequantFloatBuf_;
    TBuf<> gatherPackedBuf_;
    TBuf<> gatherExpectedMasksBuf_;
    TBuf<> gatherWaitMasksBuf_;
    TBuf<> gatherPendingTokensBuf_;
    TBuf<> receiveFlagsBuf_;
    TBuf<> receiveFlagCompareBuf_;
    TBuf<> receiveReadyFlagsBuf_;
    TBuf<> finalServerIdsBuf_;
    TBuf<> finalExpectedMasksBuf_;
    TBuf<> finalWaitMasksBuf_;
    TBuf<> finalPendingTokensBuf_;
    TBuf<> finalExpertScalesBuf_;
    TBuf<> finalExpertIdsBuf_;
    TBuf<> finalSpecialMasksBuf_;
    TBuf<> oriXFloatBuf_;
    TBuf<> constMixBuf_;
    TBuf<> finalOutputBuf_;
    TBuf<> tokenBuf_;
    TBuf<> rowTmpFloatBuf_;
    TBuf<TPosition::VECCALC> reduceFp32Buf_;
    __gm__ Mc2MoeContext* mc2Context_{nullptr};
    GM_ADDR statusDataSpaceGm_{nullptr};
    GM_ADDR statusSpaceGm_{nullptr};
    GM_ADDR epWindowGM_{nullptr};
    GM_ADDR interServerSendGM_{nullptr};
    GM_ADDR interServerReceiveGM_{nullptr};
    MoeDistributeCombineQuant<ExpandXType, ExpandXType, ExpandIdxType, QuantMode, HasAddRmsNorm> commQuant_;
    Hcomm<CommProtocol::COMM_PROTOCOL_ROCE> hcomm_;

    static constexpr uint32_t INPUT_COMPARE_ROWS_PER_REPEAT = ALIGNED_LEN_256 / sizeof(int32_t);
    static constexpr uint32_t TOKEN_BITMAPS_PER_BLOCK = UB_ALIGN / sizeof(uint16_t);
    static constexpr uint32_t PAYLOAD_BUFFER_NUM = 2U;
    static constexpr uint32_t PAYLOAD_BLOCK_BYTES = 512U;
    static constexpr uint32_t PAYLOAD_DATA_BYTES = PAYLOAD_BLOCK_BYTES - UB_ALIGN;
    static constexpr uint32_t PAYLOAD_FLAG_WORDS_PER_BLOCK = UB_ALIGN / sizeof(uint32_t);
    static constexpr uint32_t WEIGHTS_PER_REPEAT = UB_ALIGN / sizeof(float);
    static constexpr uint32_t MAX_BROADCAST_REPEATS = 255U;
    static constexpr uint32_t FLOAT_OVERFLOW_CTRL_BIT = 60U;
    static constexpr uint32_t MAX_INTER_SERVER_SEND_AIV_NUM = 8U;
    static constexpr uint32_t ROCE_BATCH_TOKENS = 5U;
    static constexpr uint32_t ROCE_WQEBB_BYTES = 64U;
    static constexpr uint32_t ROCE_HCOMM_UB_BYTES = 512U;
    static constexpr TEventID RECEIVE_FLAG_EVENT = 2U;
    static constexpr TEventID GATHER_INFO_EVENT = 3U;
    static constexpr TEventID LOCAL_EXPERT_EVENT = 3U;

    uint32_t serverNum_{0};
    uint64_t roceCombineOffset_{0};
    uint64_t interServerSendBytes_{0};
    uint32_t maxBs_{0};
    uint32_t alignedTokenNumPerRank_{0};
    uint32_t opSequence_{0};
    uint32_t aivId_{0};
    uint32_t epRankId_{0};
    uint32_t epWorldSize_{0};
    uint32_t rankSizePerServer_{0};
    uint32_t axisBS_{0};
    uint32_t axisH_{0};
    uint32_t axisK_{0};
    uint32_t commSlotNum_{0};
    uint32_t globalBS_{0};
    uint32_t aivNum_{0};
    uint32_t sendAivNum_{0};
    uint32_t forwardAivNum_{0};
    uint32_t interServerSendAivNum_{0};
    uint32_t gatherAivNum_{0};
    uint32_t interServerDataBytes_{0};
    uint32_t scatterWeightOffsetBytes_{0};
    uint32_t scatterInputStrideBytes_{0};
    uint32_t scatterPackedWeightOffsetBytes_{0};
    uint32_t scatterBlockCount_{0};
    uint32_t scatterTokenStrideBytes_{0};
    uint32_t commQuantFloatBytes_{0};
    uint32_t receiveBlockCount_{0};
    uint32_t receiveInputStrideBytes_{0};
    uint32_t receiveFloatBytes_{0};
    uint32_t receiveFlagCount_{0};
    uint32_t receiveFlagCompareCount_{0};
    uint32_t interServerBlockCount_{0};
    uint32_t interServerTokenStrideBytes_{0};
    uint32_t gatherMapBegin_{0};
    uint32_t finalTokenBegin_{0};
    uint32_t finalOutputStrideBytes_{0};
    uint8_t scatterInputRightPadding_{0};
    uint32_t moeExpertNum_{0};
    uint32_t moeExpertPerRankNum_{0};
    uint32_t sharedExpertNum_{0};
    uint32_t sharedExpertRankNum_{0};
    uint32_t zeroExpertNum_{0};
    uint32_t copyExpertNum_{0};
    uint32_t constExpertNum_{0};
    uint32_t dataState_{0};
    uint64_t totalUbSize_{0};
    uint64_t totalWinSizeEp_{0};
    uint64_t winDataSizeOffsetEp_{0};
    uint64_t winStatusOffset_{0};
    float armAvgFactor_{0.0};
    float epsilon_{0.0};
    bool hasExpertScales_{false};
    bool enablePreReduce_{true};
    bool sendExpertScales_{false};
    bool quantizeScatter_{false};
    bool isTokenMask_{false};
    bool isExpertMask_{false};
    bool hasSharedExpertX_{false};
    bool hasLocalExpertCompute_{false};
    bool loadFinalExpertScales_{false};
};

template <RoceCombineTypeClass>
__aicore__ inline GM_ADDR MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::GetWinAddrByRankId(const int32_t rankId)
{
    return (GM_ADDR)mc2Context_->epHcclBuffer_[rankId] + STATE_SIZE + winDataSizeOffsetEp_;
}

template <RoceCombineTypeClass>
__aicore__ inline GM_ADDR MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::GetWinStateAddrByRankId(const int32_t rankId)
{
    return (GM_ADDR)mc2Context_->epHcclBuffer_[rankId] + winStatusOffset_;
}

template <RoceCombineTypeClass>
__aicore__ inline uint32_t MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::GetSharedExpertRank(uint32_t sourceRank,
                                                                                                uint32_t sharedId)
{
    const uint32_t ranksPerSharedExpert = sharedExpertRankNum_ / sharedExpertNum_;
    return sharedId * ranksPerSharedExpert + sourceRank % ranksPerSharedExpert;
}

template <RoceCombineTypeClass>
__aicore__ inline uint32_t MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::GetLocalSharedSlotMask(uint32_t sourceRank)
{
    uint32_t sharedMask = 0U;
    for (uint32_t slot = axisK_; slot < commSlotNum_; ++slot) {
        const uint32_t sharedRank = GetSharedExpertRank(sourceRank, slot - axisK_);
        if (sharedRank / rankSizePerServer_ == epRankId_ / rankSizePerServer_) {
            sharedMask |= 1U << slot;
        }
    }
    return sharedMask;
}

template <RoceCombineTypeClass>
__aicore__ inline GM_ADDR MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::GetInterServerReceiveAddr(
    uint32_t computeServer, uint32_t token, uint32_t topk)
{
    if (computeServer == epRankId_ / rankSizePerServer_) {
        uint64_t sendSlot = static_cast<uint64_t>(computeServer) * maxBs_ + token;
        if (!enablePreReduce_) {
            sendSlot = sendSlot * commSlotNum_ + topk;
        }
        return interServerSendGM_ + sendSlot * interServerTokenStrideBytes_;
    }
    const uint64_t receiveSlot = static_cast<uint64_t>(token) * commSlotNum_ + topk;
    return interServerReceiveGM_ + receiveSlot * interServerTokenStrideBytes_;
}

template <RoceCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::InitTilingAttrs(
    const MoeDistributeCombineV2Info* combineV2Info)
{
    axisBS_ = combineV2Info->bs;
    axisH_ = combineV2Info->h;
    axisK_ = combineV2Info->k;
    globalBS_ = combineV2Info->globalBs;
    aivNum_ = combineV2Info->aivNum;
    sendAivNum_ = aivNum_ / 2U;
    forwardAivNum_ = aivNum_ - sendAivNum_;
    epRankId_ = combineV2Info->epRankId;
    epWorldSize_ = combineV2Info->epWorldSize;
    maxBs_ = globalBS_ / epWorldSize_;
    alignedTokenNumPerRank_ = Ceil(maxBs_, TOKEN_BITMAPS_PER_BLOCK) * TOKEN_BITMAPS_PER_BLOCK;
    moeExpertNum_ = combineV2Info->moeExpertNum;
    moeExpertPerRankNum_ = combineV2Info->moeExpertPerRankNum;
    sharedExpertNum_ = combineV2Info->sharedExpertNum;
    sharedExpertRankNum_ = combineV2Info->sharedExpertRankNum;
    commSlotNum_ = axisK_ + (sharedExpertRankNum_ > 0U ? sharedExpertNum_ : 0U);
    zeroExpertNum_ = combineV2Info->zeroExpertNum;
    copyExpertNum_ = combineV2Info->copyExpertNum;
    constExpertNum_ = combineV2Info->constExpertNum;
    hasLocalExpertCompute_ = copyExpertNum_ > 0U || constExpertNum_ > 0U;
    hasSharedExpertX_ = combineV2Info->hasSharedExpertX;
    totalUbSize_ = combineV2Info->totalUbSize;
    totalWinSizeEp_ = combineV2Info->totalWinSizeEp;
    roceCombineOffset_ = combineV2Info->roceCombineOffset;
    hasExpertScales_ = combineV2Info->hasExpertScales;
    enablePreReduce_ = combineV2Info->enablePreReduce;
    sendExpertScales_ = enablePreReduce_ && hasExpertScales_;
    quantizeScatter_ = QuantMode > UNQUANT && !enablePreReduce_;
    loadFinalExpertScales_ = hasExpertScales_ && (!enablePreReduce_ || hasLocalExpertCompute_);
    isTokenMask_ = combineV2Info->isTokenMask;
    isExpertMask_ = combineV2Info->isExpertMask;
    if constexpr (HasAddRmsNorm) {
        armAvgFactor_ = combineV2Info->armAvgFactor;
        epsilon_ = combineV2Info->epsilon;
    }
}

template <RoceCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::InitCommContext(GM_ADDR mc2Context)
{
    mc2Context_ = (__gm__ Mc2MoeContext*)mc2Context;
    const uint32_t epRankIdHccl = mc2Context_->epRankId;
    rankSizePerServer_ = mc2Context_->rankSizePerServer;
    serverNum_ = epWorldSize_ / rankSizePerServer_;
    interServerSendAivNum_ = AscendC::Std::min(serverNum_, MAX_INTER_SERVER_SEND_AIV_NUM);
    gatherAivNum_ = forwardAivNum_ - interServerSendAivNum_;
    statusDataSpaceGm_ = (GM_ADDR)mc2Context_->epHcclBuffer_[epRankIdHccl];
    GlobalTensor<uint32_t> selfDataStatusGMTensor;
    selfDataStatusGMTensor.SetGlobalBuffer(
        (__gm__ uint32_t*)(statusDataSpaceGm_ + COMBINE_STATE_WIN_OFFSET + aivId_ * WIN_ADDR_ALIGN));
    TBuf<> dataStateBuf;
    tpipe_->InitBuffer(dataStateBuf, UB_ALIGN);
    dataState_ = InitWinState(selfDataStatusGMTensor, epRankIdHccl, epWorldSize_, epRankId_, moeExpertNum_,
                              epWorldSize_, globalBS_, dataStateBuf);
    SyncFunc<HardEvent::MTE3_S>();
    opSequence_ = static_cast<uint32_t>(dataStateBuf.Get<uint64_t>().GetValue(OP_CNT_POSUL));
}

template <RoceCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::Init(
    GM_ADDR mc2Context, GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR expandIdx, GM_ADDR epSendCount, GM_ADDR residualX,
    GM_ADDR gamma, GM_ADDR expertScales, GM_ADDR expandScales, GM_ADDR xActiveMask, GM_ADDR sharedExpertX,
    GM_ADDR elasticInfo, GM_ADDR oriX, GM_ADDR constExpertAlpha1, GM_ADDR constExpertAlpha2, GM_ADDR constExpertV,
    GM_ADDR performanceInfo, GM_ADDR yOut, GM_ADDR rstdOut, GM_ADDR XOut, GM_ADDR workspaceGM, TPipe* pipe,
    const MoeDistributeCombineV2Info* combineV2Info)
{
    tpipe_ = pipe;
    aivId_ = GetBlockIdx();
    if constexpr (HasAddRmsNorm) {
        residualXGM_.SetGlobalBuffer((__gm__ XType*)residualX);
        gammaGM_.SetGlobalBuffer((__gm__ XType*)gamma);
        yOutGlobal_.SetGlobalBuffer((__gm__ XType*)yOut);
        rstdOutGlobal_.SetGlobalBuffer((__gm__ float*)rstdOut);
    }
    expandXGM_.SetGlobalBuffer((__gm__ ExpandXType*)expandX);
    expertIdsGM_.SetGlobalBuffer((__gm__ ExpandIdxType*)expertIds);
    expandIdxGM_.SetGlobalBuffer((__gm__ ExpandIdxType*)expandIdx);
    epSendCountGM_.SetGlobalBuffer((__gm__ int32_t*)epSendCount);
    expertScalesGM_.SetGlobalBuffer((__gm__ float*)expertScales);
    expandScalesGM_.SetGlobalBuffer((__gm__ float*)expandScales);
    xActiveMaskGM_.SetGlobalBuffer((__gm__ bool*)xActiveMask);
    sharedExpertXGM_.SetGlobalBuffer((__gm__ XType*)sharedExpertX);
    elasticInfoGM_.SetGlobalBuffer((__gm__ int32_t*)elasticInfo);
    oriXGM_.SetGlobalBuffer((__gm__ ExpandXType*)oriX);
    constExpertAlpha1GM_.SetGlobalBuffer((__gm__ ExpandXType*)constExpertAlpha1);
    constExpertAlpha2GM_.SetGlobalBuffer((__gm__ ExpandXType*)constExpertAlpha2);
    constExpertVGM_.SetGlobalBuffer((__gm__ ExpandXType*)constExpertV);
    performanceInfoGM_.SetGlobalBuffer((__gm__ int32_t*)performanceInfo);
    expandOutGlobal_.SetGlobalBuffer((__gm__ XType*)XOut);
    InitTilingAttrs(combineV2Info);
    InitCommContext(mc2Context);
    const uint64_t assistBitmapOffsetBytes =
        static_cast<uint64_t>(combineV2Info->a) * EXPAND_IDX_INFO * sizeof(int32_t);
    serverCommInfoGM_.SetGlobalBuffer((__gm__ uint16_t*)(expandIdx + assistBitmapOffsetBytes));
    winDataSizeOffsetEp_ = static_cast<uint64_t>(dataState_) * (totalWinSizeEp_ / 2UL);
    winStatusOffset_ = COMBINE_STATE_OFFSET + dataState_ * WIN_STATE_OFFSET;
    epWindowGM_ = GetWinAddrByRankId(epRankId_);
    statusSpaceGm_ = GetWinStateAddrByRankId(mc2Context_->epRankId);
    tpipe_->Reset();
}

template <RoceCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::AddRmsNormAddCompute(
    uint32_t tokenIndex, uint32_t tokenOffset, uint32_t numCol, LocalTensor<float>& x1TmpFloatLocal,
    LocalTensor<float>& x2TmpFloatLocal, LocalTensor<float>& addOutTmpFloatLocal,
    const DataCopyExtParams& copyExtParams, const DataCopyPadExtParams<XType>& copyPadExtParams)
{
    LocalTensor<XType> x2 = tokenBuf_.Get<XType>();
    SyncFunc<AscendC::HardEvent::V_MTE2>();
    DataCopyPad(x2, residualXGM_[tokenIndex * axisH_ + tokenOffset], copyExtParams, copyPadExtParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    Cast(x2TmpFloatLocal, x2, AscendC::RoundMode::CAST_NONE, numCol);
    AscendC::Add(addOutTmpFloatLocal, x1TmpFloatLocal, x2TmpFloatLocal, numCol);
}

template <RoceCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::AddRmsNormRmsNormCompute(
    uint32_t tokenIndex, uint32_t tokenOffset, uint32_t numCol, LocalTensor<float>& xFp32, LocalTensor<float>& sqx,
    LocalTensor<XType>& gammaLocal, const DataCopyExtParams& copyExtParams)
{
    LocalTensor<float> reduceBufLocal = reduceFp32Buf_.Get<float>();
    Mul(sqx, xFp32, xFp32, numCol);
    Muls(sqx, sqx, armAvgFactor_, numCol);
    ReduceSum(sqx, sqx, reduceBufLocal, numCol);
    Adds(sqx, sqx, epsilon_, 1);
    Sqrt(sqx, sqx, 1);
    Duplicate(reduceBufLocal, ONE, 1);
    Div(reduceBufLocal, reduceBufLocal, sqx, 1);

    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopyParams copyParams;
    copyParams.blockLen = sizeof(uint32_t);
    copyParams.blockCount = 1;
    DataCopyPad(rstdOutGlobal_[tokenIndex * 1 + tokenOffset], reduceBufLocal, copyParams);

    SyncFunc<AscendC::HardEvent::V_S>();
    float rstdValue = reduceBufLocal.GetValue(0);
    SyncFunc<AscendC::HardEvent::S_V>();
    Muls(xFp32, xFp32, rstdValue, numCol);
    LocalTensor<XType> yLocal = rowTmpFloatBuf_.Get<XType>();
    Cast(yLocal, xFp32, RoundMode::CAST_RINT, numCol);
    Cast(xFp32, yLocal, RoundMode::CAST_NONE, numCol);
    Cast(sqx, gammaLocal, RoundMode::CAST_NONE, numCol);
    Mul(xFp32, xFp32, sqx, numCol);
    Cast(yLocal, xFp32, RoundMode::CAST_RINT, numCol);

    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopyPad(yOutGlobal_[tokenIndex * axisH_ + tokenOffset], yLocal, copyExtParams);
}

template <RoceCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::InitScatterLayout()
{
    uint32_t inputAlign32Bytes = Ceil(axisH_ * sizeof(ExpandXType), UB_ALIGN) * UB_ALIGN;
    scatterInputRightPadding_ = static_cast<uint8_t>(inputAlign32Bytes / sizeof(ExpandXType) - axisH_);
    scatterInputStrideBytes_ = inputAlign32Bytes;
    interServerDataBytes_ = axisH_ * sizeof(ExpandXType);
    if constexpr (QuantMode > UNQUANT) {
        commQuantFloatBytes_ = Ceil(axisH_ * sizeof(float), ALIGNED_LEN * INT8_DIVIVE) * ALIGNED_LEN * INT8_DIVIVE;
        uint32_t scaleNum = 0U;
        uint32_t scaleAlignBytes = 0U;
        uint32_t tokenScaleCount = 0U;
        commQuant_.QuantInit(scaleNum, inputAlign32Bytes, scatterInputStrideBytes_, scaleAlignBytes,
                             commQuantFloatBytes_, tokenScaleCount, axisH_);
        interServerDataBytes_ = tokenScaleCount * sizeof(ExpandXType);
    }
    const uint32_t tokenDataBytes = quantizeScatter_ ? interServerDataBytes_ : axisH_ * sizeof(ExpandXType);
    const uint32_t weightOffsetBytes = Ceil(tokenDataBytes, UB_ALIGN) * UB_ALIGN;
    scatterWeightOffsetBytes_ = weightOffsetBytes;
    scatterPackedWeightOffsetBytes_ = weightOffsetBytes + weightOffsetBytes / PAYLOAD_DATA_BYTES * UB_ALIGN;
    const uint32_t recordBytes = sendExpertScales_ ? weightOffsetBytes + sizeof(float) : tokenDataBytes;
    scatterBlockCount_ = Ceil(recordBytes, PAYLOAD_DATA_BYTES);
    scatterTokenStrideBytes_ = scatterBlockCount_ * PAYLOAD_BLOCK_BYTES;
    if (!quantizeScatter_) {
        scatterInputStrideBytes_ = AscendC::Std::max(scatterInputStrideBytes_, scatterBlockCount_ * PAYLOAD_DATA_BYTES);
    }
}

template <RoceCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::InitInterServerLayout()
{
    InitScatterLayout();
    interServerBlockCount_ = Ceil(interServerDataBytes_, PAYLOAD_DATA_BYTES);
    interServerTokenStrideBytes_ = interServerBlockCount_ * PAYLOAD_BLOCK_BYTES;
    const uint32_t localServer = epRankId_ / rankSizePerServer_;
    interServerSendGM_ = reinterpret_cast<GM_ADDR>(mc2Context_->roceHcclBuffer_[localServer]) +
                         static_cast<uint64_t>(dataState_) * (mc2Context_->roceBufferSize / PAYLOAD_BUFFER_NUM) +
                         roceCombineOffset_;
    const uint32_t slotsPerToken = enablePreReduce_ ? 1U : commSlotNum_;
    interServerSendBytes_ = static_cast<uint64_t>(serverNum_) * maxBs_ * slotsPerToken * interServerTokenStrideBytes_;
    interServerReceiveGM_ = interServerSendGM_ + interServerSendBytes_;
}

template <RoceCombineTypeClass>
__aicore__ inline uint32_t MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::InitCommQuantBuffers()
{
    uint32_t quantBufferBytes = 0U;
    if constexpr (QuantMode > UNQUANT) {
        TBuf<> floatWorkBuf;
        tpipe_->InitBuffer(floatWorkBuf, commQuantFloatBytes_);
        LocalTensor<float> floatWork = floatWorkBuf.Get<float>();
        LocalTensor<float> absolute = floatWork;
        LocalTensor<float> maximum = floatWork;
        LocalTensor<float> broadcastScale = floatWork;
        LocalTensor<half> halfCast = floatWorkBuf.Get<half>();
        quantBufferBytes = commQuantFloatBytes_;
        if constexpr (QuantMode == INT8_COMM_QUANT) {
            TBuf<> absoluteBuf;
            TBuf<> maximumBuf;
            TBuf<> broadcastScaleBuf;
            const uint32_t absoluteBytes = commQuantFloatBytes_;
            const uint32_t maximumBytes = commQuantFloatBytes_ / REDUCE_NUM;
            const uint32_t broadcastScaleBytes = commQuantFloatBytes_;
            tpipe_->InitBuffer(absoluteBuf, absoluteBytes);
            tpipe_->InitBuffer(maximumBuf, maximumBytes);
            tpipe_->InitBuffer(broadcastScaleBuf, broadcastScaleBytes);
            absolute = absoluteBuf.Get<float>();
            maximum = maximumBuf.Get<float>();
            broadcastScale = broadcastScaleBuf.Get<float>();
            halfCast = absoluteBuf.Get<half>();
            Duplicate(absolute, 0.0f, absoluteBytes / sizeof(float));
            quantBufferBytes += absoluteBytes + maximumBytes + broadcastScaleBytes;
        }
        commQuant_.SetQuantInitParams(floatWork, halfCast, absolute, maximum, broadcastScale);
        const uint32_t quantResultBytes = Ceil(interServerDataBytes_, PAYLOAD_DATA_BYTES) * PAYLOAD_DATA_BYTES;
        tpipe_->InitBuffer(commQuantResultBuf_, quantResultBytes);
        Duplicate(commQuantResultBuf_.Get<uint8_t>(), static_cast<uint8_t>(0), quantResultBytes);
        quantBufferBytes += quantResultBytes;
    }
    return quantBufferBytes;
}

template <RoceCombineTypeClass>
__aicore__ inline uint32_t MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::InitScatterBuffers()
{
    InitScatterLayout();
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_CTRL_BIT, FLOAT_OVERFLOW_CTRL_BIT>(0);
    const uint32_t quantBufferBytes = quantizeScatter_ ? InitCommQuantBuffers() : 0U;
    const uint32_t inputBufferBytes = PAYLOAD_BUFFER_NUM * scatterInputStrideBytes_;
    const uint32_t packedBufferBytes = PAYLOAD_BUFFER_NUM * scatterTokenStrideBytes_;
    const uint32_t rowCountBufferBytes = UB_ALIGN;
    tpipe_->InitBuffer(scatterInputBuf_, inputBufferBytes);
    tpipe_->InitBuffer(scatterPackedBuf_, packedBufferBytes);
    // 每次只更新有效数据；输入及量化结果的尾部保持零，包内 flag 始终为本轮轮次号。
    Duplicate(scatterInputBuf_.Get<uint8_t>(), static_cast<uint8_t>(0), inputBufferBytes);
    Duplicate(scatterPackedBuf_.Get<uint32_t>(), opSequence_, packedBufferBytes / sizeof(uint32_t));
    return quantBufferBytes + inputBufferBytes + packedBufferBytes + rowCountBufferBytes;
}

template <RoceCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::LoadScatterBatch(
    uint32_t rowStart, uint32_t rowCount, LocalTensor<int32_t>& rows, LocalTensor<float>& weights,
    LocalTensor<float>& broadcastWeights)
{
    SyncFunc<HardEvent::S_MTE2>();
    SyncFunc<HardEvent::V_MTE2>();
    DataCopyPad(rows, expandIdxGM_[static_cast<uint64_t>(rowStart) * EXPAND_IDX_INFO],
                DataCopyExtParams{1U, rowCount * EXPAND_IDX_INFO * static_cast<uint32_t>(sizeof(int32_t)), 0U, 0U, 0U},
                DataCopyPadExtParams<int32_t>{false, 0U, 0U, 0});
    if (sendExpertScales_) {
        if (epRankId_ < sharedExpertRankNum_) {
            Duplicate(weights, 1.0f, Ceil(rowCount, WEIGHTS_PER_REPEAT) * WEIGHTS_PER_REPEAT);
        } else {
            DataCopyPad(
                weights, expandScalesGM_[rowStart],
                DataCopyExtParams{1U, rowCount * static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U},
                DataCopyPadExtParams<float>{
                    true, 0U, static_cast<uint8_t>(Ceil(rowCount, WEIGHTS_PER_REPEAT) * WEIGHTS_PER_REPEAT - rowCount),
                    0.0f});
            SyncFunc<HardEvent::MTE2_V>();
        }
        const uint32_t rowsPerBroadcast = MAX_BROADCAST_REPEATS * WEIGHTS_PER_REPEAT;
        for (uint32_t row = 0U; row < rowCount; row += rowsPerBroadcast) {
            const uint32_t repeatCount = Ceil(AscendC::Std::min(rowsPerBroadcast, rowCount - row), WEIGHTS_PER_REPEAT);
            Brcb(broadcastWeights[row * WEIGHTS_PER_REPEAT], weights[row], static_cast<uint8_t>(repeatCount),
                 {1, WEIGHTS_PER_REPEAT});
        }
    }
    SyncFunc<HardEvent::MTE2_S>();
}

template <RoceCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::CopyScatterInput(uint32_t row,
                                                                                         uint32_t bufferIndex)
{
    WaitFlag<HardEvent::V_MTE2>(bufferIndex);
    LocalTensor<ExpandXType> input =
        scatterInputBuf_.Get<ExpandXType>()[bufferIndex * scatterInputStrideBytes_ / sizeof(ExpandXType)];
    DataCopyPad(input, expandXGM_[static_cast<uint64_t>(row) * axisH_],
                DataCopyExtParams{1U, axisH_ * static_cast<uint32_t>(sizeof(ExpandXType)), 0U, 0U, 0U},
                DataCopyPadExtParams<ExpandXType>{true, 0U, scatterInputRightPadding_, 0});
    SetFlag<HardEvent::MTE2_V>(bufferIndex);
}

template <RoceCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::PackScatterToken(
    uint32_t bufferIndex, LocalTensor<float>& broadcastWeights, uint32_t weightIndex)
{
    WaitFlag<HardEvent::MTE2_V>(bufferIndex);
    LocalTensor<ExpandXType> input =
        scatterInputBuf_.Get<ExpandXType>()[bufferIndex * scatterInputStrideBytes_ / sizeof(ExpandXType)];
    LocalTensor<float> source = input.template ReinterpretCast<float>();
    if (quantizeScatter_) {
        LocalTensor<ExpandXType> quantResult = commQuantResultBuf_.Get<ExpandXType>();
        commQuant_.QuantProcess(quantResult, input);
        source = quantResult.template ReinterpretCast<float>();
        SetFlag<HardEvent::V_MTE2>(bufferIndex);
    }
    WaitFlag<HardEvent::MTE3_V>(bufferIndex);
    LocalTensor<float> packet = scatterPackedBuf_.Get<float>()[bufferIndex * scatterTokenStrideBytes_ / sizeof(float)];
    Copy(packet, source, 64UL, static_cast<uint8_t>(scatterBlockCount_), {1, 1, 16, 15});
    Copy(packet[64], source[64], 56UL, static_cast<uint8_t>(scatterBlockCount_), {1, 1, 16, 15});
    if (!quantizeScatter_) {
        SetFlag<HardEvent::V_MTE2>(bufferIndex);
    }
    if (sendExpertScales_) {
        Copy(packet[scatterPackedWeightOffsetBytes_ / sizeof(float)],
             broadcastWeights[weightIndex * WEIGHTS_PER_REPEAT], 1UL, 1U, {1, 1, 8, 8});
    }
    SetFlag<HardEvent::V_MTE3>(bufferIndex);
}

template <RoceCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::SendScatterToken(uint32_t bufferIndex,
                                                                                         GM_ADDR destination)
{
    WaitFlag<HardEvent::V_MTE3>(bufferIndex);
    LocalTensor<float> packet = scatterPackedBuf_.Get<float>()[bufferIndex * scatterTokenStrideBytes_ / sizeof(float)];
    GlobalTensor<float> remote;
    remote.SetGlobalBuffer((__gm__ float*)destination);
    DataCopy(remote, packet, scatterTokenStrideBytes_ / sizeof(float));
    SetFlag<HardEvent::MTE3_V>(bufferIndex);
}

template <RoceCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::FinishScatterBatch(
    uint32_t tokenCount, GM_ADDR pendingAddress, LocalTensor<float>& broadcastWeights, uint32_t pendingWeightIndex)
{
    if (tokenCount > 0U) {
        const uint32_t bufferIndex = (tokenCount - 1U) % PAYLOAD_BUFFER_NUM;
        PackScatterToken(bufferIndex, broadcastWeights, pendingWeightIndex);
        SendScatterToken(bufferIndex, pendingAddress);
    }
    for (uint32_t bufferIndex = 0U; bufferIndex < PAYLOAD_BUFFER_NUM; ++bufferIndex) {
        WaitFlag<HardEvent::MTE3_V>(bufferIndex);
        WaitFlag<HardEvent::V_MTE2>(bufferIndex);
    }
}

template <RoceCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::SendScatterBatch(
    uint32_t rowStart, uint32_t rowCount, LocalTensor<int32_t>& rows, LocalTensor<float>& broadcastWeights)
{
    uint32_t tokenCount = 0U;
    GM_ADDR pendingAddress = nullptr;
    uint32_t pendingWeightIndex = 0U;
    uint32_t cachedSourceRank = epWorldSize_;
    GM_ADDR sourcePayloadBase = nullptr;
    const uint32_t localServerFirstRank = epRankId_ / rankSizePerServer_ * rankSizePerServer_;
    const uint64_t payloadBytesPerSourceServer =
        static_cast<uint64_t>(maxBs_) * commSlotNum_ * scatterTokenStrideBytes_;
    for (uint32_t bufferIndex = 0U; bufferIndex < PAYLOAD_BUFFER_NUM; ++bufferIndex) {
        SetFlag<HardEvent::V_MTE2>(bufferIndex);
        SetFlag<HardEvent::MTE3_V>(bufferIndex);
    }
    for (uint32_t row = 0U; row < rowCount; ++row) {
        const uint32_t sourceRank = static_cast<uint32_t>(rows.GetValue(row * EXPAND_IDX_INFO));
        const int32_t token = rows.GetValue(row * EXPAND_IDX_INFO + 1U);
        const int32_t topk = rows.GetValue(row * EXPAND_IDX_INFO + 2U);
        if (token < 0 || token >= static_cast<int32_t>(maxBs_) || topk < 0 ||
            topk >= static_cast<int32_t>(commSlotNum_)) {
            continue;
        }
        if (sourceRank != cachedSourceRank) {
            const uint32_t sourceServer = sourceRank / rankSizePerServer_;
            const uint32_t forwardRank = localServerFirstRank + sourceRank % rankSizePerServer_;
            sourcePayloadBase = GetWinAddrByRankId(forwardRank) + sourceServer * payloadBytesPerSourceServer;
            cachedSourceRank = sourceRank;
        }
        const uint64_t slot = static_cast<uint64_t>(token) * commSlotNum_ + static_cast<uint32_t>(topk);
        GM_ADDR destination = sourcePayloadBase + slot * scatterTokenStrideBytes_;
        const uint32_t bufferIndex = tokenCount % PAYLOAD_BUFFER_NUM;
        CopyScatterInput(rowStart + row, bufferIndex);
        if (tokenCount > 0U) {
            const uint32_t previousBuffer = (tokenCount - 1U) % PAYLOAD_BUFFER_NUM;
            PackScatterToken(previousBuffer, broadcastWeights, pendingWeightIndex);
            SendScatterToken(previousBuffer, pendingAddress);
        }
        pendingAddress = destination;
        pendingWeightIndex = row;
        ++tokenCount;
    }
    FinishScatterBatch(tokenCount, pendingAddress, broadcastWeights, pendingWeightIndex);
}

template <RoceCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::ScatterTokenToLocal()
{
    TBuf<> countBuf;
    tpipe_->InitBuffer(countBuf, UB_ALIGN);
    LocalTensor<int32_t> count = countBuf.Get<int32_t>();
    const uint32_t localExpertNum = epRankId_ < sharedExpertRankNum_ ? 1U : moeExpertPerRankNum_;
    const uint32_t lastBucket = localExpertNum * epWorldSize_ - 1U;
    DataCopyPad(count, epSendCountGM_[lastBucket], DataCopyExtParams{1U, sizeof(int32_t), 0U, 0U, 0U},
                DataCopyPadExtParams<int32_t>{false, 0U, 0U, 0});
    SyncFunc<HardEvent::MTE2_S>();
    const uint32_t totalRows = static_cast<uint32_t>(count.GetValue(0));
    const uint32_t rowStart = static_cast<uint64_t>(totalRows) * aivId_ / sendAivNum_;
    const uint32_t rowEnd = static_cast<uint64_t>(totalRows) * (aivId_ + 1U) / sendAivNum_;
    if (rowStart == rowEnd) {
        return;
    }
    const uint32_t fixedUbBytes = InitScatterBuffers();
    const uint32_t rowInfoBytes = EXPAND_IDX_INFO * sizeof(int32_t);
    const uint32_t rawWeightBytes = sendExpertScales_ ? sizeof(float) : 0U;
    const uint32_t broadcastWeightBytes = sendExpertScales_ ? UB_ALIGN : 0U;
    const uint32_t rowScratchBytes = rowInfoBytes + rawWeightBytes + broadcastWeightBytes;
    const uint32_t rowsFromUb =
        (totalUbSize_ - fixedUbBytes) / rowScratchBytes / WEIGHTS_PER_REPEAT * WEIGHTS_PER_REPEAT;
    const uint32_t rowsPerBatch =
        AscendC::Std::min(Ceil(rowEnd - rowStart, WEIGHTS_PER_REPEAT) * WEIGHTS_PER_REPEAT, rowsFromUb);
    TBuf<> rowsBuf;
    TBuf<> weightsBuf;
    TBuf<> broadcastWeightsBuf;
    tpipe_->InitBuffer(rowsBuf, rowsPerBatch * rowInfoBytes);
    LocalTensor<int32_t> rows = rowsBuf.Get<int32_t>();
    LocalTensor<float> weights;
    LocalTensor<float> broadcastWeights;
    if (sendExpertScales_) {
        tpipe_->InitBuffer(weightsBuf, rowsPerBatch * rawWeightBytes);
        tpipe_->InitBuffer(broadcastWeightsBuf, rowsPerBatch * broadcastWeightBytes);
        weights = weightsBuf.Get<float>();
        broadcastWeights = broadcastWeightsBuf.Get<float>();
    }
    for (uint32_t row = rowStart; row < rowEnd; row += rowsPerBatch) {
        const uint32_t rowCount = AscendC::Std::min(rowsPerBatch, rowEnd - row);
        LoadScatterBatch(row, rowCount, rows, weights, broadcastWeights);
        SendScatterBatch(row, rowCount, rows, broadcastWeights);
    }
    tpipe_->Reset();
}

template <RoceCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::InitReceiveFlagBuffers(uint32_t blockCount)
{
    receiveBlockCount_ = blockCount;
    receiveFlagCount_ = blockCount * PAYLOAD_FLAG_WORDS_PER_BLOCK;
    receiveFlagCompareCount_ = Ceil(receiveFlagCount_, INPUT_COMPARE_ROWS_PER_REPEAT) * INPUT_COMPARE_ROWS_PER_REPEAT;
    const uint32_t flagsBytes = receiveFlagCompareCount_ * sizeof(int32_t);
    const uint32_t flagCompareBytes = Ceil(receiveFlagCompareCount_ / BITS_PER_BYTE, UB_ALIGN) * UB_ALIGN;
    const uint32_t selectedFlagsBytes = flagsBytes;
    tpipe_->InitBuffer(receiveFlagsBuf_, flagsBytes);
    tpipe_->InitBuffer(receiveFlagCompareBuf_, flagCompareBytes);
    tpipe_->InitBuffer(receiveReadyFlagsBuf_, selectedFlagsBytes);
    Duplicate(receiveFlagsBuf_.Get<int32_t>(), 0, receiveFlagCompareCount_);
}

template <RoceCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::InitReceiveBuffers(uint32_t blockCount)
{
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_CTRL_BIT, FLOAT_OVERFLOW_CTRL_BIT>(0);
    InitReceiveFlagBuffers(blockCount);
    receiveInputStrideBytes_ = blockCount * PAYLOAD_DATA_BYTES;
    receiveFloatBytes_ = Ceil(axisH_ * sizeof(float), ALIGNED_LEN_256 * INT8_DIVIVE) * ALIGNED_LEN_256 * INT8_DIVIVE;
    const uint32_t inputBytes = PAYLOAD_BUFFER_NUM * receiveInputStrideBytes_;
    const uint32_t resultBytes =
        AscendC::Std::max(interServerBlockCount_ * PAYLOAD_DATA_BYTES, scatterInputStrideBytes_);
    const uint32_t dequantFloatBytes = QuantMode == UNQUANT ? receiveFloatBytes_ : 0U;
    tpipe_->InitBuffer(receiveInputBuf_, inputBytes);
    tpipe_->InitBuffer(receiveSumBuf_, receiveFloatBytes_);
    tpipe_->InitBuffer(receiveResultBuf_, resultBytes);
    if constexpr (QuantMode == UNQUANT) {
        tpipe_->InitBuffer(receiveDequantFloatBuf_, dequantFloatBytes);
    }
    Duplicate(receiveResultBuf_.Get<uint8_t>(), static_cast<uint8_t>(0), resultBytes);
    for (uint32_t bufferIndex = 0U; bufferIndex < PAYLOAD_BUFFER_NUM; ++bufferIndex) {
        SetFlag<HardEvent::V_MTE2>(bufferIndex);
    }
}

template <RoceCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::InitGatherBuffers()
{
    if (!enablePreReduce_) {
        InitReceiveFlagBuffers(scatterBlockCount_);
        tpipe_->InitBuffer(gatherPackedBuf_, PAYLOAD_BUFFER_NUM * interServerTokenStrideBytes_);
        for (uint32_t bufferIndex = 0U; bufferIndex < PAYLOAD_BUFFER_NUM; ++bufferIndex) {
            SetFlag<HardEvent::MTE3_MTE2>(bufferIndex);
        }
        return;
    }
    InitCommQuantBuffers();
    InitReceiveBuffers(scatterBlockCount_);
    const uint32_t packedBytes = PAYLOAD_BUFFER_NUM * interServerTokenStrideBytes_;
    tpipe_->InitBuffer(gatherPackedBuf_, packedBytes);
    Duplicate(gatherPackedBuf_.Get<uint32_t>(), opSequence_, packedBytes / sizeof(uint32_t));
    for (uint32_t bufferIndex = 0U; bufferIndex < PAYLOAD_BUFFER_NUM; ++bufferIndex) {
        SetFlag<HardEvent::MTE3_V>(bufferIndex);
    }
}

template <RoceCombineTypeClass>
__aicore__ inline bool MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::IsReceiveSlotReady(GM_ADDR slotAddress)
{
    GlobalTensor<int32_t> flagsGM;
    flagsGM.SetGlobalBuffer((__gm__ int32_t*)(slotAddress + PAYLOAD_DATA_BYTES));
    LocalTensor<int32_t> flags = receiveFlagsBuf_.Get<int32_t>();
    LocalTensor<uint8_t> compare = receiveFlagCompareBuf_.Get<uint8_t>();
    SetFlag<HardEvent::V_MTE2>(RECEIVE_FLAG_EVENT);
    WaitFlag<HardEvent::V_MTE2>(RECEIVE_FLAG_EVENT);
    DataCopyPad(flags, flagsGM,
                DataCopyExtParams{static_cast<uint16_t>(receiveBlockCount_), UB_ALIGN, PAYLOAD_DATA_BYTES, 0U, 0U},
                DataCopyPadExtParams<int32_t>{false, 0U, 0U, 0});
    SetFlag<HardEvent::MTE2_V>(RECEIVE_FLAG_EVENT);
    WaitFlag<HardEvent::MTE2_V>(RECEIVE_FLAG_EVENT);
    CompareScalar(compare, flags, static_cast<int32_t>(opSequence_), CMPMODE::EQ, receiveFlagCompareCount_);
    uint64_t readyCount = 0U;
    GatherMask(receiveReadyFlagsBuf_.Get<int32_t>(), flags, compare.template ReinterpretCast<uint32_t>(), true,
               receiveFlagCount_, {1, 1, 0, 0}, readyCount);
    SyncFunc<HardEvent::V_S>();
    return readyCount == receiveFlagCount_;
}

template <RoceCombineTypeClass>
template <bool IsIntraServer>
__aicore__ inline void MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::CopyReceiveInput(GM_ADDR slotAddress,
                                                                                         uint32_t bufferIndex)
{
    WaitFlag<HardEvent::V_MTE2>(bufferIndex);
    GlobalTensor<ExpandXType> inputGM;
    inputGM.SetGlobalBuffer((__gm__ ExpandXType*)slotAddress);
    LocalTensor<ExpandXType> input =
        receiveInputBuf_.Get<ExpandXType>()[bufferIndex * receiveInputStrideBytes_ / sizeof(ExpandXType)];
    DataCopyPad(input, inputGM,
                DataCopyExtParams{static_cast<uint16_t>(receiveBlockCount_), PAYLOAD_DATA_BYTES, UB_ALIGN, 0U, 0U},
                DataCopyPadExtParams<ExpandXType>{false, 0U, 0U, 0});
    SetFlag<HardEvent::MTE2_V>(bufferIndex);
    if (IsIntraServer && hasExpertScales_) {
        SetFlag<HardEvent::MTE2_S>(bufferIndex);
    }
}

template <RoceCombineTypeClass>
template <bool IsIntraServer>
__aicore__ inline void MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::AccumulateReceiveInput(uint32_t bufferIndex,
                                                                                               float expertScale)
{
    WaitFlag<HardEvent::MTE2_V>(bufferIndex);
    LocalTensor<ExpandXType> input =
        receiveInputBuf_.Get<ExpandXType>()[bufferIndex * receiveInputStrideBytes_ / sizeof(ExpandXType)];
    LocalTensor<float> sum = receiveSumBuf_.Get<float>();
    float weight = expertScale;
    const bool applyExpertScale = hasExpertScales_ && (IsIntraServer || !enablePreReduce_);
    if (IsIntraServer && hasExpertScales_) {
        WaitFlag<HardEvent::MTE2_S>(bufferIndex);
        weight = input.template ReinterpretCast<float>().GetValue(scatterWeightOffsetBytes_ / sizeof(float));
    }
    if constexpr (QuantMode > UNQUANT && !IsIntraServer) {
        LocalTensor<float> temporary = commQuant_.winTpSendCountFloatTensor_;
        LocalTensor<ExpandXType> output = receiveResultBuf_.Get<ExpandXType>();
        if (applyExpertScale) {
            commQuant_.DeQuantProcess(input, output, temporary, sum, weight);
        } else {
            commQuant_.DeQuantProcessWithoutExpertScale(input, output, temporary, sum);
        }
    } else {
        LocalTensor<float> temporary;
        if constexpr (QuantMode > UNQUANT) {
            // 框内原始输入的 FP32 转换复用后续量化工作区。
            temporary = commQuant_.winTpSendCountFloatTensor_;
        } else {
            temporary = receiveDequantFloatBuf_.Get<float>();
        }
        Cast(temporary, input, RoundMode::CAST_NONE, axisH_);
        if (applyExpertScale) {
            Muls(temporary, temporary, weight, axisH_);
        }
        Add(sum, sum, temporary, axisH_);
    }
    SetFlag<HardEvent::V_MTE2>(bufferIndex);
}

template <RoceCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::LoadGatherExpectedMasks(uint32_t mapCount)
{
    const uint32_t bitmapCapacity = AscendC::Std::min(mapCount, alignedTokenNumPerRank_);
    TBuf<> serverInfoBuf;
    tpipe_->InitBuffer(serverInfoBuf, bitmapCapacity * sizeof(uint16_t) + UB_ALIGN);
    LocalTensor<uint16_t> bitmap = serverInfoBuf.Get<uint16_t>();
    LocalTensor<uint16_t> sharedCount = bitmap[bitmapCapacity];
    LocalTensor<uint32_t> expectedMasks = gatherExpectedMasksBuf_.Get<uint32_t>();
    for (uint32_t offset = 0U; offset < mapCount;) {
        const uint32_t sourceServer = (gatherMapBegin_ + offset) / alignedTokenNumPerRank_;
        const uint32_t tokenBegin = (gatherMapBegin_ + offset) % alignedTokenNumPerRank_;
        const uint32_t count = AscendC::Std::min(mapCount - offset, alignedTokenNumPerRank_ - tokenBegin);
        const uint64_t serverInfoOffset = static_cast<uint64_t>(sourceServer) * (alignedTokenNumPerRank_ + 1U);
        SetFlag<HardEvent::V_MTE2>(GATHER_INFO_EVENT);
        WaitFlag<HardEvent::V_MTE2>(GATHER_INFO_EVENT);
        DataCopyPad(bitmap, serverCommInfoGM_[serverInfoOffset + 1U + tokenBegin],
                    DataCopyExtParams{1U, count * static_cast<uint32_t>(sizeof(uint16_t)), 0U, 0U, 0U},
                    DataCopyPadExtParams<uint16_t>{false, 0U, 0U, 0});
        DataCopyPad(sharedCount, serverCommInfoGM_[serverInfoOffset],
                    DataCopyExtParams{1U, sizeof(uint16_t), 0U, 0U, 0U},
                    DataCopyPadExtParams<uint16_t>{false, 0U, 0U, 0});
        SetFlag<HardEvent::MTE2_V>(GATHER_INFO_EVENT);
        WaitFlag<HardEvent::MTE2_V>(GATHER_INFO_EVENT);
        Cast(expectedMasks[offset], bitmap, RoundMode::CAST_NONE, count);
        SyncFunc<HardEvent::MTE2_S>();
        const uint32_t sharedTokenCount = sharedCount.GetValue(0);
        const uint32_t sourceRank = sourceServer * rankSizePerServer_ + epRankId_ % rankSizePerServer_;
        const uint32_t sharedMask = GetLocalSharedSlotMask(sourceRank);
        if (sharedMask != 0U && sharedTokenCount > tokenBegin) {
            const uint32_t sharedTokens = AscendC::Std::min(count, sharedTokenCount - tokenBegin);
            LocalTensor<int32_t> completeMasks = expectedMasks.ReinterpretCast<int32_t>()[offset];
            Adds(completeMasks, completeMasks, static_cast<int32_t>(sharedMask), sharedTokens);
        }
        offset += count;
    }
    SyncFunc<HardEvent::V_S>();
}

template <RoceCombineTypeClass>
__aicore__ inline uint32_t MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::BuildGatherReceivePlan(uint32_t mapCount)
{
    tpipe_->InitBuffer(gatherExpectedMasksBuf_, mapCount * sizeof(uint32_t));
    tpipe_->InitBuffer(gatherWaitMasksBuf_, mapCount * sizeof(uint32_t));
    tpipe_->InitBuffer(gatherPendingTokensBuf_, mapCount * sizeof(uint32_t));
    LocalTensor<uint32_t> expectedMasks = gatherExpectedMasksBuf_.Get<uint32_t>();
    LocalTensor<uint32_t> waitMasks = gatherWaitMasksBuf_.Get<uint32_t>();
    LocalTensor<uint32_t> pendingTokens = gatherPendingTokensBuf_.Get<uint32_t>();
    LoadGatherExpectedMasks(mapCount);
    uint32_t pendingCount = 0U;
    for (uint32_t index = 0U; index < mapCount; ++index) {
        const uint32_t mask = expectedMasks.GetValue(index);
        const uint32_t token = (gatherMapBegin_ + index) % alignedTokenNumPerRank_;
        if (mask != 0U && token < maxBs_) {
            waitMasks.SetValue(index, mask);
            pendingTokens.SetValue(pendingCount++, index);
        }
    }
    return pendingCount;
}

template <RoceCombineTypeClass>
__aicore__ inline GM_ADDR MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::GetGatherTokenAddr(uint32_t mapIndex)
{
    const uint32_t sourceServer = (gatherMapBegin_ + mapIndex) / alignedTokenNumPerRank_;
    const uint32_t token = (gatherMapBegin_ + mapIndex) % alignedTokenNumPerRank_;
    const uint64_t firstSlot = (static_cast<uint64_t>(sourceServer) * maxBs_ + token) * commSlotNum_;
    return epWindowGM_ + firstSlot * scatterTokenStrideBytes_;
}

template <RoceCombineTypeClass>
__aicore__ inline bool MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::PollGatherToken(uint32_t mapIndex)
{
    LocalTensor<uint32_t> waitMasks = gatherWaitMasksBuf_.Get<uint32_t>();
    uint32_t waitMask = waitMasks.GetValue(mapIndex);
    uint32_t scanMask = waitMask;
    GM_ADDR tokenAddress = GetGatherTokenAddr(mapIndex);
    while (scanMask != 0U) {
        const uint32_t topk = static_cast<uint32_t>(ScalarGetSFFValue<1>(static_cast<uint64_t>(scanMask)));
        scanMask &= ~(1U << topk);
        if (IsReceiveSlotReady(tokenAddress + static_cast<uint64_t>(topk) * scatterTokenStrideBytes_)) {
            waitMask &= ~(1U << topk);
        }
    }
    waitMasks.SetValue(mapIndex, waitMask);
    return waitMask == 0U;
}

template <RoceCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::AccumulateGatherToken(uint32_t mapIndex)
{
    Duplicate(receiveSumBuf_.Get<float>(), 0.0f, receiveFloatBytes_ / sizeof(float));
    GM_ADDR tokenAddress = GetGatherTokenAddr(mapIndex);
    uint32_t scanMask = gatherExpectedMasksBuf_.Get<uint32_t>().GetValue(mapIndex);
    uint32_t issuedCount = 0U;
    while (scanMask != 0U) {
        const uint32_t topk = static_cast<uint32_t>(ScalarGetSFFValue<1>(static_cast<uint64_t>(scanMask)));
        scanMask &= ~(1U << topk);
        CopyReceiveInput<true>(tokenAddress + static_cast<uint64_t>(topk) * scatterTokenStrideBytes_,
                               issuedCount % PAYLOAD_BUFFER_NUM);
        if (issuedCount != 0U) {
            AccumulateReceiveInput<true>((issuedCount - 1U) % PAYLOAD_BUFFER_NUM);
        }
        ++issuedCount;
    }
    AccumulateReceiveInput<true>((issuedCount - 1U) % PAYLOAD_BUFFER_NUM);
}

template <RoceCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::WriteGatherResult(uint32_t sourceServer,
                                                                                          uint32_t token,
                                                                                          uint32_t bufferIndex)
{
    LocalTensor<ExpandXType> result = receiveResultBuf_.Get<ExpandXType>();
    Cast(result, receiveSumBuf_.Get<float>(), RoundMode::CAST_RINT, axisH_);
    LocalTensor<float> source = result.template ReinterpretCast<float>();
    if constexpr (QuantMode > UNQUANT) {
        LocalTensor<ExpandXType> quantResult = commQuantResultBuf_.Get<ExpandXType>();
        commQuant_.QuantProcess(quantResult, result);
        source = quantResult.template ReinterpretCast<float>();
    }
    WaitFlag<HardEvent::MTE3_V>(bufferIndex);
    LocalTensor<float> packet =
        gatherPackedBuf_.Get<float>()[bufferIndex * interServerTokenStrideBytes_ / sizeof(float)];
    Copy(packet, source, 64UL, static_cast<uint8_t>(interServerBlockCount_), {1, 1, 16, 15});
    Copy(packet[64], source[64], 56UL, static_cast<uint8_t>(interServerBlockCount_), {1, 1, 16, 15});
    SetFlag<HardEvent::V_MTE3>(bufferIndex);
    WaitFlag<HardEvent::V_MTE3>(bufferIndex);
    const uint64_t slot = static_cast<uint64_t>(sourceServer) * maxBs_ + token;
    GlobalTensor<float> outputGM;
    outputGM.SetGlobalBuffer((__gm__ float*)(interServerSendGM_ + slot * interServerTokenStrideBytes_));
    DataCopy(outputGM, packet, interServerTokenStrideBytes_ / sizeof(float));
    SetFlag<HardEvent::MTE3_V>(bufferIndex);
}

template <RoceCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::ForwardTokenPacket(uint64_t slotOffsetBytes,
                                                                                           uint32_t bufferIndex)
{
    WaitFlag<HardEvent::MTE3_MTE2>(bufferIndex);
    LocalTensor<uint8_t> packet = gatherPackedBuf_.Get<uint8_t>()[bufferIndex * interServerTokenStrideBytes_];
    GlobalTensor<uint8_t> input;
    input.SetGlobalBuffer((__gm__ uint8_t*)(epWindowGM_ + slotOffsetBytes));
    DataCopy(packet, input, interServerTokenStrideBytes_);
    SetFlag<HardEvent::MTE2_MTE3>(bufferIndex);
    WaitFlag<HardEvent::MTE2_MTE3>(bufferIndex);
    GlobalTensor<uint8_t> output;
    output.SetGlobalBuffer((__gm__ uint8_t*)(interServerSendGM_ + slotOffsetBytes));
    DataCopy(output, packet, interServerTokenStrideBytes_);
    SetFlag<HardEvent::MTE3_MTE2>(bufferIndex);
}

template <RoceCombineTypeClass>
__aicore__ inline bool MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::ForwardReadyTokenRoutes(
    uint32_t mapIndex, uint32_t& forwardedCount)
{
    LocalTensor<uint32_t> waitMasks = gatherWaitMasksBuf_.Get<uint32_t>();
    uint32_t waitMask = waitMasks.GetValue(mapIndex);
    uint32_t scanMask = waitMask;
    const uint32_t sourceServer = (gatherMapBegin_ + mapIndex) / alignedTokenNumPerRank_;
    const uint32_t token = (gatherMapBegin_ + mapIndex) % alignedTokenNumPerRank_;
    const uint64_t firstSlot = (static_cast<uint64_t>(sourceServer) * maxBs_ + token) * commSlotNum_;
    while (scanMask != 0U) {
        const uint32_t topk = static_cast<uint32_t>(ScalarGetSFFValue<1>(static_cast<uint64_t>(scanMask)));
        scanMask &= ~(1U << topk);
        const uint64_t slotOffsetBytes = (firstSlot + topk) * interServerTokenStrideBytes_;
        if (IsReceiveSlotReady(epWindowGM_ + slotOffsetBytes)) {
            ForwardTokenPacket(slotOffsetBytes, forwardedCount++ % PAYLOAD_BUFFER_NUM);
            waitMask &= ~(1U << topk);
        }
    }
    waitMasks.SetValue(mapIndex, waitMask);
    return waitMask == 0U;
}

template <RoceCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::GatherReadyTokens(uint32_t pendingCount)
{
    LocalTensor<uint32_t> pendingTokens = gatherPendingTokensBuf_.Get<uint32_t>();
    uint32_t cursor = 0U;
    uint32_t resultCount = 0U;
    while (pendingCount != 0U) {
        const uint32_t mapIndex = pendingTokens.GetValue(cursor);
        bool tokenComplete = false;
        if (enablePreReduce_) {
            tokenComplete = PollGatherToken(mapIndex);
            if (tokenComplete) {
                AccumulateGatherToken(mapIndex);
                const uint32_t sourceServer = (gatherMapBegin_ + mapIndex) / alignedTokenNumPerRank_;
                const uint32_t token = (gatherMapBegin_ + mapIndex) % alignedTokenNumPerRank_;
                WriteGatherResult(sourceServer, token, resultCount++ % PAYLOAD_BUFFER_NUM);
            }
        } else {
            tokenComplete = ForwardReadyTokenRoutes(mapIndex, resultCount);
        }
        if (tokenComplete) {
            pendingTokens.SetValue(cursor, pendingTokens.GetValue(--pendingCount));
        } else {
            ++cursor;
        }
        if (cursor >= pendingCount) {
            cursor = 0U;
        }
    }
}

template <RoceCombineTypeClass>
__aicore__ inline bool MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::SendReadyRoutes(
    BatchHandle<ChannelHandle>& batch, GM_ADDR remoteReceive, uint32_t serverId, uint32_t token, uint32_t& batchCount)
{
    LocalTensor<uint32_t> waitMasks = gatherWaitMasksBuf_.Get<uint32_t>();
    uint32_t remainingMask = waitMasks.GetValue(token);
    uint32_t scanMask = remainingMask;
    while (scanMask != 0U) {
        const uint32_t slot = static_cast<uint32_t>(ScalarGetSFFValue<1>(static_cast<uint64_t>(scanMask)));
        scanMask &= scanMask - 1U;
        uint64_t sendSlot = static_cast<uint64_t>(serverId) * maxBs_ + token;
        if (!enablePreReduce_) {
            sendSlot = sendSlot * commSlotNum_ + slot;
        }
        GM_ADDR source = interServerSendGM_ + sendSlot * interServerTokenStrideBytes_;
        if (!IsReceiveSlotReady(source)) {
            continue;
        }
        const uint64_t receiveSlot = static_cast<uint64_t>(token) * commSlotNum_ + slot;
        hcomm_.WriteNbi(batch, remoteReceive + receiveSlot * interServerTokenStrideBytes_, source,
                        interServerTokenStrideBytes_);
        remainingMask &= ~(1U << slot);
        if (++batchCount == ROCE_BATCH_TOKENS) {
            hcomm_.BatchCommit(batch);
            batchCount = 0U;
        }
    }
    waitMasks.SetValue(token, remainingMask);
    return remainingMask == 0U;
}

template <RoceCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::SendReadyTokens(
    BatchHandle<ChannelHandle>& batch, GM_ADDR remoteReceive, uint32_t serverId, uint32_t pendingCount)
{
    LocalTensor<uint32_t> pending = gatherPendingTokensBuf_.Get<uint32_t>();
    uint32_t batchCount = 0U;
    while (pendingCount != 0U) {
        for (uint32_t cursor = 0U; cursor < pendingCount;) {
            const uint32_t token = pending.GetValue(cursor);
            if (SendReadyRoutes(batch, remoteReceive, serverId, token, batchCount)) {
                pending.SetValue(cursor, pending.GetValue(--pendingCount));
            } else {
                ++cursor;
            }
        }
        // 不让尾批滞留到某个慢 token 到达之后。
        if (batchCount != 0U) {
            hcomm_.BatchCommit(batch);
            batchCount = 0U;
        }
    }
}

template <RoceCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::ScatterTokenToServer(uint32_t serverId)
{
    const uint32_t localServer = epRankId_ / rankSizePerServer_;
    if (serverId == localServer) {
        return;
    }
    InitReceiveFlagBuffers(interServerBlockCount_);
    gatherMapBegin_ = serverId * alignedTokenNumPerRank_;
    const uint32_t pendingCount = BuildGatherReceivePlan(alignedTokenNumPerRank_);
    LocalTensor<uint32_t> waitMasks = gatherWaitMasksBuf_.Get<uint32_t>();
    if (enablePreReduce_) {
        LocalTensor<uint32_t> pending = gatherPendingTokensBuf_.Get<uint32_t>();
        for (uint32_t index = 0U; index < pendingCount; ++index) {
            const uint32_t token = pending.GetValue(index);
            const uint32_t mask = waitMasks.GetValue(token);
            waitMasks.SetValue(token, mask & (0U - mask));
        }
    }
    if (pendingCount != 0U) {
        TBuf<> hcommBuf;
        TBuf<> batchBuf;
        tpipe_->InitBuffer(hcommBuf, ROCE_HCOMM_UB_BYTES);
        tpipe_->InitBuffer(batchBuf, (ROCE_BATCH_TOKENS + 1U) * ROCE_WQEBB_BYTES);
        hcomm_.Init(hcommBuf.Get<uint8_t>(), ROCE_HCOMM_UB_BYTES);
        LocalTensor<uint8_t> storage = batchBuf.Get<uint8_t>();
        const uint32_t padding =
            (ROCE_WQEBB_BYTES - reinterpret_cast<uintptr_t>(storage.GetPhyAddr()) % ROCE_WQEBB_BYTES) %
            ROCE_WQEBB_BYTES;
        GM_ADDR remoteReceive = reinterpret_cast<GM_ADDR>(mc2Context_->roceHcclBuffer_[serverId]) +
                                static_cast<uint64_t>(dataState_) * (mc2Context_->roceBufferSize / PAYLOAD_BUFFER_NUM) +
                                roceCombineOffset_ + interServerSendBytes_;
        const uint32_t peer = serverId > localServer ? serverId - 1U : serverId;
        const ChannelHandle channel = mc2Context_->hcommHandle_[peer * ROCE_CHANNEL_PER_SERVER];
        BatchHandle<ChannelHandle> batch = hcomm_.MakeBatchHandle(
            channel, storage[padding], ROCE_BATCH_TOKENS * ROCE_WQEBB_BYTES, remoteReceive, interServerSendGM_);
        SendReadyTokens(batch, remoteReceive, serverId, pendingCount);
    }
    tpipe_->Reset();
}

template <RoceCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::ScatterTokensToRemote()
{
    const uint32_t senderId = aivId_ - (aivNum_ - interServerSendAivNum_);
    for (uint32_t server = senderId; server < serverNum_; server += interServerSendAivNum_) {
        ScatterTokenToServer(server);
    }
}

template <RoceCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::GatherTokenFromLocal()
{
    InitInterServerLayout();
    if (aivId_ >= aivNum_ - interServerSendAivNum_) {
        ScatterTokensToRemote();
        return;
    }
    const uint32_t gatherId = aivId_ - sendAivNum_;
    const uint32_t mapBlocks = serverNum_ * alignedTokenNumPerRank_ / TOKEN_BITMAPS_PER_BLOCK;
    const uint32_t begin = static_cast<uint64_t>(mapBlocks) * gatherId / gatherAivNum_ * TOKEN_BITMAPS_PER_BLOCK;
    const uint32_t end = static_cast<uint64_t>(mapBlocks) * (gatherId + 1U) / gatherAivNum_ * TOKEN_BITMAPS_PER_BLOCK;
    if (begin == end) {
        return;
    }
    InitGatherBuffers();
    gatherMapBegin_ = begin;
    const uint32_t pendingCount = BuildGatherReceivePlan(end - begin);
    GatherReadyTokens(pendingCount);
    for (uint32_t bufferIndex = 0U; bufferIndex < PAYLOAD_BUFFER_NUM; ++bufferIndex) {
        if (enablePreReduce_) {
            WaitFlag<HardEvent::V_MTE2>(bufferIndex);
            WaitFlag<HardEvent::MTE3_V>(bufferIndex);
        } else {
            WaitFlag<HardEvent::MTE3_MTE2>(bufferIndex);
        }
    }
    tpipe_->Reset();
}

template <RoceCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::InitFinalQuantBuffers()
{
    if constexpr (QuantMode > UNQUANT) {
        TBuf<> floatWorkBuf;
        tpipe_->InitBuffer(floatWorkBuf, commQuantFloatBytes_);
        LocalTensor<float> floatWork = floatWorkBuf.Get<float>();
        LocalTensor<half> halfWork = floatWorkBuf.Get<half>();
        LocalTensor<float> broadcastScale = floatWork;
        LocalTensor<float> scaleValues = floatWork;
        if constexpr (QuantMode == INT8_COMM_QUANT) {
            TBuf<> scaleValuesBuf;
            const uint32_t scaleBytes = Ceil(commQuant_.quantScaleNum_ * sizeof(float), UB_ALIGN) * UB_ALIGN;
            tpipe_->InitBuffer(scaleValuesBuf, scaleBytes);
            scaleValues = scaleValuesBuf.Get<float>();
        }
        commQuant_.SetQuantInitParams(floatWork, halfWork, floatWork, floatWork, broadcastScale);
        commQuant_.SetDeQuantInitParams(halfWork, floatWork, broadcastScale, scaleValues);
    }
}

template <RoceCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::InitLocalExpertBuffers(uint32_t tokenCount)
{
    if (hasLocalExpertCompute_) {
        const uint32_t specialMasksBytes = Ceil(tokenCount * sizeof(uint32_t), UB_ALIGN) * UB_ALIGN;
        tpipe_->InitBuffer(finalSpecialMasksBuf_, specialMasksBytes);
        tpipe_->InitBuffer(oriXFloatBuf_, receiveFloatBytes_);
    }
    if (constExpertNum_ > 0U) {
        tpipe_->InitBuffer(constMixBuf_, receiveFloatBytes_);
    }
}

template <RoceCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::InitFinalBuffers(uint32_t tokenCount)
{
    InitFinalQuantBuffers();
    InitReceiveBuffers(interServerBlockCount_);
    InitLocalExpertBuffers(tokenCount);
    const uint32_t serverIdsBytes = Ceil(tokenCount * axisK_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    const uint32_t tokenMasksBytes = Ceil(tokenCount * sizeof(uint32_t), UB_ALIGN) * UB_ALIGN;
    const uint32_t pendingTokensBytes = Ceil(tokenCount * sizeof(uint32_t), UB_ALIGN) * UB_ALIGN;
    finalOutputStrideBytes_ = Ceil(axisH_ * sizeof(XType), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(finalServerIdsBuf_, serverIdsBytes);
    tpipe_->InitBuffer(finalExpertIdsBuf_, serverIdsBytes);
    tpipe_->InitBuffer(finalExpectedMasksBuf_, tokenMasksBytes);
    tpipe_->InitBuffer(finalWaitMasksBuf_, tokenMasksBytes);
    tpipe_->InitBuffer(finalPendingTokensBuf_, pendingTokensBytes);
    if (loadFinalExpertScales_) {
        const uint32_t expertScalesBytes = Ceil(tokenCount * axisK_ * sizeof(float), UB_ALIGN) * UB_ALIGN;
        tpipe_->InitBuffer(finalExpertScalesBuf_, expertScalesBytes);
    }
    tpipe_->InitBuffer(finalOutputBuf_, PAYLOAD_BUFFER_NUM * finalOutputStrideBytes_);
    for (uint32_t bufferIndex = 0U; bufferIndex < PAYLOAD_BUFFER_NUM; ++bufferIndex) {
        SetFlag<HardEvent::MTE3_V>(bufferIndex);
    }
}

template <RoceCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::LoadFinalRoutes(
    uint32_t tokenCount, LocalTensor<ExpandIdxType>& experts, LocalTensor<bool>& activeMask)
{
    const uint32_t routeCount = tokenCount * axisK_;
    const uint32_t routeBytes = routeCount * sizeof(ExpandIdxType);
    DataCopyPad(experts, expertIdsGM_[finalTokenBegin_ * axisK_], DataCopyExtParams{1U, routeBytes, 0U, 0U, 0U},
                DataCopyPadExtParams<ExpandIdxType>{false, 0U, 0U, 0});
    if (loadFinalExpertScales_) {
        DataCopyPad(finalExpertScalesBuf_.Get<float>(), expertScalesGM_[finalTokenBegin_ * axisK_],
                    DataCopyExtParams{1U, routeCount * static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U},
                    DataCopyPadExtParams<float>{false, 0U, 0U, 0.0f});
    }
    if (isTokenMask_ || isExpertMask_) {
        const uint32_t maskCount = isExpertMask_ ? routeCount : tokenCount;
        const uint32_t maskBegin = isExpertMask_ ? finalTokenBegin_ * axisK_ : finalTokenBegin_;
        DataCopyPad(activeMask, xActiveMaskGM_[maskBegin], DataCopyExtParams{1U, maskCount, 0U, 0U, 0U},
                    DataCopyPadExtParams<bool>{false, 0U, 0U, false});
    }
    SyncFunc<HardEvent::MTE2_V>();
}

template <RoceCombineTypeClass>
__aicore__ inline uint32_t MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::IncludeFinalReceiveSlot(
    uint32_t token, uint32_t slot, uint32_t server, LocalTensor<int32_t>& lastTokenByServer)
{
    if (!enablePreReduce_) {
        return 1U << slot;
    }
    if (lastTokenByServer.GetValue(server) == static_cast<int32_t>(token)) {
        return 0U;
    }
    lastTokenByServer.SetValue(server, static_cast<int32_t>(token));
    return 1U << slot;
}

template <RoceCombineTypeClass>
__aicore__ inline uint32_t MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::BuildTokenReceiveMask(
    uint32_t token, LocalTensor<bool>& activeMask, LocalTensor<int32_t>& lastTokenByServer, bool& needsOutput)
{
    needsOutput = false;
    if (isTokenMask_ && !activeMask.GetValue(token)) {
        return 0U;
    }
    needsOutput = hasSharedExpertX_ || commSlotNum_ > axisK_;
    LocalTensor<ExpandIdxType> experts = finalExpertIdsBuf_.Get<ExpandIdxType>();
    LocalTensor<int32_t> serverIds = finalServerIdsBuf_.Get<int32_t>();
    const uint32_t copyBegin = moeExpertNum_ + zeroExpertNum_;
    const uint32_t specialEnd = copyBegin + copyExpertNum_ + constExpertNum_;
    uint32_t expectedMask = 0U;
    uint32_t specialMask = 0U;
    for (uint32_t topk = 0U; topk < axisK_; ++topk) {
        const uint32_t route = token * axisK_ + topk;
        const int32_t expert = experts.GetValue(route);
        if (isExpertMask_ && !activeMask.GetValue(route)) {
            continue;
        }
        if (expert >= 0 && expert < static_cast<int32_t>(moeExpertNum_)) {
            expectedMask |= IncludeFinalReceiveSlot(token, topk, serverIds.GetValue(route), lastTokenByServer);
            needsOutput = true;
        } else if (expert >= static_cast<int32_t>(moeExpertNum_) && expert < static_cast<int32_t>(specialEnd)) {
            needsOutput = true;
            if (expert >= static_cast<int32_t>(copyBegin)) {
                specialMask |= 1U << topk;
            }
        }
    }
    if (hasLocalExpertCompute_) {
        finalSpecialMasksBuf_.Get<uint32_t>().SetValue(token, specialMask);
    }
    for (uint32_t slot = axisK_; slot < commSlotNum_; ++slot) {
        const uint32_t server = GetSharedExpertRank(epRankId_, slot - axisK_) / rankSizePerServer_;
        expectedMask |= IncludeFinalReceiveSlot(token, slot, server, lastTokenByServer);
    }
    return expectedMask;
}

template <RoceCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::BuildFinalServerIds(uint32_t routeCount)
{
    TBuf<> routeFloatBuf;
    tpipe_->InitBuffer(routeFloatBuf, Ceil(routeCount * sizeof(float), UB_ALIGN) * UB_ALIGN);
    LocalTensor<float> routeFloat = routeFloatBuf.Get<float>();
    Cast(routeFloat, finalExpertIdsBuf_.Get<ExpandIdxType>(), RoundMode::CAST_NONE, routeCount);
    // 取整数区间中点，避免浮点商在整除边界向下偏移；host 限制 E * P <= 2048。
    const float routeMidpointOffset = static_cast<float>(sharedExpertRankNum_ * moeExpertPerRankNum_) + 0.5f;
    const float inverseExpertsPerServer = 1.0f / static_cast<float>(moeExpertPerRankNum_ * rankSizePerServer_);
    Adds(routeFloat, routeFloat, routeMidpointOffset, routeCount);
    Muls(routeFloat, routeFloat, inverseExpertsPerServer, routeCount);
    Cast(finalServerIdsBuf_.Get<int32_t>(), routeFloat, RoundMode::CAST_FLOOR, routeCount);
    SyncFunc<HardEvent::V_S>();
}

template <RoceCombineTypeClass>
__aicore__ inline uint32_t MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::BuildFinalReceivePlan(uint32_t tokenCount)
{
    const uint32_t routeCount = tokenCount * axisK_;
    const uint32_t maskCount = isExpertMask_ ? routeCount : tokenCount;
    TBuf<> activeMaskBuf;
    TBuf<> lastTokenByServerBuf;
    LocalTensor<int32_t> lastTokenByServer;
    if (enablePreReduce_) {
        tpipe_->InitBuffer(lastTokenByServerBuf, Ceil(serverNum_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN);
        lastTokenByServer = lastTokenByServerBuf.Get<int32_t>();
        Duplicate(lastTokenByServer, -1, serverNum_);
    }
    LocalTensor<bool> activeMask;
    if (isTokenMask_ || isExpertMask_) {
        tpipe_->InitBuffer(activeMaskBuf, Ceil(maskCount * sizeof(bool), UB_ALIGN) * UB_ALIGN);
        activeMask = activeMaskBuf.Get<bool>();
    }
    LocalTensor<ExpandIdxType> experts = finalExpertIdsBuf_.Get<ExpandIdxType>();
    LoadFinalRoutes(tokenCount, experts, activeMask);
    BuildFinalServerIds(routeCount);
    LocalTensor<uint32_t> expectedMasks = finalExpectedMasksBuf_.Get<uint32_t>();
    LocalTensor<uint32_t> waitMasks = finalWaitMasksBuf_.Get<uint32_t>();
    LocalTensor<uint32_t> pendingTokens = finalPendingTokensBuf_.Get<uint32_t>();
    uint32_t pendingCount = 0U;
    for (uint32_t token = 0U; token < tokenCount; ++token) {
        bool needsOutput = false;
        const uint32_t expectedMask = BuildTokenReceiveMask(token, activeMask, lastTokenByServer, needsOutput);
        expectedMasks.SetValue(token, expectedMask);
        waitMasks.SetValue(token, expectedMask);
        if (needsOutput) {
            pendingTokens.SetValue(pendingCount++, token);
        }
    }
    return pendingCount;
}

template <RoceCombineTypeClass>
__aicore__ inline GM_ADDR MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::GetFinalReceiveAddr(uint32_t token,
                                                                                               uint32_t topk)
{
    const uint32_t computeServer = topk < axisK_ ? finalServerIdsBuf_.Get<int32_t>().GetValue(token * axisK_ + topk) :
                                                   GetSharedExpertRank(epRankId_, topk - axisK_) / rankSizePerServer_;
    return GetInterServerReceiveAddr(computeServer, finalTokenBegin_ + token, topk);
}

template <RoceCombineTypeClass>
__aicore__ inline bool MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::PollFinalToken(uint32_t token)
{
    LocalTensor<uint32_t> waitMasks = finalWaitMasksBuf_.Get<uint32_t>();
    uint32_t waitMask = waitMasks.GetValue(token);
    uint32_t scanMask = waitMask;
    while (scanMask != 0U) {
        const uint32_t topk = static_cast<uint32_t>(ScalarGetSFFValue<1>(static_cast<uint64_t>(scanMask)));
        scanMask &= ~(1U << topk);
        if (IsReceiveSlotReady(GetFinalReceiveAddr(token, topk))) {
            waitMask &= ~(1U << topk);
        }
    }
    waitMasks.SetValue(token, waitMask);
    return waitMask == 0U;
}

template <RoceCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::ReduceFinalReceiveSlots(uint32_t token,
                                                                                                uint32_t remainingMask)
{
    uint32_t issuedCount = 0U;
    float pendingWeight = 1.0f;
    while (remainingMask != 0U) {
        const uint32_t topk = static_cast<uint32_t>(ScalarGetSFFValue<1>(static_cast<uint64_t>(remainingMask)));
        remainingMask &= ~(1U << topk);
        CopyReceiveInput<false>(GetFinalReceiveAddr(token, topk), issuedCount % PAYLOAD_BUFFER_NUM);
        if (issuedCount != 0U) {
            AccumulateReceiveInput<false>((issuedCount - 1U) % PAYLOAD_BUFFER_NUM, pendingWeight);
        }
        pendingWeight = 1.0f;
        if (!enablePreReduce_ && hasExpertScales_ && topk < axisK_) {
            pendingWeight = finalExpertScalesBuf_.Get<float>().GetValue(token * axisK_ + topk);
        }
        ++issuedCount;
    }
    if (issuedCount != 0U) {
        AccumulateReceiveInput<false>((issuedCount - 1U) % PAYLOAD_BUFFER_NUM, pendingWeight);
    }
}

template <RoceCombineTypeClass>
__aicore__ inline LocalTensor<float> MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::GetFinalFloatScratch()
{
    if constexpr (QuantMode > 0U) {
        return commQuant_.winTpSendCountFloatTensor_;
    } else {
        return receiveDequantFloatBuf_.Get<float>();
    }
}

template <RoceCombineTypeClass>
template <typename InputType>
__aicore__ inline void MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::LoadFinalVector(GlobalTensor<InputType>& source,
                                                                                        uint64_t row,
                                                                                        LocalTensor<float>& destination)
{
    LocalTensor<InputType> input = receiveResultBuf_.Get<InputType>();
    SetFlag<HardEvent::V_MTE2>(LOCAL_EXPERT_EVENT);
    WaitFlag<HardEvent::V_MTE2>(LOCAL_EXPERT_EVENT);
    DataCopyPad(input, source[row * axisH_],
                DataCopyExtParams{1U, axisH_ * static_cast<uint32_t>(sizeof(InputType)), 0U, 0U, 0U},
                DataCopyPadExtParams<InputType>{false, 0U, 0U, 0});
    SetFlag<HardEvent::MTE2_V>(LOCAL_EXPERT_EVENT);
    WaitFlag<HardEvent::MTE2_V>(LOCAL_EXPERT_EVENT);
    Cast(destination, input, RoundMode::CAST_NONE, axisH_);
}

template <RoceCombineTypeClass>
__aicore__ inline float MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::CalConstExpertAlpha(
    GlobalTensor<ExpandXType>& parameters, uint32_t constId)
{
    LocalTensor<float> product = GetFinalFloatScratch();
    LoadFinalVector(parameters, constId, product);
    Mul(product, product, oriXFloatBuf_.Get<float>(), axisH_);
    const uint32_t alignedH = Ceil(axisH_ * sizeof(float), UB_ALIGN) * UB_ALIGN / sizeof(float);
    SumParams params{1U, alignedH, axisH_};
    Sum(product, product, params);
    SyncFunc<HardEvent::V_S>();
    return product.GetValue(0);
}

template <RoceCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::ProcessConstantExpert(uint32_t constId,
                                                                                              float weight)
{
    const float alpha1 = CalConstExpertAlpha(constExpertAlpha1GM_, constId);
    const float alpha2 = CalConstExpertAlpha(constExpertAlpha2GM_, constId);
    const float maxAlpha = AscendC::Std::max(alpha1, alpha2);
    LocalTensor<float> probabilities = receiveResultBuf_.Get<float>();
    probabilities.SetValue(0U, alpha1 - maxAlpha);
    probabilities.SetValue(1U, alpha2 - maxAlpha);
    SyncFunc<HardEvent::S_V>();
    Exp(probabilities, probabilities, 2U);
    SyncFunc<HardEvent::V_S>();
    const float expAlpha1 = probabilities.GetValue(0U);
    const float expAlpha2 = probabilities.GetValue(1U);
    const float alphaSum = expAlpha1 + expAlpha2;
    LocalTensor<float> constant = GetFinalFloatScratch();
    LoadFinalVector(constExpertVGM_, constId, constant);
    LocalTensor<float> mixture = constMixBuf_.Get<float>();
    Muls(mixture, oriXFloatBuf_.Get<float>(), expAlpha1 / alphaSum, axisH_);
    Muls(constant, constant, expAlpha2 / alphaSum, axisH_);
    Add(mixture, mixture, constant, axisH_);
    Muls(mixture, mixture, weight, axisH_);
    Add(receiveSumBuf_.Get<float>(), receiveSumBuf_.Get<float>(), mixture, axisH_);
}

template <RoceCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::AddLocalExpertContribution(uint32_t token,
                                                                                                   uint32_t topk)
{
    const uint32_t route = token * axisK_ + topk;
    const uint32_t expert = finalExpertIdsBuf_.Get<ExpandIdxType>().GetValue(route);
    const uint32_t constBegin = moeExpertNum_ + zeroExpertNum_ + copyExpertNum_;
    const float weight = hasExpertScales_ ? finalExpertScalesBuf_.Get<float>().GetValue(route) : 1.0f;
    if (expert < constBegin) {
        LocalTensor<float> temporary = GetFinalFloatScratch();
        Muls(temporary, oriXFloatBuf_.Get<float>(), weight, axisH_);
        Add(receiveSumBuf_.Get<float>(), receiveSumBuf_.Get<float>(), temporary, axisH_);
    } else {
        ProcessConstantExpert(expert - constBegin, weight);
    }
}

template <RoceCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::ReduceFinalToken(uint32_t token)
{
    Duplicate(receiveSumBuf_.Get<float>(), 0.0f, receiveFloatBytes_ / sizeof(float));
    uint32_t receiveMask = finalExpectedMasksBuf_.Get<uint32_t>().GetValue(token);
    if (enablePreReduce_) {
        ReduceFinalReceiveSlots(token, receiveMask);
        receiveMask = 0U;
    }
    uint32_t specialMask = hasLocalExpertCompute_ ? finalSpecialMasksBuf_.Get<uint32_t>().GetValue(token) : 0U;
    const uint64_t sourceToken = finalTokenBegin_ + token;
    if (specialMask != 0U) {
        LocalTensor<float> original = oriXFloatBuf_.Get<float>();
        LoadFinalVector(oriXGM_, sourceToken, original);
    }
    while (specialMask != 0U) {
        const uint32_t topk = static_cast<uint32_t>(ScalarGetSFFValue<1>(static_cast<uint64_t>(specialMask)));
        specialMask &= ~(1U << topk);
        // 先完成前序通信槽位的累加，再复用计算缓冲区处理当前特殊专家。
        const uint32_t precedingMask = receiveMask & ((1U << topk) - 1U);
        ReduceFinalReceiveSlots(token, precedingMask);
        receiveMask &= ~precedingMask;
        AddLocalExpertContribution(token, topk);
    }
    ReduceFinalReceiveSlots(token, receiveMask);
    if (hasSharedExpertX_) {
        LocalTensor<float> temporary = GetFinalFloatScratch();
        LoadFinalVector(sharedExpertXGM_, sourceToken, temporary);
        Add(receiveSumBuf_.Get<float>(), receiveSumBuf_.Get<float>(), temporary, axisH_);
    }
}

template <RoceCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::WriteFinalOutput(uint32_t token,
                                                                                         uint32_t bufferIndex)
{
    WaitFlag<HardEvent::MTE3_V>(bufferIndex);
    LocalTensor<XType> output = finalOutputBuf_.Get<XType>()[bufferIndex * finalOutputStrideBytes_ / sizeof(XType)];
    Cast(output, receiveSumBuf_.Get<float>(), RoundMode::CAST_RINT, axisH_);
    SetFlag<HardEvent::V_MTE3>(bufferIndex);
    WaitFlag<HardEvent::V_MTE3>(bufferIndex);
    const uint64_t outputOffset = static_cast<uint64_t>(finalTokenBegin_ + token) * axisH_;
    const uint32_t outputBytes = axisH_ * sizeof(XType);
    DataCopyPad(expandOutGlobal_[outputOffset], output, DataCopyExtParams{1U, outputBytes, 0U, 0U, 0U});
    SetFlag<HardEvent::MTE3_V>(bufferIndex);
}

template <RoceCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::CombineReadyTokens(uint32_t pendingCount)
{
    LocalTensor<uint32_t> pendingTokens = finalPendingTokensBuf_.Get<uint32_t>();
    uint32_t cursor = 0U;
    uint32_t outputCount = 0U;
    while (pendingCount != 0U) {
        const uint32_t token = pendingTokens.GetValue(cursor);
        if (PollFinalToken(token)) {
            ReduceFinalToken(token);
            WriteFinalOutput(token, outputCount % PAYLOAD_BUFFER_NUM);
            ++outputCount;
            --pendingCount;
            pendingTokens.SetValue(cursor, pendingTokens.GetValue(pendingCount));
        } else {
            ++cursor;
        }
        if (cursor >= pendingCount) {
            cursor = 0U;
        }
    }
}

template <RoceCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::ReceiveAndCombine()
{
    finalTokenBegin_ = static_cast<uint64_t>(axisBS_) * aivId_ / aivNum_;
    const uint32_t tokenEnd = static_cast<uint64_t>(axisBS_) * (aivId_ + 1U) / aivNum_;
    const uint32_t tokenCount = tokenEnd - finalTokenBegin_;
    if (tokenCount == 0U) {
        return;
    }
    InitInterServerLayout();
    InitFinalBuffers(tokenCount);
    const uint32_t pendingCount = BuildFinalReceivePlan(tokenCount);
    CombineReadyTokens(pendingCount);
    for (uint32_t bufferIndex = 0U; bufferIndex < PAYLOAD_BUFFER_NUM; ++bufferIndex) {
        WaitFlag<HardEvent::V_MTE2>(bufferIndex);
        WaitFlag<HardEvent::MTE3_V>(bufferIndex);
    }
    tpipe_->Reset();
}

template <RoceCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::Process()
{
    if ASCEND_IS_AIV {
        if (aivId_ < sendAivNum_) {
            ScatterTokenToLocal();
        } else {
            GatherTokenFromLocal();
        }
        SyncAll<true>();
        tpipe_->Reset();
        ReceiveAndCombine();
    }
}

} // namespace MoeDistributeCombineV2RoceImpl

#endif // defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
#endif
