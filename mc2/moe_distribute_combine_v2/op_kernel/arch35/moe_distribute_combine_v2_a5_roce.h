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
#include "basic_api/kernel_operator_scalar_intf.h"
#include "basic_api/reg_compute/kernel_reg_compute_intf.h"
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

static constexpr UrmaWqeEntry SENDER_WRITE_CFG = {
    .odr = 5,
    .fence = 0,
    .se = 0,
    .cqe = 0,
    .inlineEn = 0,
};
static constexpr UrmaWqeEntry SENDER_LAST_WRITE_CFG = {
    .odr = 5,
    .fence = 0,
    .se = 0,
    .cqe = 1,
    .inlineEn = 0,
};

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
    struct TokenCoreAssignment {
        uint32_t groupStart;
        uint32_t groupStride;
        uint32_t lane;
        uint32_t laneCount;
    };

    struct ScatterCardState {
        uint32_t row;
        uint32_t end;
        uint32_t cacheBegin;
        uint32_t cacheCount;
    };

    __aicore__ inline GM_ADDR GetWinAddrByRankId(const int32_t rankId);
    __aicore__ inline uint32_t GetSharedExpertRank(uint32_t sourceRank, uint32_t sharedId);
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
    __aicore__ inline void InitScatterBuffers();
    __aicore__ inline void InitInputEvents(uint32_t count);
    __aicore__ inline void InitMetadataEvents();
    __aicore__ inline void InitScatterCardBuffers(uint32_t cardCount);
    __aicore__ inline uint32_t LoadScatterCardPlan(uint32_t expert, uint32_t taskBegin, uint32_t taskEnd,
                                                   uint32_t taskStride, bool remote);
    __aicore__ inline void LoadScatterCardCache(uint32_t card, ScatterCardState& state);
    __aicore__ inline void CopyScatterInput(uint32_t row, uint32_t bufferIndex);
    __aicore__ inline void PackScatterToken(uint32_t bufferIndex, uint32_t packetIndex,
                                            LocalTensor<float>& broadcastWeights, uint32_t weightIndex);
    __aicore__ inline void SendScatterToken(uint32_t bufferIndex, GM_ADDR destination);
    __aicore__ inline void FlushScatterPending();
    __aicore__ inline void FinishScatterPipeline();
    __aicore__ inline void SendScatterCardRows(uint32_t card, const ScatterCardState& state, uint32_t rowCount);
    __aicore__ inline uint32_t ScatterCardTile(uint32_t card, uint32_t tokenEnd);
    __aicore__ inline void ScatterCardTiles(uint32_t cardCount);
    __aicore__ inline void ScatterExpertCards(uint32_t expert, uint32_t taskBegin, uint32_t taskEnd,
                                              uint32_t taskStride, bool remote, bool& active);
    __aicore__ inline void ScatterSourceServersBalanced(uint32_t experts, bool remote, bool& active);
    __aicore__ inline void ScatterSourceServers(const TokenCoreAssignment& assignment, uint32_t experts, bool remote,
                                                bool& active);
    __aicore__ inline void ScatterTokenToLocal();
    __aicore__ inline void InitReceiveFlagBuffers(uint32_t blockCount);
    __aicore__ inline void InitReceiveBuffers(uint32_t blockCount, bool clearResult = true,
                                              uint32_t sumBufferCount = 1U);
    __aicore__ inline void InitGatherBuffers(uint32_t mapCount);
    __aicore__ inline void InitReceiveFlagPrefetch(uint32_t tokenCount, uint32_t blockCount);
    template <bool Final>
    __aicore__ inline void PrefetchReceiveFlags(uint32_t token, uint32_t slot);
    template <bool Final>
    __aicore__ inline uint32_t ConsumeReceiveFlags(uint32_t slot);
    __aicore__ inline void FinishReceiveFlagPrefetch();
    template <bool Final>
    __aicore__ inline bool ProcessReadyToken(uint32_t token, uint32_t readyMask, uint32_t& resultSlot);
    template <bool Final>
    __aicore__ inline void ProcessPrefetchedTokens(LocalTensor<uint32_t>& pending, uint32_t pendingCount);
    template <bool UseVector = true>
    __aicore__ inline bool IsReceiveSlotReady(GM_ADDR slotAddress, uint32_t blockCount);
    template <bool IsIntraServer>
    __aicore__ inline void CopyReceiveInput(GM_ADDR slotAddress, uint32_t bufferIndex);
    template <bool IsIntraServer, bool Initialize = false>
    __aicore__ inline void AccumulateReceiveInput(uint32_t bufferIndex, float expertScale = 1.0f);
    __aicore__ inline void ScaleReceiveInput(LocalTensor<float>& values, LocalTensor<ExpandXType>& input);
    __aicore__ inline void LoadGatherExpectedMasks(uint32_t mapCount);
    __aicore__ inline uint32_t BuildGatherReceivePlan(uint32_t mapCount);
    __aicore__ inline uint32_t GetGatherTokenId(uint32_t mapIndex);
    __aicore__ inline GM_ADDR GetGatherTokenAddr(uint32_t mapIndex);
    __aicore__ inline void AccumulateGatherInput(uint32_t bufferIndex, bool firstInput);
    __aicore__ inline uint64_t GetGatherPartialOffset(uint32_t mapIndex);
    __aicore__ inline void PrepareGatherPartial(uint32_t mapIndex, bool resume);
    __aicore__ inline void SaveGatherPartial(uint32_t mapIndex);
    __aicore__ inline bool AccumulateReadyGatherToken(uint32_t mapIndex, uint32_t& packetSlot, uint32_t readyMask);
    __aicore__ inline void FinishGatherToken(uint32_t mapIndex, uint32_t remaining, uint32_t& packetSlot);
    __aicore__ inline void WriteGatherResult(uint32_t sourceServer, uint32_t token, uint32_t bufferIndex);
    __aicore__ inline void PublishLocalTokenReady(uint32_t serverId, uint32_t token);
    __aicore__ inline void ForwardTokenPacket(uint64_t slotOffsetBytes, uint32_t bufferIndex, uint32_t token,
                                              bool tokenReady);
    __aicore__ inline bool ForwardReadyTokenRoutes(uint32_t mapIndex, uint32_t& packetSlot);
    __aicore__ inline void GatherReadyTokens(uint32_t pendingCount);
    __aicore__ inline void InitSenderFlagBuffers();
    __aicore__ inline void LoadSenderFlags(uint32_t serverId);
    __aicore__ inline void CompareSenderFlags();
    __aicore__ inline void InitSenderPendingBitmap(uint32_t pendingCount);
    __aicore__ inline void WriteSenderPacket(BatchHandle<ChannelHandle>& batch, GM_ADDR destination, GM_ADDR source,
                                             bool lastPacket);
    __aicore__ inline void SendTokenRoutes(BatchHandle<ChannelHandle>& batch, GM_ADDR remoteReceive, uint32_t serverId,
                                           uint32_t token, uint32_t& batchCount, uint32_t remainingTokens);
    __aicore__ inline uint32_t SendReadySnapshot(BatchHandle<ChannelHandle>& batch, GM_ADDR remoteReceive,
                                                 uint32_t serverId, uint32_t pendingCount);
    __aicore__ inline void SendReadyTokens(BatchHandle<ChannelHandle>& batch, GM_ADDR remoteReceive, uint32_t serverId,
                                           uint32_t pendingCount);
    __aicore__ inline bool IsSenderGroupReady(uint32_t begin, uint32_t count);
    __aicore__ inline void WriteSenderGroup(BatchHandle<ChannelHandle>& batch, GM_ADDR remoteReceive, uint32_t serverId,
                                            uint32_t begin, uint32_t count, bool lastGroup);
    __aicore__ inline uint32_t SendReadyGroups(BatchHandle<ChannelHandle>& batch, GM_ADDR remoteReceive,
                                               uint32_t serverId, uint32_t pendingCount);
    __aicore__ inline void ScatterTokenToServer(uint32_t serverId);
    __aicore__ inline void ScatterTokensToRemote();
    __aicore__ inline TokenCoreAssignment GetTokenCoreAssignment(uint32_t core, uint32_t cores, uint32_t groups);
    __aicore__ inline void GatherServerTokens(uint32_t serverId);
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
    __aicore__ inline uint32_t GetFinalComputeServer(uint32_t token, uint32_t topk);
    __aicore__ inline GM_ADDR GetFinalReceiveAddr(uint32_t token, uint32_t topk);
    __aicore__ inline void BuildLocalPartial(uint32_t token, uint32_t localMask);
    __aicore__ inline void AccumulateLocalPartial();
    __aicore__ inline void AccumulateFinalReceiveInput(uint32_t bufferIndex, bool localPartial, float weight);
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
    GlobalTensor<uint32_t> serverCommInfoGM_;
    GlobalTensor<float> expertScalesGM_;
    GlobalTensor<float> expandScalesGM_;
    GlobalTensor<float> gatherPartialsGM_;
    GlobalTensor<uint64_t> localTokenReadyGM_;
    GlobalTensor<bool> xActiveMaskGM_;
    GlobalTensor<XType> gammaGM_;
    GlobalTensor<XType> sharedExpertXGM_;
    GlobalTensor<XType> expandOutGlobal_;
    GlobalTensor<XType> residualXGM_;
    GlobalTensor<XType> yOutGlobal_;
    GlobalTensor<float> rstdOutGlobal_;
    TBuf<> scatterInputBuf_;
    TBuf<> scatterPackedBuf_;
    TBuf<> scatterPrefixBuf_;
    TBuf<> scatterCardStatesBuf_;
    TBuf<> scatterCardBasesBuf_;
    TBuf<> scatterRowsBuf_;
    TBuf<> scatterTokenIdsBuf_;
    TBuf<> scatterWeightsBuf_;
    TBuf<> scatterBroadcastWeightsBuf_;
    TBuf<> scatterIndexOffsetsBuf_;
    TBuf<> commQuantResultBuf_;
    TBuf<> receiveInputBuf_;
    TBuf<> receiveSumBuf_;
    LocalTensor<float> receiveSum_;
    TBuf<> receiveResultBuf_;
    TBuf<> receiveDequantFloatBuf_;
    TBuf<> gatherPackedBuf_;
    TBuf<> gatherExpectedMasksBuf_;
    TBuf<> gatherWaitMasksBuf_;
    TBuf<> gatherPendingTokensBuf_;
    TBuf<> receiveFlagsBuf_;
    TBuf<> receiveFlagCompareBuf_;
    TBuf<> receiveReadyFlagsBuf_;
    TBuf<> receivePrefetchFlagsBuf_;
    TBuf<> finalServerIdsBuf_;
    TBuf<> finalExpectedMasksBuf_;
    TBuf<> finalLocalMasksBuf_;
    TBuf<> finalWaitMasksBuf_;
    TBuf<> finalPendingTokensBuf_;
    TBuf<> finalExpertScalesBuf_;
    TBuf<> finalExpertIdsBuf_;
    TBuf<> finalSpecialMasksBuf_;
    TBuf<> oriXFloatBuf_;
    LocalTensor<float> constMix_;
    TBuf<> finalOutputBuf_;
    TBuf<> tokenBuf_;
    TBuf<> rowTmpFloatBuf_;
    TBuf<TPosition::VECCALC> reduceFp32Buf_;
    __gm__ Mc2MoeContext* mc2Context_{nullptr};
    GM_ADDR statusDataSpaceGm_{nullptr};
    GM_ADDR epWindowGM_{nullptr};
    GM_ADDR interServerSendGM_{nullptr};
    GM_ADDR interServerReceiveGM_{nullptr};
    MoeDistributeCombineQuant<ExpandXType, ExpandXType, ExpandIdxType, QuantMode, HasAddRmsNorm> commQuant_;
    Hcomm<CommProtocol::COMM_PROTOCOL_ROCE> hcomm_;

    static constexpr uint32_t INPUT_COMPARE_ROWS_PER_REPEAT = ALIGNED_LEN_256 / sizeof(int32_t);
    static constexpr uint32_t TOKEN_BITMAPS_PER_BLOCK = UB_ALIGN / sizeof(uint32_t);
    static constexpr uint32_t SENDER_TOKENS_PER_WORD = BITS_PER_BYTE * sizeof(uint64_t);
    static constexpr uint32_t WINDOW_BANK_NUM = 2U;
    static constexpr uint32_t GATHER_SUM_BUFFER_NUM = 2U;
    static constexpr uint32_t MAX_PIPELINE_BUFFERS = 8U;
    static constexpr uint32_t PAYLOAD_BLOCK_BYTES = 512U;
    static constexpr uint32_t PAYLOAD_DATA_BYTES = PAYLOAD_BLOCK_BYTES - UB_ALIGN;
    static constexpr uint32_t WEIGHTS_PER_REPEAT = UB_ALIGN / sizeof(float);
    static constexpr uint32_t SCATTER_TOKEN_TILE = 16U;
    static constexpr uint32_t SCATTER_CACHE_ROWS = 8U;
    static constexpr uint32_t SCATTER_CARD_STATE_WORDS = 5U;
    static constexpr uint32_t FLOAT_OVERFLOW_CTRL_BIT = 60U;
    static constexpr uint32_t MAX_INTER_SERVER_SEND_AIV_NUM = 8U;
    static constexpr uint32_t ROCE_WQEBB_BYTES = 64U;
    static constexpr uint32_t ROCE_HCOMM_UB_BYTES = 512U;
    static constexpr uint32_t SENDER_GROUP_COUNT_MIN = 3U;
    static constexpr TEventID GATHER_PARTIAL_EVENT_BASE = 6U;
    TEventID metadataFreeEvent_;
    TEventID metadataReadyEvent_;
    TEventID metadataScalarEvent_;
    TEventID inputFreeEvents_[MAX_PIPELINE_BUFFERS];
    TEventID inputReadyEvents_[MAX_PIPELINE_BUFFERS];
    TEventID receiveFlagFreeEvents_[MAX_PIPELINE_BUFFERS];
    TEventID receiveFlagReadyEvents_[MAX_PIPELINE_BUFFERS];
    uint32_t receiveFlagTokens_[MAX_PIPELINE_BUFFERS];
    uint32_t receiveFlagMasks_[MAX_PIPELINE_BUFFERS];
    uint32_t receiveFlagBufferNum_{0U};
    uint32_t receiveFlagStride_{0U};

    uint32_t serverNum_{0};
    uint64_t roceCombineOffset_{0};
    uint64_t interServerSendBytes_{0};
    uint32_t maxBs_{0};
    uint32_t alignedTokenNumPerRank_{0};
    uint64_t opSequence_{0};
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
    uint32_t scatterIssuedCount_{0};
    uint32_t scatterCompletedCount_{0};
    uint32_t scatterPacketSlot_{0};
    uint32_t scatterBroadcastSlot_{0};
    uint32_t scatterPendingWeightIndex_[MAX_PIPELINE_BUFFERS];
    GM_ADDR scatterPendingAddress_[MAX_PIPELINE_BUFFERS];
    uint32_t scatterBlockCount_{0};
    uint32_t scatterTokenStrideBytes_{0};
    uint32_t commQuantFloatBytes_{0};
    uint32_t receiveInputStrideBytes_{0};
    uint32_t receiveFloatBytes_{0};
    uint32_t scatterPacketBufferNum_{0};
    uint32_t scatterInputBufferNum_{0};
    uint32_t gatherPacketBufferNum_{0};
    uint32_t gatherInputBufferNum_{0};
    uint32_t finalOutputBufferNum_{0};
    uint32_t finalInputBufferNum_{0};
    uint32_t receiveInputBufferNum_{0};
    uint32_t receiveInputSlot_{0};
    uint32_t receiveFlagCompareCount_{0};
    uint32_t interServerBlockCount_{0};
    uint32_t interServerTokenStrideBytes_{0};
    uint32_t roceBatchCapacity_{0U};
    uint32_t senderTokenCount_{0U};
    uint32_t senderTokensPerGroup_{0U};
    uint32_t gatherServerId_{0};
    uint32_t gatherTokenStart_{0};
    uint32_t gatherTokenStride_{1U};
    uint32_t gatherSumBufferIndex_{0U};
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
    uint64_t totalWinSizeEp_{0};
    uint64_t winDataSizeOffsetEp_{0};
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
__aicore__ inline uint32_t MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::GetSharedExpertRank(uint32_t sourceRank,
                                                                                                uint32_t sharedId)
{
    const uint32_t ranksPerSharedExpert = sharedExpertRankNum_ / sharedExpertNum_;
    return sharedId * ranksPerSharedExpert + sourceRank % ranksPerSharedExpert;
}

template <RoceCombineTypeClass>
__aicore__ inline GM_ADDR MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::GetInterServerReceiveAddr(
    uint32_t computeServer, uint32_t token, uint32_t topk)
{
    if (computeServer == epRankId_ / rankSizePerServer_) {
        const uint64_t slot = (static_cast<uint64_t>(computeServer) * maxBs_ + token) * commSlotNum_ + topk;
        return epWindowGM_ + slot * scatterTokenStrideBytes_;
    }
    const uint64_t receiveSlot = enablePreReduce_ ? static_cast<uint64_t>(computeServer) * maxBs_ + token :
                                                    static_cast<uint64_t>(token) * commSlotNum_ + topk;
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
    totalWinSizeEp_ = combineV2Info->totalWinSizeEp;
    scatterPacketBufferNum_ = combineV2Info->roceScatterBufferNum;
    scatterInputBufferNum_ = combineV2Info->roceScatterInputBufferNum;
    gatherPacketBufferNum_ = combineV2Info->roceGatherBufferNum;
    gatherInputBufferNum_ = combineV2Info->roceGatherInputBufferNum;
    finalOutputBufferNum_ = combineV2Info->bufferNum;
    finalInputBufferNum_ = combineV2Info->roceFinalInputBufferNum;
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
    opSequence_ = dataStateBuf.Get<uint64_t>().GetValue(OP_CNT_POSUL);
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
    expandOutGlobal_.SetGlobalBuffer((__gm__ XType*)XOut);
    gatherPartialsGM_.SetGlobalBuffer((__gm__ float*)workspaceGM);
    InitTilingAttrs(combineV2Info);
    InitCommContext(mc2Context);
    const uint64_t assistBitmapOffsetBytes =
        static_cast<uint64_t>(combineV2Info->a) * EXPAND_IDX_INFO * sizeof(int32_t);
    serverCommInfoGM_.SetGlobalBuffer((__gm__ uint32_t*)(expandIdx + assistBitmapOffsetBytes));
    winDataSizeOffsetEp_ = static_cast<uint64_t>(dataState_) * (totalWinSizeEp_ / 2UL);
    epWindowGM_ = GetWinAddrByRankId(epRankId_);
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
    DataCopyPad(x2, residualXGM_[static_cast<uint64_t>(tokenIndex) * axisH_ + tokenOffset], copyExtParams,
                copyPadExtParams);
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
    DataCopyPad(rstdOutGlobal_[static_cast<uint64_t>(tokenIndex) + tokenOffset], reduceBufLocal, copyParams);

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
    DataCopyPad(yOutGlobal_[static_cast<uint64_t>(tokenIndex) * axisH_ + tokenOffset], yLocal, copyExtParams);
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
                         static_cast<uint64_t>(dataState_) * (mc2Context_->roceBufferSize / WINDOW_BANK_NUM) +
                         roceCombineOffset_;
    const uint32_t slotsPerToken = enablePreReduce_ ? 1U : commSlotNum_;
    interServerSendBytes_ = static_cast<uint64_t>(serverNum_) * maxBs_ * slotsPerToken * interServerTokenStrideBytes_;
    interServerReceiveGM_ = interServerSendGM_ + interServerSendBytes_;
    const uint32_t receiveGroups = enablePreReduce_ ? serverNum_ : commSlotNum_;
    const uint64_t receiveBytes = static_cast<uint64_t>(maxBs_) * receiveGroups * interServerTokenStrideBytes_;
    localTokenReadyGM_.SetGlobalBuffer((__gm__ uint64_t*)(interServerReceiveGM_ + receiveBytes));
    localTokenReadyGM_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
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
__aicore__ inline void MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::InitMetadataEvents()
{
    metadataFreeEvent_ = tpipe_->AllocEventID<HardEvent::V_MTE2>();
    metadataReadyEvent_ = tpipe_->AllocEventID<HardEvent::MTE2_V>();
    metadataScalarEvent_ = tpipe_->AllocEventID<HardEvent::MTE2_S>();
}

template <RoceCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::InitInputEvents(uint32_t count)
{
    // Register persistent events so SyncFunc inside quantization cannot borrow an in-flight payload event.
    for (uint32_t slot = 0U; slot < count; ++slot) {
        inputFreeEvents_[slot] = tpipe_->AllocEventID<HardEvent::V_MTE2>();
        inputReadyEvents_[slot] = tpipe_->AllocEventID<HardEvent::MTE2_V>();
        SetFlag<HardEvent::V_MTE2>(inputFreeEvents_[slot]);
    }
}

template <RoceCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::InitScatterBuffers()
{
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_CTRL_BIT, FLOAT_OVERFLOW_CTRL_BIT>(0);
    if (quantizeScatter_) {
        if constexpr (QuantMode == MXFP8_E5M2_COMM_QUANT || QuantMode == MXFP8_E4M3_COMM_QUANT) {
            scatterInputStrideBytes_ = Align256(axisH_) * sizeof(ExpandXType);
        }
        InitCommQuantBuffers();
    }
    const uint32_t inputBufferBytes = scatterInputBufferNum_ * scatterInputStrideBytes_;
    const uint32_t packedBufferBytes = scatterPacketBufferNum_ * scatterTokenStrideBytes_;
    tpipe_->InitBuffer(scatterInputBuf_, inputBufferBytes);
    tpipe_->InitBuffer(scatterPackedBuf_, packedBufferBytes);
    // 每次只更新有效数据；输入及量化结果的尾部保持零，包内 flag 始终为本轮轮次号。
    Duplicate(scatterInputBuf_.Get<uint8_t>(), static_cast<uint8_t>(0), inputBufferBytes);
    Duplicate(scatterPackedBuf_.Get<uint64_t>(), opSequence_, packedBufferBytes / sizeof(uint64_t));
    LocalTensor<int32_t> offsets = scatterIndexOffsetsBuf_.Get<int32_t>();
    CreateVecIndex(offsets, int32_t(0), SCATTER_CACHE_ROWS);
    Muls(offsets, offsets, int32_t(EXPAND_IDX_INFO * sizeof(int32_t)), SCATTER_CACHE_ROWS);
    Adds(offsets, offsets, int32_t(sizeof(int32_t)), SCATTER_CACHE_ROWS);
    scatterIssuedCount_ = 0U;
    scatterCompletedCount_ = 0U;
    scatterPacketSlot_ = 0U;
    scatterBroadcastSlot_ = 0U;
    InitMetadataEvents();
    InitInputEvents(scatterInputBufferNum_);
    for (uint32_t bufferIndex = 0U; bufferIndex < scatterPacketBufferNum_; ++bufferIndex) {
        SetFlag<HardEvent::MTE3_V>(bufferIndex);
    }
}

template <RoceCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::InitScatterCardBuffers(uint32_t cardCount)
{
    const uint32_t cacheWords = cardCount * SCATTER_CACHE_ROWS;
    tpipe_->InitBuffer(scatterPrefixBuf_,
                       Ceil((epWorldSize_ + WEIGHTS_PER_REPEAT) * sizeof(int32_t), UB_ALIGN) * UB_ALIGN);
    tpipe_->InitBuffer(scatterCardStatesBuf_,
                       Ceil(cardCount * SCATTER_CARD_STATE_WORDS * sizeof(uint32_t), UB_ALIGN) * UB_ALIGN);
    tpipe_->InitBuffer(scatterCardBasesBuf_, Ceil(cardCount * sizeof(uint64_t), UB_ALIGN) * UB_ALIGN);
    tpipe_->InitBuffer(scatterRowsBuf_, cacheWords * EXPAND_IDX_INFO * sizeof(int32_t));
    tpipe_->InitBuffer(scatterTokenIdsBuf_, cacheWords * sizeof(int32_t));
    tpipe_->InitBuffer(scatterIndexOffsetsBuf_, SCATTER_CACHE_ROWS * sizeof(int32_t));
    if (sendExpertScales_) {
        tpipe_->InitBuffer(scatterWeightsBuf_, cacheWords * sizeof(float));
        tpipe_->InitBuffer(scatterBroadcastWeightsBuf_, scatterInputBufferNum_ * SCATTER_CACHE_ROWS * UB_ALIGN);
    }
}

template <RoceCombineTypeClass>
__aicore__ inline uint32_t MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::LoadScatterCardPlan(
    uint32_t expert, uint32_t taskBegin, uint32_t taskEnd, uint32_t taskStride, bool remote)
{
    LocalTensor<int32_t> prefixes = scatterPrefixBuf_.Get<int32_t>();
    const uint32_t prefixStart = expert == 0U ? WEIGHTS_PER_REPEAT - 1U : 0U;
    const uint32_t copyStart = expert == 0U ? WEIGHTS_PER_REPEAT : 0U;
    const uint32_t copyCount = epWorldSize_ + (expert == 0U ? 0U : 1U);
    const uint32_t firstBucket = expert == 0U ? 0U : expert * epWorldSize_ - 1U;
    if (expert == 0U) {
        prefixes.SetValue(prefixStart, 0);
    }
    DataCopyPad(prefixes[copyStart], epSendCountGM_[firstBucket],
                DataCopyExtParams{1U, copyCount * uint32_t(sizeof(int32_t)), 0U, 0U, 0U},
                DataCopyPadExtParams<int32_t>{false, 0U, 0U, 0});
    SyncFunc<HardEvent::MTE2_S>();
    const uint32_t localServer = epRankId_ / rankSizePerServer_;
    const uint32_t sourceServers = remote ? serverNum_ - 1U : 1U;
    const uint64_t serverBytes = static_cast<uint64_t>(maxBs_) * commSlotNum_ * scatterTokenStrideBytes_;
    LocalTensor<uint32_t> states = scatterCardStatesBuf_.Get<uint32_t>();
    LocalTensor<uint64_t> bases = scatterCardBasesBuf_.Get<uint64_t>();
    uint32_t cards = 0U;
    for (uint32_t task = taskBegin; task < taskEnd; task += taskStride) {
        const uint32_t peer = task % sourceServers;
        const uint32_t server = remote ? (peer < localServer ? peer : peer + 1U) : localServer;
        const uint32_t card = task / sourceServers;
        const uint32_t rank = server * rankSizePerServer_ + card;
        const uint32_t begin = static_cast<uint32_t>(prefixes.GetValue(prefixStart + rank));
        const uint32_t end = static_cast<uint32_t>(prefixes.GetValue(prefixStart + rank + 1U));
        if (begin == end) {
            continue;
        }
        const uint32_t state = cards * SCATTER_CARD_STATE_WORDS;
        states.SetValue(state, begin);
        states.SetValue(state + 1U, end);
        states.SetValue(state + 2U, begin);
        states.SetValue(state + 3U, 0U);
        states.SetValue(state + 4U, 0U);
        bases.SetValue(cards, reinterpret_cast<uint64_t>(GetWinAddrByRankId(localServer * rankSizePerServer_ + card)) +
                                  server * serverBytes);
        ++cards;
    }
    return cards;
}

template <RoceCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::LoadScatterCardCache(uint32_t card,
                                                                                             ScatterCardState& state)
{
    FlushScatterPending();
    state.cacheBegin = state.row;
    state.cacheCount = AscendC::Std::min(SCATTER_CACHE_ROWS, state.end - state.row);
    LocalTensor<int32_t> rows = scatterRowsBuf_.Get<int32_t>()[card * SCATTER_CACHE_ROWS * EXPAND_IDX_INFO];
    SyncFunc<HardEvent::S_MTE2>();
    SetFlag<HardEvent::V_MTE2>(metadataFreeEvent_);
    WaitFlag<HardEvent::V_MTE2>(metadataFreeEvent_);
    DataCopyPad(rows, expandIdxGM_[static_cast<uint64_t>(state.row) * EXPAND_IDX_INFO],
                DataCopyExtParams{1U, state.cacheCount * EXPAND_IDX_INFO * uint32_t(sizeof(int32_t)), 0U, 0U, 0U},
                DataCopyPadExtParams<int32_t>{false, 0U, 0U, 0});
    if (sendExpertScales_) {
        LocalTensor<float> weights = scatterWeightsBuf_.Get<float>()[card * SCATTER_CACHE_ROWS];
        if (epRankId_ < sharedExpertRankNum_) {
            Duplicate(weights, 1.0f, SCATTER_CACHE_ROWS);
        } else {
            DataCopyPad(weights, expandScalesGM_[state.row],
                        DataCopyExtParams{1U, state.cacheCount * uint32_t(sizeof(float)), 0U, 0U, 0U},
                        DataCopyPadExtParams<float>{false, 0U, 0U, 0.0f});
        }
    }
    SetFlag<HardEvent::MTE2_V>(metadataReadyEvent_);
    WaitFlag<HardEvent::MTE2_V>(metadataReadyEvent_);
    LocalTensor<int32_t> tokens = scatterTokenIdsBuf_.Get<int32_t>()[card * SCATTER_CACHE_ROWS];
    Gather(tokens, rows, scatterIndexOffsetsBuf_.Get<uint32_t>(), 0U, state.cacheCount);
    SyncFunc<HardEvent::V_S>();
}

template <RoceCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::CopyScatterInput(uint32_t row,
                                                                                         uint32_t bufferIndex)
{
    WaitFlag<HardEvent::V_MTE2>(inputFreeEvents_[bufferIndex]);
    LocalTensor<ExpandXType> input =
        scatterInputBuf_.Get<ExpandXType>()[bufferIndex * scatterInputStrideBytes_ / sizeof(ExpandXType)];
    DataCopyPad(input, expandXGM_[static_cast<uint64_t>(row) * axisH_],
                DataCopyExtParams{1U, axisH_ * static_cast<uint32_t>(sizeof(ExpandXType)), 0U, 0U, 0U},
                DataCopyPadExtParams<ExpandXType>{true, 0U, scatterInputRightPadding_, 0});
    SetFlag<HardEvent::MTE2_V>(inputReadyEvents_[bufferIndex]);
}

template <RoceCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::PackScatterToken(
    uint32_t bufferIndex, uint32_t packetIndex, LocalTensor<float>& broadcastWeights, uint32_t weightIndex)
{
    WaitFlag<HardEvent::MTE2_V>(inputReadyEvents_[bufferIndex]);
    LocalTensor<ExpandXType> input =
        scatterInputBuf_.Get<ExpandXType>()[bufferIndex * scatterInputStrideBytes_ / sizeof(ExpandXType)];
    LocalTensor<float> source = input.template ReinterpretCast<float>();
    if (quantizeScatter_) {
        LocalTensor<ExpandXType> quantResult = commQuantResultBuf_.Get<ExpandXType>();
        commQuant_.QuantProcess(quantResult, input);
        source = quantResult.template ReinterpretCast<float>();
        SetFlag<HardEvent::V_MTE2>(inputFreeEvents_[bufferIndex]);
    }
    WaitFlag<HardEvent::MTE3_V>(packetIndex);
    LocalTensor<float> packet = scatterPackedBuf_.Get<float>()[packetIndex * scatterTokenStrideBytes_ / sizeof(float)];
    Copy(packet, source, 64UL, static_cast<uint8_t>(scatterBlockCount_), {1, 1, 16, 15});
    Copy(packet[64], source[64], 56UL, static_cast<uint8_t>(scatterBlockCount_), {1, 1, 16, 15});
    if (!quantizeScatter_) {
        SetFlag<HardEvent::V_MTE2>(inputFreeEvents_[bufferIndex]);
    }
    if (sendExpertScales_) {
        Copy(packet[scatterPackedWeightOffsetBytes_ / sizeof(float)],
             broadcastWeights[weightIndex * WEIGHTS_PER_REPEAT], 1UL, 1U, {1, 1, 8, 8});
    }
    SetFlag<HardEvent::V_MTE3>(packetIndex);
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
__aicore__ inline void MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::FlushScatterPending()
{
    if (scatterCompletedCount_ != scatterIssuedCount_) {
        const uint32_t bufferIndex = scatterCompletedCount_ % scatterInputBufferNum_;
        LocalTensor<float> broadcastWeights;
        if (sendExpertScales_) {
            broadcastWeights = scatterBroadcastWeightsBuf_.Get<float>();
        }
        PackScatterToken(bufferIndex, scatterPacketSlot_, broadcastWeights, scatterPendingWeightIndex_[bufferIndex]);
        SendScatterToken(scatterPacketSlot_, scatterPendingAddress_[bufferIndex]);
        if (++scatterPacketSlot_ == scatterPacketBufferNum_) {
            scatterPacketSlot_ = 0U;
        }
        ++scatterCompletedCount_;
    }
}

template <RoceCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::FinishScatterPipeline()
{
    while (scatterCompletedCount_ != scatterIssuedCount_) {
        FlushScatterPending();
    }
    for (uint32_t bufferIndex = 0U; bufferIndex < scatterInputBufferNum_; ++bufferIndex) {
        WaitFlag<HardEvent::V_MTE2>(inputFreeEvents_[bufferIndex]);
    }
    for (uint32_t bufferIndex = 0U; bufferIndex < scatterPacketBufferNum_; ++bufferIndex) {
        WaitFlag<HardEvent::MTE3_V>(bufferIndex);
    }
    SyncFunc<HardEvent::MTE3_MTE2>();
}

template <RoceCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::SendScatterCardRows(
    uint32_t card, const ScatterCardState& state, uint32_t rowCount)
{
    const uint32_t weightBase = scatterBroadcastSlot_ * SCATTER_CACHE_ROWS;
    if (sendExpertScales_) {
        Brcb(scatterBroadcastWeightsBuf_.Get<float>()[weightBase * WEIGHTS_PER_REPEAT],
             scatterWeightsBuf_.Get<float>()[card * SCATTER_CACHE_ROWS],
             static_cast<uint8_t>(SCATTER_CACHE_ROWS / WEIGHTS_PER_REPEAT), {1, 8});
    }
    LocalTensor<int32_t> rows = scatterRowsBuf_.Get<int32_t>()[card * SCATTER_CACHE_ROWS * EXPAND_IDX_INFO];
    LocalTensor<int32_t> tokens = scatterTokenIdsBuf_.Get<int32_t>()[card * SCATTER_CACHE_ROWS];
    GM_ADDR base = reinterpret_cast<GM_ADDR>(scatterCardBasesBuf_.Get<uint64_t>().GetValue(card));
    const uint32_t begin = state.row - state.cacheBegin;
    for (uint32_t row = 0U; row < rowCount; ++row) {
        const uint32_t index = begin + row;
        const uint32_t token = static_cast<uint32_t>(tokens.GetValue(index));
        const uint32_t topk = static_cast<uint32_t>(rows.GetValue(index * EXPAND_IDX_INFO + 2U));
        const uint64_t slot = static_cast<uint64_t>(token) * commSlotNum_ + topk;
        const uint32_t bufferIndex = scatterIssuedCount_ % scatterInputBufferNum_;
        CopyScatterInput(state.row + row, bufferIndex);
        scatterPendingAddress_[bufferIndex] = base + slot * scatterTokenStrideBytes_;
        scatterPendingWeightIndex_[bufferIndex] = weightBase + index;
        ++scatterIssuedCount_;
        if (scatterInputBufferNum_ == 1U || scatterIssuedCount_ - scatterCompletedCount_ > 1U) {
            FlushScatterPending();
        }
    }
    scatterBroadcastSlot_ = (scatterBroadcastSlot_ + 1U) % scatterInputBufferNum_;
}

template <RoceCombineTypeClass>
__aicore__ inline uint32_t MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::ScatterCardTile(uint32_t card,
                                                                                            uint32_t tokenEnd)
{
    LocalTensor<uint32_t> states = scatterCardStatesBuf_.Get<uint32_t>()[card * SCATTER_CARD_STATE_WORDS];
    ScatterCardState state = {states.GetValue(0U), states.GetValue(1U), states.GetValue(2U), states.GetValue(3U)};
    uint32_t nextToken = maxBs_;
    while (state.row < state.end) {
        if (state.row == state.cacheBegin + state.cacheCount) {
            LoadScatterCardCache(card, state);
        }
        const uint32_t consumed = state.row - state.cacheBegin;
        LocalTensor<int32_t> tokens = scatterTokenIdsBuf_.Get<int32_t>()[card * SCATTER_CACHE_ROWS];
        uint32_t readyEnd = consumed;
        while (readyEnd < state.cacheCount && tokens.GetValue(readyEnd) < static_cast<int32_t>(tokenEnd)) {
            ++readyEnd;
        }
        const uint32_t rowCount = readyEnd - consumed;
        if (rowCount > 0U) {
            SendScatterCardRows(card, state, rowCount);
            state.row += rowCount;
        }
        if (state.row < state.cacheBegin + state.cacheCount) {
            nextToken = static_cast<uint32_t>(
                scatterTokenIdsBuf_.Get<int32_t>().GetValue(card * SCATTER_CACHE_ROWS + state.row - state.cacheBegin));
            break;
        }
    }
    states.SetValue(0U, state.row);
    states.SetValue(2U, state.cacheBegin);
    states.SetValue(3U, state.cacheCount);
    states.SetValue(4U, nextToken);
    return nextToken;
}

template <RoceCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::ScatterCardTiles(uint32_t cardCount)
{
    uint32_t tokenEnd = SCATTER_TOKEN_TILE;
    uint32_t nextToken = 0U;
    LocalTensor<uint32_t> states = scatterCardStatesBuf_.Get<uint32_t>();
    while (nextToken < maxBs_) {
        nextToken = maxBs_;
        for (uint32_t card = 0U; card < cardCount; ++card) {
            uint32_t cardToken = states.GetValue(card * SCATTER_CARD_STATE_WORDS + 4U);
            if (cardToken < tokenEnd) {
                cardToken = ScatterCardTile(card, tokenEnd);
            }
            nextToken = AscendC::Std::min(nextToken, cardToken);
        }
        tokenEnd = (nextToken / SCATTER_TOKEN_TILE + 1U) * SCATTER_TOKEN_TILE;
    }
}

template <RoceCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::ScatterExpertCards(
    uint32_t expert, uint32_t taskBegin, uint32_t taskEnd, uint32_t taskStride, bool remote, bool& active)
{
    if (taskBegin >= taskEnd) {
        return;
    }
    const uint32_t cardCount = LoadScatterCardPlan(expert, taskBegin, taskEnd, taskStride, remote);
    if (cardCount == 0U) {
        return;
    }
    if (!active) {
        InitScatterBuffers();
        active = true;
    }
    ScatterCardTiles(cardCount);
    // Submit the cached tail before replacing its metadata; packet slots remain in flight.
    FlushScatterPending();
}

template <RoceCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::ScatterSourceServersBalanced(uint32_t experts,
                                                                                                     bool remote,
                                                                                                     bool& active)
{
    const uint32_t sourceCards = (remote ? serverNum_ - 1U : 1U) * rankSizePerServer_;
    const uint64_t totalTasks = static_cast<uint64_t>(experts) * sourceCards;
    const uint64_t beginTask = totalTasks * aivId_ / sendAivNum_;
    const uint64_t endTask = totalTasks * (aivId_ + 1U) / sendAivNum_;
    for (uint32_t expert = static_cast<uint32_t>(beginTask / sourceCards);
         expert <= static_cast<uint32_t>((endTask - 1U) / sourceCards); ++expert) {
        const uint64_t expertBase = static_cast<uint64_t>(expert) * sourceCards;
        const uint32_t cardBegin = beginTask > expertBase ? static_cast<uint32_t>(beginTask - expertBase) : 0U;
        const uint64_t expertLimit = expertBase + sourceCards;
        const uint32_t cardEnd = endTask < expertLimit ? static_cast<uint32_t>(endTask - expertBase) : sourceCards;
        ScatterExpertCards(expert, cardBegin, cardEnd, 1U, remote, active);
    }
}

template <RoceCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::ScatterSourceServers(
    const TokenCoreAssignment& assignment, uint32_t experts, bool remote, bool& active)
{
    const uint32_t sourceCards = (remote ? serverNum_ - 1U : 1U) * rankSizePerServer_;
    const uint64_t totalTasks = static_cast<uint64_t>(experts) * sourceCards;
    if (!quantizeScatter_ && totalTasks > sendAivNum_) {
        ScatterSourceServersBalanced(experts, remote, active);
        return;
    }
    for (uint32_t expert = assignment.groupStart; expert < experts; expert += assignment.groupStride) {
        ScatterExpertCards(expert, assignment.lane, sourceCards, assignment.laneCount, remote, active);
    }
}

template <RoceCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::ScatterTokenToLocal()
{
    const uint32_t experts = epRankId_ < sharedExpertRankNum_ ? 1U : moeExpertPerRankNum_;
    const TokenCoreAssignment assignment = GetTokenCoreAssignment(aivId_, sendAivNum_, experts);
    InitScatterLayout();
    const uint32_t phaseCards = AscendC::Std::max(serverNum_ - 1U, 1U) * rankSizePerServer_;
    InitScatterCardBuffers(phaseCards);
    bool active = false;
    // Both source-server phases share input/packet slots until the final UB handoff.
    ScatterSourceServers(assignment, experts, true, active);
    ScatterSourceServers(assignment, experts, false, active);
    if (active) {
        FinishScatterPipeline();
    }
}

template <RoceCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::InitReceiveFlagBuffers(uint32_t blockCount)
{
    InitMetadataEvents();
    const uint32_t flagsBytes = Ceil(blockCount * sizeof(uint64_t), ALIGNED_LEN_256) * ALIGNED_LEN_256;
    receiveFlagCompareCount_ = flagsBytes / sizeof(uint64_t);
    const uint32_t flagCompareBytes = Ceil(receiveFlagCompareCount_ / BITS_PER_BYTE, UB_ALIGN) * UB_ALIGN;
    const uint32_t selectedFlagsBytes = Ceil(blockCount * sizeof(uint32_t), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(receiveFlagsBuf_, flagsBytes);
    tpipe_->InitBuffer(receiveFlagCompareBuf_, flagCompareBytes);
    tpipe_->InitBuffer(receiveReadyFlagsBuf_, selectedFlagsBytes);
}

template <RoceCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::InitReceiveBuffers(uint32_t blockCount,
                                                                                           bool clearResult,
                                                                                           uint32_t sumBufferCount)
{
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_CTRL_BIT, FLOAT_OVERFLOW_CTRL_BIT>(0);
    InitReceiveFlagBuffers(blockCount);
    receiveInputStrideBytes_ = blockCount * PAYLOAD_DATA_BYTES;
    receiveFloatBytes_ = Ceil(axisH_ * sizeof(float), ALIGNED_LEN_256 * INT8_DIVIVE) * ALIGNED_LEN_256 * INT8_DIVIVE;
    const uint32_t inputBytes = receiveInputBufferNum_ * receiveInputStrideBytes_;
    uint32_t resultBytes = AscendC::Std::max(interServerBlockCount_ * PAYLOAD_DATA_BYTES, scatterInputStrideBytes_);
    if constexpr (QuantMode == MXFP8_E5M2_COMM_QUANT || QuantMode == MXFP8_E4M3_COMM_QUANT) {
        if (enablePreReduce_) {
            const uint32_t mxInputBytes = Align256(axisH_) * sizeof(ExpandXType);
            resultBytes = AscendC::Std::max(resultBytes, mxInputBytes);
        }
    }
    const uint32_t dequantFloatBytes = QuantMode == UNQUANT ? receiveFloatBytes_ : 0U;
    tpipe_->InitBuffer(receiveInputBuf_, inputBytes);
    tpipe_->InitBuffer(receiveSumBuf_, sumBufferCount * receiveFloatBytes_);
    receiveSum_ = receiveSumBuf_.Get<float>();
    tpipe_->InitBuffer(receiveResultBuf_, resultBytes);
    if constexpr (QuantMode == UNQUANT) {
        tpipe_->InitBuffer(receiveDequantFloatBuf_, dequantFloatBytes);
    }
    if (clearResult) {
        Duplicate(receiveResultBuf_.Get<uint8_t>(), static_cast<uint8_t>(0), resultBytes);
    }
    receiveInputSlot_ = 0U;
    InitInputEvents(receiveInputBufferNum_);
}

template <RoceCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::InitReceiveFlagPrefetch(uint32_t tokenCount,
                                                                                                uint32_t blockCount)
{
    receiveFlagBufferNum_ = AscendC::Std::min(receiveInputBufferNum_, tokenCount);
    receiveFlagStride_ = Ceil(blockCount * sizeof(uint64_t), UB_ALIGN) * UB_ALIGN / sizeof(uint64_t);
    tpipe_->InitBuffer(receivePrefetchFlagsBuf_,
                       receiveFlagBufferNum_ * commSlotNum_ * receiveFlagStride_ * sizeof(uint64_t));
    for (uint32_t slot = 0U; slot < receiveFlagBufferNum_; ++slot) {
        receiveFlagFreeEvents_[slot] = tpipe_->AllocEventID<HardEvent::S_MTE2>();
        receiveFlagReadyEvents_[slot] = tpipe_->AllocEventID<HardEvent::MTE2_S>();
        SetFlag<HardEvent::S_MTE2>(receiveFlagFreeEvents_[slot]);
    }
}

template <RoceCombineTypeClass>
template <bool Final>
__aicore__ inline void MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::PrefetchReceiveFlags(uint32_t token,
                                                                                             uint32_t slot)
{
    WaitFlag<HardEvent::S_MTE2>(receiveFlagFreeEvents_[slot]);
    LocalTensor<uint32_t> waits = Final ? finalWaitMasksBuf_.Get<uint32_t>() : gatherWaitMasksBuf_.Get<uint32_t>();
    uint32_t mask = waits.GetValue(token);
    receiveFlagTokens_[slot] = token;
    receiveFlagMasks_[slot] = mask;
    LocalTensor<uint64_t> flags = receivePrefetchFlagsBuf_.Get<uint64_t>()[slot * commSlotNum_ * receiveFlagStride_];
    while (mask != 0U) {
        const uint32_t bit = static_cast<uint32_t>(ScalarGetSFFValue<1>(static_cast<uint64_t>(mask)));
        mask &= mask - 1U;
        GM_ADDR address;
        uint32_t blocks = scatterBlockCount_;
        if constexpr (Final) {
            address = GetFinalReceiveAddr(token, bit);
            if (GetFinalComputeServer(token, bit) != epRankId_ / rankSizePerServer_) {
                blocks = interServerBlockCount_;
            }
        } else {
            address = GetGatherTokenAddr(token) + static_cast<uint64_t>(bit) * scatterTokenStrideBytes_;
        }
        GlobalTensor<uint64_t> source;
        source.SetGlobalBuffer((__gm__ uint64_t*)(address + PAYLOAD_DATA_BYTES));
        source.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        DataCopyPad<uint64_t, PaddingMode::Compact>(flags[bit * receiveFlagStride_], source,
                                                    DataCopyExtParams{static_cast<uint16_t>(blocks), sizeof(uint64_t),
                                                                      PAYLOAD_BLOCK_BYTES - sizeof(uint64_t), 0U, 0U},
                                                    DataCopyPadExtParams<uint64_t>{false, 0U, 0U, 0U});
    }
    SetFlag<HardEvent::MTE2_S>(receiveFlagReadyEvents_[slot]);
}

template <RoceCombineTypeClass>
template <bool Final>
__aicore__ inline uint32_t MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::ConsumeReceiveFlags(uint32_t slot)
{
    WaitFlag<HardEvent::MTE2_S>(receiveFlagReadyEvents_[slot]);
    uint32_t ready = 0U;
    uint32_t mask = receiveFlagMasks_[slot];
    LocalTensor<uint64_t> flags = receivePrefetchFlagsBuf_.Get<uint64_t>()[slot * commSlotNum_ * receiveFlagStride_];
    while (mask != 0U) {
        const uint32_t bit = static_cast<uint32_t>(ScalarGetSFFValue<1>(static_cast<uint64_t>(mask)));
        mask &= mask - 1U;
        uint32_t blocks = scatterBlockCount_;
        if constexpr (Final) {
            if (GetFinalComputeServer(receiveFlagTokens_[slot], bit) != epRankId_ / rankSizePerServer_) {
                blocks = interServerBlockCount_;
            }
        }
        uint32_t block = 0U;
        while (block < blocks && flags.GetValue(bit * receiveFlagStride_ + block) == opSequence_) {
            ++block;
        }
        if (block == blocks) {
            ready |= 1U << bit;
        }
    }
    SetFlag<HardEvent::S_MTE2>(receiveFlagFreeEvents_[slot]);
    return ready;
}

template <RoceCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::FinishReceiveFlagPrefetch()
{
    for (uint32_t slot = 0U; slot < receiveFlagBufferNum_; ++slot) {
        WaitFlag<HardEvent::S_MTE2>(receiveFlagFreeEvents_[slot]);
        tpipe_->ReleaseEventID<HardEvent::S_MTE2>(receiveFlagFreeEvents_[slot]);
        tpipe_->ReleaseEventID<HardEvent::MTE2_S>(receiveFlagReadyEvents_[slot]);
    }
}

template <RoceCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::InitGatherBuffers(uint32_t mapCount)
{
    if (!enablePreReduce_) {
        InitReceiveFlagBuffers(scatterBlockCount_);
        tpipe_->InitBuffer(gatherPackedBuf_, gatherPacketBufferNum_ * interServerTokenStrideBytes_);
        for (uint32_t bufferIndex = 0U; bufferIndex < gatherPacketBufferNum_; ++bufferIndex) {
            SetFlag<HardEvent::MTE3_MTE2>(bufferIndex);
        }
        return;
    }
    InitCommQuantBuffers();
    receiveInputBufferNum_ = gatherInputBufferNum_;
    InitReceiveBuffers(scatterBlockCount_, true, GATHER_SUM_BUFFER_NUM);
    gatherSumBufferIndex_ = 0U;
    const uint32_t packedBytes = gatherPacketBufferNum_ * interServerTokenStrideBytes_;
    tpipe_->InitBuffer(gatherPackedBuf_, packedBytes);
    Duplicate(gatherPackedBuf_.Get<uint64_t>(), opSequence_, packedBytes / sizeof(uint64_t));
    for (uint32_t bufferIndex = 0U; bufferIndex < gatherPacketBufferNum_; ++bufferIndex) {
        SetFlag<HardEvent::MTE3_V>(bufferIndex);
    }
    for (uint32_t bufferIndex = 0U; bufferIndex < GATHER_SUM_BUFFER_NUM; ++bufferIndex) {
        const TEventID partialEvent = GATHER_PARTIAL_EVENT_BASE + bufferIndex;
        SetFlag<HardEvent::MTE3_V>(partialEvent);
        SetFlag<HardEvent::MTE3_MTE2>(partialEvent);
    }
    InitReceiveFlagPrefetch(mapCount, scatterBlockCount_);
}

template <RoceCombineTypeClass>
template <bool UseVector>
__aicore__ inline bool MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::IsReceiveSlotReady(GM_ADDR slotAddress,
                                                                                           uint32_t blockCount)
{
    GlobalTensor<uint64_t> flagsGM;
    flagsGM.SetGlobalBuffer((__gm__ uint64_t*)(slotAddress + PAYLOAD_DATA_BYTES));
    flagsGM.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    LocalTensor<uint64_t> flags = receiveFlagsBuf_.Get<uint64_t>();
    if constexpr (UseVector) {
        SetFlag<HardEvent::V_MTE2>(metadataFreeEvent_);
        WaitFlag<HardEvent::V_MTE2>(metadataFreeEvent_);
    }
    DataCopyPad<uint64_t, PaddingMode::Compact>(flags, flagsGM,
                                                DataCopyExtParams{static_cast<uint16_t>(blockCount), sizeof(uint64_t),
                                                                  PAYLOAD_BLOCK_BYTES - sizeof(uint64_t), 0U, 0U},
                                                DataCopyPadExtParams<uint64_t>{false, 0U, 0U, 0U});
    if constexpr (UseVector) {
        SetFlag<HardEvent::MTE2_V>(metadataReadyEvent_);
        WaitFlag<HardEvent::MTE2_V>(metadataReadyEvent_);
        LocalTensor<uint8_t> compare = receiveFlagCompareBuf_.Get<uint8_t>();
        Compares(compare, flags, opSequence_, CMPMODE::EQ, receiveFlagCompareCount_);
        uint64_t readyCount = 0U;
        GatherMask(receiveReadyFlagsBuf_.Get<uint32_t>(), flags.ReinterpretCast<uint32_t>(),
                   compare.ReinterpretCast<uint32_t>(), true, blockCount, {1, 1, 0, 0}, readyCount);
        SyncFunc<HardEvent::V_S>();
        return readyCount == blockCount;
    } else {
        SetFlag<HardEvent::MTE2_S>(metadataScalarEvent_);
        WaitFlag<HardEvent::MTE2_S>(metadataScalarEvent_);
        for (uint32_t block = 0U; block < blockCount; ++block) {
            if (flags.GetValue(block) != opSequence_) {
                return false;
            }
        }
        return true;
    }
}

template <RoceCombineTypeClass>
template <bool IsIntraServer>
__aicore__ inline void MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::CopyReceiveInput(GM_ADDR slotAddress,
                                                                                         uint32_t bufferIndex)
{
    WaitFlag<HardEvent::V_MTE2>(inputFreeEvents_[bufferIndex]);
    GlobalTensor<ExpandXType> inputGM;
    inputGM.SetGlobalBuffer((__gm__ ExpandXType*)slotAddress);
    LocalTensor<ExpandXType> input =
        receiveInputBuf_.Get<ExpandXType>()[bufferIndex * receiveInputStrideBytes_ / sizeof(ExpandXType)];
    const uint32_t blockCount = IsIntraServer ? scatterBlockCount_ : interServerBlockCount_;
    DataCopyPad(input, inputGM,
                DataCopyExtParams{static_cast<uint16_t>(blockCount), PAYLOAD_DATA_BYTES, UB_ALIGN, 0U, 0U},
                DataCopyPadExtParams<ExpandXType>{false, 0U, 0U, 0});
    SetFlag<HardEvent::MTE2_V>(inputReadyEvents_[bufferIndex]);
}

template <RoceCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::ScaleReceiveInput(
    LocalTensor<float>& values, LocalTensor<ExpandXType>& input)
{
    __ubuf__ float* data = (__ubuf__ float*)values.GetPhyAddr();
    __ubuf__ float* weight = (__ubuf__ float*)input.GetPhyAddr() + scatterWeightOffsetBytes_ / sizeof(float);
    const uint32_t elementsPerRepeat = GetVecLen() / sizeof(float);
    const uint16_t repeatTimes = static_cast<uint16_t>(Ceil(axisH_, elementsPerRepeat));
    uint32_t remaining = axisH_;
    __VEC_SCOPE__
    {
        Reg::RegTensor<float> weightReg;
        Reg::RegTensor<float> valueReg;
        Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(weightReg, weight);
        for (uint16_t repeat = 0U; repeat < repeatTimes; ++repeat) {
            Reg::MaskReg mask = Reg::UpdateMask<float>(remaining);
            Reg::DataCopy<float, Reg::LoadDist::DIST_NORM>(valueReg, data + repeat * elementsPerRepeat);
            Reg::Mul(valueReg, valueReg, weightReg, mask);
            Reg::DataCopy(data + repeat * elementsPerRepeat, valueReg, mask);
        }
    }
}

template <RoceCombineTypeClass>
template <bool IsIntraServer, bool Initialize>
__aicore__ inline void MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::AccumulateReceiveInput(uint32_t bufferIndex,
                                                                                               float expertScale)
{
    WaitFlag<HardEvent::MTE2_V>(inputReadyEvents_[bufferIndex]);
    LocalTensor<ExpandXType> input =
        receiveInputBuf_.Get<ExpandXType>()[bufferIndex * receiveInputStrideBytes_ / sizeof(ExpandXType)];
    LocalTensor<float> sum = receiveSum_;
    const bool applyExpertScale = hasExpertScales_ && (IsIntraServer || !enablePreReduce_);
    if constexpr (QuantMode > UNQUANT && !IsIntraServer) {
        const uint32_t quantScaleNum = commQuant_.quantScaleNum_;
        LocalTensor<float> temporary = commQuant_.winTpSendCountFloatTensor_;
        LocalTensor<ExpandXType> output = receiveResultBuf_.Get<ExpandXType>();
        if (applyExpertScale) {
            commQuant_.DeQuantProcess(input, output, temporary, sum, expertScale);
        } else {
            commQuant_.DeQuantProcessWithoutExpertScale(input, output, temporary, sum);
        }
        commQuant_.axisH_ = axisH_;
        commQuant_.quantScaleNum_ = quantScaleNum;
    } else {
        LocalTensor<float> temporary;
        if constexpr (Initialize) {
            temporary = sum;
        } else if constexpr (QuantMode > UNQUANT) {
            // 框内原始输入的 FP32 转换复用后续量化工作区。
            temporary = commQuant_.winTpSendCountFloatTensor_;
        } else {
            temporary = receiveDequantFloatBuf_.Get<float>();
        }
        Cast(temporary, input, RoundMode::CAST_NONE, axisH_);
        if (applyExpertScale) {
            if constexpr (IsIntraServer) {
                ScaleReceiveInput(temporary, input);
            } else {
                Muls(temporary, temporary, expertScale, axisH_);
            }
        }
        if constexpr (Initialize) {
            // Preserve the original first addition to +0, including signed-zero canonicalization.
            Adds(sum, temporary, 0.0f, axisH_);
        } else {
            Add(sum, sum, temporary, axisH_);
        }
    }
    SetFlag<HardEvent::V_MTE2>(inputFreeEvents_[bufferIndex]);
}

template <RoceCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::LoadGatherExpectedMasks(uint32_t mapCount)
{
    const uint64_t offset =
        static_cast<uint64_t>(gatherServerId_) * (alignedTokenNumPerRank_ + UB_ALIGN / sizeof(uint32_t)) +
        UB_ALIGN / sizeof(uint32_t);
    SetFlag<HardEvent::V_MTE2>(metadataFreeEvent_);
    WaitFlag<HardEvent::V_MTE2>(metadataFreeEvent_);
    LocalTensor<uint32_t> expectedMasks = gatherExpectedMasksBuf_.Get<uint32_t>();
    if (gatherTokenStride_ == 1U) {
        DataCopyPad(expectedMasks, serverCommInfoGM_[offset + gatherTokenStart_],
                    DataCopyExtParams{1U, mapCount * uint32_t(sizeof(uint32_t)), 0U, 0U, 0U},
                    DataCopyPadExtParams<uint32_t>{false, 0U, 0U, 0U});
    } else {
        DataCopyPad<uint32_t, PaddingMode::Compact>(
            expectedMasks, serverCommInfoGM_[offset + gatherTokenStart_],
            DataCopyExtParams{static_cast<uint16_t>(mapCount), sizeof(uint32_t),
                              (gatherTokenStride_ - 1U) * uint32_t(sizeof(uint32_t)), 0U, 0U},
            DataCopyPadExtParams<uint32_t>{false, 0U, 0U, 0U});
    }
    SetFlag<HardEvent::MTE2_S>(metadataScalarEvent_);
    WaitFlag<HardEvent::MTE2_S>(metadataScalarEvent_);
}

template <RoceCombineTypeClass>
__aicore__ inline uint32_t MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::BuildGatherReceivePlan(uint32_t mapCount)
{
    const uint32_t mapBytes = Ceil(mapCount * sizeof(uint32_t), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(gatherExpectedMasksBuf_, mapBytes);
    tpipe_->InitBuffer(gatherWaitMasksBuf_, mapBytes);
    tpipe_->InitBuffer(gatherPendingTokensBuf_, mapBytes);
    LocalTensor<uint32_t> expectedMasks = gatherExpectedMasksBuf_.Get<uint32_t>();
    LocalTensor<uint32_t> waitMasks = gatherWaitMasksBuf_.Get<uint32_t>();
    LocalTensor<uint32_t> pendingTokens = gatherPendingTokensBuf_.Get<uint32_t>();
    LoadGatherExpectedMasks(mapCount);
    uint32_t pendingCount = 0U;
    for (uint32_t index = 0U; index < mapCount; ++index) {
        const uint32_t mask = expectedMasks.GetValue(index);
        const uint32_t token = GetGatherTokenId(index);
        if (mask != 0U && token < maxBs_) {
            waitMasks.SetValue(index, mask);
            pendingTokens.SetValue(pendingCount++, index);
        }
    }
    return pendingCount;
}

template <RoceCombineTypeClass>
__aicore__ inline uint32_t MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::GetGatherTokenId(uint32_t mapIndex)
{
    return gatherTokenStart_ + mapIndex * gatherTokenStride_;
}

template <RoceCombineTypeClass>
__aicore__ inline GM_ADDR MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::GetGatherTokenAddr(uint32_t mapIndex)
{
    const uint32_t token = GetGatherTokenId(mapIndex);
    const uint64_t firstSlot = (static_cast<uint64_t>(gatherServerId_) * maxBs_ + token) * commSlotNum_;
    return epWindowGM_ + firstSlot * scatterTokenStrideBytes_;
}

template <RoceCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::AccumulateGatherInput(uint32_t bufferIndex,
                                                                                              bool firstInput)
{
    if constexpr (QuantMode == UNQUANT) {
        if (firstInput) {
            AccumulateReceiveInput<true, true>(bufferIndex);
            return;
        }
    }
    AccumulateReceiveInput<true>(bufferIndex);
}

template <RoceCombineTypeClass>
__aicore__ inline uint64_t MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::GetGatherPartialOffset(uint32_t mapIndex)
{
    const uint64_t slot = static_cast<uint64_t>(gatherServerId_) * maxBs_ + GetGatherTokenId(mapIndex);
    return slot * (receiveFloatBytes_ / sizeof(float));
}

template <RoceCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::PrepareGatherPartial(uint32_t mapIndex,
                                                                                             bool resume)
{
    const TEventID partialEvent = GATHER_PARTIAL_EVENT_BASE + gatherSumBufferIndex_;
    WaitFlag<HardEvent::MTE3_V>(partialEvent);
    WaitFlag<HardEvent::MTE3_MTE2>(partialEvent);
    receiveSum_ = receiveSumBuf_.Get<float>()[gatherSumBufferIndex_ * receiveFloatBytes_ / sizeof(float)];
    if (resume) {
        SyncFunc<HardEvent::MTE3_MTE2>();
        const uint32_t partialBytes = axisH_ * sizeof(float);
        DataCopyPad(receiveSum_, gatherPartialsGM_[GetGatherPartialOffset(mapIndex)],
                    DataCopyExtParams{1U, partialBytes, 0U, 0U, 0U}, DataCopyPadExtParams<float>{false, 0U, 0U, 0.0f});
    } else if constexpr (QuantMode > UNQUANT) {
        Duplicate(receiveSum_, 0.0f, receiveFloatBytes_ / sizeof(float));
    }
}

template <RoceCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::SaveGatherPartial(uint32_t mapIndex)
{
    const TEventID partialEvent = GATHER_PARTIAL_EVENT_BASE + gatherSumBufferIndex_;
    SetFlag<HardEvent::V_MTE3>(partialEvent);
    WaitFlag<HardEvent::V_MTE3>(partialEvent);
    const uint32_t partialBytes = axisH_ * sizeof(float);
    DataCopyPad(gatherPartialsGM_[GetGatherPartialOffset(mapIndex)], receiveSum_,
                DataCopyExtParams{1U, partialBytes, 0U, 0U, 0U});
}

template <RoceCombineTypeClass>
__aicore__ inline bool MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::AccumulateReadyGatherToken(uint32_t mapIndex,
                                                                                                   uint32_t& packetSlot,
                                                                                                   uint32_t readyMask)
{
    LocalTensor<uint32_t> waitMasks = gatherWaitMasksBuf_.Get<uint32_t>();
    uint32_t remaining = waitMasks.GetValue(mapIndex);
    bool firstInput = remaining == gatherExpectedMasksBuf_.Get<uint32_t>().GetValue(mapIndex);
    GM_ADDR tokenAddress = GetGatherTokenAddr(mapIndex);
    uint32_t issuedCount = 0U;
    uint32_t consumedCount = 0U;
    while (remaining != 0U) {
        const uint32_t topk = static_cast<uint32_t>(ScalarGetSFFValue<1>(static_cast<uint64_t>(remaining)));
        GM_ADDR input = tokenAddress + static_cast<uint64_t>(topk) * scatterTokenStrideBytes_;
        if ((readyMask & (1U << topk)) == 0U) {
            break;
        }
        if (issuedCount == 0U) {
            PrepareGatherPartial(mapIndex, !firstInput);
        }
        CopyReceiveInput<true>(input, (receiveInputSlot_ + issuedCount) % receiveInputBufferNum_);
        ++issuedCount;
        if (receiveInputBufferNum_ == 1U || issuedCount - consumedCount > 1U) {
            AccumulateGatherInput((receiveInputSlot_ + consumedCount++) % receiveInputBufferNum_, firstInput);
            firstInput = false;
        }
        remaining &= remaining - 1U;
    }
    if (issuedCount == 0U) {
        return false;
    }
    while (consumedCount != issuedCount) {
        AccumulateGatherInput((receiveInputSlot_ + consumedCount++) % receiveInputBufferNum_, firstInput);
        firstInput = false;
    }
    receiveInputSlot_ = (receiveInputSlot_ + issuedCount) % receiveInputBufferNum_;
    FinishGatherToken(mapIndex, remaining, packetSlot);
    return remaining == 0U;
}

template <RoceCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::FinishGatherToken(uint32_t mapIndex,
                                                                                          uint32_t remaining,
                                                                                          uint32_t& packetSlot)
{
    LocalTensor<uint32_t> waitMasks = gatherWaitMasksBuf_.Get<uint32_t>();
    waitMasks.SetValue(mapIndex, remaining);
    if (remaining == 0U) {
        WriteGatherResult(gatherServerId_, GetGatherTokenId(mapIndex), packetSlot);
        if (++packetSlot == gatherPacketBufferNum_) {
            packetSlot = 0U;
        }
    } else {
        SaveGatherPartial(mapIndex);
    }
    const TEventID partialEvent = GATHER_PARTIAL_EVENT_BASE + gatherSumBufferIndex_;
    SetFlag<HardEvent::MTE3_V>(partialEvent);
    SetFlag<HardEvent::MTE3_MTE2>(partialEvent);
    gatherSumBufferIndex_ = (gatherSumBufferIndex_ + 1U) % GATHER_SUM_BUFFER_NUM;
}

template <RoceCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::WriteGatherResult(uint32_t sourceServer,
                                                                                          uint32_t token,
                                                                                          uint32_t bufferIndex)
{
    LocalTensor<ExpandXType> result = receiveResultBuf_.Get<ExpandXType>();
    Cast(result, receiveSum_, RoundMode::CAST_RINT, axisH_);
    LocalTensor<float> source = result.template ReinterpretCast<float>();
    if constexpr (QuantMode > UNQUANT) {
        LocalTensor<ExpandXType> quantResult = commQuantResultBuf_.Get<ExpandXType>();
        commQuant_.QuantProcess(quantResult, result);
        source = quantResult.template ReinterpretCast<float>();
    }
    const TEventID packetEvent = bufferIndex;
    WaitFlag<HardEvent::MTE3_V>(packetEvent);
    LocalTensor<float> packet =
        gatherPackedBuf_.Get<float>()[bufferIndex * interServerTokenStrideBytes_ / sizeof(float)];
    Copy(packet, source, 64UL, static_cast<uint8_t>(interServerBlockCount_), {1, 1, 16, 15});
    Copy(packet[64], source[64], 56UL, static_cast<uint8_t>(interServerBlockCount_), {1, 1, 16, 15});
    SetFlag<HardEvent::V_MTE3>(packetEvent);
    WaitFlag<HardEvent::V_MTE3>(packetEvent);
    const uint64_t slot = static_cast<uint64_t>(sourceServer) * maxBs_ + token;
    GlobalTensor<float> outputGM;
    outputGM.SetGlobalBuffer((__gm__ float*)(interServerSendGM_ + slot * interServerTokenStrideBytes_));
    DataCopy(outputGM, packet, interServerTokenStrideBytes_ / sizeof(float));
    PublishLocalTokenReady(sourceServer, token);
    SetFlag<HardEvent::MTE3_V>(packetEvent);
}

template <RoceCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::PublishLocalTokenReady(uint32_t serverId,
                                                                                               uint32_t token)
{
    const uint64_t slot = static_cast<uint64_t>(serverId) * alignedTokenNumPerRank_ + token;
    // The scalar notification must follow payload completion on MTE3.
    SyncFunc<HardEvent::MTE3_S>();
    AscendC::WriteGmByPassDCache(localTokenReadyGM_.GetPhyAddr(slot), opSequence_);
}

template <RoceCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::ForwardTokenPacket(uint64_t slotOffsetBytes,
                                                                                           uint32_t bufferIndex,
                                                                                           uint32_t token,
                                                                                           bool tokenReady)
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
    if (tokenReady) {
        PublishLocalTokenReady(gatherServerId_, token);
    }
    SetFlag<HardEvent::MTE3_MTE2>(bufferIndex);
}

template <RoceCombineTypeClass>
__aicore__ inline bool MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::ForwardReadyTokenRoutes(uint32_t mapIndex,
                                                                                                uint32_t& packetSlot)
{
    LocalTensor<uint32_t> waitMasks = gatherWaitMasksBuf_.Get<uint32_t>();
    uint32_t waitMask = waitMasks.GetValue(mapIndex);
    uint32_t scanMask = waitMask;
    const uint32_t token = GetGatherTokenId(mapIndex);
    const uint64_t firstSlot = (static_cast<uint64_t>(gatherServerId_) * maxBs_ + token) * commSlotNum_;
    while (scanMask != 0U) {
        const uint32_t topk = static_cast<uint32_t>(ScalarGetSFFValue<1>(static_cast<uint64_t>(scanMask)));
        scanMask &= ~(1U << topk);
        const uint64_t slotOffsetBytes = (firstSlot + topk) * interServerTokenStrideBytes_;
        if (IsReceiveSlotReady(epWindowGM_ + slotOffsetBytes, scatterBlockCount_)) {
            waitMask &= ~(1U << topk);
            ForwardTokenPacket(slotOffsetBytes, packetSlot, token, waitMask == 0U);
            if (++packetSlot == gatherPacketBufferNum_) {
                packetSlot = 0U;
            }
        }
    }
    waitMasks.SetValue(mapIndex, waitMask);
    return waitMask == 0U;
}

template <RoceCombineTypeClass>
template <bool Final>
__aicore__ inline bool MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::ProcessReadyToken(uint32_t token,
                                                                                          uint32_t readyMask,
                                                                                          uint32_t& resultSlot)
{
    if constexpr (!Final) {
        return AccumulateReadyGatherToken(token, resultSlot, readyMask);
    } else {
        LocalTensor<uint32_t> waits = finalWaitMasksBuf_.Get<uint32_t>();
        const uint32_t remaining = waits.GetValue(token) & ~readyMask;
        waits.SetValue(token, remaining);
        if (remaining != 0U) {
            return false;
        }
        ReduceFinalToken(token);
        WriteFinalOutput(token, resultSlot);
        resultSlot = (resultSlot + 1U) % finalOutputBufferNum_;
        return true;
    }
}

template <RoceCombineTypeClass>
template <bool Final>
__aicore__ inline void MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::ProcessPrefetchedTokens(
    LocalTensor<uint32_t>& pending, uint32_t pendingCount)
{
    uint32_t resultSlot = 0U;
    while (pendingCount != 0U) {
        uint32_t next = 0U;
        uint32_t remaining = 0U;
        const uint32_t depth = AscendC::Std::min(receiveFlagBufferNum_, pendingCount);
        for (; next < depth; ++next) {
            PrefetchReceiveFlags<Final>(pending.GetValue(next), next);
        }
        for (uint32_t done = 0U; done < pendingCount; ++done) {
            const uint32_t slot = done % depth;
            const uint32_t token = receiveFlagTokens_[slot];
            const uint32_t ready = ConsumeReceiveFlags<Final>(slot);
            if (!ProcessReadyToken<Final>(token, ready, resultSlot)) {
                pending.SetValue(remaining++, token);
            }
            if (next < pendingCount) {
                PrefetchReceiveFlags<Final>(pending.GetValue(next++), slot);
            }
        }
        pendingCount = remaining;
    }
    FinishReceiveFlagPrefetch();
}

template <RoceCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::GatherReadyTokens(uint32_t pendingCount)
{
    LocalTensor<uint32_t> pendingTokens = gatherPendingTokensBuf_.Get<uint32_t>();
    if (enablePreReduce_) {
        ProcessPrefetchedTokens<false>(pendingTokens, pendingCount);
        return;
    }
    uint32_t cursor = 0U;
    uint32_t packetSlot = 0U;
    while (pendingCount != 0U) {
        const uint32_t mapIndex = pendingTokens.GetValue(cursor);
        if (ForwardReadyTokenRoutes(mapIndex, packetSlot)) {
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
__aicore__ inline void MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::InitSenderFlagBuffers()
{
    InitMetadataEvents();
    receiveFlagCompareCount_ = alignedTokenNumPerRank_;
    tpipe_->InitBuffer(receiveFlagsBuf_, receiveFlagCompareCount_ * sizeof(uint64_t));
    tpipe_->InitBuffer(receiveFlagCompareBuf_, Ceil(receiveFlagCompareCount_ / BITS_PER_BYTE, UB_ALIGN) * UB_ALIGN);
}

template <RoceCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::LoadSenderFlags(uint32_t serverId)
{
    SetFlag<HardEvent::V_MTE2>(metadataFreeEvent_);
    WaitFlag<HardEvent::V_MTE2>(metadataFreeEvent_);
    LocalTensor<uint64_t> flags = receiveFlagsBuf_.Get<uint64_t>();
    const uint64_t offset = static_cast<uint64_t>(serverId) * receiveFlagCompareCount_;
    DataCopy(flags, localTokenReadyGM_[offset], receiveFlagCompareCount_);
    SetFlag<HardEvent::MTE2_V>(metadataReadyEvent_);
    WaitFlag<HardEvent::MTE2_V>(metadataReadyEvent_);
}

template <RoceCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::CompareSenderFlags()
{
    SyncFunc<HardEvent::S_V>();
    LocalTensor<uint64_t> flags = receiveFlagsBuf_.Get<uint64_t>();
    Compares(receiveFlagCompareBuf_.Get<uint8_t>(), flags, opSequence_, CMPMODE::EQ, receiveFlagCompareCount_);
    SyncFunc<HardEvent::V_S>();
}

template <RoceCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::InitSenderPendingBitmap(uint32_t pendingCount)
{
    LocalTensor<uint64_t> pendingBitmap = gatherExpectedMasksBuf_.Get<uint64_t>();
    const uint32_t wordCount = Ceil(maxBs_, SENDER_TOKENS_PER_WORD);
    for (uint32_t word = 0U; word < wordCount; ++word) {
        pendingBitmap.SetValue(word, enablePreReduce_ ? ~uint64_t{0} : 0UL);
    }
    if (enablePreReduce_) {
        senderTokenCount_ = pendingCount;
        const uint32_t maxTokensPerGroup = AscendC::Std::max(1U, pendingCount / SENDER_GROUP_COUNT_MIN);
        const uint32_t groupCount = Ceil(pendingCount, maxTokensPerGroup);
        // Balance valid tokens within max(1, floor(N/3)); holes do not count and the tail may be shorter.
        senderTokensPerGroup_ = Ceil(pendingCount, groupCount);
        return;
    }
    LocalTensor<uint32_t> pendingTokens = gatherPendingTokensBuf_.Get<uint32_t>();
    for (uint32_t index = 0U; index < pendingCount; ++index) {
        const uint32_t token = pendingTokens.GetValue(index);
        const uint32_t word = token / SENDER_TOKENS_PER_WORD;
        const uint64_t bit = uint64_t{1} << (token % SENDER_TOKENS_PER_WORD);
        pendingBitmap.SetValue(word, pendingBitmap.GetValue(word) | bit);
    }
}

template <RoceCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::WriteSenderPacket(
    BatchHandle<ChannelHandle>& batch, GM_ADDR destination, GM_ADDR source, bool lastPacket)
{
    if (lastPacket) {
        hcomm_.WriteNbi<SENDER_LAST_WRITE_CFG>(batch, destination, source, interServerTokenStrideBytes_);
    } else {
        hcomm_.WriteNbi<SENDER_WRITE_CFG>(batch, destination, source, interServerTokenStrideBytes_);
    }
}

template <RoceCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::SendTokenRoutes(
    BatchHandle<ChannelHandle>& batch, GM_ADDR remoteReceive, uint32_t serverId, uint32_t token, uint32_t& batchCount,
    uint32_t remainingTokens)
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
        const uint64_t receiveSlot = static_cast<uint64_t>(token) * commSlotNum_ + slot;
        remainingMask &= ~(1U << slot);
        GM_ADDR destination = remoteReceive + receiveSlot * interServerTokenStrideBytes_;
        // Readiness can reorder tokens and routes: signal the last actual Write, not token BS-1.
        WriteSenderPacket(batch, destination, source, remainingTokens == 1U && remainingMask == 0U);
        if (++batchCount == roceBatchCapacity_) {
            hcomm_.BatchCommit(batch);
            batchCount = 0U;
        }
    }
}

template <RoceCombineTypeClass>
__aicore__ inline uint32_t MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::SendReadySnapshot(
    BatchHandle<ChannelHandle>& batch, GM_ADDR remoteReceive, uint32_t serverId, uint32_t pendingCount)
{
    LocalTensor<uint64_t> pendingBitmap = gatherExpectedMasksBuf_.Get<uint64_t>();
    LocalTensor<uint64_t> readyBitmap = receiveFlagCompareBuf_.Get<uint64_t>();
    uint32_t batchCount = 0U;
    const uint32_t wordCount = Ceil(maxBs_, SENDER_TOKENS_PER_WORD);
    for (uint32_t word = 0U; word < wordCount && pendingCount != 0U; ++word) {
        uint64_t remaining = pendingBitmap.GetValue(word);
        if (remaining == 0UL) {
            continue;
        }
        uint64_t ready = readyBitmap.GetValue(word) & remaining;
        if (ready == 0UL) {
            continue;
        }
        while (ready != 0UL) {
            const uint32_t bit = static_cast<uint32_t>(ScalarGetSFFValue<1>(ready));
            const uint32_t token = word * SENDER_TOKENS_PER_WORD + bit;
            SendTokenRoutes(batch, remoteReceive, serverId, token, batchCount, pendingCount);
            --pendingCount;
            remaining &= ~(uint64_t{1} << bit);
            ready &= ready - 1UL;
        }
        pendingBitmap.SetValue(word, remaining);
    }
    if (batchCount != 0U) {
        hcomm_.BatchCommit(batch);
    }
    return pendingCount;
}

template <RoceCombineTypeClass>
__aicore__ inline bool MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::IsSenderGroupReady(uint32_t begin,
                                                                                           uint32_t count)
{
    LocalTensor<uint32_t> tokens = gatherPendingTokensBuf_.Get<uint32_t>();
    LocalTensor<uint64_t> ready = receiveFlagCompareBuf_.Get<uint64_t>();
    for (uint32_t index = 0U; index < count; ++index) {
        const uint32_t token = tokens.GetValue(begin + index);
        if ((ready.GetValue(token / SENDER_TOKENS_PER_WORD) & (uint64_t{1} << (token % SENDER_TOKENS_PER_WORD))) ==
            0U) {
            return false;
        }
    }
    return true;
}

template <RoceCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::WriteSenderGroup(
    BatchHandle<ChannelHandle>& batch, GM_ADDR remoteReceive, uint32_t serverId, uint32_t begin, uint32_t count,
    bool lastGroup)
{
    LocalTensor<uint32_t> tokens = gatherPendingTokensBuf_.Get<uint32_t>();
    const uint32_t firstToken = tokens.GetValue(begin);
    const uint32_t lastToken = tokens.GetValue(begin + count - 1U);
    const uint32_t length = (lastToken - firstToken + 1U) * interServerTokenStrideBytes_;
    const uint64_t sendSlot = static_cast<uint64_t>(serverId) * maxBs_ + firstToken;
    const uint64_t receiveSlot = static_cast<uint64_t>(epRankId_ / rankSizePerServer_) * maxBs_ + firstToken;
    GM_ADDR source = interServerSendGM_ + sendSlot * interServerTokenStrideBytes_;
    GM_ADDR destination = remoteReceive + receiveSlot * interServerTokenStrideBytes_;
    // Both windows retain token IDs. Copy holes within the span; Final only reads routed tokens.
    if (lastGroup) {
        hcomm_.WriteNbi<SENDER_LAST_WRITE_CFG>(batch, destination, source, length);
    } else {
        hcomm_.WriteNbi<SENDER_WRITE_CFG>(batch, destination, source, length);
    }
}

template <RoceCombineTypeClass>
__aicore__ inline uint32_t MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::SendReadyGroups(
    BatchHandle<ChannelHandle>& batch, GM_ADDR remoteReceive, uint32_t serverId, uint32_t pendingCount)
{
    LocalTensor<uint64_t> pending = gatherExpectedMasksBuf_.Get<uint64_t>();
    const uint32_t groupCount = Ceil(senderTokenCount_, senderTokensPerGroup_);
    uint32_t batchCount = 0U;
    for (uint32_t group = 0U; group < groupCount; ++group) {
        const uint32_t word = group / SENDER_TOKENS_PER_WORD;
        const uint64_t bit = uint64_t{1} << (group % SENDER_TOKENS_PER_WORD);
        const uint64_t remaining = pending.GetValue(word);
        const uint32_t begin = group * senderTokensPerGroup_;
        const uint32_t count = AscendC::Std::min(senderTokensPerGroup_, senderTokenCount_ - begin);
        if ((remaining & bit) == 0U || !IsSenderGroupReady(begin, count)) {
            continue;
        }
        WriteSenderGroup(batch, remoteReceive, serverId, begin, count, pendingCount == count);
        if (++batchCount == roceBatchCapacity_) {
            hcomm_.BatchCommit(batch);
            batchCount = 0U;
        }
        pendingCount -= count;
        pending.SetValue(word, remaining & ~bit);
    }
    if (batchCount != 0U) {
        hcomm_.BatchCommit(batch);
    }
    return pendingCount;
}

template <RoceCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::SendReadyTokens(
    BatchHandle<ChannelHandle>& batch, GM_ADDR remoteReceive, uint32_t serverId, uint32_t pendingCount)
{
    while (pendingCount != 0U) {
        LoadSenderFlags(serverId);
        CompareSenderFlags();
        pendingCount = enablePreReduce_ ? SendReadyGroups(batch, remoteReceive, serverId, pendingCount) :
                                          SendReadySnapshot(batch, remoteReceive, serverId, pendingCount);
    }
}

template <RoceCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::ScatterTokenToServer(uint32_t serverId)
{
    const uint32_t localServer = epRankId_ / rankSizePerServer_;
    if (serverId == localServer) {
        return;
    }
    InitSenderFlagBuffers();
    gatherServerId_ = serverId;
    gatherTokenStart_ = 0U;
    gatherTokenStride_ = 1U;
    const uint32_t pendingCount = BuildGatherReceivePlan(alignedTokenNumPerRank_);
    if (pendingCount != 0U) {
        InitSenderPendingBitmap(pendingCount);
        roceBatchCapacity_ = maxBs_;
        TBuf<> hcommBuf;
        TBuf<> batchBuf;
        tpipe_->InitBuffer(hcommBuf, ROCE_HCOMM_UB_BYTES);
        tpipe_->InitBuffer(batchBuf, (roceBatchCapacity_ + 1U) * ROCE_WQEBB_BYTES);
        hcomm_.Init(hcommBuf.Get<uint8_t>(), ROCE_HCOMM_UB_BYTES);
        LocalTensor<uint8_t> storage = batchBuf.Get<uint8_t>();
        const uint32_t padding =
            (ROCE_WQEBB_BYTES - reinterpret_cast<uintptr_t>(storage.GetPhyAddr()) % ROCE_WQEBB_BYTES) %
            ROCE_WQEBB_BYTES;
        GM_ADDR remoteReceive = reinterpret_cast<GM_ADDR>(mc2Context_->roceHcclBuffer_[serverId]) +
                                static_cast<uint64_t>(dataState_) * (mc2Context_->roceBufferSize / WINDOW_BANK_NUM) +
                                roceCombineOffset_ + interServerSendBytes_;
        const uint32_t peer = serverId > localServer ? serverId - 1U : serverId;
        const ChannelHandle channel = mc2Context_->hcommHandle_[peer * ROCE_CHANNEL_PER_SERVER];
        BatchHandle<ChannelHandle> batch = hcomm_.MakeBatchHandle(
            channel, storage[padding], roceBatchCapacity_ * ROCE_WQEBB_BYTES, remoteReceive, interServerSendGM_);
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
__aicore__ inline typename MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::TokenCoreAssignment
MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::GetTokenCoreAssignment(uint32_t core, uint32_t cores, uint32_t groups)
{
    if (cores < groups) {
        return {core, cores, 0U, 1U};
    }
    const uint32_t coresPerGroup = cores / groups;
    const uint32_t extraGroups = cores % groups;
    const uint32_t extraCores = extraGroups * (coresPerGroup + 1U);
    if (core < extraCores) {
        return {core / (coresPerGroup + 1U), groups, core % (coresPerGroup + 1U), coresPerGroup + 1U};
    }
    const uint32_t remainingId = core - extraCores;
    return {extraGroups + remainingId / coresPerGroup, groups, remainingId % coresPerGroup, coresPerGroup};
}

template <RoceCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::GatherServerTokens(uint32_t serverId)
{
    gatherServerId_ = serverId;
    const uint32_t mapCount = Ceil(maxBs_ - gatherTokenStart_, gatherTokenStride_);
    InitGatherBuffers(mapCount);
    const uint32_t pendingCount = BuildGatherReceivePlan(mapCount);
    GatherReadyTokens(pendingCount);
    for (uint32_t bufferIndex = 0U; bufferIndex < gatherPacketBufferNum_; ++bufferIndex) {
        if (enablePreReduce_) {
            WaitFlag<HardEvent::MTE3_V>(bufferIndex);
        } else {
            WaitFlag<HardEvent::MTE3_MTE2>(bufferIndex);
        }
    }
    if (enablePreReduce_) {
        for (uint32_t bufferIndex = 0U; bufferIndex < receiveInputBufferNum_; ++bufferIndex) {
            WaitFlag<HardEvent::V_MTE2>(inputFreeEvents_[bufferIndex]);
        }
        for (uint32_t bufferIndex = 0U; bufferIndex < GATHER_SUM_BUFFER_NUM; ++bufferIndex) {
            const TEventID partialEvent = GATHER_PARTIAL_EVENT_BASE + bufferIndex;
            WaitFlag<HardEvent::MTE3_V>(partialEvent);
            WaitFlag<HardEvent::MTE3_MTE2>(partialEvent);
        }
    }
    tpipe_->Reset();
}

template <RoceCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::GatherTokenFromLocal()
{
    InitInterServerLayout();
    if (aivId_ >= aivNum_ - interServerSendAivNum_) {
        ScatterTokensToRemote();
        return;
    }
    if (serverNum_ == 1U) {
        return;
    }
    const uint32_t localServer = epRankId_ / rankSizePerServer_;
    const uint32_t remoteServers = serverNum_ - 1U;
    const TokenCoreAssignment assignment = GetTokenCoreAssignment(aivId_ - sendAivNum_, gatherAivNum_, remoteServers);
    if (assignment.lane >= maxBs_) {
        return;
    }
    gatherTokenStart_ = assignment.lane;
    gatherTokenStride_ = assignment.laneCount;
    for (uint32_t peer = assignment.groupStart; peer < remoteServers; peer += assignment.groupStride) {
        GatherServerTokens(peer < localServer ? peer : peer + 1U);
    }
}

template <RoceCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::InitFinalQuantBuffers()
{
    if (enablePreReduce_) {
        if constexpr (QuantMode == UNQUANT) {
            tpipe_->InitBuffer(commQuantResultBuf_, Ceil(axisH_ * sizeof(ExpandXType), UB_ALIGN) * UB_ALIGN);
        } else {
            InitCommQuantBuffers();
            commQuant_.SetDeQuantInitParams(commQuant_.fp16CastTensor_, commQuant_.absFloatTensor_,
                                            commQuant_.scaleDupLocalTensor_, commQuant_.reduceMaxFloatTensor_);
        }
        return;
    }
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
        if constexpr (QuantMode == INT8_COMM_QUANT) {
            if (enablePreReduce_) {
                // Pre-reduce completes communication before constant expert computation.
                constMix_ = commQuant_.scaleDupLocalTensor_;
                return;
            }
        }
        TBuf<> constMixBuf;
        tpipe_->InitBuffer(constMixBuf, receiveFloatBytes_);
        constMix_ = constMixBuf.Get<float>();
    }
}

template <RoceCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::InitFinalBuffers(uint32_t tokenCount)
{
    InitFinalQuantBuffers();
    receiveInputBufferNum_ = finalInputBufferNum_;
    InitReceiveBuffers(AscendC::Std::max(scatterBlockCount_, interServerBlockCount_), QuantMode > UNQUANT);
    InitLocalExpertBuffers(tokenCount);
    const uint32_t serverIdsBytes = Ceil(tokenCount * axisK_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    const uint32_t tokenMasksBytes = Ceil(tokenCount * sizeof(uint32_t), UB_ALIGN) * UB_ALIGN;
    const uint32_t pendingTokensBytes = Ceil(tokenCount * sizeof(uint32_t), UB_ALIGN) * UB_ALIGN;
    finalOutputStrideBytes_ = Ceil(axisH_ * sizeof(XType), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(finalServerIdsBuf_, serverIdsBytes);
    tpipe_->InitBuffer(finalExpertIdsBuf_, serverIdsBytes);
    tpipe_->InitBuffer(finalExpectedMasksBuf_, tokenMasksBytes);
    if (enablePreReduce_) {
        tpipe_->InitBuffer(finalLocalMasksBuf_, tokenMasksBytes);
    }
    tpipe_->InitBuffer(finalWaitMasksBuf_, tokenMasksBytes);
    tpipe_->InitBuffer(finalPendingTokensBuf_, pendingTokensBytes);
    if (loadFinalExpertScales_) {
        const uint32_t expertScalesBytes = Ceil(tokenCount * axisK_ * sizeof(float), UB_ALIGN) * UB_ALIGN;
        tpipe_->InitBuffer(finalExpertScalesBuf_, expertScalesBytes);
    }
    tpipe_->InitBuffer(finalOutputBuf_, finalOutputBufferNum_ * finalOutputStrideBytes_);
    for (uint32_t bufferIndex = 0U; bufferIndex < finalOutputBufferNum_; ++bufferIndex) {
        SetFlag<HardEvent::MTE3_V>(bufferIndex);
    }
    InitReceiveFlagPrefetch(tokenCount, AscendC::Std::max(scatterBlockCount_, interServerBlockCount_));
}

template <RoceCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::LoadFinalRoutes(
    uint32_t tokenCount, LocalTensor<ExpandIdxType>& experts, LocalTensor<bool>& activeMask)
{
    const uint32_t routeCount = tokenCount * axisK_;
    const uint32_t routeBytes = routeCount * sizeof(ExpandIdxType);
    DataCopyPad(experts, expertIdsGM_[static_cast<uint64_t>(finalTokenBegin_) * axisK_],
                DataCopyExtParams{1U, routeBytes, 0U, 0U, 0U}, DataCopyPadExtParams<ExpandIdxType>{false, 0U, 0U, 0});
    if (loadFinalExpertScales_) {
        DataCopyPad(finalExpertScalesBuf_.Get<float>(),
                    expertScalesGM_[static_cast<uint64_t>(finalTokenBegin_) * axisK_],
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
    if (!enablePreReduce_ || server == epRankId_ / rankSizePerServer_) {
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
    uint32_t localMask = 0U;
    uint32_t specialMask = 0U;
    for (uint32_t topk = 0U; topk < axisK_; ++topk) {
        const uint32_t route = token * axisK_ + topk;
        const int32_t expert = experts.GetValue(route);
        if (isExpertMask_ && !activeMask.GetValue(route)) {
            continue;
        }
        if (expert >= 0 && expert < static_cast<int32_t>(moeExpertNum_)) {
            const uint32_t server = serverIds.GetValue(route);
            expectedMask |= IncludeFinalReceiveSlot(token, topk, server, lastTokenByServer);
            if (enablePreReduce_ && server == epRankId_ / rankSizePerServer_) {
                localMask |= 1U << topk;
            }
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
        if (enablePreReduce_ && server == epRankId_ / rankSizePerServer_) {
            localMask |= 1U << slot;
        }
    }
    if (enablePreReduce_) {
        finalLocalMasksBuf_.Get<uint32_t>().SetValue(token, localMask);
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
__aicore__ inline uint32_t MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::GetFinalComputeServer(uint32_t token,
                                                                                                  uint32_t topk)
{
    return topk < axisK_ ? finalServerIdsBuf_.Get<int32_t>().GetValue(token * axisK_ + topk) :
                           GetSharedExpertRank(epRankId_, topk - axisK_) / rankSizePerServer_;
}

template <RoceCombineTypeClass>
__aicore__ inline GM_ADDR MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::GetFinalReceiveAddr(uint32_t token,
                                                                                               uint32_t topk)
{
    return GetInterServerReceiveAddr(GetFinalComputeServer(token, topk), finalTokenBegin_ + token, topk);
}

template <RoceCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::BuildLocalPartial(uint32_t token,
                                                                                          uint32_t localMask)
{
    if constexpr (QuantMode > UNQUANT) {
        Duplicate(receiveSumBuf_.Get<float>(), 0.0f, receiveFloatBytes_ / sizeof(float));
    }
    uint32_t issuedCount = 0U;
    uint32_t consumedCount = 0U;
    while (localMask != 0U) {
        const uint32_t topk = static_cast<uint32_t>(ScalarGetSFFValue<1>(static_cast<uint64_t>(localMask)));
        localMask &= ~(1U << topk);
        CopyReceiveInput<true>(GetFinalReceiveAddr(token, topk),
                               (receiveInputSlot_ + issuedCount) % receiveInputBufferNum_);
        ++issuedCount;
        if (receiveInputBufferNum_ == 1U || issuedCount - consumedCount > 1U) {
            AccumulateGatherInput((receiveInputSlot_ + consumedCount) % receiveInputBufferNum_, consumedCount == 0U);
            ++consumedCount;
        }
    }
    while (consumedCount != issuedCount) {
        AccumulateGatherInput((receiveInputSlot_ + consumedCount) % receiveInputBufferNum_, consumedCount == 0U);
        ++consumedCount;
    }
    receiveInputSlot_ = (receiveInputSlot_ + issuedCount) % receiveInputBufferNum_;
    LocalTensor<ExpandXType> partial = commQuantResultBuf_.Get<ExpandXType>();
    if constexpr (QuantMode == UNQUANT) {
        Cast(partial, receiveSumBuf_.Get<float>(), RoundMode::CAST_RINT, axisH_);
    } else {
        LocalTensor<ExpandXType> result = receiveResultBuf_.Get<ExpandXType>();
        Cast(result, receiveSumBuf_.Get<float>(), RoundMode::CAST_RINT, axisH_);
        commQuant_.QuantProcess(partial, result);
    }
    PipeBarrier<PIPE_V>();
}

template <RoceCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::AccumulateLocalPartial()
{
    LocalTensor<ExpandXType> partial = commQuantResultBuf_.Get<ExpandXType>();
    LocalTensor<float> sum = receiveSumBuf_.Get<float>();
    if constexpr (QuantMode == UNQUANT) {
        LocalTensor<float> temporary = receiveDequantFloatBuf_.Get<float>();
        Cast(temporary, partial, RoundMode::CAST_NONE, axisH_);
        Add(sum, sum, temporary, axisH_);
    } else {
        LocalTensor<ExpandXType> result = receiveResultBuf_.Get<ExpandXType>();
        const uint32_t quantScaleNum = commQuant_.quantScaleNum_;
        LocalTensor<float> temporary = commQuant_.winTpSendCountFloatTensor_;
        commQuant_.DeQuantProcessWithoutExpertScale(partial, result, temporary, sum);
        commQuant_.axisH_ = axisH_;
        commQuant_.quantScaleNum_ = quantScaleNum;
    }
}

template <RoceCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::AccumulateFinalReceiveInput(
    uint32_t bufferIndex, bool localPartial, float weight)
{
    if (localPartial) {
        AccumulateLocalPartial();
    } else {
        AccumulateReceiveInput<false>(bufferIndex, weight);
    }
}

template <RoceCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::ReduceFinalReceiveSlots(uint32_t token,
                                                                                                uint32_t remainingMask)
{
    uint32_t issuedCount = 0U;
    uint32_t consumedCount = 0U;
    float pendingWeights[MAX_PIPELINE_BUFFERS];
    bool pendingLocalPartials[MAX_PIPELINE_BUFFERS];
    while (remainingMask != 0U) {
        const uint32_t topk = static_cast<uint32_t>(ScalarGetSFFValue<1>(static_cast<uint64_t>(remainingMask)));
        remainingMask &= ~(1U << topk);
        const bool localPartial =
            enablePreReduce_ && GetFinalComputeServer(token, topk) == epRankId_ / rankSizePerServer_;
        const uint32_t slot = (receiveInputSlot_ + issuedCount) % receiveInputBufferNum_;
        if (!localPartial) {
            CopyReceiveInput<false>(GetFinalReceiveAddr(token, topk), slot);
        }
        pendingLocalPartials[slot] = localPartial;
        pendingWeights[slot] = 1.0f;
        if (!enablePreReduce_ && hasExpertScales_ && topk < axisK_) {
            pendingWeights[slot] = finalExpertScalesBuf_.Get<float>().GetValue(token * axisK_ + topk);
        }
        ++issuedCount;
        if (receiveInputBufferNum_ == 1U || issuedCount - consumedCount > 1U) {
            const uint32_t ready = (receiveInputSlot_ + consumedCount++) % receiveInputBufferNum_;
            AccumulateFinalReceiveInput(ready, pendingLocalPartials[ready], pendingWeights[ready]);
        }
    }
    while (consumedCount != issuedCount) {
        const uint32_t ready = (receiveInputSlot_ + consumedCount++) % receiveInputBufferNum_;
        AccumulateFinalReceiveInput(ready, pendingLocalPartials[ready], pendingWeights[ready]);
    }
    receiveInputSlot_ = (receiveInputSlot_ + issuedCount) % receiveInputBufferNum_;
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
    const uint32_t inputBytes = axisH_ * static_cast<uint32_t>(sizeof(InputType));
    const uint8_t rightPadding =
        static_cast<uint8_t>((Ceil(inputBytes, UB_ALIGN) * UB_ALIGN - inputBytes) / sizeof(InputType));
    SetFlag<HardEvent::V_MTE2>(metadataFreeEvent_);
    WaitFlag<HardEvent::V_MTE2>(metadataFreeEvent_);
    // Keep the tail zero when this buffer is reused for MX quantization of the next token.
    DataCopyPad(input, source[row * axisH_], DataCopyExtParams{1U, inputBytes, 0U, 0U, 0U},
                DataCopyPadExtParams<InputType>{true, 0U, rightPadding, 0});
    SetFlag<HardEvent::MTE2_V>(metadataReadyEvent_);
    WaitFlag<HardEvent::MTE2_V>(metadataReadyEvent_);
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
    LocalTensor<float> mixture = constMix_;
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
    uint32_t receiveMask = finalExpectedMasksBuf_.Get<uint32_t>().GetValue(token);
    if (enablePreReduce_) {
        const uint32_t localMask = finalLocalMasksBuf_.Get<uint32_t>().GetValue(token);
        if (localMask != 0U) {
            BuildLocalPartial(token, localMask);
            receiveMask = (receiveMask & ~localMask) | (localMask & (0U - localMask));
        }
    }
    Duplicate(receiveSumBuf_.Get<float>(), 0.0f, receiveFloatBytes_ / sizeof(float));
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
    const uint64_t outputOffset = (static_cast<uint64_t>(finalTokenBegin_) + token) * axisH_;
    const uint32_t outputBytes = axisH_ * sizeof(XType);
    DataCopyPad(expandOutGlobal_[outputOffset], output, DataCopyExtParams{1U, outputBytes, 0U, 0U, 0U});
    SetFlag<HardEvent::MTE3_V>(bufferIndex);
}

template <RoceCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::CombineReadyTokens(uint32_t pendingCount)
{
    LocalTensor<uint32_t> pendingTokens = finalPendingTokensBuf_.Get<uint32_t>();
    ProcessPrefetchedTokens<true>(pendingTokens, pendingCount);
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
    for (uint32_t bufferIndex = 0U; bufferIndex < receiveInputBufferNum_; ++bufferIndex) {
        WaitFlag<HardEvent::V_MTE2>(inputFreeEvents_[bufferIndex]);
    }
    for (uint32_t bufferIndex = 0U; bufferIndex < finalOutputBufferNum_; ++bufferIndex) {
        WaitFlag<HardEvent::MTE3_V>(bufferIndex);
    }
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
        tpipe_->Reset();
        ReceiveAndCombine();
    }
}

} // namespace MoeDistributeCombineV2RoceImpl

#endif // defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
#endif
