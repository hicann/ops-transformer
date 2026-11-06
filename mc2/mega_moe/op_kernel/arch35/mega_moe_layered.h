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
 * \file mega_moe_layered.h
 * \brief
 */

#ifndef MEGA_MOE_LAYERED_H
#define MEGA_MOE_LAYERED_H

#include "../arch35_92/common/mega_moe_sync.h"
#include "kernel_operator.h"

#if __has_include("version/asc_devkit_version.h") && __has_include("version/hcomm_version.h")
#include "version/asc_devkit_version.h"
#include "version/hcomm_version.h"

#if (ASC_DEVKIT_VERSION_NUM >= 90200000) && (HCOMM_VERSION_NUM >= 90200000)
#define ENABLE_MEGA_MOE_LAYERED_KERNEL
#endif

#endif

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator_list_tensor_intf.h"
#include "mega_moe_vf/mega_moe_vf.h"

#include "../../../common/op_kernel/mc2_kernel_utils.h"
#include "../../../common/op_kernel/quantize_functions.h"

#include "../arch35_92/common/mega_moe_types.h"
#include "../arch35_92/common/mega_moe_gm_layout.h"
#include "../arch35_92/common/mega_moe_utils.h"
#include "../arch35_92/common/mega_moe_mxfp8_utils.h"
#include "../arch35_92/common/mega_moe_layered_comm.h"

#include "../arch35_92/blaze/epilogue/mega_moe_block_epilogue_activation_mx_quant.h"

#include "../arch35_92/stage/mega_moe_token_quant.h"
#include "../arch35_92/stage/mega_moe_expert_count.h"
#include "../arch35_92/mega_moe_vf/mega_moe_layered_vf.h"
#include "../arch35_92/stage/mega_moe_gmm1_activation.h"
#include "../arch35_92/stage/mega_moe_gmm2_combine.h"
#include "../arch35_92/stage/mega_moe_layered_combine_send.h"

namespace MegaMoeImpl {

using namespace AscendC;

#if defined(ENABLE_MEGA_MOE_LAYERED_KERNEL)
using TupleShape = Shape<int64_t, int64_t, int64_t, int64_t>;

struct DispatchRowRange {
    int32_t beginRow = 0;
    int32_t endRow = 0;
};

struct DispatchRoundSlice {
    int32_t validRowCount = 0;
    int32_t sourceRowBegin = 0;
    int32_t copyRowCount = 0;
    int32_t gatheredOffset = 0;
};

#define TemplateMegaMoeLayeredTypeClass \
    typename XType, typename OutputType, typename TopkWeightsType, typename Weight1Type, int32_t QuantMode, \
        int32_t Weight1Format, int32_t Weight2Format, int32_t CombineQuantMode, bool TopkWeightsPrefetch, \
        typename CommPolicy
#define TemplateMegaMoeLayeredTypeFunc \
    XType, OutputType, TopkWeightsType, Weight1Type, QuantMode, Weight1Format, Weight2Format, CombineQuantMode, \
        TopkWeightsPrefetch, CommPolicy

template <typename XType, typename OutputType, typename TopkWeightsType, typename Weight1Type, int32_t QuantMode,
          int32_t Weight1Format, int32_t Weight2Format, int32_t CombineQuantMode, bool TopkWeightsPrefetch,
          typename CommPolicy = LayeredUrmaComm>
class MegaMoeLayered {
public:
    using Channel = typename CommPolicy::Channel;
    using BatchHandle = typename CommPolicy::BatchHandle;
    using CombineBatchState = CombineImpl::LayeredCombineBatchState<CommPolicy>;
    using MoeQuantConfig = QuantConfig<Weight1Type, QuantMode>;
    using QuantOutType = typename MoeQuantConfig::QuantOutType;
    using ActivationType = typename MoeQuantConfig::QuantStorageType;
    using QuantScaleOutType = typename MoeQuantConfig::QuantScaleType;
    using ActivationQuantOutType = typename MoeQuantConfig::ActivationQuantOutType;
    __aicore__ inline explicit MegaMoeLayered(GM_ADDR context)
        : mc2Context_(reinterpret_cast<__gm__ Mc2MoeContext*>(context)),
          comm_(mc2Context_)
    {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR topkIds, GM_ADDR topkWeights, GM_ADDR weight1, GM_ADDR weight2,
                                GM_ADDR xActiveMask, GM_ADDR weightScales1, GM_ADDR weightScales2, GM_ADDR scales,
                                GM_ADDR sharedWeight1, GM_ADDR sharedWeight2, GM_ADDR sharedWeightScales1,
                                GM_ADDR sharedWeightScales2, GM_ADDR yOut, GM_ADDR expertTokenNumsOut,
                                GM_ADDR workspaceGM, const MegaMoeTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void PrepareWithoutSharedExpert();
    __aicore__ inline void PrepareSharedExpertInput(uint32_t maskWorkerCount);
    __aicore__ inline void DrainAndUnlockMaskChannels(const AivJobContext& job);
    __aicore__ inline void PrepareWithSharedExpert(Gmm1ActivationSync& sharedGmm1ActivationSync);

    __aicore__ inline bool IsCombineServerDedupEnabled() const
    {
        return IsServerCombineDedupOn(params_.tilingData->compute.dedupMode);
    }
    __aicore__ inline bool RecordHasWeights() const
    {
        return TopkWeightsPrefetch || IsCombineServerDedupEnabled();
    }
    __aicore__ inline uint64_t GatherCombineTokenIds(uint32_t originServer, const WorkRange& tokenRange,
                                                     const LocalTensor<int32_t>& tokenBitmaps,
                                                     const LocalTensor<int32_t>& tokenIds,
                                                     const LocalTensor<int32_t>& selectedTokenIds,
                                                     const LocalTensor<int32_t>& selectedTokenBitmaps,
                                                     const LocalTensor<uint8_t>& compareMask);
    __aicore__ inline void BuildServerCombineLists();
    __aicore__ inline bool IsServerContributionsReady(uint32_t originServer,
                                                      const LocalTensor<uint32_t>& tokenIdsTensor,
                                                      const LocalTensor<uint32_t>& expectedTopkBitmapsTensor,
                                                      uint32_t batchTokenCount, uint32_t workUbOffset) const;
    __aicore__ inline uint64_t WaitForServerCombineTokenCounts(const LocalTensor<uint32_t>& tokenCountsTensor,
                                                               uint64_t tokenListStrideBytes) const;
    __aicore__ inline void ProcessCombineRelay();
    __aicore__ inline void ReduceRelayContributions();
    __aicore__ inline void ReduceRelayContributionsInRange(const WorkRange& tokenRange,
                                                           const LocalTensor<uint32_t>& tokenCountsTensor,
                                                           const LocalTensor<uint32_t>& tokenIdsTensor,
                                                           const LocalTensor<uint32_t>& expectedTopkBitmapsTensor,
                                                           uint64_t tokenListStrideBytes, uint32_t workUbOffset);
    __aicore__ inline void ReduceRelayContributionsForBatch(uint32_t originServer, const WorkRange& batchRange,
                                                            __gm__ uint32_t* tokenIdsGlobalTensor,
                                                            __gm__ uint32_t* expectedTopkBitmapsGlobalTensor,
                                                            const LocalTensor<uint32_t>& tokenIdsTensor,
                                                            const LocalTensor<uint32_t>& expectedTopkBitmapsTensor,
                                                            uint32_t workUbOffset);
    __aicore__ inline void ReduceRelayContributionsForToken(uint32_t originServer, uint32_t tokenIdx,
                                                            uint32_t expectedTopkBitmap, uint32_t workUbOffset);
    __aicore__ inline void AccumulateRelayContributions(uint32_t expectedTopkBitmap,
                                                        __gm__ bfloat16_t* contributionGlobalTensor,
                                                        const LocalTensor<bfloat16_t>& contributionTensor,
                                                        const LocalTensor<float>& inputTensor,
                                                        const LocalTensor<float>& sumTensor,
                                                        const LocalTensor<float>& topkWeightsTensor);
    __aicore__ inline void SendServerCombineResults();
    __aicore__ inline void SendRelayResultsToTargetRank(uint32_t targetServer, uint32_t targetRank, uint32_t count,
                                                        uint32_t batchTokens, __gm__ uint32_t* tokenIdsGlobalTensor,
                                                        const LocalTensor<uint32_t>& tokenIdsTensor);
    __aicore__ inline void UnpermuteServerCombine();
    __aicore__ inline void InitTopology(const MegaMoeTilingData* tilingData);
    __aicore__ inline void InitDispatchLayout(const MegaMoeTilingData* tilingData);
    __aicore__ inline void InitWorkspace(GM_ADDR workspaceGM, const MegaMoeTilingData* tilingData);
    __aicore__ inline uint32_t InitResetBatch();
    __aicore__ inline void InitSendMaskRounds(uint32_t resetTensorSize, uint32_t countTensorSize);
    __aicore__ inline void InitDispatchRounds(uint32_t fixedBefore, uint32_t dispatchFixedCost);
    __aicore__ inline uint32_t InitDispatchIndexBuffers(uint32_t fixedBefore);
    __aicore__ inline void InitDispatchReceiveScratch(uint32_t copyBufferUbOffset, uint32_t copyBufferTotalBytes,
                                                      uint32_t expertCountBufferBytes);
    __aicore__ inline GM_ADDR GetSendMaskSlot(uint32_t expertId) const;
    __aicore__ inline void UpdateLocalRelayWeight(int32_t copyIdx, int32_t outIdx, TEventID eventId);
    __aicore__ inline void UnpermuteTokenChunk(const WorkRange& tokenRange,
                                               const MegaMoeUnpermuteBufferConfig& bufferConfig,
                                               uint32_t& inputBufferIdx);
    __aicore__ inline void LoadUnpermuteExpertInput(__gm__ bfloat16_t* expandedX, int32_t tokenIdx, int32_t expId,
                                                    TEventID event, __ubuf__ bfloat16_t* dataInBf16,
                                                    __ubuf__ float* dataInFp32);
    __aicore__ inline void AccumulateUnpermuteExperts(__gm__ bfloat16_t* expandedX, int32_t tokenIdx, int32_t localIdx,
                                                      const MegaMoeUnpermuteBufferConfig& bufferConfig,
                                                      uint32_t& inputBufferIdx);
    __aicore__ inline void LoadLocalRelayToken(uint32_t srcServer, uint32_t tokenIndex, int32_t bufferIdx,
                                               uint32_t copyInNum, TEventID eventId);
    __aicore__ inline uint32_t GetChannelOwnerBlock(uint32_t rank) const;
    __aicore__ inline bool IsChannelOwner(uint32_t rank) const;
    __aicore__ inline void PrepareExpertTokenCountTable();
    __aicore__ inline void InitDispatchSourceReadyCache();
    __aicore__ inline void EnsureDispatchSourceReady(uint32_t relayRank, uint32_t srcServer);
    __aicore__ inline ExpertTokenRange PlanTokenWave(const ExpertTokenPosition& begin, uint32_t groupTarget);
    __aicore__ inline void AdjustStartBlockIdxForNextWave(const ExpertTokenRange& wave, uint32_t waveStartBlockIdx,
                                                          uint32_t gmm1EndBlockIdx);
    template <AddrUpdateMode Mode>
    __aicore__ inline void ProcessGmmTokenRange(const ExpertTokenRange& range, ExpertLoopState& state,
                                                GMMAddrInfo& addrInfo, int32_t& vecSetSyncCom);
    __aicore__ inline void ProcessMoeExpertWaveLoop(ExpertLoopState& gmm1State, ExpertLoopState& gmm2State,
                                                    GMMAddrInfo& gmm1AddrInfo, GMMAddrInfo& gmm2AddrInfo,
                                                    int32_t& vecSetSyncCom, ExpertTokenRange currentWave);
    __aicore__ inline void ProcessMoeExpertWave(Gmm1ActivationSync& gmm1ActivationSync);
    __aicore__ inline void DispatchBuffInit();
    __aicore__ inline void DispatchPrepareBuffInit();
    __aicore__ inline void SendAndQuantBuffInit(uint32_t maskWorkerCount);
    __aicore__ inline MegaMoeUnpermuteBufferConfig UnpermuteBuffInit();
    __aicore__ inline void ResetState();
    __aicore__ inline void CopyExpertMaskCounts(const WorkRange& expertRange);
    __aicore__ inline void ComputeExpertMasks(const WorkRange& expertRange);
    __aicore__ inline void SendExpertMasks(const AivJobContext& job);
    __aicore__ inline void PrepareDispatchReceive();
    __aicore__ inline DispatchRowRange CalcDispatchExpertSlice(uint32_t expertIdx, uint32_t expertBegin,
                                                               const ExpertTokenRange& range);
    __aicore__ inline uint64_t LoadAndGatherDispatchRound(__gm__ uint8_t* rankMaskBasePtr, uint32_t roundIdx);
    __aicore__ inline void CopyLocalDispatchRound(uint32_t expertIdx, uint32_t srcRank, int32_t rowStart,
                                                  int32_t copyBegin, int32_t roundCopyCount, int32_t gatheredOffset);
    __aicore__ inline void AppendRemoteDispatchRound(uint32_t relayRank, uint32_t expertIdx, uint32_t srcServer,
                                                     uint32_t srcRank, int32_t rowStart, int32_t copyBegin,
                                                     int32_t roundCopyCount, int32_t gatheredOffset,
                                                     uint32_t& remoteBatchTokenCount);
    __aicore__ inline void ReceiveDispatchTokenRange(const ExpertTokenRange& range);
    template <bool IsLocalServer>
    __aicore__ inline void ReceiveDispatchServerPhase(const ExpertTokenRange& range);
    template <bool IsLocalServer>
    __aicore__ inline void ReceiveDispatchExpertFromServers(uint32_t relayRank, uint32_t expertIdx,
                                                            const ExpertTokenRange& range,
                                                            uint32_t& remoteBatchTokenCount);
    template <bool IsLocalServer>
    __aicore__ inline void ReceiveDispatchFromSourceRank(uint32_t relayRank, uint32_t expertIdx, uint32_t srcServer,
                                                         __gm__ uint8_t* expertMaskBase,
                                                         const DispatchRowRange& expertRows,
                                                         uint32_t& remoteBatchTokenCount);
    __aicore__ inline uint32_t FindDispatchSegmentEnd(uint32_t segmentStart, uint32_t batchTokenCount);
    __aicore__ inline void ReceiveRemoteDispatchBatch(uint32_t relayRank, uint32_t batchTokenCount);
    __aicore__ inline void PublishDispatchRows(uint32_t expertIdx, int32_t globalRowStart, int32_t rowCount);
    __aicore__ inline void UpdateGmm1GlobalBuffer(GMMAddrInfo& gmmAddrInfo, const ExpertLoopState& state,
                                                  uint32_t rowOffsetInExpert = 0U);
    __aicore__ inline void UpdateGmm2GlobalBuffer(GMMAddrInfo& gmmAddrInfo, const ExpertLoopState& state,
                                                  uint32_t rowOffsetInExpert = 0U);
    template <AddrUpdateMode Mode>
    __aicore__ inline void UpdateSharedGlobalBuffer(GMMAddrInfo& gmmAddrInfo, const ExpertLoopState& state);
    __aicore__ inline void Unpermute(const MegaMoeUnpermuteBufferConfig& bufferConfig);
    __aicore__ inline void InitCombineBuffers();
    __aicore__ inline void ProcessCombineTokenRange(const ExpertLoopState& waveBeginState,
                                                    const ExpertTokenRange& range);
    template <bool ReuseChannelLock>
    __aicore__ inline void CrossRankSyncInWorldSize();
    __aicore__ inline void ExpertTokenNumCopyOut();
    __aicore__ inline void CopyLocalDispatchTokens(int32_t dstRow, int32_t srcRank, int32_t tokenOffset,
                                                   int32_t tokenCount);
    __aicore__ inline void PublishLocalDispatchRelayCount();
    __aicore__ inline GM_ADDR PrepareDispatchReadySource(uint32_t targetServer);
    __aicore__ inline void SendDispatchToRemoteServer(uint32_t targetServer, uint32_t relayRank);
    __aicore__ inline void StartInterServerDispatch();
    __aicore__ inline uint64_t DispatchBulkReadyFlagOffset(uint32_t sourceServer) const;
    template <bool CombineServerDedup>
    __aicore__ inline void BuildDispatchRelayCandidates(const LocalTensor<int32_t>& topkIds,
                                                        const LocalTensor<uint32_t>& masks, uint32_t tokenCount,
                                                        const WorkRange& serverRange);
    __aicore__ inline void EnqueueDispatchRelayCandidates(const LocalTensor<uint32_t>& masks,
                                                          const WorkRange& tokenRange, const WorkRange& serverRange);
    __aicore__ inline uint32_t TransposeDispatchRelayBitmaps(const LocalTensor<uint32_t>& masks,
                                                             const LocalTensor<uint32_t>& transposedMasks,
                                                             uint32_t tokenCount);
    __aicore__ inline void PublishDispatchRelayBitmaps(const LocalTensor<uint32_t>& masks, const WorkRange& tokenRange,
                                                       const WorkRange& serverRange, uint32_t rowElements);
    __aicore__ inline void PrepareDispatchRelayQueues(const AivJobContext& job);
    template <bool CombineServerDedup>
    __aicore__ inline void BuildDispatchRelayQueues(const LocalTensor<int32_t>& topkIds,
                                                    const LocalTensor<uint32_t>& masks,
                                                    const LocalTensor<uint32_t>& transposedMasks,
                                                    const WorkRange& tokenRange, uint32_t batchTokens);
    __aicore__ inline void EnqueueDispatchRelayToken(uint32_t targetServer, uint32_t tokenIdx);
    __aicore__ inline void LoadTokenFromLocalRelay(uint32_t srcServer, uint32_t tokenIndex, int32_t bufferIdx,
                                                   uint32_t copyInNum);
    __aicore__ inline void CopyTokensFromLocalRelay(int32_t rowDstOffsetInCore, uint32_t srcServer, int32_t copyNum,
                                                    int64_t widthA, int64_t widthAScale, uint32_t copyInNum);
    __aicore__ inline void QuantizeTokenInUb(__ubuf__ bfloat16_t* input, __ubuf__ ActivationType* output,
                                             __ubuf__ uint16_t* scratch);
    __aicore__ inline void QuantizeLocalTokensToRelay();
    __aicore__ inline uint64_t DispatchRelayQueueServerOffset(uint32_t targetServer) const;
    __aicore__ inline uint64_t RelayTokenOffset(uint32_t sourceServer, uint32_t tokenId) const;
    __aicore__ inline uint64_t RelayFlagOffset(uint32_t sourceServer, uint32_t tokenId) const;
    __aicore__ inline void ProcessSharedExpertGmm1Loop(GMMAddrInfo& sharedGmm1AddrInfo,
                                                       ExpertLoopState& sharedGmm1State, int32_t& vecSetSyncCom);
    __aicore__ inline void ProcessSharedExpertGmm1(Gmm1ActivationSync& sharedGmm1ActivationSync);
    __aicore__ inline void ProcessSharedExpertGmm2Loop(GMMAddrInfo& sharedGmm2AddrInfo,
                                                       ExpertLoopState& sharedGmm2State);
    __aicore__ inline void ProcessSharedExpertGmm2();
    __aicore__ inline void UnpermuteSharedExpert(int32_t tokenIdx, const MegaMoeUnpermuteBufferConfig& bufferConfig,
                                                 uint32_t& inputBufferIdx);
    __aicore__ inline void LoadTopkWeightsToUb(__ubuf__ ActivationType* xOutTensor, int32_t currentOffset,
                                               int32_t index, TEventID event);
    template <bool IsShared, typename Epilogue, typename... LayoutArgs>
    __aicore__ inline void RunGmm1WithEpilogue(Epilogue& epilogue, const GMMAddrInfo& gmmAddrInfo,
                                               const ExpertLoopState& state, uint32_t expertIdx, int32_t& vecSetSyncCom,
                                               uint32_t rowOffsetInExpert = 0U, const LayoutArgs&... layoutArgs);
    template <bool IsShared = false>
    __aicore__ inline void GroupMatmulWithActivationQuant(const GMMAddrInfo& gmmAddrInfo, const ExpertLoopState& state,
                                                          uint32_t expertIdx, int32_t& vecSetSyncCom,
                                                          uint32_t rowOffsetInExpert = 0U);
    template <bool IsShared = false>
    __aicore__ inline void GroupMatmulWithCombine(const GMMAddrInfo& gmmAddrInfo, const ExpertLoopState& state,
                                                  uint32_t expertTokenCount, uint32_t rowOffsetInExpert = 0U);
    __aicore__ inline bool GetCombineRankRange(uint32_t expertIdx, uint32_t dstRank, uint32_t& rowStart,
                                               uint32_t& tokenCount);
    template <bool CombineServerDedup>
    __aicore__ inline void CombineTokensOfExpert(const GMMAddrInfo& gmmAddrInfo, const ExpertLoopState& gmm2State,
                                                 uint32_t expertIdx, const WorkRange& expertRowRange,
                                                 CombineBatchState& batchState);
    template <bool CombineServerDedup>
    __aicore__ inline void ProcessCombineToTargetRank(ExpertLoopState waveBeginState, const ExpertTokenRange& range,
                                                      uint32_t targetRank);
    template <bool CombineServerDedup>
    __aicore__ inline void CombineTokensInRange(const GMMAddrInfo& gmmAddrInfo, const ExpertLoopState& gmm2State,
                                                uint32_t expertIdx, const WorkRange& rowRange,
                                                CombineBatchState& batchState);
    __aicore__ inline void LockOwnedRemoteChannels();
    __aicore__ inline void DrainAndUnlockOwnedRemoteChannels();

    __gm__ Mc2MoeContext* mc2Context_{nullptr};
    CommPolicy comm_;
    Params params_{};

    __gm__ int32_t* expertTokenNumsOut_ = nullptr;
    __gm__ int32_t* metadataGm_ = nullptr;
    __gm__ int32_t* expertRecvCountsGm_ = nullptr;
    // 完整前缀和在建表时备份到 GM，Dispatch 接收和 CopyOut 在 UB 被计算覆盖后从 GM 恢复。
    __gm__ int32_t* cumsumGm_ = nullptr;

    uint32_t m_ = 0;
    uint32_t k_ = 0;
    uint32_t activationOutputDivisor_ = ACTIVATION_N_HALF;
    // 专家循环状态与地址绑定的公共阶段配置，InitTopology 里一次填好。
    MoeStageCommonConfig commonConfig_{};
    uint32_t aicNum_ = 0;
    uint32_t topK_ = 0;
    uint32_t rankId_ = 0;
    uint32_t worldSize_ = 0;
    uint32_t rankNumPerServer_ = 0;
    uint32_t serverNum_ = 0;
    uint32_t serverId_ = 0;
    uint32_t rankIdInServer_ = 0;
    int64_t hiddenDim_ = 0;
    uint64_t maxOutputSize_ = 0;
    uint16_t gmm1PingPongIdx_ = 0;
    uint32_t startBlockIdx_ = 0;
    uint32_t mGroupsPerWave_ = 1U;
    // 由本核 AIV1 在建表时计算，仅用于划分通信 wave。
    uint64_t gmmWaveCount_ = 0U;
    int32_t dispatchFlagSlotsPerExpert_ = 0;
    int32_t maxWavesPerExpert_ = 0;
    // A5 混合核启动约定：GetBlockNum() 等于 AIC 数，每个逻辑核对应两个 AIV 子核。
    // 因此 Host 侧仍按 blockAivNum / 2 计算逻辑核数。
    uint32_t blockNum_ = GetBlockNum();
    uint32_t blockAivNum_ = GetBlockNum() * 2;
    uint32_t blockIdx_ = GetBlockIdx() / GetTaskRation();
    uint32_t aivCoreIdx_ = GetBlockIdx();
    uint32_t subBlockIdx_ = GetSubBlockIdx();
    uint32_t mxQuantScaleNumAlignPerToken_ = 0;
    uint32_t mxQuantTokenAlignBytes_ = 0;
    uint32_t mxQuantScaleAlignBytes_ = 0;
    uint32_t mxQuantTokenScaleAlignBytes_ = 0;
    uint32_t weightAlignBytes_ = 0;
    uint32_t ubBufferUsedAddr_ = 0;
    uint64_t sendTotalNum_ = 0;      // 本 Rank 真实 route 数：bs * topK
    uint64_t maskRouteCapacity_ = 0; // 对称 mask 容量：numMaxTokensPerRank * topK
    uint32_t maskAlignSize_ = 0;
    uint32_t maskSlotSize_ = 0;      // 单个 win 槽位 = maskAlignSize_(mask) + 32B(count)
    uint32_t roundSendTotalNum_ = 0; // 分轮次：每轮处理的 token*topK 数量（256 对齐保证 GM 写 32B 对齐）
    uint32_t roundCompareCount_ = 0;          // 分轮次：每轮 CompareScalar 的元素数
    uint32_t roundMaskAlignSize_ = 0;         // 分轮次：每轮部分 mask 的字节大小（32B 对齐）
    uint32_t roundMaskSlotSize_ = 0;          // 分轮次：每轮部分 [mask|count] 槽位字节大小
    uint32_t totalRounds_ = 0;                // 分轮次：总轮数 = CeilDiv(sendTotalNum, roundSendTotalNum)
    uint32_t dispatchRoundSendTotalNum_ = 0;  // dispatch分轮次：每轮处理的 token*topK 数量（256 对齐）
    uint32_t dispatchTotalRounds_ = 0;        // dispatch分轮次：容量上界总轮数
    uint32_t dispatchRoundMaskAlignSize_ = 0; // dispatch分轮次：每轮mask的字节大小（32B对齐）
    uint32_t relayRecordBytes_ = 0;
    uint64_t dispatchRelayQueueBytesPerServer_ = 0;
    uint64_t dispatchRelayFlagSnapshotBytesPerBlock_ = 0;
    int64_t combineUbTensorSize_ = 0; // combineUbTensor 的大小（元素数）
    uint32_t topKWeightsTempAddr_ = 0;
    uint32_t sharedExpertNum_ = 0;
    uint32_t moeExpertPerRank_ = 0;

    static constexpr uint32_t A_ELEMS_PER_BYTE = MoeQuantConfig::A_ELEMS_PER_BYTE;
    static constexpr uint32_t B_ELEMS_PER_BYTE = MoeQuantConfig::B_ELEMS_PER_BYTE;
    static constexpr uint32_t C_ELEMS_PER_BYTE = MoeQuantConfig::C_ELEMS_PER_BYTE;
    // 来源就绪位图每个 rank 占一个 bit；按 512B 预留，使下移后的 WQE 区继续保持对齐。
    static constexpr uint32_t DISPATCH_READY_CACHE_RESERVE_BYTES = 512U;
    static constexpr int32_t DISPATCH_COPY_BUFFER_COUNT = 5;
    static constexpr uint32_t MX_QUANT_TEMP_UB_BYTES = 2U * 1024U;
    static constexpr uint32_t DISPATCH_MAX_TOPK = 32U;
    // 二级读取在 TopkWeightsPrefetch 路径下每 token 需要 data/scale/weight 三条 WQE。
    // 64 KiB 可容纳 1024 条 WQE，足够一次提交完整的 256-token 批次（最多 768 条 WQE）。
    static_assert(LAYERED_HCOMM_BATCH_WQE_CAPACITY >= L1_TILE_M_256 * 3U,
                  "Layered Hcomm UB must hold 256 three-WQE token records");
    // Wave 通信需要同时保留批量描述符和其他搬运张量，因此动态 UB 预算止于批量区起点。
    static constexpr uint32_t COMM_WORK_UB_LIMIT = LAYERED_HCOMM_BATCH_UB_OFFSET - DISPATCH_READY_CACHE_RESERVE_BYTES;
    // AIV1 私有持久存储；Dispatch/Combine 临时缓冲的上界不得进入该区域。
    __ubuf__ uint32_t* dispatchSourceReadyBitsUb_ = nullptr;
    // 一级按协议长度限制连续分块；二级按最多 256 条 metadata 组织批次。
    static constexpr uint32_t DISPATCH_RELAY_QUEUE_ENTRY_BYTES = ALIGN_32;
    static constexpr uint32_t DISPATCH_RECEIVE_BATCH_TOKEN_CAPACITY = L1_TILE_M_256;
    static constexpr uint32_t DISPATCH_BATCH_DST_ROW = 4U;
    static constexpr uint32_t DISPATCH_BATCH_EXPERT_ID = 5U;
    static constexpr uint32_t DISPATCH_BATCH_SRC_SERVER = 6U;
    __ubuf__ int32_t* topkIndexTensor_ = nullptr;
    __ubuf__ uint8_t* gatherMaskTensor_ = nullptr;
    __ubuf__ uint32_t* gatherMaskInt32Tensor_ = nullptr;
    __ubuf__ int32_t* validTopkIndexTensor_ = nullptr;
    __ubuf__ int32_t* cumsumInfoTensor_ = nullptr;
    __ubuf__ ActivationType* copyTmpTensors_[DISPATCH_COPY_BUFFER_COUNT]{}; // 5 路软流水：占用 EVENT_ID1..EVENT_ID5。
    __ubuf__ int32_t* metaInfoTensor_ = nullptr;
    __ubuf__ bfloat16_t* xInTensor1_ = nullptr;
    __ubuf__ bfloat16_t* xInTensor2_ = nullptr;
    __ubuf__ ActivationType* xOutTensor1_ = nullptr;
    __ubuf__ ActivationType* xOutTensor2_ = nullptr;
    __ubuf__ uint16_t* mxTempTensor_ = nullptr;
    __ubuf__ int32_t* resetTensor_ = nullptr;
    int32_t resetBatchElementCount_ = 0;
    __ubuf__ int32_t* topkIdsTensor_ = nullptr;
    __ubuf__ uint8_t* sendMaskTensor_[DOUBLE_BUFFER]{}; // ComputeExpertMasks 位图搬出的 ping-pong 缓冲
    __ubuf__ int32_t* sendMaskCounts_ = nullptr;        // 每个任务一个 32B count 槽
    __ubuf__ int32_t* expertTokenNumsOutTensor_ = nullptr;
    __ubuf__ bfloat16_t* dataResTensor_ = nullptr;
    __ubuf__ float* dataResFp32Tensor_ = nullptr;
    __ubuf__ float* topKWeightsTensor_ = nullptr;
    __ubuf__ float* fp32ScaleTensor_ = nullptr;
    __ubuf__ bfloat16_t* bf16ScaleTensor_ = nullptr;

    // GMM1 固定按 256 行切块；权重前移时后处理可按 128 行细分。
    static constexpr uint32_t GMM1_TILE_M = L1_TILE_M_256;
    static constexpr uint32_t EPILOGUE_TILE_M = TopkWeightsPrefetch ? L1_TILE_M_128 : L1_TILE_M_256;

    // GMM1 的每个 tile 固定为 [gate128, up128]，由同一 epilogue 消费。
    using A8W4Config = GmmKernel::Config<true, 0, ActivationQuantOutType, Weight1Type, bfloat16_t, QuantScaleOutType,
                                         QuantScaleOutType>;
    typename A8W4Config::BlockContext blockContext_{};
    using BlockEpilogue = BlockEpilogueActivationMxQuant<ActivationQuantOutType, bfloat16_t, EPILOGUE_TILE_M, L1_TILE_N,
                                                         TopkWeightsPrefetch>;
    using SharedBlockEpilogue =
        BlockEpilogueActivationMxQuant<ActivationQuantOutType, bfloat16_t, L1_TILE_M_256, L1_TILE_N, false>;
    BlockEpilogue epilogueOp_;
    SharedBlockEpilogue sharedEpilogueOp_;
};

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline uint32_t MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::GetChannelOwnerBlock(uint32_t rank) const
{
    // 通道归属在逻辑 AIC 域内计算，不使用物理 AIV 编号。
    // Dispatch 两级通信、Combine 写入和最终 Drain 共用此映射，且只有归属 block 的 AIV1 可访问通道。
    return rank % blockNum_;
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline bool MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::IsChannelOwner(uint32_t rank) const
{
    if constexpr (g_coreType == AIC) {
        return false;
    }
    return subBlockIdx_ == 1U && blockIdx_ == GetChannelOwnerBlock(rank);
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::InitDispatchSourceReadyCache()
{
    if constexpr (g_coreType == AIV) {
        if (subBlockIdx_ != 1U) {
            return;
        }
        const uint32_t readyCacheBytes =
            Ops::Base::CeilAlign(Ops::Base::CeilDiv(worldSize_, uint32_t(32)) * static_cast<uint32_t>(sizeof(uint32_t)),
                                 static_cast<uint32_t>(ALIGN_32));
        dispatchSourceReadyBitsUb_ =
            reinterpret_cast<__ubuf__ uint32_t*>(LAYERED_USABLE_UB_BYTES - DISPATCH_READY_CACHE_RESERVE_BYTES);
        // 每次调用初始化一次；计数表准备阶段的 V→S 同步保证首次标量访问前清零完成。
        // 后续仅由当前 AIV1 以标量方式读写。
        const uint32_t cacheWordCount = readyCacheBytes / sizeof(uint32_t);
        LocalTensor<uint32_t> readyCache(TPosition::VECCALC, reinterpret_cast<uint64_t>(dispatchSourceReadyBitsUb_),
                                         cacheWordCount);
        Duplicate<uint32_t>(readyCache, 0U, cacheWordCount);
    }
}

// 完整计数表在入口就绪后，按调用方给定的 M-group 预算规划 wave。
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline ExpertTokenRange MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::PlanTokenWave(
    const ExpertTokenPosition& begin, uint32_t groupTarget)
{
    const auto& workspace = params_.workspace.layout;
    const auto& buffer = params_.workspace.base;
    ExpertTokenRange range{begin, begin};
    uint32_t groupCount = 0U;
    const BlockWorkspaceContext countWorkspace{blockIdx_, aicNum_};
    while (range.end.expertIdx < moeExpertPerRank_ && groupCount < groupTarget) {
        uint32_t rowCount = GetExpertTokenCountFromWorkspace(workspace.routing.recvCounts.get_ptr(buffer),
                                                             countWorkspace, moeExpertPerRank_, range.end.expertIdx);
        if (rowCount == 0U || range.end.tokenIndexInExpert >= rowCount) {
            ++range.end.expertIdx;
            range.end.tokenIndexInExpert = 0U;
            continue;
        }
        AdvanceExpertTokenPositionInWave<GMM1_TILE_M>(rowCount, groupTarget, groupCount, range.end);
    }
    return range;
}

// 保持原 GMM wave 的固定角色共振：GMM1 结束时若整体 block 游标发生偏移，则下一 wave 从该偏移开始。
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::AdjustStartBlockIdxForNextWave(
    const ExpertTokenRange& wave, uint32_t waveStartBlockIdx, uint32_t gmm1EndBlockIdx)
{
    if constexpr (MoeQuantConfig::AXW_MODE != AxWMode::A8W4) {
        const bool hasNextWave = wave.end.expertIdx < moeExpertPerRank_;
        const bool fixedRoleResonance = startBlockIdx_ == waveStartBlockIdx && gmm1EndBlockIdx != waveStartBlockIdx;
        if (hasNextWave && fixedRoleResonance) {
            startBlockIdx_ = gmm1EndBlockIdx;
        }
    }
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::InitTopology(const MegaMoeTilingData* tilingData)
{
    m_ = static_cast<uint32_t>(tilingData->compute.bs);
    k_ = static_cast<uint32_t>(tilingData->compute.h);
    aicNum_ = static_cast<uint32_t>(tilingData->compute.aicNum);
    topK_ = static_cast<uint32_t>(tilingData->compute.topK);
    sendTotalNum_ = static_cast<uint64_t>(m_) * static_cast<uint64_t>(topK_);
    maskRouteCapacity_ = static_cast<uint64_t>(tilingData->compute.numMaxTokensPerRank) * static_cast<uint64_t>(topK_);
    worldSize_ = static_cast<uint32_t>(tilingData->compute.epWorldSize);
    moeExpertPerRank_ = static_cast<uint32_t>(tilingData->compute.moeExpertPerRank);
    mGroupsPerWave_ =
        tilingData->compute.mGroupsPerWave == 0 ? 1U : static_cast<uint32_t>(tilingData->compute.mGroupsPerWave);
    sharedExpertNum_ = static_cast<uint32_t>(tilingData->compute.sharedExpertNum);
    maxOutputSize_ = static_cast<uint64_t>(tilingData->compute.maxOutputSize);
    // 与 WorkspaceInfo 构造里 flagDispatchToGmm1Ptr 的分配公式保持一致。
    maxWavesPerExpert_ = static_cast<int32_t>(
        Ops::Base::CeilDiv(static_cast<int64_t>(maxOutputSize_), static_cast<int64_t>(L1_TILE_M_256)));
    dispatchFlagSlotsPerExpert_ = maxWavesPerExpert_ * INT_CACHELINE;
    hiddenDim_ = tilingData->compute.hiddenDim;
    activationOutputDivisor_ = GetActivationOutputDivisor(tilingData->actMode);
    rankId_ = mc2Context_->epRankId;
    rankNumPerServer_ = tilingData->rankNumPerServer;
    if (rankNumPerServer_ == 0U || rankNumPerServer_ > worldSize_) {
        rankNumPerServer_ = worldSize_;
    }
    serverNum_ = Ops::Base::CeilDiv(worldSize_, rankNumPerServer_);
    serverId_ = rankId_ / rankNumPerServer_;
    rankIdInServer_ = rankId_ % rankNumPerServer_;
    for (int i = 0; i < worldSize_; i++) {
        g_winRankAddr_[i] = (GM_ADDR)mc2Context_->epHcclBuffer_[i];
    }
    commonConfig_.rankId = rankId_;
    commonConfig_.worldSize = worldSize_;
    commonConfig_.moeExpertPerRank = moeExpertPerRank_;
    commonConfig_.sharedExpertNum = sharedExpertNum_;
    commonConfig_.tokenNum = m_;
    commonConfig_.topK = topK_;
    commonConfig_.tokenHiddenDim = k_;
    commonConfig_.gmm1OutputDim = static_cast<uint32_t>(hiddenDim_);
    commonConfig_.activationOutputDivisor = activationOutputDivisor_;
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::InitDispatchLayout(
    const MegaMoeTilingData* tilingData)
{
    // mask 是跨 Rank 共享的槽布局，几何必须与 PeermemInfo 一样按全卡容量计算。
    maskAlignSize_ = static_cast<uint32_t>(
        CalcDispatchMaskAlignSize(tilingData->compute.numMaxTokensPerRank, tilingData->compute.topK));
    // 每个 win 槽位再追加 32B 存 count（源卡 ComputeExpertMasks 同时算好）, 须与 PeermemInfo 的 maskSlotSize 一致。
    maskSlotSize_ = maskAlignSize_ + static_cast<uint32_t>(ALIGN_32);
    mxQuantScaleNumAlignPerToken_ = Ops::Base::CeilDiv(k_, static_cast<uint32_t>(ALIGN_32));
    mxQuantTokenAlignBytes_ =
        Ops::Base::CeilAlign(static_cast<uint32_t>(k_ / A_ELEMS_PER_BYTE), static_cast<uint32_t>(ALIGN_256)) *
        sizeof(ActivationType);
    // 有效 scale 数仅用于量化；存储段须对齐，使后续权重的 UB 起点满足 32B 要求。
    mxQuantScaleAlignBytes_ = Ops::Base::CeilAlign(
        mxQuantScaleNumAlignPerToken_ * static_cast<uint32_t>(sizeof(uint8_t)), static_cast<uint32_t>(ALIGN_32));
    mxQuantTokenScaleAlignBytes_ =
        Ops::Base::CeilAlign(mxQuantTokenAlignBytes_ + mxQuantScaleAlignBytes_, static_cast<uint32_t>(ALIGN_32));
    if (RecordHasWeights()) {
        weightAlignBytes_ =
            Ops::Base::CeilAlign(static_cast<uint32_t>(topK_ * sizeof(float)), static_cast<uint32_t>(ALIGN_32));
        mxQuantTokenScaleAlignBytes_ += weightAlignBytes_;
    }
    relayRecordBytes_ =
        Ops::Base::CeilAlign(static_cast<uint64_t>(mxQuantTokenScaleAlignBytes_), static_cast<uint64_t>(ALIGN_512));
    dispatchRelayQueueBytesPerServer_ =
        static_cast<uint64_t>(ALIGN_32) + static_cast<uint64_t>(m_) * DISPATCH_RELAY_QUEUE_ENTRY_BYTES;
    // 保留 GM 快照原有行宽，来源级 ready 轮询只使用行首一个 uint64_t。
    uint64_t flagSnapshotBytes = static_cast<uint64_t>(DISPATCH_RECEIVE_BATCH_TOKEN_CAPACITY) * sizeof(uint64_t);
    dispatchRelayFlagSnapshotBytesPerBlock_ = Ops::Base::CeilAlign(flagSnapshotBytes, static_cast<uint64_t>(ALIGN_512));
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::InitWorkspace(
    GM_ADDR workspaceGM, const MegaMoeTilingData* tilingData)
{
    (void)workspaceGM;
    params_.communication.base = g_winRankAddr_[rankId_];
    params_.workspace.layout = WorkspaceLayout(tilingData->compute);
    params_.communication.layout = PeermemInfo(tilingData->comm, params_.workspace.layout.workspaceSize);
    params_.workspace.base = params_.communication.layout.workspace.get_ptr(params_.communication.base);
#if defined(ASCENDC_OOM) && ASCENDC_OOM == 1
    // HCCL 窗口: 逐 rank 登记整窗(ccl_buffer_size), 覆盖通信布局区与 workspace 区
    for (int i = 0; i < worldSize_; i++) {
        OOMRegisterRange(reinterpret_cast<GM_ADDR>(mc2Context_->epHcclBuffer_[i]), tilingData->cclBufferSize);
    }
#endif
    params_.tilingData = tilingData;
    const auto& workspace = params_.workspace.layout;
    const auto& buffer = params_.workspace.base;
    expertTokenNumsOut_ = (__gm__ int32_t*)params_.expertTokenNumsOutGmAddr;
    expertRecvCountsGm_ = workspace.routing.recvCounts.template get_ptr<int32_t>(buffer);
    metadataGm_ = workspace.routing.receivedTokenMetadata.template get_ptr<int32_t>(buffer);
    // 每个 block 备份完整专家表，每个专家包含 worldSize 个 rank 前缀值。
    uint64_t cumsumStride =
        Ops::Base::CeilAlign(static_cast<int64_t>(worldSize_ * moeExpertPerRank_ * sizeof(int32_t)), ALIGN_32);
    cumsumGm_ = reinterpret_cast<__gm__ int32_t*>(workspace.routing.recvPrefixBackup.get_ptr(buffer) +
                                                  cumsumStride * blockIdx_);
    epilogueOp_.Init({.yGmAddr = workspace.routed.gmm2Input.data.get_ptr(buffer),
                      .yScaleGmAddr = workspace.routed.gmm2Input.scale.get_ptr(buffer),
                      .clampLimit = tilingData->clampLimit,
                      .actMode = tilingData->actMode,
                      .actSubMode = tilingData->actSubMode,
                      .activationAlpha = tilingData->activationAlpha,
                      .activationBeta = tilingData->activationBeta});
    InitDispatchLayout(tilingData);
}

// ========================
// Init：初始化 & 偏移计算
// ========================
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::Init(
    GM_ADDR x, GM_ADDR topkIds, GM_ADDR topkWeights, GM_ADDR weight1, GM_ADDR weight2, GM_ADDR xActiveMask,
    GM_ADDR weightScales1, GM_ADDR weightScales2, GM_ADDR scales, GM_ADDR sharedWeight1, GM_ADDR sharedWeight2,
    GM_ADDR sharedWeightScales1, GM_ADDR sharedWeightScales2, GM_ADDR yOut, GM_ADDR expertTokenNumsOut,
    GM_ADDR workspaceGM, const MegaMoeTilingData* tilingData)
{
    if constexpr (g_coreType == AIV) {
        InitCopyPadding();
    }
    InitTopology(tilingData);
    params_.aGmAddr = x;
    params_.expertIdxGmAddr = topkIds;
    // 保留 TensorList 入口；每个 expert 更新地址时再按 2D-list/3D-stacked 布局解析。
    params_.bGmAddr = weight1;
    params_.b2GmAddr = weight2;
    params_.bScaleGmAddr = weightScales1;
    params_.b2ScaleGmAddr = weightScales2;
    params_.sharedBGmAddr = sharedWeight1;
    params_.sharedB2GmAddr = sharedWeight2;
    params_.sharedBScaleGmAddr = sharedWeightScales1;
    params_.sharedB2ScaleGmAddr = sharedWeightScales2;

    params_.y2GmAddr = yOut;
    params_.expertTokenNumsOutGmAddr = expertTokenNumsOut;
    params_.probsGmAddr = topkWeights;
    InitWorkspace(workspaceGM, tilingData);
}

// ==================================================================================
// UpdateGmm1GlobalBuffer：绑定 GMM1 的输入、权重、输出、tile 状态与 flag 地址。
// ==================================================================================
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::UpdateGmm1GlobalBuffer(
    GMMAddrInfo& gmmAddrInfo, const ExpertLoopState& state, uint32_t rowOffsetInExpert)
{
    const auto& workspace = params_.workspace.layout;
    const auto& buffer = params_.workspace.base;
    // 地址 = 专家前累计行数 + 专家内偏移；行相关项随行数线性增长，权重/flag 只依赖专家序号。
    const uint64_t rows = static_cast<uint64_t>(state.globalTokenStartIndex) + rowOffsetInExpert;
    const uint32_t expertIdx = state.expertIdx;
    const uint64_t n = static_cast<uint64_t>(Get<N_VALUE>(state.problemShape));
    const uint64_t k = static_cast<uint64_t>(Get<K_VALUE>(state.problemShape));
    const uint64_t activationCols = n / activationOutputDivisor_;
    const uint64_t scaleK = Ops::Base::CeilDiv(k, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
    const uint64_t scaleN =
        Ops::Base::CeilDiv(activationCols, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
    // guard 与 WorkspaceInfo 分配条件一致，由 TilingKey 保证同步。
    if constexpr (MoeQuantConfig::AXW_MODE == AxWMode::A8W4 || TopkWeightsPrefetch) {
        gmmAddrInfo.gmm1OutGlobal = workspace.routed.gmm1Output.get_ptr(buffer) + rows * n * sizeof(bfloat16_t);
    }
    gmmAddrInfo.metaInfoGlobal = workspace.routing.receivedTokenMetadata.get_ptr(buffer);
    gmmAddrInfo.aGlobal =
        workspace.routed.gmm1Input.data.get_ptr(buffer) + rows * k / A_ELEMS_PER_BYTE * sizeof(ActivationType);
    gmmAddrInfo.aScaleGlobal =
        workspace.routed.gmm1Input.scale.get_ptr(buffer) + rows * scaleK * sizeof(QuantScaleOutType);
    gmmAddrInfo.bGlobal =
        GetExpertWeightAddr<Weight1Type>(params_.bGmAddr, params_.tilingData->isPerExpertWeightTensor, expertIdx,
                                         static_cast<uint64_t>(expertIdx) * (n * k / B_ELEMS_PER_BYTE));
    gmmAddrInfo.bScaleGlobal =
        GetExpertWeightAddr<QuantScaleOutType>(params_.bScaleGmAddr, params_.tilingData->isPerExpertWeightTensor,
                                               expertIdx, static_cast<uint64_t>(expertIdx) * (n * scaleK));
    if constexpr (g_coreType == AIV) {
        AscendC::Coord<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> vecBaseOffset{
            static_cast<int64_t>(rows * activationCols / C_ELEMS_PER_BYTE),
            static_cast<int64_t>(rows * scaleN),
            static_cast<int64_t>(expertIdx) * maxWavesPerExpert_ + rowOffsetInExpert / GMM1_TILE_M,
            0L,
            0L,
            0L};
        epilogueOp_.UpdateGlobalAddr(vecBaseOffset);
    }
    gmmAddrInfo.activationToGmm2Flag = (__gm__ int32_t*)workspace.sync.activationReady.get_ptr(buffer) +
                                       static_cast<uint64_t>(expertIdx) * dispatchFlagSlotsPerExpert_ +
                                       static_cast<uint64_t>(rowOffsetInExpert / GMM1_TILE_M) * INT_CACHELINE;
    // wave-grain dispatch-gmm1 flag: per-expert 步长是 dispatchFlagSlotsPerExpert_,而不是 INT_CACHELINE。
    gmmAddrInfo.dispatchToGmm1Flag = (__gm__ int32_t*)workspace.sync.dispatchReady.get_ptr(buffer) +
                                     static_cast<uint64_t>(expertIdx) * dispatchFlagSlotsPerExpert_ +
                                     static_cast<uint64_t>(rowOffsetInExpert / GMM1_TILE_M) * INT_CACHELINE;
}

// ==================================================================================
// UpdateGmm2GlobalBuffer：绑定 GMM2 的激活输入、权重、输出与 flag 地址。
// ==================================================================================
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::UpdateGmm2GlobalBuffer(
    GMMAddrInfo& gmmAddrInfo, const ExpertLoopState& state, uint32_t rowOffsetInExpert)
{
    const auto& workspace = params_.workspace.layout;
    const auto& buffer = params_.workspace.base;
    // 地址 = 专家前累计行数 + 专家内偏移；行相关项随行数线性增长，权重/flag 只依赖专家序号。
    const uint64_t rows = static_cast<uint64_t>(state.globalTokenStartIndex) + rowOffsetInExpert;
    const uint32_t expertIdx = state.expertIdx;
    const uint64_t k = static_cast<uint64_t>(Get<K_VALUE>(state.problemShape));
    const uint64_t activationCols = static_cast<uint64_t>(Get<N_VALUE>(state.problemShape)) / activationOutputDivisor_;
    const uint64_t scaleN =
        Ops::Base::CeilDiv(activationCols, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
    gmmAddrInfo.gmm2OutGlobal = workspace.routed.gmm2Output.get_ptr(buffer) + rows * k * sizeof(bfloat16_t);
    gmmAddrInfo.metaInfoGlobal =
        workspace.routing.receivedTokenMetadata.get_ptr(buffer) + rows * META_INFO_SIZE * sizeof(int32_t);
    gmmAddrInfo.aGlobal = workspace.routed.gmm2Input.data.get_ptr(buffer) +
                          rows * activationCols / C_ELEMS_PER_BYTE * sizeof(ActivationType);
    gmmAddrInfo.aScaleGlobal =
        workspace.routed.gmm2Input.scale.get_ptr(buffer) + rows * scaleN * sizeof(QuantScaleOutType);
    gmmAddrInfo.bGlobal =
        GetExpertWeightAddr<Weight1Type>(params_.b2GmAddr, params_.tilingData->isPerExpertWeightTensor, expertIdx,
                                         static_cast<uint64_t>(expertIdx) * (k * activationCols / B_ELEMS_PER_BYTE));
    gmmAddrInfo.bScaleGlobal =
        GetExpertWeightAddr<QuantScaleOutType>(params_.b2ScaleGmAddr, params_.tilingData->isPerExpertWeightTensor,
                                               expertIdx, static_cast<uint64_t>(expertIdx) * (k * scaleN));
    gmmAddrInfo.gmm2CombineSyncCounter = (__gm__ int32_t*)workspace.urma.sync.combineCounters.get_ptr(buffer) +
                                         static_cast<uint64_t>(expertIdx) *
                                             params_.tilingData->compute.combineSyncSlotCountPerExpert *
                                             static_cast<uint64_t>(INT_CACHELINE);
    gmmAddrInfo.gmm2CombineLogicalCoreCount = blockNum_;
    gmmAddrInfo.activationToGmm2Flag = (__gm__ int32_t*)workspace.sync.activationReady.get_ptr(buffer) +
                                       static_cast<uint64_t>(expertIdx) * dispatchFlagSlotsPerExpert_ +
                                       static_cast<uint64_t>(rowOffsetInExpert / GMM1_TILE_M) * INT_CACHELINE;
    gmmAddrInfo.dispatchToGmm1Flag = (__gm__ int32_t*)workspace.sync.dispatchReady.get_ptr(buffer) +
                                     static_cast<uint64_t>(expertIdx) * dispatchFlagSlotsPerExpert_ +
                                     static_cast<uint64_t>(rowOffsetInExpert / GMM1_TILE_M) * INT_CACHELINE;
}

// ==================================================================================
// UpdateSharedGlobalBuffer：共享 GMM1 复用本卡量化记录，其余中间结果使用 shared workspace。
// ==================================================================================
template <TemplateMegaMoeLayeredTypeClass>
template <AddrUpdateMode Mode>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::UpdateSharedGlobalBuffer(
    GMMAddrInfo& gmmAddrInfo, const ExpertLoopState& state)
{
    const auto& workspace = params_.workspace.layout;
    const auto& buffer = params_.workspace.base;
    const uint64_t rows = static_cast<uint64_t>(state.globalTokenStartIndex);
    const uint32_t expertIdx = state.expertIdx;

    if constexpr (Mode == AddrUpdateMode::GMM1) {
        const uint64_t n = static_cast<uint64_t>(Get<N_VALUE>(state.problemShape));
        const uint64_t k = static_cast<uint64_t>(Get<K_VALUE>(state.problemShape));
        const uint64_t scaleK = Ops::Base::CeilDiv(k, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
        gmmAddrInfo.aGlobal = params_.communication.layout.dispatchReceive.get_ptr(params_.communication.base) +
                              RelayTokenOffset(serverId_, 0U);
        gmmAddrInfo.aScaleGlobal = gmmAddrInfo.aGlobal + mxQuantTokenAlignBytes_;
        if constexpr (MoeQuantConfig::AXW_MODE == AxWMode::A8W4) {
            gmmAddrInfo.gmm1OutGlobal = workspace.shared.gmm1Output.get_ptr(buffer) + rows * n * sizeof(bfloat16_t);
        }
        gmmAddrInfo.bGlobal =
            GetExpertWeightAddr<Weight1Type>(params_.sharedBGmAddr, params_.tilingData->isPerExpertWeightTensor,
                                             expertIdx, static_cast<uint64_t>(expertIdx) * (n * k / B_ELEMS_PER_BYTE));
        gmmAddrInfo.bScaleGlobal = GetExpertWeightAddr<QuantScaleOutType>(
            params_.sharedBScaleGmAddr, params_.tilingData->isPerExpertWeightTensor, expertIdx,
            static_cast<uint64_t>(expertIdx) * (n * scaleK));
    } else if constexpr (Mode == AddrUpdateMode::GMM2) {
        const uint64_t k = static_cast<uint64_t>(Get<K_VALUE>(state.problemShape));
        const uint64_t activationCols =
            static_cast<uint64_t>(Get<N_VALUE>(state.problemShape)) / activationOutputDivisor_;
        const uint64_t scaleN =
            Ops::Base::CeilDiv(activationCols, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
        gmmAddrInfo.gmm2OutGlobal = workspace.shared.gmm2Output.get_ptr(buffer) + rows * k * sizeof(bfloat16_t);
        gmmAddrInfo.aGlobal = workspace.shared.gmm2Input.data.get_ptr(buffer) +
                              rows * activationCols / C_ELEMS_PER_BYTE * sizeof(ActivationType);
        gmmAddrInfo.aScaleGlobal =
            workspace.shared.gmm2Input.scale.get_ptr(buffer) + rows * scaleN * sizeof(QuantScaleOutType);
        gmmAddrInfo.bGlobal = GetExpertWeightAddr<Weight1Type>(
            params_.sharedB2GmAddr, params_.tilingData->isPerExpertWeightTensor, expertIdx,
            static_cast<uint64_t>(expertIdx) * (k * activationCols / B_ELEMS_PER_BYTE));
        gmmAddrInfo.bScaleGlobal = GetExpertWeightAddr<QuantScaleOutType>(
            params_.sharedB2ScaleGmAddr, params_.tilingData->isPerExpertWeightTensor, expertIdx,
            static_cast<uint64_t>(expertIdx) * (k * scaleN));
    }
    gmmAddrInfo.activationToGmm2Flag = nullptr;
    gmmAddrInfo.dispatchToGmm1Flag = nullptr;
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::LockOwnedRemoteChannels()
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    if (subBlockIdx_ != 1U) {
        return;
    }
    // Wave 内 Dispatch 一级写、Dispatch 二级读和 Combine 写始终使用同一套固定归属核。
    // 在 Wave 开始前一次性取得全部归属通道，整个 Wave 内不重复 Lock/Unlock。
    for (uint32_t rank = 0; rank < worldSize_; ++rank) {
        if (rank == rankId_ || !IsChannelOwner(rank)) {
            continue;
        }
        comm_.Lock(comm_.GetChannel(rank));
    }
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::DrainAndUnlockOwnedRemoteChannels()
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    if (subBlockIdx_ != 1U) {
        return;
    }
    // Drain 全部归属远端 rank，不能只处理最后一个专家的活跃 rank。
    // 否则当最后一个专家无 token 时，早期专家的 WriteNbi 可能遗留到流水结束之后。
    // Drain 后 Unlock 发布通道 counter 并释放所有权，允许后续同步阶段换核使用同一通道。
    for (uint32_t rank = 0; rank < worldSize_; ++rank) {
        if (rank == rankId_ || !IsChannelOwner(rank)) {
            continue;
        }
        Channel channel = comm_.GetChannel(rank);
        BatchHandle batchHandle = comm_.MakeBatchHandle(channel, GetRankWinAddrWithOffset(rank, 0));
        // 同一通道可能同时残留 Dispatch 一级和 Combine 的已提交批次，统一按批量句柄等待完成。
        comm_.Drain(batchHandle);
        comm_.Unlock(channel);
    }
}

// ==============================================================================================
// CrossRankSyncInWorldSize：全卡同步，rank 槽位之后的独立区域记录各 AIV 的 syncCnt。
// ==============================================================================================
template <TemplateMegaMoeLayeredTypeClass>
template <bool ReuseChannelLock>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::CrossRankSyncInWorldSize()
{
    const auto& peermem = params_.communication.layout;
    const auto& commBuffer = params_.communication.base;
    if constexpr (g_coreType == AIC) {
        return;
    }
    __gm__ int32_t* syncRank = (__gm__ int32_t*)peermem.rankSync.get_ptr(commBuffer);
    __gm__ int32_t* syncCount =
        (__gm__ int32_t*)(peermem.syncCount.get_ptr(commBuffer) + aivCoreIdx_ * PEERMEM_SYNC_SLOT_SIZE);
    int count = GmRead(syncCount) + 1;
    GmWrite(syncCount, count);
    // E_dsburma1: 旁路 GM 写只进本核写队列，WriteNbi 的 DMA 引擎直读 GM，不嗅探未排空的写队列；
    // 不加 DSB 则 DMA 可能把上一轮 count 发往远端，对端 GmSignalWaitBarrier 等值等待永不命中即全卡挂死。
    GmDataBarrier();
    // 先向本 AIV 负责的所有 peer 发出通知，再进入等待。256P 场景下可同时维持多个
    // channel 在途，避免原实现逐 peer“写-等-Drain”造成的串行握手。
    for (int rankIndex = aivCoreIdx_; rankIndex < worldSize_; rankIndex += blockAivNum_) {
        if (rankIndex == rankId_) {
            continue;
        }
        Channel channel = comm_.GetChannel(rankIndex);
        if constexpr (!ReuseChannelLock) {
            // Wave 结束后同步任务重新按物理 AIV 分核，可能不同于 Wave 的固定归属核，因此先取得通道锁。
            comm_.Lock(channel);
        }
        __gm__ int32_t* syncRemoteAddr = (__gm__ int32_t*)(g_winRankAddr_[rankIndex]) + rankId_ * 16;
        comm_.WriteNbi(channel, (GM_ADDR)syncRemoteAddr, (GM_ADDR)syncCount, static_cast<int64_t>(sizeof(int32_t)));
    }
    for (int rankIndex = aivCoreIdx_; rankIndex < worldSize_; rankIndex += blockAivNum_) {
        if (rankIndex == rankId_) {
            continue;
        }
        Channel channel = comm_.GetChannel(rankIndex);
        auto syncCheck = syncRank + rankIndex * 16;
        GmSignalWaitBarrier(syncCheck, count);
        // 等待完成后释放通道锁：复用调用方已持有的锁时由本函数统一释放，否则释放本函数取得的锁。
        comm_.Drain(channel);
        comm_.Unlock(channel);
    }
    PipeBarrier<PIPE_ALL>();
    AllVectorBarrier();
}

// ===============================================================
// GroupMatmulWithActivationQuant：按实现路径分发到 A8W4 或 generic GMM1。
//                            A8W4 按量化配置选择；generic 路径的 subBlockIdx 判断已下沉到函数内部。
// ===============================================================
template <TemplateMegaMoeLayeredTypeClass>
template <bool IsShared, typename Epilogue, typename... LayoutArgs>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::RunGmm1WithEpilogue(
    Epilogue& epilogue, const GMMAddrInfo& gmmAddrInfo, const ExpertLoopState& state, uint32_t expertIdx,
    int32_t& vecSetSyncCom, uint32_t rowOffsetInExpert, const LayoutArgs&... layoutArgs)
{
    // Shared experts use the 256-row, non-prefetch epilogue; the caller binds A8W4 GM-output notifications.
    constexpr uint32_t epilogueTileM = IsShared ? L1_TILE_M_256 : EPILOGUE_TILE_M;
    constexpr bool prefetchWeights = !IsShared && TopkWeightsPrefetch;
    const uint32_t expertRowStart = static_cast<uint32_t>(state.globalTokenStartIndex) + rowOffsetInExpert;
    if constexpr (g_coreType == AIV) {
        const uint64_t rows = static_cast<uint64_t>(state.globalTokenStartIndex) + rowOffsetInExpert;
        const uint64_t n = static_cast<uint64_t>(Get<N_VALUE>(state.problemShape));
        const uint64_t activationCols = n / activationOutputDivisor_;
        const uint64_t scaleN =
            Ops::Base::CeilDiv(activationCols, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
        AscendC::Coord<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> vecBaseOffset{
            static_cast<int64_t>(rows * activationCols / C_ELEMS_PER_BYTE),
            static_cast<int64_t>(rows * scaleN),
            IsShared ? static_cast<int64_t>(state.expertIdx) :
                       static_cast<int64_t>(state.expertIdx) * maxWavesPerExpert_ + rowOffsetInExpert / GMM1_TILE_M,
            0L,
            0L,
            0L};
        epilogue.UpdateGlobalAddr(vecBaseOffset);
    }
    const BlockJobContext blockJob{MegaMoeHardwarePolicy::GetAicBlockIdx(), blockNum_};
    if constexpr (MoeQuantConfig::AXW_MODE == AxWMode::A8W4) {
        RunGmm1A8W4<QuantOutType, Weight1Type, bfloat16_t, QuantScaleOutType, QuantScaleOutType, GMM1_TILE_M,
                    epilogueTileM, prefetchWeights, IsShared, !IsShared>(
            blockContext_, epilogue, params_, state.problemShape, gmmAddrInfo, startBlockIdx_, blockJob, expertRowStart,
            expertIdx, activationOutputDivisor_, layoutArgs...);
    } else {
        RunGmm1Generic<QuantOutType, ActivationQuantOutType, QuantOutType, bfloat16_t, QuantScaleOutType,
                       QuantScaleOutType, Weight1Format != FORMAT_ND, GMM1_TILE_M, epilogueTileM, prefetchWeights,
                       IsShared, !IsShared>(epilogue, params_, state.problemShape, gmmAddrInfo, startBlockIdx_,
                                            vecSetSyncCom, blockJob, expertRowStart, expertIdx, gmm1PingPongIdx_,
                                            activationOutputDivisor_, nullptr, false, layoutArgs...);
    }
}

template <TemplateMegaMoeLayeredTypeClass>
template <bool IsShared>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::GroupMatmulWithActivationQuant(
    const GMMAddrInfo& gmmAddrInfo, const ExpertLoopState& state, uint32_t expertIdx, int32_t& vecSetSyncCom,
    uint32_t rowOffsetInExpert)
{
    if constexpr (IsShared) {
        // 量化记录交织存储，GMM 输入和 scale 的行跨度分别以各自逻辑元素为单位。
        const StridedAConfig inputLayout{static_cast<uint32_t>(relayRecordBytes_) * A_ELEMS_PER_BYTE,
                                         static_cast<uint32_t>(relayRecordBytes_ / sizeof(QuantScaleOutType))};
        RunGmm1WithEpilogue<IsShared>(sharedEpilogueOp_, gmmAddrInfo, state, expertIdx, vecSetSyncCom, 0U, inputLayout);
    } else {
        RunGmm1WithEpilogue<IsShared>(epilogueOp_, gmmAddrInfo, state, expertIdx, vecSetSyncCom, rowOffsetInExpert);
    }
}

// ===============================================================
// GroupMatmulWithCombine：先按实现路径分发，再按 combine 模式分发。
// IsShared=true 时跳过 activation flag 等待和 Combine 后处理，供共享专家使用。
// ===============================================================
template <TemplateMegaMoeLayeredTypeClass>
template <bool IsShared>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::GroupMatmulWithCombine(
    const GMMAddrInfo& gmmAddrInfo, const ExpertLoopState& state, uint32_t expertTokenCount, uint32_t rowOffsetInExpert)
{
    const BlockJobContext blockJob{blockIdx_, blockNum_};
    if constexpr (MoeQuantConfig::AXW_MODE == AxWMode::A8W4) {
        RunGmm2A8W4<ActivationQuantOutType, Weight1Type, bfloat16_t, QuantScaleOutType, QuantScaleOutType,
                    L1_TILE_M_256, TopkWeightsPrefetch, IsShared, true, !IsShared, false>(
            blockContext_, state.problemShape, gmmAddrInfo, startBlockIdx_, blockJob, expertTokenCount,
            rowOffsetInExpert, nullptr, activationOutputDivisor_);
    } else if constexpr (MoeQuantConfig::AXW_MODE == AxWMode::A4W4) {
        // 空 slice 三角色整对跳过，避免每 slice 局部构造的 BlockMmad/BlockPrologue 跨核额度收支失配。
        // 跳过后额度 flag（SYNC_AIV_AIC_FLAG）未消费计数恒 <16：AIC 侧构造 Set 与逐 chunk 就绪 Wait
        // 同在 PIPE_MTE1，AIV0 侧额度 Wait、就绪 Set 与析构回收 Wait 同在 PIPE_MTE3；两条管内序使
        // AIC 发第 N 个 slice 的构造额度前必已走完第 N-1 个 slice 的就绪 Wait，彼时 AIV0 已回收完
        // 第 N-2 个及更早 slice 的全部额度，在途至多两批即 2*bBufferNum；Host L1 布局（512KiB，
        // A/B 槽各 64KiB，另留两组最小 scale）给出 bBufferNum<=6，峰值 12<16。
        if (SkipProblemWithoutWorkOnAllRoles(
                GetGmmProblemTileCount(Get<M_VALUE>(state.problemShape), Get<K_VALUE>(state.problemShape)), blockJob,
                startBlockIdx_)) {
            return;
        }
        // A4W4 GMM1 uses a different block; keep the FP8/W4 objects local to GMM2.
        if constexpr (g_coreType == AscendC::AIC) {
            typename A8W4Config::BlockMmad block(params_.tilingData->a8w4L1Layout);
            typename A8W4Config::BlockContext context{&block};
            RunGmm2A8W4<ActivationQuantOutType, Weight1Type, bfloat16_t, QuantScaleOutType, QuantScaleOutType,
                        L1_TILE_M_256, TopkWeightsPrefetch, IsShared, true, !IsShared, false>(
                context, state.problemShape, gmmAddrInfo, startBlockIdx_, blockJob, expertTokenCount, rowOffsetInExpert,
                nullptr, activationOutputDivisor_);
        } else {
            if (GetSubBlockIdx() == 0U) {
                typename A8W4Config::BlockPrologue block(params_.tilingData->a8w4L1Layout);
                typename A8W4Config::BlockContext context{&block};
                RunGmm2A8W4<ActivationQuantOutType, Weight1Type, bfloat16_t, QuantScaleOutType, QuantScaleOutType,
                            L1_TILE_M_256, TopkWeightsPrefetch, IsShared, true, !IsShared, false>(
                    context, state.problemShape, gmmAddrInfo, startBlockIdx_, blockJob, expertTokenCount,
                    rowOffsetInExpert, nullptr, activationOutputDivisor_);
            } else {
                RunGmm2A8W4<ActivationQuantOutType, Weight1Type, bfloat16_t, QuantScaleOutType, QuantScaleOutType,
                            L1_TILE_M_256, TopkWeightsPrefetch, IsShared, true, !IsShared, false>(
                    typename A8W4Config::BlockContext{}, state.problemShape, gmmAddrInfo, startBlockIdx_, blockJob,
                    expertTokenCount, rowOffsetInExpert, nullptr, activationOutputDivisor_);
            }
        }
    } else {
        RunGmm2Generic<CombineQuantMode, QuantOutType, QuantOutType, bfloat16_t, QuantScaleOutType, QuantScaleOutType,
                       Weight2Format != FORMAT_ND, true, L1_TILE_M_256, TopkWeightsPrefetch, IsShared, !IsShared>(
            state.problemShape, gmmAddrInfo, startBlockIdx_, nullptr, false, rowOffsetInExpert, nullptr,
            activationOutputDivisor_, expertTokenCount);
    }
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::ProcessSharedExpertGmm1Loop(
    GMMAddrInfo& sharedGmm1AddrInfo, ExpertLoopState& sharedGmm1State, int32_t& vecSetSyncCom)
{
    for (uint32_t sharedIdx = 0; sharedIdx < sharedExpertNum_; sharedIdx++) {
        UpdateExpertLoopState(sharedGmm1State, sharedIdx, m_);
        UpdateSharedGlobalBuffer<AddrUpdateMode::GMM1>(sharedGmm1AddrInfo, sharedGmm1State);
        GroupMatmulWithActivationQuant<true>(sharedGmm1AddrInfo, sharedGmm1State, sharedIdx, vecSetSyncCom);
    }
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::ProcessSharedExpertGmm1(
    Gmm1ActivationSync& sharedGmm1ActivationSync)
{
    const auto& workspace = params_.workspace.layout;
    const auto& buffer = params_.workspace.base;
    sharedEpilogueOp_.Init({.yGmAddr = workspace.shared.gmm2Input.data.get_ptr(buffer),
                            .yScaleGmAddr = workspace.shared.gmm2Input.scale.get_ptr(buffer),
                            .clampLimit = params_.tilingData->clampLimit,
                            .actMode = params_.tilingData->actMode,
                            .actSubMode = params_.tilingData->actSubMode,
                            .activationAlpha = params_.tilingData->activationAlpha,
                            .activationBeta = params_.tilingData->activationBeta});

    GMMAddrInfo sharedGmm1AddrInfo{};
    if constexpr (MoeQuantConfig::AXW_MODE == AxWMode::A8W4) {
        sharedGmm1AddrInfo.gmm1ActivationSync = &sharedGmm1ActivationSync;
    }
    ExpertLoopState sharedGmm1State = CreateExpertLoopState(commonConfig_);
    int32_t vecSetSyncCom = 0;
    if constexpr (MoeQuantConfig::AXW_MODE == AxWMode::A8W4) {
        if (g_coreType == AscendC::AIC || GetSubBlockIdx() == 0U) {
            typename A8W4Config::BlockContext::Block block(params_.tilingData->a8w4L1Layout);
            blockContext_ = {&block};
            ProcessSharedExpertGmm1Loop(sharedGmm1AddrInfo, sharedGmm1State, vecSetSyncCom);
            blockContext_ = {};
        } else {
            ProcessSharedExpertGmm1Loop(sharedGmm1AddrInfo, sharedGmm1State, vecSetSyncCom);
        }
    } else {
        ProcessSharedExpertGmm1Loop(sharedGmm1AddrInfo, sharedGmm1State, vecSetSyncCom);
    }
    Gmm1UbActivationSync::EndSync(vecSetSyncCom, gmm1PingPongIdx_);
    gmm1PingPongIdx_ = 0U;
    startBlockIdx_ = 0; // 共享专家GMM1修改了startBlockIdx_，重置给GMM1使用
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::ProcessSharedExpertGmm2Loop(
    GMMAddrInfo& sharedGmm2AddrInfo, ExpertLoopState& sharedGmm2State)
{
    for (uint32_t sharedIdx = 0; sharedIdx < sharedExpertNum_; sharedIdx++) {
        UpdateExpertLoopState(sharedGmm2State, sharedIdx, m_);
        UpdateSharedGlobalBuffer<AddrUpdateMode::GMM2>(sharedGmm2AddrInfo, sharedGmm2State);
        GroupMatmulWithCombine<true>(sharedGmm2AddrInfo, sharedGmm2State,
                                     static_cast<uint32_t>(Get<M_VALUE>(sharedGmm2State.problemShape)));
    }
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::ProcessSharedExpertGmm2()
{
    GMMAddrInfo sharedGmm2AddrInfo{};
    ExpertLoopState sharedGmm2State = CreateExpertLoopState(commonConfig_);
    if constexpr (MoeQuantConfig::AXW_MODE == AxWMode::A8W4) {
        if (g_coreType == AscendC::AIC || GetSubBlockIdx() == 0U) {
            typename A8W4Config::BlockContext::Block block(params_.tilingData->a8w4L1Layout);
            blockContext_ = {&block};
            ProcessSharedExpertGmm2Loop(sharedGmm2AddrInfo, sharedGmm2State);
            blockContext_ = {};
        } else {
            ProcessSharedExpertGmm2Loop(sharedGmm2AddrInfo, sharedGmm2State);
        }
    } else {
        ProcessSharedExpertGmm2Loop(sharedGmm2AddrInfo, sharedGmm2State);
    }
    AllCoreBarrier();
}

template <TemplateMegaMoeLayeredTypeClass>
template <AddrUpdateMode Mode>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::ProcessGmmTokenRange(
    const ExpertTokenRange& range, ExpertLoopState& state, GMMAddrInfo& addrInfo, int32_t& vecSetSyncCom)
{
    uint32_t expertEnd = range.end.expertIdx + (range.end.tokenIndexInExpert == 0U ? 0U : 1U);
    for (uint32_t expertIdx = range.begin.expertIdx; expertIdx < expertEnd; ++expertIdx) {
        uint32_t rowBegin = expertIdx == range.begin.expertIdx ? range.begin.tokenIndexInExpert : 0U;
        if (rowBegin == 0U) {
            // 按专家顺序推进累计行偏移：空专家直接跳过，非空专家把 M 换成当前专家行数。
            const BlockWorkspaceContext countWorkspace{blockIdx_, aicNum_};
            const uint32_t expertTokenCount = GetExpertTokenCountFromWorkspace(
                params_.workspace.layout.routing.recvCounts.get_ptr(params_.workspace.base), countWorkspace,
                moeExpertPerRank_, expertIdx);
            UpdateExpertLoopState(state, expertIdx, expertTokenCount);
            if (expertTokenCount == 0U) {
                continue;
            }
        }
        uint32_t expertRows = static_cast<uint32_t>(Get<M_VALUE>(state.problemShape));
        uint32_t rowEnd = expertIdx == range.end.expertIdx ? range.end.tokenIndexInExpert : expertRows;
        if (rowEnd <= rowBegin) {
            continue;
        }
        // 临时计算视图只切 M；地址由 globalTokenStartIndex + 专家内偏移在绑定时计算。
        ExpertLoopState problem = state;
        Get<M_VALUE>(problem.problemShape) = static_cast<int64_t>(rowEnd - rowBegin);
        if constexpr (Mode == AddrUpdateMode::GMM1) {
            UpdateGmm1GlobalBuffer(addrInfo, problem, rowBegin);
            GroupMatmulWithActivationQuant(addrInfo, problem, expertIdx, vecSetSyncCom, rowBegin);
        } else {
            UpdateGmm2GlobalBuffer(addrInfo, problem, rowBegin);
            GroupMatmulWithCombine(addrInfo, problem, expertRows, rowBegin);
        }
    }
}

// =================================================================================================
// ProcessMoeExpertWaveLoop：按“计算 wave + 通信 wave”两层结构滚动执行 GMM1/GMM2/Combine。
//   - 计算 wave（GMM wave）：一次 GMM1 + GMM2，消费一个 ExpertTokenRange；
//   - 通信 wave：固定每个 comm wave 承载 gmmWavesPerCommunicationWave = ceil(gmmWaveCount /
//     MAX_COMMUNICATION_WAVES) 个计算 wave；最后不足的部分作为尾块；
//     每个 comm wave 执行一次 Dispatch 接收、gmmWavesPerCommunicationWave 个计算 wave、一次 Combine 提交；
//   - currentWave：下一个待执行的计算 wave；communicationWave：当前通信 wave 已覆盖的范围；
//   - combineWaveBeginState：通信 wave 起点处的 GMM2 游标；
//   - gmmWavesPerCommunicationWave 只在 AIV1 上计算，AIC/AIV0 固定为 1；
//   - AIV1 是唯一通信/Combine 角色；AIC/AIV0 只执行计算 wave。
// =================================================================================================
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::ProcessMoeExpertWaveLoop(
    ExpertLoopState& gmm1State, ExpertLoopState& gmm2State, GMMAddrInfo& gmm1AddrInfo, GMMAddrInfo& gmm2AddrInfo,
    int32_t& vecSetSyncCom, ExpertTokenRange currentWave)
{
    constexpr uint32_t MAX_COMMUNICATION_WAVES = 2U;
    uint64_t gmmWavesPerCommunicationWave = 1U;
    if constexpr (g_coreType == AIV) {
        if (subBlockIdx_ == 1U) {
            // 固定每个 comm wave 承载 gmmWavesPerCommunicationWave = ceil(gmmWaveCount / MAX_COMMUNICATION_WAVES)，
            // 最后不足的部分作为尾块。
            gmmWavesPerCommunicationWave =
                gmmWaveCount_ == 0U ? 1U : (gmmWaveCount_ - 1U) / MAX_COMMUNICATION_WAVES + 1U;
        }
    }
    // [prefetch] 通信预取：AIV1 在第 1 段计算期间就把第 2 段的 Dispatch 接收发起，隐藏远端到达延迟。
    // dispatchRevData 按 maxOutputSize 整块分配，提前接收不增加内存占用。
    bool dispatchPrefetched = false;
    while (currentWave.begin.expertIdx < moeExpertPerRank_) {
        // 外层通信 wave：AIV1 一次性接收一个 comm wave 的 gmmWavesPerCommunicationWave 个计算 wave 范围；
        // AIC/AIV0 固定按 1 个计算 wave 处理。
        ExpertTokenRange communicationWave = currentWave;
        ExpertLoopState combineWaveBeginState = gmm2State;
        if constexpr (g_coreType == AIV) {
            if (subBlockIdx_ == 1U) {
                // 整个 comm wave 的 M-group 预算 = gmmWavesPerCommunicationWave * mGroupsPerWave_；
                // 该值受 maxOutputSize/256 约束，uint32 足够。
                const uint64_t groupTarget64 = gmmWavesPerCommunicationWave * static_cast<uint64_t>(mGroupsPerWave_);
                if (!dispatchPrefetched) {
                    ExpertTokenRange dispatchWave =
                        PlanTokenWave(currentWave.begin, static_cast<uint32_t>(groupTarget64));
                    ReceiveDispatchTokenRange(dispatchWave);
                    // 预取下一通信 wave 的范围：从本段起点按计算 wave 推进 gmmWavesPerCommunicationWave 次。
                    ExpertTokenRange nextWaveBegin = currentWave;
                    for (uint64_t i = 0U; i < gmmWavesPerCommunicationWave; ++i) {
                        nextWaveBegin = PlanTokenWave(nextWaveBegin.end, mGroupsPerWave_);
                    }
                    if (nextWaveBegin.begin.expertIdx < moeExpertPerRank_) {
                        ExpertTokenRange nextDispatch =
                            PlanTokenWave(nextWaveBegin.begin, static_cast<uint32_t>(groupTarget64));
                        ReceiveDispatchTokenRange(nextDispatch);
                        dispatchPrefetched = true;
                    }
                }
            }
        }

        // 内层计算 wave：满 comm wave 预算的 gmmWavesPerCommunicationWave 个走正常路径，
        // 最后不足的部分作为尾块处理。
        for (uint64_t gmmWaveIdx = 0U;
             gmmWaveIdx < gmmWavesPerCommunicationWave && currentWave.begin.expertIdx < moeExpertPerRank_;
             ++gmmWaveIdx) {
            // 单个计算 wave：GMM1、GMM2 使用同一 token 区间；GMM1 结束后的 block 游标用于固定角色共振。
            const uint32_t waveStartBlockIdx = startBlockIdx_;
            ProcessGmmTokenRange<AddrUpdateMode::GMM1>(currentWave, gmm1State, gmm1AddrInfo, vecSetSyncCom);
            const uint32_t gmm1EndBlockIdx = startBlockIdx_;
            ProcessGmmTokenRange<AddrUpdateMode::GMM2>(currentWave, gmm2State, gmm2AddrInfo, vecSetSyncCom);
            communicationWave.end = currentWave.end;
            AdjustStartBlockIdxForNextWave(currentWave, waveStartBlockIdx, gmm1EndBlockIdx);
            // 当前计算 wave 的 GMM 调用结束后，再推进下一计算 wave 的边界。
            currentWave = PlanTokenWave(currentWave.end, mGroupsPerWave_);
        }

        // 通信 wave 尾：以当前通信 wave 范围执行 Combine；WQE 可跨计算 wave 累积，尾批在此提交。
        ProcessCombineTokenRange(combineWaveBeginState, communicationWave);
    }
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::ProcessMoeExpertWave(
    Gmm1ActivationSync& gmm1ActivationSync)
{
    const auto& workspace = params_.workspace.layout;
    const auto& buffer = params_.workspace.base;
    DispatchBuffInit();
    InitCombineBuffers();
    InitDispatchSourceReadyCache();
    LockOwnedRemoteChannels();
    ExpertLoopState gmm1State = CreateExpertLoopState(commonConfig_);
    ExpertLoopState gmm2State = CreateExpertLoopState(commonConfig_);
    GMMAddrInfo gmm1AddrInfo{};
    GMMAddrInfo gmm2AddrInfo{};
    // GM 结果交接使用逐 tile 硬同步；A8W4 由 AIV1 激活，generic 预取由 AIV0 激活。
    if constexpr (MoeQuantConfig::AXW_MODE == AxWMode::A8W4 || TopkWeightsPrefetch) {
        gmm1AddrInfo.gmm1ActivationSync = &gmm1ActivationSync;
    }
    int32_t vecSetSyncCom = 0;
    if constexpr (g_coreType == AIV) {
        if (subBlockIdx_ == 1U) {
            // 量化和 Combine 元数据已准备完成，一级通信异步提交，
            // 使本 Server 的二级接收能够与其重叠。
            StartInterServerDispatch();
            PrepareExpertTokenCountTable();
        }
    }
    // 每个 block 只等待一次完整计数表；输入数据仍按 256 行分组独立等待就绪。
    if (moeExpertPerRank_ != 0U && (g_coreType == AIC || subBlockIdx_ == 0U)) {
        const BlockWorkspaceContext countWorkspace{blockIdx_, aicNum_};
        WaitForMoeExpertTokenCountReady(workspace.sync.countsReady.get_ptr(buffer), countWorkspace, 0U);
    }
    ExpertTokenRange currentWave = PlanTokenWave({}, mGroupsPerWave_);

    if constexpr (MoeQuantConfig::AXW_MODE == AxWMode::A8W4) {
        if (g_coreType == AscendC::AIC || GetSubBlockIdx() == 0U) {
            typename A8W4Config::BlockContext::Block block(params_.tilingData->a8w4L1Layout);
            blockContext_ = {&block};
            ProcessMoeExpertWaveLoop(gmm1State, gmm2State, gmm1AddrInfo, gmm2AddrInfo, vecSetSyncCom, currentWave);
            blockContext_ = {};
        } else {
            ProcessMoeExpertWaveLoop(gmm1State, gmm2State, gmm1AddrInfo, gmm2AddrInfo, vecSetSyncCom, currentWave);
        }
    } else {
        ProcessMoeExpertWaveLoop(gmm1State, gmm2State, gmm1AddrInfo, gmm2AddrInfo, vecSetSyncCom, currentWave);
    }

    // UB 双缓冲在本阶段回收；GM 硬同步的剩余 ACK 由 Process 末尾统一回收。
    if constexpr (MoeQuantConfig::AXW_MODE != AxWMode::A8W4 && !TopkWeightsPrefetch) {
        Gmm1UbActivationSync::EndSync(vecSetSyncCom, gmm1PingPongIdx_);
    }
    gmm1PingPongIdx_ = 0U;
    if constexpr (g_coreType == AIV) {
        if (IsCombineServerDedupEnabled()) {
            ProcessCombineRelay();
        }
        if (subBlockIdx_ == 1U) {
            PipeBarrier<PIPE_ALL>();
            ExpertTokenNumCopyOut();
            DrainAndUnlockOwnedRemoteChannels();
        }
        PipeBarrier<PIPE_ALL>();
        AllVectorBarrier();
    }
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::PrepareWithoutSharedExpert()
{
    const AivJobContext maskJob{aivCoreIdx_, blockAivNum_};
    if constexpr (g_coreType == AIV) {
        SendAndQuantBuffInit(maskJob.totalJobs);
        ResetState();
        const auto expertRange = GetBalancedWorkRange(worldSize_ * moeExpertPerRank_, maskJob);
        if (expertRange.count != 0U) {
            ComputeExpertMasks(expertRange);
        }
        // 同一 MTE3 队列上的清零、mask 和 count 写回完成后，再允许 owner 提交通信。
        SyncInterArrive<PIPE_MTE3>(ALL_AIV_BARRIER_EVENT);
        SyncInterWait<PIPE_S>(ALL_AIV_BARRIER_EVENT);
        SendExpertMasks(maskJob);
        PipeBarrier<PIPE_ALL>();
    }
    AllCoreBarrier(); // AIC 等待同步标志清零完成

    if constexpr (g_coreType == AIV) {
        // 所有物理 AIV 在 CrossRankSync 前完成各自 token 分片的量化。
        DispatchPrepareBuffInit();
        QuantizeLocalTokensToRelay();
        // Server Combine 仍需位图和有效 token 数量；
        // 普通整批发送路径不再构造或读取 token 队列。
        if (IsCombineServerDedupEnabled()) {
            PrepareDispatchRelayQueues(maskJob);
        }
        // 首次跨卡同步前收口 SendMask 的发送完成，使同步通知代表本 AIV 的 mask 已发送。
        for (int rankIndex = aivCoreIdx_; rankIndex < worldSize_; rankIndex += blockAivNum_) {
            if (rankIndex != rankId_) {
                comm_.Drain(comm_.GetChannel(rankIndex));
            }
        }
        // 每个对端的跨卡同步通知由一个 AIV 发送，发送前须确保本卡所有 AIV 均已完成量化，
        // 使本 Server 的二级接收可以直接读取原始记录。
        AllVectorBarrier();
    }

    CrossRankSyncInWorldSize<true>(); // 同一物理 AIV 继续使用 SendMask 已取得的锁，并在同步结束后释放。
}

/* 全部 AIV 清理本轮状态，并量化各自负责的本卡输入。 */
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::PrepareSharedExpertInput(
    uint32_t maskWorkerCount)
{
    SendAndQuantBuffInit(maskWorkerCount);
    ResetState();
    PipeFence<PIPE_MTE3, PIPE_MTE2, SYNC_EVENT_ID3>(); // 清零搬出读完 UB 后，允许量化搬入覆盖。
    DispatchPrepareBuffInit();
    QuantizeLocalTokensToRelay();
}

/* mask 发送核交还通道锁，供首次跨卡同步重新分配 owner。 */
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::DrainAndUnlockMaskChannels(
    const AivJobContext& job)
{
    for (uint32_t rankIndex = job.jobIndex; rankIndex < worldSize_; rankIndex += job.totalJobs) {
        if (rankIndex != rankId_) {
            Channel channel = comm_.GetChannel(rankIndex);
            comm_.Drain(channel);  // 等待该通道的 mask 写请求完成。
            comm_.Unlock(channel); // 交还通道，供同步核重新持锁。
        }
    }
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::PrepareWithSharedExpert(
    Gmm1ActivationSync& sharedGmm1ActivationSync)
{
    const AivJobContext maskJob{blockIdx_, blockNum_};
    if constexpr (g_coreType == AIV) {
        PrepareSharedExpertInput(maskJob.totalJobs);
        GmDataBarrier(); // 发布输入就绪前，确认 Scalar 队列计数清零已写回 GM。
        SyncInterArrive<PIPE_MTE3>(TOKEN_DATA_READY_FLAG); // 本核量化写回后到达，空任务核也参与。
    }

    if constexpr (g_coreType == AIC) {
        // AIC：共享 GMM1 计算。
        SyncIntraWait<PIPE_MTE2>(INPUT_PREPARE_DONE_FLAG); // 等 AIV0 确认全体量化完成，再读取输入。
        PipeNotify<PIPE_MTE2, PIPE_S>(SYNC_EVENT_ID3);     // 记录输入就绪，也承接全体清零完成条件。
        ProcessSharedExpertGmm1(sharedGmm1ActivationSync);
        PipeWait<PIPE_MTE2, PIPE_S>(SYNC_EVENT_ID3); // 后续 Scalar 读 countsReady 前，确认本轮已清零。
    } else if (subBlockIdx_ == 0U) {
        // AIV0：共享 GMM1 的权重转换或激活。
        SyncInterWait<PIPE_MTE3>(TOKEN_DATA_READY_FLAG);     // 等全体量化写回，约束后续输入通知。
        SyncIntraArrive<PIPE_MTE3>(INPUT_PREPARE_DONE_FLAG); // 放行配对 AIC 的 MTE2 输入读取。
        if constexpr (MoeQuantConfig::AXW_MODE == AxWMode::A8W4) {
            PipeFence<PIPE_MTE3, PIPE_MTE2, SYNC_EVENT_ID3>(); // 量化搬出读完 UB 后，允许权重搬入。
        } else {
            PipeFence<PIPE_MTE3, PIPE_V, SYNC_EVENT_ID3>(); // 量化搬出读完 UB 后，允许激活写入。
        }
        SyncInterArrive<PIPE_MTE3>(EXPERT_MASK_READY_FLAG); // AIV0 无 mask 任务，提前到达。
        ProcessSharedExpertGmm1(sharedGmm1ActivationSync);
        SyncInterWait<PIPE_MTE3>(EXPERT_MASK_READY_FLAG); // 等 mask 完成；等待延至共享计算之后。
    } else {
        // AIV1：mask、建表，随后执行 A8W4 共享激活。
        PipeFence<PIPE_MTE3, PIPE_V, SYNC_EVENT_ID3>(); // mask 写 UB 前，等量化搬出读完。
        const auto expertRange = GetBalancedWorkRange(worldSize_ * moeExpertPerRank_, maskJob);
        if (expertRange.count != 0U) {
            ComputeExpertMasks(expertRange);
        }
        SyncInterWait<PIPE_MTE3>(TOKEN_DATA_READY_FLAG);    // 等全体量化完成，随后发布 mask 就绪。
        SyncInterArrive<PIPE_MTE3>(EXPERT_MASK_READY_FLAG); // 本核 mask/count 写回后发布就绪。
        SyncInterWait<PIPE_S>(EXPERT_MASK_READY_FLAG);      // 提交通信前，等全部 mask/count 写回。
        SendExpertMasks(maskJob);
        if (IsCombineServerDedupEnabled()) {
            PrepareDispatchRelayQueues(maskJob);
            if constexpr (MoeQuantConfig::AXW_MODE == AxWMode::A8W4) {
                PipeFence<PIPE_MTE3, PIPE_MTE2, SYNC_EVENT_ID3>(); // 激活搬入覆盖 UB 前，等位图搬出读完。
            }
        }
        ProcessSharedExpertGmm1(sharedGmm1ActivationSync);
        DrainAndUnlockMaskChannels(maskJob);
    }

    if constexpr (g_coreType == AIV) {
        AllVectorBarrier(); // 等全体 AIV 完成共享工作、建表及 mask 通道解锁。
    }
    CrossRankSyncInWorldSize<false>(); // 跨卡确认量化与 mask 就绪，再进入 MoE wave。
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::Process()
{
    int64_t oriOverflowMode = GetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>();
    SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(0);
    Gmm1ActivationSync sharedGmm1ActivationSync(1U, GmmEventPair::SHARED_GMM1);
    if (sharedExpertNum_ > 0) {
        PrepareWithSharedExpert(sharedGmm1ActivationSync);
    } else {
        PrepareWithoutSharedExpert();
    }

    // URMA Layered 的 A8W8、A4W4 和 A8W4 共用同一专家 Wave 状态机。
    Gmm1ActivationSync gmm1ActivationSync(MoeQuantConfig::AXW_MODE == AxWMode::A8W4 ? 1U : 0U, GmmEventPair::MOE_GMM1);
    ProcessMoeExpertWave(gmm1ActivationSync);

    // 3.5: 共享专家 GMM2 (MoE GMM2 之后, 复用 MoE 函数)
    if (sharedExpertNum_ > 0) {
        ProcessSharedExpertGmm2();
    }

    // 4. 本卡数据Unpermute
    if constexpr (g_coreType == AIV) {
        CrossRankSyncInWorldSize<false>(); // Wave 已释放固定归属锁，同步核重新取得通道后确认 Combine 完成。
        if (IsCombineServerDedupEnabled()) {
            UnpermuteServerCombine();
        } else {
            const auto unpermuteBufferConfig = UnpermuteBuffInit();
            Unpermute(unpermuteBufferConfig);
        }
    }
    // 两条 GM 激活同步链独立，在主流程末尾回收剩余 ACK，让共享 GMM2 先推进。
    sharedGmm1ActivationSync.EndSync();
    gmm1ActivationSync.EndSync();
    SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(oriOverflowMode);
}

#endif

} // namespace MegaMoeImpl

#if defined(ENABLE_MEGA_MOE_LAYERED_KERNEL)
#include "../arch35_92/stage/mega_moe_layered_dispatch.h"
#include "../arch35_92/stage/mega_moe_layered_combine.h"
#include "../arch35_92/stage/mega_moe_layered_combine_relay.h"
#endif

#undef TemplateMegaMoeLayeredTypeClass
#undef TemplateMegaMoeLayeredTypeFunc
#endif
