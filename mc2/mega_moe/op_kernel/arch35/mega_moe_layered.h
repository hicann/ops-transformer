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

#if __has_include("version/asc_devkit_version.h") && __has_include("version/hcomm_version.h")
#include "version/asc_devkit_version.h"
#include "version/hcomm_version.h"

#if (ASC_DEVKIT_MAJOR > 9 || (ASC_DEVKIT_MAJOR == 9 && ASC_DEVKIT_MINOR > 0)) && \
    (HCOMM_MAJOR > 9 || (HCOMM_MAJOR == 9 && HCOMM_MINOR > 0))
#define ENABLE_MEGA_MOE_LAYERED_KERNEL
#endif

#endif

#include "kernel_tiling/kernel_tiling.h"
#if __has_include("../../common/mc2_kernel_utils.h")
#include "../../common/mc2_kernel_utils.h"
#else
#include "../../../common/op_kernel/mc2_kernel_utils.h"
#endif
#include "kernel_operator_list_tensor_intf.h"
#include "common/mega_moe_types.h"
#include "common/mega_moe_workspace.h"
#include "common/mega_moe_utils.h"
#include "blaze/epilogue/block_epilogue_activation_mx_quant.h"
#include "stage/mega_moe_token_quant.h"
#include "stage/mega_moe_gmm1_activation.h"
#include "stage/mega_moe_gmm2_combine.h"
#include "common/mega_moe_mxfp8_utils.h"
#if __has_include("../../common/quantize_functions.h")
#include "../../common/quantize_functions.h"
#else
#include "../../../common/op_kernel/quantize_functions.h"
#endif

namespace MegaMoeImpl {

using namespace AscendC;

constexpr uint32_t UNPERMUTE_LIST_NUM = 3U;
constexpr int64_t HALF_TO_FP32 = 2U;
constexpr int64_t DEQUANT_SCALE_EXPAND = 2U;

#if defined(ENABLE_MEGA_MOE_LAYERED_KERNEL)
using TupleShape = Shape<int64_t, int64_t, int64_t, int64_t>;
using BlockOffset = Shape<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                          int64_t, int64_t, int64_t, int64_t>;

// 预留：XType OutputType TopkWeightsType Weight1Type
#define TemplateMegaMoeLayeredTypeClass \
    typename XType, typename OutputType, typename TopkWeightsType, typename Weight1Type, int32_t QuantMode, \
        int32_t CombineQuantMode, bool TopkWeightsPrefetch, bool IsGmm1Interleaved
#define TemplateMegaMoeLayeredTypeFunc \
    XType, OutputType, TopkWeightsType, Weight1Type, QuantMode, CombineQuantMode, TopkWeightsPrefetch, IsGmm1Interleaved

template <TemplateMegaMoeLayeredTypeClass>
class MegaMoeLayered {
public:
    template <int32_t QM>
    struct QuantTraits {
        using OutType = fp8_e4m3fn_t;
    };
    template <>
    struct QuantTraits<E5M2_QUANT> {
        using OutType = fp8_e5m2_t;
    };
    template <>
    struct QuantTraits<E2M1_QUANT> {
        using OutType = fp4x2_e2m1_t;
    };
    using QuantOutType = typename QuantTraits<QuantMode>::OutType;
    using ActivationType =
        typename std::conditional<Std::IsSame<QuantOutType, fp4x2_e2m1_t>::value, uint8_t, QuantOutType>::type;
    using QuantScaleOutType = typename std::conditional<(QuantMode >= E5M2_QUANT), fp8_e8m0_t, float>::type;
    struct ExpertLoopState {
        TupleShape problemShape;
        BlockOffset baseOffset;
        // Rows before the current expert, kept per cursor for dispatch/GMM prefetch state split.
        uint32_t expertBeforeCnt = 0;
    };
    __aicore__ inline MegaMoeLayered(){};
    __aicore__ inline void Init(GM_ADDR context, GM_ADDR x, GM_ADDR topkIds, GM_ADDR topkWeights, GM_ADDR weight1,
                                GM_ADDR weight2, GM_ADDR xActiveMask, GM_ADDR weightScales1, GM_ADDR weightScales2,
                                GM_ADDR scales, GM_ADDR sharedWeight1, GM_ADDR sharedWeight2,
                                GM_ADDR sharedWeightScales1, GM_ADDR sharedWeightScales2, GM_ADDR yOut,
                                GM_ADDR expertTokenNumsOut, GM_ADDR workspaceGM, MegaMoeTilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitTopology(GM_ADDR context, const MegaMoeTilingData *tilingData);
    __aicore__ inline void InitDispatchLayout(const MegaMoeTilingData *tilingData);
    __aicore__ inline void InitWorkspace(GM_ADDR workspaceGM, MegaMoeTilingData *tilingData);
    __aicore__ inline uint32_t InitResetBatch();
    __aicore__ inline void InitSendMaskRounds(uint32_t resetTensorSize);
    __aicore__ inline void InitDispatchRounds(uint32_t fixedBefore, uint32_t dispatchFixedCost);
    __aicore__ inline uint32_t InitDispatchIndexBuffers(uint32_t fixedBefore);
    __aicore__ inline void InitDispatchCopyBuffers(uint32_t copyTmpBaseAddr, uint32_t relayFlagTensorSize,
                                                   uint32_t dispatchWorkTensorSize,
                                                   uint32_t expertTokenNumsOutTensorSize);
    __aicore__ inline void BuildExpertMaskRound(int32_t curExpertId, uint32_t roundIdx, uint64_t roundStart,
                                                uint32_t roundLen, uint32_t curRoundCompareCount,
                                                uint64_t &totalSendCnt);
    __aicore__ inline uint64_t BuildExpertMask(uint32_t curRankId, int32_t curExpertId, uint64_t srcOffset,
                                               uint64_t dstOffset);
    __aicore__ inline void UpdateLocalRelayWeight(int32_t copyIdx, int32_t outIdx, TEventID eventId);
    __aicore__ inline QuantProcessScratch<ActivationType> InitSharedExpertInputBuffers();
    __aicore__ inline void InitUnpermuteWeightChunk(uint32_t dataResBufAlign, uint32_t dataResFp32BufAlign);
    __aicore__ inline void LoadUnpermuteWeights(int32_t chunkStart, int32_t chunkTokenCnt);
    __aicore__ inline void LoadUnpermuteExpertInput(const GlobalTensor<bfloat16_t> &expandedX, int32_t tokenIdx,
                                                    int32_t expId, TEventID event, LocalTensor<bfloat16_t> &dataInBf16,
                                                    LocalTensor<float> &dataInFp32);
    __aicore__ inline void AccumulateUnpermuteExperts(const GlobalTensor<bfloat16_t> &expandedX, int32_t tokenIdx,
                                                      int32_t localIdx);
    __aicore__ inline void LoadLocalRelayToken(uint32_t srcServer, uint32_t tokenIndex, int32_t bufferIdx,
                                               uint32_t copyInNum, TEventID eventId);
    __aicore__ inline uint32_t GetChannelOwnerBlock(uint32_t rank) const;
    __aicore__ inline bool IsChannelOwner(uint32_t rank) const;
    __aicore__ inline uint32_t CalcTargetWaveCount() const;
    __aicore__ inline uint32_t CalcFirstWaveExpertCount(uint32_t targetWaveCount) const;
    __aicore__ inline uint32_t CalcSteadyWaveExpertCount(uint32_t firstWaveExpertCount, uint32_t targetWaveCount) const;
    __aicore__ inline void ProcessMoeExpertWave(const TupleShape &initShape, const BlockOffset &initOffset,
                                                int32_t &gmTileSequence);
    __aicore__ inline void PrepareDispatch(uint32_t firstWaveExpertCount);
    __aicore__ inline void RunGmm1ActivationForExpert(ExpertLoopState &state, GMMAddrInfo &gmmAddrInfo,
                                                      int32_t &vecSetSyncCom, int32_t &gmTileSequence,
                                                      uint32_t expertIdx);
    __aicore__ inline void RunGmm2ForExpert(ExpertLoopState &state, GMMAddrInfo &gmmAddrInfo, uint32_t expertIdx);
    __aicore__ inline void PublishOrWaitGmm1AllDone();
    __aicore__ inline void DispatchBuffInit();
    __aicore__ inline void DispatchPrepareBuffInit();
    __aicore__ inline void SendAndQuantBuffInit();
    __aicore__ inline void UnpermuteBuffInit();
    __aicore__ inline void ResetFlagList();
    __aicore__ inline void ResetGmm2CombineSyncCounters();
    __aicore__ inline void SendMaskCal();
    __aicore__ inline void PublishExpertTokenCount(uint32_t expertIdx);
    __aicore__ inline void ReceiveDispatchExpertRange(uint32_t expertBegin, uint32_t expertEnd);
    __aicore__ inline uint32_t FindDispatchSegmentEnd(uint32_t segmentStart, uint32_t batchTokenCount);
    __aicore__ inline void ReceiveRemoteDispatchBatch(uint32_t relayRank, uint32_t batchTokenCount);
    __aicore__ inline void PublishDispatchRows(uint32_t expertIdx, int32_t globalRowStart, int32_t rowCount);
    __aicore__ inline void AdvanceExpertOffsets(ExpertLoopState &state, uint32_t expertIdx);
    template <AddrUpdateMode Mode>
    __aicore__ inline bool UpdateGroupParams(ExpertLoopState &state, uint32_t expertIdx);
    __aicore__ inline bool UpdateSharedGroupParams(ExpertLoopState &state, uint32_t expertIdx);
    template <AddrUpdateMode Mode>
    __aicore__ inline void UpdateGlobalBuffer(GMMAddrInfo &gmmAddrInfo, const ExpertLoopState &state);
    template <AddrUpdateMode Mode>
    __aicore__ inline void UpdateSharedGlobalBuffer(GMMAddrInfo &gmmAddrInfo, const ExpertLoopState &state);
    __aicore__ inline void Unpermute();
    __aicore__ inline void InitCombineBuffers();
    __aicore__ inline void ProcessCombineExpertRange(ExpertLoopState waveBeginState, uint32_t expertBegin,
                                                     uint32_t expertEnd);
    __aicore__ inline void CrossRankSyncInWorldSize(bool reuseSendMaskChannelLock);
    __aicore__ inline void ExpertTokenNumCopyOut();
    __aicore__ inline void CopyLocalDispatchTokens(int32_t dstRow, int32_t srcRank, int32_t tokenOffset,
                                                   int32_t tokenCount);
    __aicore__ inline void ResetContiguousGm(GM_ADDR dstAddr, uint64_t sizeBytes);
    __aicore__ inline void ResetDispatchState();
    __aicore__ inline void SendDispatchRelayQueues();
    __aicore__ inline void BuildDispatchRelayQueues();
    __aicore__ inline void AppendTokenToDispatchRelayQueues(GlobalTensor<int32_t> &topkIdsGlobal, uint32_t tokenIdx);
    __aicore__ inline void AppendDispatchRelayData(uint32_t targetServer, uint32_t relayRank,
                                                   HcommBatchHandle &batchHandle, int32_t slot);
    __aicore__ inline void AppendDispatchRelayFlag(uint32_t targetServer, uint32_t relayRank,
                                                   HcommBatchHandle &batchHandle, int32_t slot, bool firstFlag);
    __aicore__ inline void CommitDispatchRelayBatch(uint32_t targetServer, uint32_t relayRank,
                                                    HcommBatchHandle &batchHandle, int32_t batchStart,
                                                    uint32_t batchTokenCount);
    __aicore__ inline void LoadTokenFromLocalRelay(uint32_t srcServer, uint32_t tokenIndex, int32_t bufferIdx,
                                                   uint32_t copyInNum);
    __aicore__ inline void CopyTokensFromLocalRelay(int32_t rowDstOffsetInCore, uint32_t srcServer, int32_t copyNum,
                                                    int64_t widthA, int64_t widthAScale, uint32_t copyInNum);
    __aicore__ inline void QuantizeTokenInUb(const LocalTensor<bfloat16_t> &input,
                                             const LocalTensor<ActivationType> &output,
                                             const LocalTensor<uint16_t> &scratch);
    __aicore__ inline void QuantizeLocalTokensToRelay();
    __aicore__ inline void SetLocalRelayReadyFlags();
    __aicore__ inline uint64_t DispatchRelayQueueServerOffset(uint32_t targetServer) const;
    __aicore__ inline uint64_t RelayTokenOffset(uint32_t sourceServer, uint32_t tokenId) const;
    __aicore__ inline uint64_t RelayFlagOffset(uint32_t sourceServer, uint32_t tokenId) const;
    __aicore__ inline void SharedExpertCopyInput();
    __aicore__ inline void ProcessSharedExpertGmm1(const TupleShape &initShape, const BlockOffset &initOffset,
                                                   int32_t &gmTileSequence);
    __aicore__ inline void ProcessSharedExpertGmm2(const TupleShape &initShape, const BlockOffset &initOffset);
    __aicore__ inline void UnpermuteSharedExpert(int32_t tokenIdx);
    __aicore__ inline void LoadTopkWeightsToUb(const LocalTensor<ActivationType> &xOutTensor, int32_t currentOffset,
                                               int32_t index, TEventID event);
    template <bool IsShared, typename Epilogue>
    __aicore__ inline void RunGmm1WithEpilogue(Epilogue &epilogue, const GMMAddrInfo &gmmAddrInfo,
                                               const ExpertLoopState &state, uint32_t expertIdx, int32_t &vecSetSyncCom,
                                               int32_t &gmTileSequence);
    template <bool IsShared = false>
    __aicore__ inline void GroupMatmulWithActivationQuant(const GMMAddrInfo &gmmAddrInfo, const ExpertLoopState &state,
                                                          uint32_t expertIdx, int32_t &vecSetSyncCom,
                                                          int32_t &gmTileSequence);
    template <bool IsShared = false>
    __aicore__ inline void GroupMatmulWithCombine(const GMMAddrInfo &gmmAddrInfo, const ExpertLoopState &state);
    __aicore__ inline bool GetCombineRankRange(uint32_t expertIdx, uint32_t dstRank, uint32_t &rowStart,
                                               uint32_t &tokenCount);
    __aicore__ inline void ProcessCombineRank(const GMMAddrInfo &gmmAddrInfo, const ExpertLoopState &gmm2State,
                                              uint32_t expertIdx, CombineImpl::LayeredCombineBatchState &batchState);
    __aicore__ inline void LockOwnedRemoteChannels();
    __aicore__ inline void DrainAndUnlockOwnedRemoteChannels();

    __gm__ Mc2MoeContext *mc2Context_{nullptr};
    __gm__ int32_t *gmmToEpilogueFlag_{nullptr};
    Hcomm<COMM_PROTOCOL_UBC_CTP> hcomm_;
    LocalTensor<uint8_t> hcommBatchWqeTensor_;
    Params params_{};

    GlobalTensor<int32_t> expertTokenNumsOut_;
    GlobalTensor<int32_t> metaInfoGlobalTensor_;
    GlobalTensor<int32_t> expertRevNumsGlobalTensor_;
    // A8W4 路径下 RunGmm1A8W4 会覆盖 V1 UB，导致 UB 上跨 expert 的状态
    // 无法保持。cumsumInfoGlobalTensor_ 作为 cumsum 数据的 GM 持久备份：
    // PublishExpertTokenCount 中 Load → 计算 → Store；Dispatch 接收和 CopyOut 从 GM 恢复。
    GlobalTensor<int32_t> cumsumInfoGlobalTensor_;

    uint32_t m_ = 0;
    uint32_t k_ = 0;
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
    uint64_t maskWinOffset_ = 0;              // maskRecvPtr 相对 win 基址(rankSyncInWorldPtr)的偏移
    uint64_t dispatchWinOffset_ = 0;          // peermemInfo dispatchReceivePtr 相对 URMA win 基址的偏移
    uint64_t dispatchFlagWinOffset_ = 0;      // peermemInfo dispatchFlagPtr 相对 URMA win 基址的偏移
    uint32_t relayRecordBytes_ = 0;
    uint64_t dispatchRelayQueueBytesPerServer_ = 0;
    uint64_t dispatchRelayFlagSnapshotBytesPerBlock_ = 0;
    uint64_t cumsumRevCntInRank_ = 0;
    int64_t combineUbTensorSize_ = 0; // combineUbTensor 的大小（元素数）
    uint32_t topKWeightsChunkLen_ = 0;
    uint32_t topKWeightsTempAddr_ = 0;
    uint32_t sharedExpertNum_ = 0;
    uint32_t moeExpertPerRank_ = 0;

    static constexpr uint32_t A_ELEMS_PER_BYTE = PackedElementTraits<QuantOutType>::ELEMENTS_PER_BYTE;
    static constexpr uint32_t B_ELEMS_PER_BYTE = PackedElementTraits<Weight1Type>::ELEMENTS_PER_BYTE;
    // ENABLE_A8W4：A8W4 路径（FP8 激活 + FP4 权重），GMM1 使用 A8W4 前处理（W4→W8 + MMAD）。
    static constexpr bool ENABLE_A8W4 =
        Std::IsSame<Weight1Type, fp4x2_e2m1_t>::value && Std::IsSame<QuantOutType, fp8_e4m3fn_t>::value;
    // ENABLE_A4W4：A4W4 路径（FP4 激活 + FP4 权重），GMM2 复用 A8W4 前处理。
    // A4W4 场景下 GMM1 走通用 A4W4、GMM2 走 A8W4，避免两段都使用 A4W4 导致精度损失过大。
    static constexpr bool ENABLE_A4W4 =
        Std::IsSame<Weight1Type, fp4x2_e2m1_t>::value && Std::IsSame<QuantOutType, fp4x2_e2m1_t>::value;
    // 仅通用 A8W8 路径支持 gate/up 交织布局，FP4 配对路径保持专用布局和同步协议。
    static constexpr bool GMM1_INTERLEAVED = IsGmm1Interleaved && !ENABLE_A8W4 && !ENABLE_A4W4;
    static constexpr int32_t DISPATCH_COPY_BUFFER_COUNT = 5;
    static constexpr uint32_t MX_QUANT_TEMP_UB_BYTES = 2U * 1024U;
    static constexpr uint32_t DISPATCH_READY_FLAG_BATCH_TOKENS = 256U;
    static constexpr uint32_t DISPATCH_MAX_TOPK = 32U;
    static constexpr uint32_t SEND_MASK_UB_LIMIT = LAYERED_USABLE_UB_BYTES;
    // 二级读取在 TopkWeightsPrefetch 路径下每 token 需要 data/scale/weight 三条 WQE。
    // 64 KiB 可容纳 1024 条 WQE，足够一次提交完整的 256-token 批次（最多 768 条 WQE）。
    static_assert(LAYERED_HCOMM_BATCH_WQE_CAPACITY >= L1_TILE_M_256 * 3U,
                  "Layered Hcomm UB must hold 256 three-WQE token records");
    // Wave 通信需要同时保留批量描述符和其他搬运张量，因此动态 UB 预算止于批量区起点。
    static constexpr uint32_t COMM_WORK_UB_LIMIT = LAYERED_HCOMM_BATCH_UB_OFFSET;
    // 计算宏 Wave、一级发送批次和二级就绪段使用独立粒度。
    // 二级读取/ready 仍以 256 行为正确性边界；一级发送尽量用满 64 KiB WQE 区。
    static constexpr uint32_t LAYERED_FIRST_WAVE_ROWS = 1024U;
    static constexpr uint32_t LAYERED_LATENCY_WAVE_COUNT = 2U;
    static constexpr uint32_t LAYERED_BALANCED_WAVE_COUNT = 6U;
    static constexpr uint32_t LAYERED_THROUGHPUT_WAVE_COUNT = 4U;
    static constexpr uint32_t LAYERED_LATENCY_ROWS_PER_EXPERT = L1_TILE_M_256;
    static constexpr uint32_t LAYERED_THROUGHPUT_ROWS_PER_EXPERT = 2048U;
    static constexpr uint32_t LAYERED_FEW_EXPERT_THRESHOLD = 8U;
    // Dispatch 一级通信严格按“数据批次先提交、就绪标志批次后提交”发布。
    // 仅这一段关闭 CQE；Dispatch 二级读和 Combine 写均使用 Hcomm 默认 WQE 配置。
    static constexpr struct UrmaWqeEntry DISPATCH_DATA_WQE_CONFIG = {
        .odr = 5, .fence = 1, .se = 0, .cqe = 0, .inlineEn = 0};
    static constexpr struct UrmaWqeEntry DISPATCH_FIRST_FLAG_WQE_CONFIG = {
        .odr = 6, .fence = 1, .se = 0, .cqe = 0, .inlineEn = 0};
    static constexpr struct UrmaWqeEntry DISPATCH_FLAG_WQE_CONFIG = {
        .odr = 5, .fence = 1, .se = 0, .cqe = 0, .inlineEn = 0};
    // 一级发送尽量吃满 64 KiB WQE 区；二级接收仍以 256 token 作为完整就绪边界。
    static constexpr uint32_t DISPATCH_RELAY_QUEUE_ENTRY_BYTES = ALIGN_32;
    static constexpr uint32_t DISPATCH_SEND_BATCH_TOKEN_CAPACITY = LAYERED_HCOMM_BATCH_WQE_CAPACITY;
    static constexpr uint32_t DISPATCH_RECEIVE_BATCH_TOKEN_CAPACITY = L1_TILE_M_256;
    static constexpr uint32_t DISPATCH_BATCH_DST_ROW = 4U;
    static constexpr uint32_t DISPATCH_BATCH_EXPERT_ID = 5U;
    static constexpr uint32_t DISPATCH_BATCH_SRC_SERVER = 6U;
    LocalTensor<int32_t> topkIndexTensor_;
    LocalTensor<uint8_t> gatherMaskTensor_;
    LocalTensor<uint32_t> gatherMaskInt32Tensor_;
    LocalTensor<int32_t> expertTokenCntTensor_;
    LocalTensor<int32_t> validTopkIndexTensor_;
    LocalTensor<int32_t> cumsumInfoTensor_;
    LocalTensor<ActivationType> copyTmpTensors_[DISPATCH_COPY_BUFFER_COUNT]; // 5 路软流水：占用 EVENT_ID1..EVENT_ID5。
    // 当前 256-token 中继就绪标志窗口，复用本地搬运缓冲起始地址。
    LocalTensor<uint64_t> relayFlagTensor_;
    LocalTensor<int32_t> metaInfoTensor_;
    LocalTensor<bfloat16_t> xInTensor1_;
    LocalTensor<bfloat16_t> xInTensor2_;
    LocalTensor<ActivationType> xOutTensor1_;
    LocalTensor<ActivationType> xOutTensor2_;
    LocalTensor<uint16_t> mxTempTensor_;
    LocalTensor<uint64_t> relayReadyFillTensor_;
    LocalTensor<int32_t> resetTensor_;
    int32_t resetBatchElementCount_ = 0;
    LocalTensor<int32_t> topkIdsTensor_;
    LocalTensor<uint8_t> sendMaskTensor_[DOUBLE_BUFFER]; // SendMaskCal 源卡算 [mask|count] 的 ping-pong 缓冲
    LocalTensor<int32_t> sendGatherOutTensor_;           // SendMaskCal GatherMask 计 count 的废弃输出 scratch
    LocalTensor<int32_t> expertTokenNumsOutTensor_;
    LocalTensor<bfloat16_t> dataResTensor_;
    LocalTensor<float> dataResFp32Tensor_;
    LocalTensor<float> topKWeightsTensor_;
    LocalTensor<float> fp32ScaleTensor_;
    LocalTensor<bfloat16_t> bf16ScaleTensor_;

    // GMM2 走 A8W4 且 QuantMode 为 a4w4（E2M1）时，ActivationQuant 输出需提升为 fp8_e4m3fn_t。
    // 同时当 Weight2 非 fp4 但 QuantMode==E2M1 时（generic GMM2 路径），也需 promotion，
    // 否则会出现 A=QuantOutType(fp4) vs B=Weight1Type(fp8) 的类型不匹配。
    using ActivationQuantOutType =
        typename std::conditional<(QuantMode == E2M1_QUANT), fp8_e4m3fn_t, QuantOutType>::type;

    // ActivationQuant 输出的元素字节密度：fp4 时为 2elem/B，fp8 时为 1elem/B。
    static constexpr uint32_t C_ELEMS_PER_BYTE = PackedElementTraits<ActivationQuantOutType>::ELEMENTS_PER_BYTE;

    // GMM1 固定按 256 行切块；权重前移时后处理可按 128 行细分。
    static constexpr uint32_t GMM1_TILE_M = L1_TILE_M_256;
    static constexpr uint32_t EPILOGUE_TILE_M = TopkWeightsPrefetch ? L1_TILE_M_128 : L1_TILE_M_256;

    using BlockEpilogue = BlockEpilogueActivationMxQuant<ActivationQuantOutType, bfloat16_t, EPILOGUE_TILE_M, L1_TILE_N,
                                                         TopkWeightsPrefetch, GMM1_INTERLEAVED>;
    using SharedBlockEpilogue = BlockEpilogueActivationMxQuant<ActivationQuantOutType, bfloat16_t, L1_TILE_M_256,
                                                               L1_TILE_N, false, GMM1_INTERLEAVED>;
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
__aicore__ inline uint32_t MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::CalcTargetWaveCount() const
{
    if (moeExpertPerRank_ <= 1U) {
        return moeExpertPerRank_;
    }
    uint64_t totalRows = static_cast<uint64_t>(m_) * static_cast<uint64_t>(topK_);
    if (totalRows <= static_cast<uint64_t>(LAYERED_FIRST_WAVE_ROWS)) {
        return 1U;
    }
    uint64_t estimatedRowsPerExpert =
        (totalRows + static_cast<uint64_t>(moeExpertPerRank_) - 1U) / static_cast<uint64_t>(moeExpertPerRank_);
    uint32_t targetWaveCount = LAYERED_BALANCED_WAVE_COUNT;
    if (estimatedRowsPerExpert <= LAYERED_LATENCY_ROWS_PER_EXPERT) {
        targetWaveCount = LAYERED_LATENCY_WAVE_COUNT;
    } else if (estimatedRowsPerExpert >= LAYERED_THROUGHPUT_ROWS_PER_EXPERT) {
        targetWaveCount = LAYERED_THROUGHPUT_WAVE_COUNT;
    } else if (moeExpertPerRank_ <= LAYERED_FEW_EXPERT_THRESHOLD) {
        targetWaveCount = LAYERED_LATENCY_WAVE_COUNT;
    }
    return targetWaveCount < moeExpertPerRank_ ? targetWaveCount : moeExpertPerRank_;
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline uint32_t MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::CalcFirstWaveExpertCount(
    uint32_t targetWaveCount) const
{
    if (moeExpertPerRank_ == 0U || targetWaveCount == 0U) {
        return 0U;
    }
    if (targetWaveCount == 1U) {
        return moeExpertPerRank_;
    }
    uint64_t totalRows = static_cast<uint64_t>(m_) * static_cast<uint64_t>(topK_);
    uint64_t estimatedRowsPerExpert =
        (totalRows + static_cast<uint64_t>(moeExpertPerRank_) - 1U) / static_cast<uint64_t>(moeExpertPerRank_);
    if (estimatedRowsPerExpert == 0U) {
        estimatedRowsPerExpert = 1U;
    }
    uint32_t expertsByWaveBudget = (moeExpertPerRank_ + targetWaveCount - 1U) / targetWaveCount;
    uint32_t expertsByWarmupRows = static_cast<uint32_t>(
        (static_cast<uint64_t>(LAYERED_FIRST_WAVE_ROWS) + estimatedRowsPerExpert - 1U) / estimatedRowsPerExpert);
    uint32_t firstWaveExperts = expertsByWarmupRows < expertsByWaveBudget ? expertsByWarmupRows : expertsByWaveBudget;
    if (firstWaveExperts == 0U) {
        firstWaveExperts = 1U;
    }
    return firstWaveExperts < moeExpertPerRank_ ? firstWaveExperts : moeExpertPerRank_;
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline uint32_t MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::CalcSteadyWaveExpertCount(
    uint32_t firstWaveExpertCount, uint32_t targetWaveCount) const
{
    if (firstWaveExpertCount >= moeExpertPerRank_ || targetWaveCount <= 1U) {
        return moeExpertPerRank_;
    }
    uint32_t remainingExperts = moeExpertPerRank_ - firstWaveExpertCount;
    uint32_t remainingWaves = targetWaveCount - 1U;
    uint32_t steadyWaveExperts = (remainingExperts + remainingWaves - 1U) / remainingWaves;
    return steadyWaveExperts == 0U ? 1U : steadyWaveExperts;
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::InitTopology(GM_ADDR context,
                                                                                    const MegaMoeTilingData *tilingData)
{
    m_ = tilingData->bs;
    k_ = tilingData->h;
    aicNum_ = tilingData->aicNum;
    topK_ = tilingData->topK;
    sendTotalNum_ = static_cast<uint64_t>(m_) * static_cast<uint64_t>(topK_);
    maskRouteCapacity_ = static_cast<uint64_t>(tilingData->numMaxTokensPerRank) * static_cast<uint64_t>(topK_);
    worldSize_ = tilingData->epWorldSize;
    moeExpertPerRank_ = tilingData->moeExpertPerRank;
    sharedExpertNum_ = tilingData->sharedExpertNum;
    maxOutputSize_ = tilingData->maxOutputSize;
    // 与 WorkspaceInfo 构造里 flagDispatchToGmm1Ptr 的分配公式保持一致。
    maxWavesPerExpert_ = static_cast<int32_t>(
        Ops::Base::CeilDiv(static_cast<int64_t>(maxOutputSize_), static_cast<int64_t>(L1_TILE_M_256)));
    dispatchFlagSlotsPerExpert_ = maxWavesPerExpert_ * INT_CACHELINE;
    hiddenDim_ = tilingData->hiddenDim;
    mc2Context_ = reinterpret_cast<__gm__ Mc2MoeContext *>(context);
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
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::InitDispatchLayout(
    const MegaMoeTilingData *tilingData)
{
    // 各 win 区相对 win 基址(rankSyncInWorldPtr)的偏移; 所有卡 win 布局一致, 跨卡读写用同一偏移。
    maskWinOffset_ = static_cast<uint64_t>(params_.peermemInfo.maskRecvPtr - params_.peermemInfo.rankSyncInWorldPtr);
    dispatchWinOffset_ =
        static_cast<uint64_t>(params_.peermemInfo.dispatchReceivePtr - params_.peermemInfo.rankSyncInWorldPtr);
    dispatchFlagWinOffset_ =
        static_cast<uint64_t>(params_.peermemInfo.dispatchFlagPtr - params_.peermemInfo.rankSyncInWorldPtr);
    // mask 是跨 Rank 共享的槽布局，几何必须与 PeermemInfo 一样按全卡容量计算。
    maskAlignSize_ = static_cast<uint32_t>(CalcDispatchMaskAlignSize(tilingData));
    // 每个 win 槽位再追加 32B 存 count(源卡 SendMaskCal 同步算好), 须与 PeermemInfo 的 maskSlotSize 一致。
    maskSlotSize_ = maskAlignSize_ + static_cast<uint32_t>(ALIGN_32);
    mxQuantScaleNumAlignPerToken_ = Ops::Base::CeilDiv(k_, static_cast<uint32_t>(ALIGN_32));
    mxQuantTokenAlignBytes_ =
        Ops::Base::CeilAlign(static_cast<uint32_t>(k_ / A_ELEMS_PER_BYTE), static_cast<uint32_t>(ALIGN_256)) *
        sizeof(ActivationType);
    mxQuantScaleAlignBytes_ = mxQuantScaleNumAlignPerToken_ * sizeof(uint8_t);
    mxQuantTokenScaleAlignBytes_ =
        Ops::Base::CeilAlign(mxQuantTokenAlignBytes_ + mxQuantScaleAlignBytes_, static_cast<uint32_t>(ALIGN_32));
    if constexpr (TopkWeightsPrefetch) {
        weightAlignBytes_ =
            Ops::Base::CeilAlign(static_cast<uint32_t>(topK_ * sizeof(float)), static_cast<uint32_t>(ALIGN_32));
        mxQuantTokenScaleAlignBytes_ += weightAlignBytes_;
    }
    relayRecordBytes_ =
        Ops::Base::CeilAlign(static_cast<uint64_t>(mxQuantTokenScaleAlignBytes_), static_cast<uint64_t>(ALIGN_512));
    dispatchRelayQueueBytesPerServer_ =
        static_cast<uint64_t>(ALIGN_32) + static_cast<uint64_t>(m_) * DISPATCH_RELAY_QUEUE_ENTRY_BYTES;
    // 远端 flag 按 256-token 连续窗口读取，UB 占用不随 BS 增长。
    uint64_t flagSnapshotBytes = static_cast<uint64_t>(DISPATCH_RECEIVE_BATCH_TOKEN_CAPACITY) * sizeof(uint64_t);
    dispatchRelayFlagSnapshotBytesPerBlock_ = Ops::Base::CeilAlign(flagSnapshotBytes, static_cast<uint64_t>(ALIGN_512));
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::InitWorkspace(GM_ADDR workspaceGM,
                                                                                     MegaMoeTilingData *tilingData)
{
    {
        WorkspaceLayout workspaceLayout(tilingData, serverNum_);
        params_.workspaceInfo.Bind(workspaceGM, workspaceLayout);
    }
    params_.peermemInfo = PeermemInfo(g_winRankAddr_[rankId_], tilingData, A_ELEMS_PER_BYTE, serverNum_);
    params_.tilingData = tilingData;
    if constexpr (ENABLE_A8W4 || ENABLE_A4W4) {
        gmmToEpilogueFlag_ = reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.flagGmmToEpiloguePtr) +
                             static_cast<uint64_t>(blockIdx_) * INT_CACHELINE;
    }
    expertTokenNumsOut_.SetGlobalBuffer((__gm__ int32_t *)params_.expertTokenNumsOutGmAddr);
    expertRevNumsGlobalTensor_.SetGlobalBuffer((__gm__ int32_t *)params_.workspaceInfo.expertRevTokenNumsPtr);
    metaInfoGlobalTensor_.SetGlobalBuffer((__gm__ int32_t *)params_.workspaceInfo.metaInfoPtr);
    // 每个 block 负责一个专家，cumsumInfo 中每个专家占 worldSize 个
    // int32_t 存 rank 维度的 cumsum 结果，blockIdx 决定了负责哪个专家。
    uint64_t cumsumStride =
        Ops::Base::CeilAlign(static_cast<int64_t>(worldSize_ * moeExpertPerRank_ * sizeof(int32_t)), ALIGN_32);
    cumsumInfoGlobalTensor_.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.cumsumInfoPtr + cumsumStride * blockIdx_));
    epilogueOp_.Init({.yGmAddr = params_.workspaceInfo.activationQuantDataPtr,
                      .yScaleGmAddr = params_.workspaceInfo.activationQuantScalePtr,
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
    GM_ADDR context, GM_ADDR x, GM_ADDR topkIds, GM_ADDR topkWeights, GM_ADDR weight1, GM_ADDR weight2,
    GM_ADDR xActiveMask, GM_ADDR weightScales1, GM_ADDR weightScales2, GM_ADDR scales, GM_ADDR sharedWeight1,
    GM_ADDR sharedWeight2, GM_ADDR sharedWeightScales1, GM_ADDR sharedWeightScales2, GM_ADDR yOut,
    GM_ADDR expertTokenNumsOut, GM_ADDR workspaceGM, MegaMoeTilingData *tilingData)
{
    InitTopology(context, tilingData);
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
    params_.combineCommParams.hcomm = &hcomm_;

    params_.y2GmAddr = yOut;
    params_.expertTokenNumsOutGmAddr = expertTokenNumsOut;
    params_.probsGmAddr = topkWeights;
    InitWorkspace(workspaceGM, tilingData);
}

// Advance the GMM cursor using the previous expert's shape, including empty experts.
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::AdvanceExpertOffsets(ExpertLoopState &state,
                                                                                            uint32_t expertIdx)
{
    if (expertIdx != 0) {
        uint64_t m = Get<M_VALUE>(state.problemShape);
        uint64_t n = Get<N_VALUE>(state.problemShape);
        uint64_t k = Get<K_VALUE>(state.problemShape);
        state.expertBeforeCnt += m;
        Get<IDX_A_OFFSET>(state.baseOffset) += m * k / A_ELEMS_PER_BYTE;
        Get<IDX_B_OFFSET>(state.baseOffset) += n * k / B_ELEMS_PER_BYTE;
        auto scaleK = Ops::Base::CeilDiv(k, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
        Get<IDX_A_SCALE_OFFSET>(state.baseOffset) += m * scaleK;
        Get<IDX_B_SCALE_OFFSET>(state.baseOffset) += n * scaleK;
        Get<IDX_C_OFFSET>(state.baseOffset) += m * n / ACTIVATION_N_HALF / C_ELEMS_PER_BYTE;
        Get<IDX_C_SCALE_OFFSET>(state.baseOffset) +=
            m * Ops::Base::CeilDiv(n / ACTIVATION_N_HALF, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) *
            MXFP_MULTI_BASE_SIZE;
        Get<IDX_FLAG_OFFSET>(state.baseOffset) += 1;
        Get<IDX_B2_OFFSET>(state.baseOffset) += k * n / ACTIVATION_N_HALF / B_ELEMS_PER_BYTE;
        Get<IDX_B2_SCALE_OFFSET>(state.baseOffset) +=
            k * Ops::Base::CeilDiv(n / ACTIVATION_N_HALF, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) *
            MXFP_MULTI_BASE_SIZE;
        Get<IDX_Y2_OFFSET>(state.baseOffset) += m * k;
        Get<IDX_M_OFFSET>(state.baseOffset) += m;
        Get<IDX_GMM1_OFFSET>(state.baseOffset) += m * n;
        Get<IDX_GMM2_OFFSET>(state.baseOffset) += m * k;
    }
}

template <TemplateMegaMoeLayeredTypeClass>
template <AddrUpdateMode Mode>
__aicore__ inline bool MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::UpdateGroupParams(ExpertLoopState &state,
                                                                                         uint32_t expertIdx)
{
    AdvanceExpertOffsets(state, expertIdx);

    // gmm1中当前专家收到的count数是由subBlockIdx_=1的aiv计算出并写入expertRevNumsGlobalTensor_，通知后续aic/aiv0读取该值
    if constexpr (Mode == AddrUpdateMode::GMM1) {
        if (subBlockIdx_ == 0) { // aiv1进行SendCntCal计算完成后atomicAddFlag，aic/aiv0等到该flag位后读取cnt值
            __gm__ int32_t *sendCntFlag = (__gm__ int32_t *)params_.workspaceInfo.flagSendCntCalToUpdParamsPtr +
                                          static_cast<uint64_t>(expertIdx) * aicNum_ * INT_CACHELINE +
                                          static_cast<uint64_t>(blockIdx_) * INT_CACHELINE;
            WaitUntilGmFlagIsNonZero(sendCntFlag);
        }
    }
    uint64_t offsetInCnt = expertIdx * 8 * aicNum_ + 8 * blockIdx_;
    DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(
        expertRevNumsGlobalTensor_[offsetInCnt]);
    Get<M_VALUE>(state.problemShape) = expertRevNumsGlobalTensor_.GetValue(offsetInCnt);

    if (Get<M_VALUE>(state.problemShape) == 0) {
        return false;
    }
    return true;
}

// =====================================================================================================
// UpdateSharedGroupParams：共享专家专用，M 恒为 m_，无 flag 等待与 DCache 操作。
// =====================================================================================================
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline bool MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::UpdateSharedGroupParams(ExpertLoopState &state,
                                                                                               uint32_t expertIdx)
{
    AdvanceExpertOffsets(state, expertIdx);

    Get<M_VALUE>(state.problemShape) = m_;
    return true;
}

// ==================================================================================
// UpdateGlobalBuffer：更新当前 expert 的 GMM 地址视图。
//                     GMM1 始终写 gmm1MmadResPtr；
//                     GMM2 始终写 gmm2MmadResPtr。
// ==================================================================================
template <TemplateMegaMoeLayeredTypeClass>
template <AddrUpdateMode Mode>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::UpdateGlobalBuffer(GMMAddrInfo &gmmAddrInfo,
                                                                                          const ExpertLoopState &state)
{
    if constexpr (Mode == AddrUpdateMode::GMM1) {
        // guard 与 WorkspaceInfo 分配条件一致，由 TilingKey 保证同步。
        if constexpr (ENABLE_A8W4 || TopkWeightsPrefetch) {
            gmmAddrInfo.gmm1OutGlobal =
                params_.workspaceInfo.gmm1MmadResPtr + Get<IDX_GMM1_OFFSET>(state.baseOffset) * sizeof(bfloat16_t);
        }
        gmmAddrInfo.metaInfoGlobal = params_.workspaceInfo.metaInfoPtr;
        gmmAddrInfo.aGlobal =
            params_.workspaceInfo.dispatchRevDataPtr + Get<IDX_A_OFFSET>(state.baseOffset) * sizeof(ActivationType);
        gmmAddrInfo.aScaleGlobal = params_.workspaceInfo.dispatchRevScalePtr +
                                   Get<IDX_A_SCALE_OFFSET>(state.baseOffset) * sizeof(QuantScaleOutType);

        uint32_t expertIdx = static_cast<uint32_t>(Get<IDX_FLAG_OFFSET>(state.baseOffset));
        if constexpr (TopkWeightsPrefetch) {
            gmmAddrInfo.gmm1TileStatus =
                reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.gmm1TileStatusPtr) +
                static_cast<uint64_t>(expertIdx) * params_.tilingData->maxTilesPerExpert * INT_CACHELINE;
        }
        gmmAddrInfo.bGlobal =
            GetExpertWeightAddr<Weight1Type>(params_.bGmAddr, params_.tilingData->isPerExpertWeightTensor, expertIdx,
                                             Get<IDX_B_OFFSET>(state.baseOffset));
        gmmAddrInfo.bScaleGlobal =
            GetExpertWeightAddr<QuantScaleOutType>(params_.bScaleGmAddr, params_.tilingData->isPerExpertWeightTensor,
                                                   expertIdx, Get<IDX_B_SCALE_OFFSET>(state.baseOffset));

        if constexpr (g_coreType == AIV) {
            AscendC::Coord<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> vecBaseOffset{
                Get<IDX_C_OFFSET>(state.baseOffset),
                Get<IDX_C_SCALE_OFFSET>(state.baseOffset),
                Get<IDX_FLAG_OFFSET>(state.baseOffset),
                0L,
                0L,
                0L};
            epilogueOp_.UpdateGlobalAddr(vecBaseOffset);
        }
    } else if constexpr (Mode == AddrUpdateMode::GMM2) {
        // guard 与 WorkspaceInfo 分配条件一致，由 TilingKey 保证同步。
        gmmAddrInfo.gmm2OutGlobal =
            params_.workspaceInfo.gmm2MmadResPtr + Get<IDX_GMM2_OFFSET>(state.baseOffset) * sizeof(bfloat16_t);
        gmmAddrInfo.metaInfoGlobal = params_.workspaceInfo.metaInfoPtr +
                                     static_cast<uint64_t>(state.expertBeforeCnt) * META_INFO_SIZE * sizeof(int32_t);
        gmmAddrInfo.aGlobal =
            params_.workspaceInfo.activationQuantDataPtr + Get<IDX_C_OFFSET>(state.baseOffset) * sizeof(ActivationType);
        gmmAddrInfo.aScaleGlobal = params_.workspaceInfo.activationQuantScalePtr +
                                   Get<IDX_C_SCALE_OFFSET>(state.baseOffset) * sizeof(QuantScaleOutType);
        uint32_t expertIdx = static_cast<uint32_t>(Get<IDX_FLAG_OFFSET>(state.baseOffset));
        gmmAddrInfo.bGlobal =
            GetExpertWeightAddr<Weight1Type>(params_.b2GmAddr, params_.tilingData->isPerExpertWeightTensor, expertIdx,
                                             Get<IDX_B2_OFFSET>(state.baseOffset));
        gmmAddrInfo.bScaleGlobal =
            GetExpertWeightAddr<QuantScaleOutType>(params_.b2ScaleGmAddr, params_.tilingData->isPerExpertWeightTensor,
                                                   expertIdx, Get<IDX_B2_SCALE_OFFSET>(state.baseOffset));
        uint64_t expertSyncSlotOffset = static_cast<uint64_t>(Get<IDX_FLAG_OFFSET>(state.baseOffset)) *
                                        params_.tilingData->combineSyncSlotCountPerExpert;
        gmmAddrInfo.gmm2CombineSyncCounter = (__gm__ int32_t *)params_.workspaceInfo.gmm2CombineSyncCounterPtr +
                                             expertSyncSlotOffset * static_cast<uint64_t>(INT_CACHELINE);
        gmmAddrInfo.gmm2CombineLogicalCoreCount = blockNum_;
    }
    gmmAddrInfo.activationToGmm2Flag = (__gm__ int32_t *)params_.workspaceInfo.flagActivationToGmm2Ptr +
                                       Get<IDX_FLAG_OFFSET>(state.baseOffset) * INT_CACHELINE;
    // wave-grain dispatch-gmm1 flag: per-expert 步长是 dispatchFlagSlotsPerExpert_,而不是 INT_CACHELINE。
    gmmAddrInfo.dispatchToGmm1Flag = (__gm__ int32_t *)params_.workspaceInfo.flagDispatchToGmm1Ptr +
                                     Get<IDX_FLAG_OFFSET>(state.baseOffset) * dispatchFlagSlotsPerExpert_;
    if constexpr (ENABLE_A8W4 || ENABLE_A4W4) {
        gmmAddrInfo.gmmToEpilogueFlag = gmmToEpilogueFlag_;
    }
}

// ==================================================================================
// UpdateSharedGlobalBuffer：共享专家专用，地址来自 shared* workspace，flags 为 nullptr。
// ==================================================================================
template <TemplateMegaMoeLayeredTypeClass>
template <AddrUpdateMode Mode>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::UpdateSharedGlobalBuffer(
    GMMAddrInfo &gmmAddrInfo, const ExpertLoopState &state)
{
    if constexpr (Mode == AddrUpdateMode::GMM1) {
        gmmAddrInfo.aGlobal = params_.workspaceInfo.sharedExpertInputDataPtr;
        gmmAddrInfo.aScaleGlobal = params_.workspaceInfo.sharedExpertInputScalePtr;
        if constexpr (ENABLE_A8W4) {
            gmmAddrInfo.gmm1OutGlobal = params_.workspaceInfo.sharedExpertGmm1OutPtr +
                                        Get<IDX_GMM1_OFFSET>(state.baseOffset) * sizeof(bfloat16_t);
        }
        uint32_t expertIdx = static_cast<uint32_t>(Get<IDX_FLAG_OFFSET>(state.baseOffset));
        gmmAddrInfo.bGlobal =
            GetExpertWeightAddr<Weight1Type>(params_.sharedBGmAddr, params_.tilingData->isPerExpertWeightTensor,
                                             expertIdx, Get<IDX_B_OFFSET>(state.baseOffset));
        gmmAddrInfo.bScaleGlobal = GetExpertWeightAddr<QuantScaleOutType>(
            params_.sharedBScaleGmAddr, params_.tilingData->isPerExpertWeightTensor, expertIdx,
            Get<IDX_B_SCALE_OFFSET>(state.baseOffset));
    } else if constexpr (Mode == AddrUpdateMode::GMM2) {
        gmmAddrInfo.gmm2OutGlobal =
            params_.workspaceInfo.sharedExpertResultPtr + Get<IDX_GMM2_OFFSET>(state.baseOffset) * sizeof(bfloat16_t);
        gmmAddrInfo.aGlobal = params_.workspaceInfo.sharedExpertActivationDataPtr +
                              Get<IDX_C_OFFSET>(state.baseOffset) * sizeof(ActivationType);
        gmmAddrInfo.aScaleGlobal = params_.workspaceInfo.sharedExpertActivationScalePtr +
                                   Get<IDX_C_SCALE_OFFSET>(state.baseOffset) * sizeof(QuantScaleOutType);
        uint32_t expertIdx = static_cast<uint32_t>(Get<IDX_FLAG_OFFSET>(state.baseOffset));
        gmmAddrInfo.bGlobal =
            GetExpertWeightAddr<Weight1Type>(params_.sharedB2GmAddr, params_.tilingData->isPerExpertWeightTensor,
                                             expertIdx, Get<IDX_B2_OFFSET>(state.baseOffset));
        gmmAddrInfo.bScaleGlobal = GetExpertWeightAddr<QuantScaleOutType>(
            params_.sharedB2ScaleGmAddr, params_.tilingData->isPerExpertWeightTensor, expertIdx,
            Get<IDX_B2_SCALE_OFFSET>(state.baseOffset));
    }
    gmmAddrInfo.activationToGmm2Flag = nullptr;
    gmmAddrInfo.dispatchToGmm1Flag = nullptr;
    if constexpr (ENABLE_A8W4 || ENABLE_A4W4) {
        gmmAddrInfo.gmmToEpilogueFlag = gmmToEpilogueFlag_;
    }
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
        hcomm_.Lock(GetUrmaCommHandle(mc2Context_, rank, rankId_));
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
        ChannelHandle channel = GetUrmaCommHandle(mc2Context_, rank, rankId_);
        HcommBatchHandle batchHandle = hcomm_.MakeBatchHandle(
            channel, hcommBatchWqeTensor_, LAYERED_HCOMM_BATCH_UB_BYTES, GetRankWinAddrWithOffset(rank, 0));
        // 同一通道可能同时残留 Dispatch 一级和 Combine 的已提交批次，统一按批量句柄等待完成。
        hcomm_.Drain(batchHandle);
        hcomm_.Unlock(channel);
    }
}

// ==============================================================================================
// CrossRankSyncInWorldSize：全卡同步，rank 槽位之后的独立区域记录各 AIV 的 syncCnt。
// ==============================================================================================
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::CrossRankSyncInWorldSize(
    bool reuseSendMaskChannelLock)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    __gm__ int32_t *syncRank = (__gm__ int32_t *)params_.peermemInfo.rankSyncInWorldPtr;
    int64_t syncCountOffset = CalcUrmaSyncCountOffset(static_cast<int64_t>(worldSize_));
    __gm__ int32_t *syncCount = (__gm__ int32_t *)(params_.peermemInfo.rankSyncInWorldPtr + syncCountOffset +
                                                   aivCoreIdx_ * PEERMEM_SYNC_SLOT_SIZE);
    int count = ReadGmByPassDCache(syncCount) + 1;
    WriteGmByPassDCache(syncCount, count);
    // 先向本 AIV 负责的所有 peer 发出通知，再进入等待。256P 场景下可同时维持多个
    // channel 在途，避免原实现逐 peer“写-等-Drain”造成的串行握手。
    for (int rankIndex = aivCoreIdx_; rankIndex < worldSize_; rankIndex += blockAivNum_) {
        if (rankIndex == rankId_) {
            continue;
        }
        ChannelHandle channel = GetUrmaCommHandle(mc2Context_, rankIndex, rankId_);
        if (!reuseSendMaskChannelLock) {
            // Wave 结束后同步任务重新按物理 AIV 分核，可能不同于 Wave 的固定归属核，因此先取得通道锁。
            hcomm_.Lock(channel);
        }
        __gm__ int32_t *syncRemoteAddr = (__gm__ int32_t *)(g_winRankAddr_[rankIndex]) + rankId_ * 16;
        hcomm_.WriteNbi(channel, (GM_ADDR)syncRemoteAddr, (GM_ADDR)syncCount, static_cast<int64_t>(sizeof(int32_t)));
    }
    for (int rankIndex = aivCoreIdx_; rankIndex < worldSize_; rankIndex += blockAivNum_) {
        if (rankIndex == rankId_) {
            continue;
        }
        ChannelHandle channel = GetUrmaCommHandle(mc2Context_, rankIndex, rankId_);
        auto syncCheck = syncRank + rankIndex * 16;
        GmSignalWaitBarrier(syncCheck, count);
        // Drain 后 Unlock 发布通道 counter。第一次调用释放 SendMask 延续的锁，第二次调用释放本函数取得的锁。
        hcomm_.Drain(channel);
        hcomm_.Unlock(channel);
    }
    PipeBarrier<PIPE_ALL>();
    SyncAll<true>();
}

// ===============================================================
// GroupMatmulWithActivationQuant：按实现路径分发到 A8W4 或 generic GMM1。
//                            A8W4 由 ENABLE_A8W4 控制；generic 路径的 subBlockIdx 判断已下沉到函数内部。
// ===============================================================
template <TemplateMegaMoeLayeredTypeClass>
template <bool IsShared, typename Epilogue>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::RunGmm1WithEpilogue(
    Epilogue &epilogue, const GMMAddrInfo &gmmAddrInfo, const ExpertLoopState &state, uint32_t expertIdx,
    int32_t &vecSetSyncCom, int32_t &gmTileSequence)
{
    // Shared experts retain the 256-row, non-prefetch epilogue and its synchronization path.
    constexpr uint32_t epilogueTileM = IsShared ? L1_TILE_M_256 : EPILOGUE_TILE_M;
    constexpr bool prefetchWeights = !IsShared && TopkWeightsPrefetch;
    if constexpr (g_coreType == AIV) {
        AscendC::Coord<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> vecBaseOffset{
            Get<IDX_C_OFFSET>(state.baseOffset),
            Get<IDX_C_SCALE_OFFSET>(state.baseOffset),
            Get<IDX_FLAG_OFFSET>(state.baseOffset),
            0L,
            0L,
            0L};
        epilogue.UpdateGlobalAddr(vecBaseOffset);
    }
    if constexpr (ENABLE_A8W4) {
        RunGmm1A8W4<QuantOutType, Weight1Type, bfloat16_t, QuantScaleOutType, QuantScaleOutType, GMM1_TILE_M,
                    epilogueTileM, prefetchWeights, IsShared>(epilogue, params_, state.problemShape, gmmAddrInfo,
                                                              startBlockIdx_, gmTileSequence, state.expertBeforeCnt,
                                                              expertIdx);
    } else {
        if (params_.tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A8W8_NZ ||
            params_.tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A4W4_NZ) {
            RunGmm1Generic<QuantOutType, ActivationQuantOutType, QuantOutType, bfloat16_t, QuantScaleOutType,
                           QuantScaleOutType, true, GMM1_TILE_M, epilogueTileM, prefetchWeights, IsShared,
                           GMM1_INTERLEAVED>(epilogue, params_, state.problemShape, gmmAddrInfo, startBlockIdx_,
                                             vecSetSyncCom, state.expertBeforeCnt, expertIdx, gmm1PingPongIdx_);
        } else {
            RunGmm1Generic<QuantOutType, ActivationQuantOutType, QuantOutType, bfloat16_t, QuantScaleOutType,
                           QuantScaleOutType, false, GMM1_TILE_M, epilogueTileM, prefetchWeights, IsShared,
                           GMM1_INTERLEAVED>(epilogue, params_, state.problemShape, gmmAddrInfo, startBlockIdx_,
                                             vecSetSyncCom, state.expertBeforeCnt, expertIdx, gmm1PingPongIdx_);
        }
    }
}

template <TemplateMegaMoeLayeredTypeClass>
template <bool IsShared>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::GroupMatmulWithActivationQuant(
    const GMMAddrInfo &gmmAddrInfo, const ExpertLoopState &state, uint32_t expertIdx, int32_t &vecSetSyncCom,
    int32_t &gmTileSequence)
{
    if constexpr (IsShared) {
        RunGmm1WithEpilogue<IsShared>(sharedEpilogueOp_, gmmAddrInfo, state, expertIdx, vecSetSyncCom, gmTileSequence);
    } else {
        RunGmm1WithEpilogue<IsShared>(epilogueOp_, gmmAddrInfo, state, expertIdx, vecSetSyncCom, gmTileSequence);
    }
}

// ===============================================================
// GroupMatmulWithCombine：先按实现路径分发，再按 combine 模式分发。
// IsShared=true 时跳过 activation flag 等待和 Combine 后处理，供共享专家使用。
// ===============================================================
template <TemplateMegaMoeLayeredTypeClass>
template <bool IsShared>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::GroupMatmulWithCombine(
    const GMMAddrInfo &gmmAddrInfo, const ExpertLoopState &state)
{
    if constexpr (ENABLE_A8W4 || ENABLE_A4W4) {
        RunGmm2A8W4<ActivationQuantOutType, Weight1Type, bfloat16_t, QuantScaleOutType, QuantScaleOutType,
                    L1_TILE_M_256, TopkWeightsPrefetch, IsShared, true>(state.problemShape, gmmAddrInfo,
                                                                        startBlockIdx_);
    } else {
        // A8W8_NZ / Generic 共用 RunGmm2Generic，仅 LayoutB 不同（ZN/ND）。
        if (params_.tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A8W8_NZ) {
            RunGmm2Generic<CombineQuantMode, QuantOutType, QuantOutType, bfloat16_t, QuantScaleOutType,
                           QuantScaleOutType, true, true, L1_TILE_M_256, TopkWeightsPrefetch, IsShared,
                           GMM1_INTERLEAVED, false>(state.problemShape, gmmAddrInfo, startBlockIdx_);
        } else {
            RunGmm2Generic<CombineQuantMode, QuantOutType, QuantOutType, bfloat16_t, QuantScaleOutType,
                           QuantScaleOutType, false, true, L1_TILE_M_256, TopkWeightsPrefetch, IsShared,
                           GMM1_INTERLEAVED, false>(state.problemShape, gmmAddrInfo, startBlockIdx_);
        }
    }
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::ProcessSharedExpertGmm1(
    const TupleShape &initShape, const BlockOffset &initOffset, int32_t &gmTileSequence)
{
    sharedEpilogueOp_.Init({.yGmAddr = params_.workspaceInfo.sharedExpertActivationDataPtr,
                            .yScaleGmAddr = params_.workspaceInfo.sharedExpertActivationScalePtr,
                            .clampLimit = params_.tilingData->clampLimit,
                            .actMode = params_.tilingData->actMode,
                            .actSubMode = params_.tilingData->actSubMode,
                            .activationAlpha = params_.tilingData->activationAlpha,
                            .activationBeta = params_.tilingData->activationBeta});

    GMMAddrInfo sharedGmm1AddrInfo{};
    ExpertLoopState sharedGmm1State{initShape, initOffset, 0};
    int32_t vecSetSyncCom = 0;
    for (uint32_t sharedIdx = 0; sharedIdx < sharedExpertNum_; sharedIdx++) {
        if (!UpdateSharedGroupParams(sharedGmm1State, sharedIdx)) {
            continue;
        }
        UpdateSharedGlobalBuffer<AddrUpdateMode::GMM1>(sharedGmm1AddrInfo, sharedGmm1State);
        GroupMatmulWithActivationQuant<true>(sharedGmm1AddrInfo, sharedGmm1State, sharedIdx, vecSetSyncCom,
                                             gmTileSequence);
    }
    EndSync<GMM1_INTERLEAVED>(vecSetSyncCom, gmm1PingPongIdx_);
    gmm1PingPongIdx_ = 0U;
    startBlockIdx_ = 0; // 共享专家GMM1修改了startBlockIdx_，重置给GMM1使用
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::ProcessSharedExpertGmm2(
    const TupleShape &initShape, const BlockOffset &initOffset)
{
    GMMAddrInfo sharedGmm2AddrInfo{};
    ExpertLoopState sharedGmm2State{initShape, initOffset, 0};
    for (uint32_t sharedIdx = 0; sharedIdx < sharedExpertNum_; sharedIdx++) {
        if (!UpdateSharedGroupParams(sharedGmm2State, sharedIdx)) {
            continue;
        }
        UpdateSharedGlobalBuffer<AddrUpdateMode::GMM2>(sharedGmm2AddrInfo, sharedGmm2State);
        GroupMatmulWithCombine<true>(sharedGmm2AddrInfo, sharedGmm2State);
    }
    SyncAll<false>();
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::RunGmm1ActivationForExpert(
    ExpertLoopState &state, GMMAddrInfo &gmmAddrInfo, int32_t &vecSetSyncCom, int32_t &gmTileSequence,
    uint32_t expertIdx)
{
    if (!UpdateGroupParams<AddrUpdateMode::GMM1>(state, expertIdx)) {
        return;
    }
    UpdateGlobalBuffer<AddrUpdateMode::GMM1>(gmmAddrInfo, state);
    GroupMatmulWithActivationQuant(gmmAddrInfo, state, expertIdx, vecSetSyncCom, gmTileSequence);
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::RunGmm2ForExpert(ExpertLoopState &state,
                                                                                        GMMAddrInfo &gmmAddrInfo,
                                                                                        uint32_t expertIdx)
{
    if (!UpdateGroupParams<AddrUpdateMode::GMM2>(state, expertIdx)) {
        return;
    }
    UpdateGlobalBuffer<AddrUpdateMode::GMM2>(gmmAddrInfo, state);
    GroupMatmulWithCombine(gmmAddrInfo, state);
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::PublishOrWaitGmm1AllDone()
{
    if constexpr (!TopkWeightsPrefetch) {
        return;
    }
    int32_t allDoneTag = static_cast<int32_t>(moeExpertPerRank_ + 1U);
    __gm__ int32_t *allDoneAddr =
        reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.gmm1TileStatusPtr) +
        static_cast<uint64_t>(moeExpertPerRank_) * params_.tilingData->maxTilesPerExpert * INT_CACHELINE;
    if constexpr (g_coreType == AIV) {
        constexpr uint32_t epilogueSubIdx = ENABLE_A8W4 ? 1U : 0U;
        if (subBlockIdx_ == epilogueSubIdx) {
            AscendC::WriteGmByPassDCache(allDoneAddr, allDoneTag);
        }
    } else {
        WaitUntilGmFlagEquals(allDoneAddr, allDoneTag);
    }
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::ProcessMoeExpertWave(
    const TupleShape &initShape, const BlockOffset &initOffset, int32_t &gmTileSequence)
{
    DispatchBuffInit();
    InitCombineBuffers();
    LockOwnedRemoteChannels();
    ExpertLoopState gmm1State{initShape, initOffset, 0};
    ExpertLoopState gmm2State{initShape, initOffset, 0};
    GMMAddrInfo gmm1AddrInfo{};
    GMMAddrInfo gmm2AddrInfo{};
    int32_t vecSetSyncCom = 0;
    // 计算宏 Wave 只负责组织 GMM 阶段，不再定义一级通信提交边界或二级 256-token ready 边界。
    // 小负载控制在 1～2 个有效 Wave；中等负载约 6 个；单专家很大的吞吐场景约 4 个。
    const uint32_t targetWaveCount = CalcTargetWaveCount();
    const uint32_t firstWaveExpertCount = CalcFirstWaveExpertCount(targetWaveCount);
    const uint32_t steadyWaveExpertCount = CalcSteadyWaveExpertCount(firstWaveExpertCount, targetWaveCount);
    uint32_t currentWaveBegin = 0U;
    uint32_t currentWaveEnd = firstWaveExpertCount;

    if constexpr (g_coreType == AIV) {
        if (subBlockIdx_ == 1U) {
            PrepareDispatch(firstWaveExpertCount);
            // 首个宏 Wave 就绪后即可启动计算，不再等待全部专家的二级 Dispatch。
            ReceiveDispatchExpertRange(currentWaveBegin, currentWaveEnd);
        }
    }

    while (currentWaveBegin < moeExpertPerRank_) {
        const uint32_t waveStartBlockIdx = startBlockIdx_;
        const uint32_t nextWaveBegin = currentWaveEnd;
        const uint32_t proposedWaveEnd = nextWaveBegin + steadyWaveExpertCount;
        const uint32_t nextWaveEnd = proposedWaveEnd < moeExpertPerRank_ ? proposedWaveEnd : moeExpertPerRank_;

        if constexpr (g_coreType == AIV) {
            if (subBlockIdx_ == 1U && nextWaveBegin < moeExpertPerRank_) {
                // 通信核预取下一宏 Wave；AIC/AIV0 同时计算当前宏 Wave。
                ReceiveDispatchExpertRange(nextWaveBegin, nextWaveEnd);
            }
        }

        for (uint32_t expertIdx = currentWaveBegin; expertIdx < currentWaveEnd; ++expertIdx) {
            RunGmm1ActivationForExpert(gmm1State, gmm1AddrInfo, vecSetSyncCom, gmTileSequence, expertIdx);
        }
        const uint32_t gmm1EndBlockIdx = startBlockIdx_;
        ExpertLoopState combineWaveBeginState = gmm2State;
        for (uint32_t expertIdx = currentWaveBegin; expertIdx < currentWaveEnd; ++expertIdx) {
            RunGmm2ForExpert(gmm2State, gmm2AddrInfo, expertIdx);
        }
        ProcessCombineExpertRange(combineWaveBeginState, currentWaveBegin, currentWaveEnd);
        if constexpr (!ENABLE_A8W4) {
            const bool hasNextWave = currentWaveEnd < moeExpertPerRank_;
            const bool fixedRoleResonance = startBlockIdx_ == waveStartBlockIdx && gmm1EndBlockIdx != waveStartBlockIdx;
            if (hasNextWave && fixedRoleResonance) {
                startBlockIdx_ = gmm1EndBlockIdx;
            }
        }
        currentWaveBegin = nextWaveBegin;
        currentWaveEnd = nextWaveEnd;
    }

    if constexpr (TopkWeightsPrefetch) {
        PublishOrWaitGmm1AllDone();
    } else {
        EndSync<GMM1_INTERLEAVED>(vecSetSyncCom, gmm1PingPongIdx_);
    }
    gmm1PingPongIdx_ = 0U;
    if constexpr (g_coreType == AIV) {
        if (subBlockIdx_ == 1U) {
            PipeBarrier<PIPE_ALL>();
            ExpertTokenNumCopyOut();
            DrainAndUnlockOwnedRemoteChannels();
        }
        PipeBarrier<PIPE_ALL>();
        SyncAll<true>();
    }
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::Process()
{
    // 1.本卡数据处理
    int64_t oriOverflowMode = GetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>();
    SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(0);
    SendAndQuantBuffInit();
    SendMaskCal();        // 源卡按全局专家计算掩码，并推送到目标专家卡
    ResetFlagList();      // 清零工作空间中的同步标志
    ResetDispatchState(); // 清零跨 Server URMA Dispatch 队列和中继就绪标志
    if (sharedExpertNum_ > 0) {
        SharedExpertCopyInput();
    }
    if constexpr (g_coreType == AIV) {
        PipeBarrier<PIPE_ALL>();
    }
    SyncAll<false>(); // AIC 等待同步标志清零完成

    // 共享专家 GMM1+SwiGLU 前移到 MoE 之前执行，复用 MoE 函数。
    TupleShape initShape;
    Get<N_VALUE>(initShape) = hiddenDim_;
    Get<K_VALUE>(initShape) = k_;
    BlockOffset initOffset{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    int32_t gmTileSequence = 0; // Specialized A8W4/A4W4 GMM1 AIC-AIV1 tile ready sequence.
    if (sharedExpertNum_ > 0) {
        ProcessSharedExpertGmm1(initShape, initOffset, gmTileSequence);
        SyncAll<false>();
    }

    if constexpr (g_coreType == AIV) {
        // 全部物理 AIV 先量化各自 token 分片，再批量发布本 Server ready flag 并构造一级队列。
        // 量化/flag 的 MTE3 与 Scalar 建队列允许重叠；CrossRankSync 尾部统一排空并同步全部 AIV。
        DispatchPrepareBuffInit();
        QuantizeLocalTokensToRelay();
        SetLocalRelayReadyFlags();
        BuildDispatchRelayQueues();
    }

    CrossRankSyncInWorldSize(true); // 同一物理 AIV 继续使用 SendMask 已取得的锁，并在同步结束后释放。

    // URMA Layered 的 A8W8、A4W4 和 A8W4 共用同一专家 Wave 状态机。
    ProcessMoeExpertWave(initShape, initOffset, gmTileSequence);

    // 3.5: 共享专家 GMM2 (MoE GMM2 之后, 复用 MoE 函数)
    if (sharedExpertNum_ > 0) {
        ProcessSharedExpertGmm2(initShape, initOffset);
    }

    // 4. 本卡数据Unpermute
    if constexpr (g_coreType == AIV) {
        UnpermuteBuffInit();
        CrossRankSyncInWorldSize(false); // Wave 已释放固定归属锁，同步核重新取得通道后确认 Combine 完成。
        Unpermute();
    }
    SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(oriOverflowMode);
}

#endif

} // namespace MegaMoeImpl

#if defined(ENABLE_MEGA_MOE_LAYERED_KERNEL)
#include "stage/mega_moe_layered_dispatch.h"
#include "stage/mega_moe_layered_combine.h"
#endif

#undef TemplateMegaMoeLayeredTypeClass
#undef TemplateMegaMoeLayeredTypeFunc
#endif
