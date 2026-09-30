/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file ffn_to_attention_urma.h
 * \brief FFNToAttentionV2 URMA implementation.
 */
#ifndef FFN_TO_ATTENTION_URMA_H
#define FFN_TO_ATTENTION_URMA_H

#if __has_include("version/asc_devkit_version.h") && __has_include("version/hcomm_version.h")
#include "version/asc_devkit_version.h"
#include "version/hcomm_version.h"

#if (ASC_DEVKIT_MAJOR > 9 || (ASC_DEVKIT_MAJOR == 9 && ASC_DEVKIT_MINOR > 1)) && \
    (HCOMM_MAJOR > 9 || (HCOMM_MAJOR == 9 && HCOMM_MINOR > 1))
#define ENABLE_FFN_TO_ATTENTION_V2_KERNEL
#endif

#endif

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "kernel_tiling/kernel_tiling.h"
#if __has_include("adv_api/hcomm/hcomm.h")
#include "adv_api/hcomm/hcomm.h"
#endif
#include "adv_api/reduce/reduce.h"
#include "adv_api/reduce/sum.h"
#include "ffn_to_attention_v2_tiling.h"

#include "../../common/op_kernel/moe_distribute_base.h"
#include "../../common/op_kernel/attention_ffn_context.h"
#include "../../common/op_kernel/mc2_kernel_utils.h"

namespace FFNToAttentionImpl {

#if defined(ENABLE_FFN_TO_ATTENTION_V2_KERNEL)

// x only supports FLOAT16/BFLOAT16, both of which occupy 2 bytes per element.
// The host tiling mirrors this value in ffn_to_attention_v2_tilling.cpp; keep both sides in sync.
constexpr uint32_t TOKEN_DTYPE_BYTES = 2U;

constexpr uint8_t URMA_BUFFER_NUM = 2U;
constexpr uint32_t URMA_UB_ALIGN = 32U;
constexpr uint64_t URMA_WIN_ADDR_ALIGN = 512UL;

constexpr uint32_t URMA_HCOMM_INIT_SIZE = 512U;
constexpr uint32_t URMA_FLAG_SLOT_SIZE = 32U;
constexpr int32_t URMA_FLAG_VALUE = 1;

constexpr uint32_t URMA_WQE_SIZE = 64U;
constexpr uint32_t URMA_BATCH_WQE_CAPACITY = 256U;
constexpr uint32_t URMA_BATCH_BUFFER_SIZE = URMA_BATCH_WQE_CAPACITY * URMA_WQE_SIZE;
constexpr uint32_t HCOMM_SQ_MAX_PENDING = 32767U;
constexpr uint32_t METADATA_CHUNK_TOKEN_NUM = 256U;
constexpr uint32_t METADATA_FIELD_NUM = 4U;
constexpr uint32_t METADATA_TOKEN_BATCH_OFFSET_FIELD_INDEX = 2U;
constexpr uint32_t METADATA_TOKEN_TOPK_OFFSET_FIELD_INDEX = 3U;
constexpr uint32_t METADATA_CHUNK_BUFFER_SIZE = METADATA_CHUNK_TOKEN_NUM * METADATA_FIELD_NUM * sizeof(int32_t);
constexpr uint32_t REMOTE_ADDRESS_ENTRY_FIELD_NUM = 2U;
constexpr uint32_t REMOTE_ADDRESS_ENTRY_SIZE = REMOTE_ADDRESS_ENTRY_FIELD_NUM * sizeof(uint64_t);
constexpr uint32_t ADDRESS_TABLE_BUFFER_SIZE = 40U * 1024U;

// rankInfoBuf_ layout: [own rank counts | preceding-core prefix | per-batch reduce sums], one row each.
constexpr uint32_t RANK_INFO_SEGMENT_NUM = 3U;
constexpr uint32_t RANK_INFO_PREFIX_SEG_IDX = 1U;
constexpr uint32_t RANK_INFO_REDUCE_SUM_SEG_IDX = 2U;
static constexpr AscendC::UrmaWqeEntry remoteWqeConfig = {.odr = 5, .fence = 1, .se = 0, .cqe = 0, .inlineEn = 0};
static constexpr AscendC::UrmaWqeEntry remoteCqeWqeConfig = {.odr = 5, .fence = 1, .se = 0, .cqe = 1, .inlineEn = 0};
static constexpr AscendC::UrmaWqeEntry remoteOrderedFlagWqeConfig = {
    .odr = 6, .fence = 1, .se = 0, .cqe = 0, .inlineEn = 0};
static constexpr AscendC::UrmaWqeEntry remoteOrderedFlagCqeWqeConfig = {
    .odr = 6, .fence = 1, .se = 0, .cqe = 1, .inlineEn = 0};

#define TemplateFFNToAttentionUrmaTypeClass typename xType, bool isInputRankTable
#define TemplateFFNToAttentionUrmaTypeFunc xType, isInputRankTable

using namespace AscendC;

using HcommChannelHandle = AscendC::ChannelHandle;

struct RemoteAddressEntry {
    uint64_t tokenSlot;
    uint64_t sourceTokenIndex;
};

template <TemplateFFNToAttentionUrmaTypeClass>
class FFNToAttentionUrma {
public:
    __aicore__ inline FFNToAttentionUrma(){};
    __aicore__ inline void Init(GM_ADDR mc2Context, GM_ADDR x, GM_ADDR sessionIds, GM_ADDR microBatchIds,
                                GM_ADDR tokenIds, GM_ADDR expertOffsets, GM_ADDR actualTokenNum, GM_ADDR attnRankTable,
                                GM_ADDR workspaceGM, TPipe* pipe, const FFNToAttentionV2TilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void PrefetchTokenMetaData(uint64_t tokenStart, uint32_t tokenNum);
    __aicore__ inline void ReadTokenMetaDataFromLocal(ReadTokenMetaDataStruct& metaDataStruct, uint32_t localOffset);
    __aicore__ inline GM_ADDR GetWindowAddr(uint32_t curAttenWorkRank);
    __aicore__ inline uint64_t GetUrmaCommHandle(uint32_t dstRank, uint32_t channelIndex);
    __aicore__ inline void HcommInit();
    __aicore__ inline void InitLocalFlag();
    __aicore__ inline void SendToLocal(GM_ADDR remoteDataAddr, GM_ADDR remoteFlagAddr, uint64_t yOffset,
                                       const DataCopyExtParams& xCopyParams);
    template <typename BatchHandleType>
    __aicore__ inline void AppendRemoteDataWrite(BatchHandleType& batchHandle, GM_ADDR remoteWindowAddr,
                                                 const RemoteAddressEntry& addressEntry, bool enableCqe);
    template <typename BatchHandleType>
    __aicore__ inline void AppendRemoteFlagWrite(BatchHandleType& batchHandle, GM_ADDR remoteWindowAddr,
                                                 const RemoteAddressEntry& addressEntry, bool enableCqe);
    template <typename BatchHandleType>
    __aicore__ inline void ClearRemoteFlags(BatchHandleType& batchHandle, GM_ADDR remoteWindowAddr,
                                            HcommChannelHandle channel, GlobalTensor<uint64_t> rankAddressTable,
                                            uint32_t entryCount, uint32_t channelIdx);
    __aicore__ inline void BuildAddressTable();
    __aicore__ inline uint32_t GetAddressTableCount(uint32_t targetRank);
    __aicore__ inline void SendAddressTable(uint32_t targetRank, uint32_t entryCount, uint32_t channelIdx);
    template <typename BatchHandleType>
    __aicore__ inline void CommitAndDrainChannel(BatchHandleType& batchHandle, HcommChannelHandle channel,
                                                 uint32_t& preparedWqeCount, uint32_t& sqWriteCount);
    __aicore__ inline GM_ADDR GetRankAddressTableAddr(uint32_t targetRank)
    {
        return addressTableWinAddr_ + static_cast<uint64_t>(targetRank) * addressTableStridePerRank_;
    }

    TPipe* tpipe_{nullptr};
    Hcomm<COMM_PROTOCOL_UBC_CTP> hcomm_;
    __gm__ Mc2Aclnn::AttentionFFNContext* mc2Context_{nullptr};

    GM_ADDR inputDataAddr_{nullptr};
    GM_ADDR localFlagAddr_{nullptr};
    GM_ADDR zeroFlagAddr_{nullptr};
    GM_ADDR localWinAddr_{nullptr};
    GM_ADDR addressTableWinAddr_{nullptr};
    GM_ADDR totalRankCountWinAddr_{nullptr};
    GM_ADDR perCoreRankCountWinAddr_{nullptr};

    GlobalTensor<xType> xGMTensor_;
    GlobalTensor<int32_t> sessionIdsGMTensor_;
    GlobalTensor<int32_t> microBatchIdsGMTensor_;
    GlobalTensor<int32_t> tokenIdsGMTensor_;
    GlobalTensor<int32_t> expertOffsetsGMTensor_;
    GlobalTensor<int64_t> actualTokenNumGMTensor_;
    GlobalTensor<int32_t> attnRankTableGMTensor_;
    LocalTensor<xType> xTmpTensor_;
    LocalTensor<int32_t> statusTensor_;
    LocalTensor<uint8_t> hcommTensor_;
    LocalTensor<uint8_t> hcommBatchTensor_;
    LocalTensor<int32_t> metadataTensor_;
    LocalTensor<int32_t> rankTableTensor_;

    TBuf<> statusBuf_;
    TBuf<> hcommBuf_;
    TBuf<> metadataBuf_;
    TBuf<> addressTableBuf_;
    TBuf<> rankInfoBuf_;
    TBuf<> rankTableBuf_;
    TBuf<TPosition::VECOUT> hcommBatchBuf_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> xQueue_;

    uint32_t aivId_{0};
    uint32_t rankId_{0};
    uint64_t actualTokenNum_{0};
    uint64_t maxTokenNum_{0};
    uint32_t axisH_{0};
    uint32_t axisHS_{0};
    uint32_t axisA_{0};
    uint32_t microBatchNum_{0};
    uint32_t axisBS_{0};
    uint32_t expertNumPerToken_{0};
    uint32_t aivNum_{0};
    uint32_t worldSize_{0};
    uint32_t channelsPerRank_{1};
    uint64_t batchSizeSendCnt_{0};
    uint64_t tokenSlotNum_{0};
    uint64_t hSize_{0};
    uint64_t winTokenInfoTableSize_{0};
    uint64_t addressTableStridePerRank_{0};
    uint64_t addressTableWinOffset_{0};
    uint64_t perCoreRankCountStride_{0};
    uint32_t rankCountElements_{0};
};

template <TemplateFFNToAttentionUrmaTypeClass>
__aicore__ inline void FFNToAttentionUrma<TemplateFFNToAttentionUrmaTypeFunc>::Init(
    GM_ADDR mc2Context, GM_ADDR x, GM_ADDR sessionIds, GM_ADDR microBatchIds, GM_ADDR tokenIds, GM_ADDR expertOffsets,
    GM_ADDR actualTokenNum, GM_ADDR attnRankTable, GM_ADDR workspaceGM, TPipe* pipe,
    const FFNToAttentionV2TilingData* tilingData)
{
    tpipe_ = pipe;
    aivId_ = GetBlockIdx();
    static_assert(sizeof(xType) == TOKEN_DTYPE_BYTES,
                  "token dtype must match TOKEN_DTYPE_BYTES assumed by host tiling");
    mc2Context_ = reinterpret_cast<__gm__ Mc2Aclnn::AttentionFFNContext*>(mc2Context);
    rankId_ = mc2Context_->epRankId;

    axisBS_ = tilingData->ffnToAttentionV2Info.BS;
    axisH_ = tilingData->ffnToAttentionV2Info.H;
    axisHS_ = tilingData->ffnToAttentionV2Info.HS;
    axisA_ = tilingData->ffnToAttentionV2Info.A;
    microBatchNum_ = tilingData->ffnToAttentionV2Info.microBatchNum;
    expertNumPerToken_ = tilingData->ffnToAttentionV2Info.expertNumPerToken;
    aivNum_ = tilingData->ffnToAttentionV2Info.aivNum;
    worldSize_ = tilingData->ffnToAttentionV2Info.worldSize;
    maxTokenNum_ = tilingData->ffnToAttentionV2Info.maxTokenNum;
    // Multi-channel: hcommHandle_ is laid out by rank * channelsPerRank + channel. Guard against
    // an unset or oversized context value so a degenerate context still falls back to 1 channel.
    channelsPerRank_ = mc2Context_->channelsPerRank;
    if (worldSize_ == 0U || channelsPerRank_ == 0U || channelsPerRank_ > Mc2Aclnn::HCCL_MAX_RANK_SIZE / worldSize_) {
        channelsPerRank_ = 1U;
    }

    inputDataAddr_ = x;
    xGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ xType*>(x));
    sessionIdsGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(sessionIds));
    microBatchIdsGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(microBatchIds));
    tokenIdsGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(tokenIds));
    expertOffsetsGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(expertOffsets));
    actualTokenNumGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(actualTokenNum));
    if constexpr (isInputRankTable) {
        attnRankTableGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(attnRankTable));
    }

    int64_t actualTokenNumValue = actualTokenNumGMTensor_.GetValue(0UL);
    ascendc_assert(actualTokenNumValue >= 0, "actualTokenNum must be non-negative, actualTokenNumValue=%ld",
                   actualTokenNumValue);
    ascendc_assert(static_cast<uint64_t>(actualTokenNumValue) <= maxTokenNum_,
                   "actualTokenNum exceeds x dim0, actualTokenNum=%ld, maxTokenNum=%lu", actualTokenNumValue,
                   maxTokenNum_);
    actualTokenNum_ = static_cast<uint64_t>(actualTokenNumValue);
    hSize_ = static_cast<uint64_t>(axisH_) * sizeof(xType);
    batchSizeSendCnt_ = static_cast<uint64_t>(axisBS_) * expertNumPerToken_;
    const uint64_t tokenSlotNum = static_cast<uint64_t>(microBatchNum_) * batchSizeSendCnt_;
    tokenSlotNum_ = tokenSlotNum;
    winTokenInfoTableSize_ = Ceil(tokenSlotNum * sizeof(int32_t), URMA_WIN_ADDR_ALIGN) * URMA_WIN_ADDR_ALIGN;

    // Follow MoeEpCombine's layout: one independent compact address table per rank, followed by the total-count row
    // and the per-core count matrix. Every core publishes only its own count row.
    addressTableStridePerRank_ =
        Ceil(maxTokenNum_ * REMOTE_ADDRESS_ENTRY_SIZE, URMA_WIN_ADDR_ALIGN) * URMA_WIN_ADDR_ALIGN;
    (void)workspaceGM;
    localWinAddr_ = GetWindowAddr(rankId_);
    addressTableWinOffset_ = tilingData->ffnToAttentionV2Info.addressTableWinOffset;
    addressTableWinAddr_ = localWinAddr_ + addressTableWinOffset_;
    uint64_t addressTableBytes = static_cast<uint64_t>(worldSize_) * addressTableStridePerRank_;
    rankCountElements_ = static_cast<uint32_t>(
        Ceil(static_cast<uint64_t>(worldSize_) * sizeof(int32_t), URMA_UB_ALIGN) * URMA_UB_ALIGN / sizeof(int32_t));
    uint64_t rankCountRowBytes = static_cast<uint64_t>(rankCountElements_) * sizeof(int32_t);
    uint64_t totalRankCountBytes = Ceil(rankCountRowBytes, URMA_WIN_ADDR_ALIGN) * URMA_WIN_ADDR_ALIGN;
    perCoreRankCountStride_ = totalRankCountBytes;
    totalRankCountWinAddr_ = addressTableWinAddr_ + addressTableBytes;
    perCoreRankCountWinAddr_ = totalRankCountWinAddr_ + totalRankCountBytes;

    const uint64_t flagWindowSize = static_cast<uint64_t>(aivNum_) * URMA_FLAG_SLOT_SIZE;
    ascendc_assert(
        addressTableWinOffset_ >= flagWindowSize + winTokenInfoTableSize_ + tokenSlotNum * axisHS_ * sizeof(xType),
        "address table offset must reserve flag slots after the receiving area, "
        "addressTableWinOffset=%lu",
        addressTableWinOffset_);
    GM_ADDR localScratchAddr = addressTableWinAddr_ - flagWindowSize;
    localFlagAddr_ = localScratchAddr + static_cast<uint64_t>(aivId_) * URMA_FLAG_SLOT_SIZE;
    // Zero source for the per-round flag-clear phase; lives inside this AIV's own 32B source slot
    // (word1), so the reserved flagWindowSize = aivNum_ * URMA_FLAG_SLOT_SIZE stays unchanged.
    zeroFlagAddr_ = localFlagAddr_ + sizeof(int32_t);

    // x may be a view of registered memory, but its payload must not alias the scratch tail.
    const uint64_t inputAddr = reinterpret_cast<uint64_t>(inputDataAddr_);
    const uint64_t scratchBegin = reinterpret_cast<uint64_t>(localScratchAddr);
    const uint64_t scratchEnd = reinterpret_cast<uint64_t>(localScratchAddr) + flagWindowSize;
    ascendc_assert(inputAddr >= scratchEnd || inputAddr + maxTokenNum_ * hSize_ <= scratchBegin,
                   "x overlaps local flag window tail");

    tpipe_->InitBuffer(xQueue_, URMA_BUFFER_NUM, Ceil(hSize_, static_cast<uint64_t>(URMA_UB_ALIGN)) * URMA_UB_ALIGN);
    tpipe_->InitBuffer(statusBuf_, URMA_UB_ALIGN);
    statusTensor_ = statusBuf_.Get<int32_t>();
    tpipe_->InitBuffer(metadataBuf_, METADATA_CHUNK_BUFFER_SIZE);
    metadataTensor_ = metadataBuf_.Get<int32_t>();
    tpipe_->InitBuffer(addressTableBuf_, ADDRESS_TABLE_BUFFER_SIZE);
    tpipe_->InitBuffer(rankInfoBuf_, RANK_INFO_SEGMENT_NUM * rankCountElements_ * sizeof(int32_t));
    if constexpr (isInputRankTable) {
        tpipe_->InitBuffer(rankTableBuf_,
                           Ceil(static_cast<uint64_t>(axisA_) * sizeof(int32_t), URMA_UB_ALIGN) * URMA_UB_ALIGN);
    }
}

template <TemplateFFNToAttentionUrmaTypeClass>
__aicore__ inline void FFNToAttentionUrma<TemplateFFNToAttentionUrmaTypeFunc>::HcommInit()
{
    tpipe_->InitBuffer(hcommBuf_, URMA_HCOMM_INIT_SIZE);
    hcommTensor_ = hcommBuf_.Get<uint8_t>();
    hcomm_.Init(hcommTensor_, URMA_HCOMM_INIT_SIZE);

    tpipe_->InitBuffer(hcommBatchBuf_, URMA_BATCH_BUFFER_SIZE);
    hcommBatchTensor_ = hcommBatchBuf_.Get<uint8_t>();
    Duplicate<uint8_t>(hcommBatchTensor_, 0U, URMA_BATCH_BUFFER_SIZE);

    SyncFunc<HardEvent::V_S>();
}

template <TemplateFFNToAttentionUrmaTypeClass>
__aicore__ inline void FFNToAttentionUrma<TemplateFFNToAttentionUrmaTypeFunc>::InitLocalFlag()
{
    constexpr uint32_t flagSlotElementNum = URMA_FLAG_SLOT_SIZE / sizeof(int32_t);
    Duplicate<int32_t>(statusTensor_, URMA_FLAG_VALUE, flagSlotElementNum);
    // word0 keeps the flag value (1) used by the flag phase; word1 becomes the constant zero
    // source consumed by the per-round clear phase (ClearRemoteFlags / SendToLocal).
    statusTensor_.SetValue(1, 0);
    SyncFunc<HardEvent::V_MTE3>();
    SyncFunc<HardEvent::S_MTE3>();
    GlobalTensor<int32_t> localFlagGMTensor;
    localFlagGMTensor.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(localFlagAddr_));
    DataCopy(localFlagGMTensor, statusTensor_, flagSlotElementNum);
}

template <TemplateFFNToAttentionUrmaTypeClass>
__aicore__ inline GM_ADDR FFNToAttentionUrma<TemplateFFNToAttentionUrmaTypeFunc>::GetWindowAddr(
    uint32_t curAttenWorkRank)
{
    return (GM_ADDR)mc2Context_->epHcclBuffer_[curAttenWorkRank];
}

template <TemplateFFNToAttentionUrmaTypeClass>
__aicore__ inline uint64_t FFNToAttentionUrma<TemplateFFNToAttentionUrmaTypeFunc>::GetUrmaCommHandle(
    uint32_t dstRank, uint32_t channelIndex)
{
    return mc2Context_->hcommHandle_[dstRank * channelsPerRank_ + channelIndex];
}

template <TemplateFFNToAttentionUrmaTypeClass>
__aicore__ inline void FFNToAttentionUrma<TemplateFFNToAttentionUrmaTypeFunc>::PrefetchTokenMetaData(
    uint64_t tokenStart, uint32_t tokenNum)
{
    SyncFunc<HardEvent::S_MTE2>(); // Finish scalar reads before reusing the chunk buffer.
    DataCopyExtParams copyParams = {1U, static_cast<uint32_t>(tokenNum * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> padParams{false, 0U, 0U, 0U};
    DataCopyPad(metadataTensor_, sessionIdsGMTensor_[tokenStart], copyParams, padParams);
    DataCopyPad(metadataTensor_[METADATA_CHUNK_TOKEN_NUM], microBatchIdsGMTensor_[tokenStart], copyParams, padParams);
    DataCopyPad(metadataTensor_[METADATA_TOKEN_BATCH_OFFSET_FIELD_INDEX * METADATA_CHUNK_TOKEN_NUM],
                tokenIdsGMTensor_[tokenStart], copyParams, padParams);
    DataCopyPad(metadataTensor_[METADATA_TOKEN_TOPK_OFFSET_FIELD_INDEX * METADATA_CHUNK_TOKEN_NUM],
                expertOffsetsGMTensor_[tokenStart], copyParams, padParams);
    SyncFunc<HardEvent::MTE2_S>();
}

template <TemplateFFNToAttentionUrmaTypeClass>
__aicore__ inline void FFNToAttentionUrma<TemplateFFNToAttentionUrmaTypeFunc>::ReadTokenMetaDataFromLocal(
    ReadTokenMetaDataStruct& metaDataStruct, uint32_t localOffset)
{
    int32_t curAttenWorkId = metadataTensor_.GetValue(localOffset);
    metaDataStruct.curAttenWorkIds = static_cast<uint32_t>(curAttenWorkId);
    metaDataStruct.curAttenWorkRank = metaDataStruct.curAttenWorkIds;

    if constexpr (isInputRankTable) {
        ascendc_assert(curAttenWorkId >= 0 && static_cast<uint32_t>(curAttenWorkId) < axisA_,
                       "sessionId out of range, sessionId=%d, axisA=%u", curAttenWorkId, axisA_);
        int32_t curAttenWorkRank = rankTableTensor_.GetValue(metaDataStruct.curAttenWorkIds);
        metaDataStruct.curAttenWorkRank = static_cast<uint32_t>(curAttenWorkRank);
    }

    int32_t curMicroBatchId = metadataTensor_.GetValue(METADATA_CHUNK_TOKEN_NUM + localOffset);
    metaDataStruct.curMicroBatchIds = static_cast<uint32_t>(curMicroBatchId);

    int32_t curTokenBatchOffset =
        metadataTensor_.GetValue(METADATA_TOKEN_BATCH_OFFSET_FIELD_INDEX * METADATA_CHUNK_TOKEN_NUM + localOffset);
    metaDataStruct.curTokenBatchOffset = static_cast<uint32_t>(curTokenBatchOffset);

    int32_t curTokenTopkOffset =
        metadataTensor_.GetValue(METADATA_TOKEN_TOPK_OFFSET_FIELD_INDEX * METADATA_CHUNK_TOKEN_NUM + localOffset);
    metaDataStruct.curTokenTopkOffset = static_cast<uint32_t>(curTokenTopkOffset);
}

template <TemplateFFNToAttentionUrmaTypeClass>
__aicore__ inline void FFNToAttentionUrma<TemplateFFNToAttentionUrmaTypeFunc>::SendToLocal(
    GM_ADDR remoteDataAddr, GM_ADDR remoteFlagAddr, uint64_t yOffset, const DataCopyExtParams& xCopyParams)
{
    DataCopyPadExtParams<xType> copyPadExtParams{false, 0U, 0U, 0U};
    xTmpTensor_ = xQueue_.AllocTensor<xType>();
    DataCopyPad(xTmpTensor_, xGMTensor_[static_cast<uint64_t>(yOffset) * axisH_], xCopyParams, copyPadExtParams);
    xQueue_.EnQue(xTmpTensor_);
    xTmpTensor_ = xQueue_.DeQue<xType>();

    // Per-round flag clear for self-delivered tokens: write 0 (statusTensor_ word1) into the local
    // window slot before data/flag, so the consumer sees a fresh 0->1 edge every round. MTE3 is
    // in-order, so clear lands before data and data lands before the flag.
    GlobalTensor<int32_t> tokenInfoTableGMTensor;
    tokenInfoTableGMTensor.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(remoteFlagAddr));
    DataCopyExtParams flagCopyParams = {1U, sizeof(int32_t), 0U, 0U, 0U};
    DataCopyPad(tokenInfoTableGMTensor, statusTensor_[1], flagCopyParams);

    GlobalTensor<xType> tokenDataGMTensor;
    tokenDataGMTensor.SetGlobalBuffer(reinterpret_cast<__gm__ xType*>(remoteDataAddr));
    DataCopyPad(tokenDataGMTensor, xTmpTensor_, xCopyParams);
    PipeBarrier<PIPE_MTE3>();
    xQueue_.FreeTensor<xType>(xTmpTensor_);

    DataCopyPad(tokenInfoTableGMTensor, statusTensor_, flagCopyParams);
    SyncFunc<HardEvent::MTE3_S>();
}

template <TemplateFFNToAttentionUrmaTypeClass>
template <typename BatchHandleType>
__aicore__ inline void FFNToAttentionUrma<TemplateFFNToAttentionUrmaTypeFunc>::AppendRemoteDataWrite(
    BatchHandleType& batchHandle, GM_ADDR remoteWindowAddr, const RemoteAddressEntry& addressEntry, bool enableCqe)
{
    uint64_t remoteDataOffset =
        winTokenInfoTableSize_ + addressEntry.tokenSlot * static_cast<uint64_t>(axisHS_) * sizeof(xType);
    GM_ADDR sourceDataAddr = inputDataAddr_ + addressEntry.sourceTokenIndex * hSize_;
    if (enableCqe) {
        (void)hcomm_.WriteNbi<remoteCqeWqeConfig>(batchHandle, remoteWindowAddr + remoteDataOffset, sourceDataAddr,
                                                  static_cast<uint32_t>(hSize_));
    } else {
        (void)hcomm_.WriteNbi<remoteWqeConfig>(batchHandle, remoteWindowAddr + remoteDataOffset, sourceDataAddr,
                                               static_cast<uint32_t>(hSize_));
    }
}

template <TemplateFFNToAttentionUrmaTypeClass>
template <typename BatchHandleType>
__aicore__ inline void FFNToAttentionUrma<TemplateFFNToAttentionUrmaTypeFunc>::AppendRemoteFlagWrite(
    BatchHandleType& batchHandle, GM_ADDR remoteWindowAddr, const RemoteAddressEntry& addressEntry, bool enableCqe)
{
    uint64_t remoteFlagOffset = addressEntry.tokenSlot * sizeof(int32_t);
    // Every flag write uses the ordered config (same as attention_to_ffn's flag phase): each flag
    // must land after this round's clear/data WQEs on the same channel, otherwise a late clear
    // could erase an already-visible flag on the same slot.
    if (enableCqe) {
        (void)hcomm_.WriteNbi<remoteOrderedFlagCqeWqeConfig>(batchHandle, remoteWindowAddr + remoteFlagOffset,
                                                             localFlagAddr_, static_cast<uint32_t>(sizeof(int32_t)));
    } else {
        (void)hcomm_.WriteNbi<remoteOrderedFlagWqeConfig>(batchHandle, remoteWindowAddr + remoteFlagOffset,
                                                          localFlagAddr_, static_cast<uint32_t>(sizeof(int32_t)));
    }
}

template <TemplateFFNToAttentionUrmaTypeClass>
__aicore__ inline void FFNToAttentionUrma<TemplateFFNToAttentionUrmaTypeFunc>::BuildAddressTable()
{
    if constexpr (isInputRankTable) {
        // Fetch the whole attention rank table once; token-level lookups below become UB reads
        // instead of blocking per-token GM scalar loads.
        rankTableTensor_ = rankTableBuf_.Get<int32_t>();
        SyncFunc<HardEvent::S_MTE2>();
        DataCopyPad(rankTableTensor_, attnRankTableGMTensor_,
                    DataCopyExtParams{1U, static_cast<uint32_t>(axisA_ * sizeof(int32_t)), 0U, 0U, 0U},
                    DataCopyPadExtParams<int32_t>{false, 0U, 0U, 0U});
        SyncFunc<HardEvent::MTE2_S>();
    }

    LocalTensor<int32_t> rankCounts = rankInfoBuf_.Get<int32_t>();
    LocalTensor<int32_t> rankPrefix = rankCounts[RANK_INFO_PREFIX_SEG_IDX * rankCountElements_];
    LocalTensor<int32_t> prefixSums = rankCounts[RANK_INFO_REDUCE_SUM_SEG_IDX * rankCountElements_];
    Duplicate<int32_t>(rankCounts, 0, rankCountElements_);
    Duplicate<int32_t>(rankPrefix, 0, rankCountElements_);
    SyncFunc<HardEvent::V_S>();

    uint64_t tokensPerCore = actualTokenNum_ / aivNum_;
    uint64_t remainder = actualTokenNum_ % aivNum_;
    uint64_t scanStart = static_cast<uint64_t>(aivId_) * tokensPerCore + ((aivId_ < remainder) ? aivId_ : remainder);
    uint64_t scanEnd = scanStart + tokensPerCore + ((aivId_ < remainder) ? 1U : 0U);

    // Pass 1: every AIV counts its own remote records per rank and publishes only its own count row.
    for (uint64_t tokenStart = scanStart; tokenStart < scanEnd; tokenStart += METADATA_CHUNK_TOKEN_NUM) {
        uint32_t tokenNum = static_cast<uint32_t>(
            (tokenStart + METADATA_CHUNK_TOKEN_NUM > scanEnd) ? scanEnd - tokenStart : METADATA_CHUNK_TOKEN_NUM);
        PrefetchTokenMetaData(tokenStart, tokenNum);

        for (uint32_t localOffset = 0U; localOffset < tokenNum; ++localOffset) {
            ReadTokenMetaDataStruct metaDataStruct;
            ReadTokenMetaDataFromLocal(metaDataStruct, localOffset);
            ascendc_assert(metaDataStruct.curAttenWorkRank < worldSize_, "attention rank is out of range");
            if (metaDataStruct.curAttenWorkRank == rankId_) {
                continue;
            }
            uint32_t dstRank = metaDataStruct.curAttenWorkRank;
            rankCounts.SetValue(dstRank, rankCounts.GetValue(dstRank) + 1);
        }
    }

    GlobalTensor<int32_t> perCoreRankCounts;
    GM_ADDR countRowAddr = perCoreRankCountWinAddr_ + static_cast<uint64_t>(aivId_) * perCoreRankCountStride_;
    perCoreRankCounts.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(countRowAddr));
    SyncFunc<HardEvent::S_MTE3>();
    DataCopy(perCoreRankCounts, rankCounts, rankCountElements_);
    SyncFunc<HardEvent::MTE3_S>();
    SyncAll<true>();

    // Every core independently sums the preceding core rows. Core 0 is not a centralized prefix calculator.
    uint32_t rankCountRowBytes = rankCountElements_ * sizeof(int32_t);
    uint32_t prefixRowsPerBatch = ADDRESS_TABLE_BUFFER_SIZE / rankCountRowBytes;
    LocalTensor<int32_t> prefixRows = addressTableBuf_.Get<int32_t>();
    GlobalTensor<int32_t> allPerCoreRankCounts;
    allPerCoreRankCounts.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(perCoreRankCountWinAddr_));
    const DataCopyPadExtParams<int32_t> prefixPadParams{false, 0U, 0U, 0U};
    for (uint32_t coreStart = 0U; coreStart < aivId_; coreStart += prefixRowsPerBatch) {
        uint32_t copyRows = (coreStart + prefixRowsPerBatch > aivId_) ? aivId_ - coreStart : prefixRowsPerBatch;
        DataCopyExtParams prefixCopyParams{static_cast<uint16_t>(copyRows), rankCountRowBytes,
                                           static_cast<uint32_t>(perCoreRankCountStride_ - rankCountRowBytes), 0U, 0U};
        DataCopyPad(prefixRows,
                    allPerCoreRankCounts[static_cast<uint64_t>(coreStart) * perCoreRankCountStride_ / sizeof(int32_t)],
                    prefixCopyParams, prefixPadParams);
        SyncFunc<HardEvent::MTE2_V>();
        const uint32_t prefixShape[] = {copyRows, rankCountElements_};
        ReduceSum<int32_t, AscendC::Pattern::Reduce::RA, true>(prefixSums, prefixRows, prefixShape, false);
        Add(rankPrefix, rankPrefix, prefixSums, rankCountElements_);
        if (coreStart + copyRows < aivId_) {
            SyncFunc<HardEvent::V_MTE2>();
        }
    }
    SyncFunc<HardEvent::V_S>();

    // As in MoeEpCombine, the last AIV publishes prefix + its own row as the total count per rank.
    if (aivId_ == aivNum_ - 1U) {
        for (uint32_t rank = 0U; rank < worldSize_; ++rank) {
            rankCounts.SetValue(rank, rankPrefix.GetValue(rank) + rankCounts.GetValue(rank));
        }
        GlobalTensor<int32_t> totalRankCounts;
        totalRankCounts.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(totalRankCountWinAddr_));
        SyncFunc<HardEvent::S_MTE3>();
        DataCopy(totalRankCounts, rankCounts, rankCountElements_);
        SyncFunc<HardEvent::MTE3_S>();
    }

    // Pass 2: bucket this core's records by rank in UB, then batch-copy each bucket into its rank table.
    // Each UB bucket starts on a 32B boundary; GM entries remain tightly packed 16B records.
    uint32_t bucketBytes = ADDRESS_TABLE_BUFFER_SIZE / worldSize_ / URMA_UB_ALIGN * URMA_UB_ALIGN;
    uint32_t entriesPerRank = bucketBytes / REMOTE_ADDRESS_ENTRY_SIZE;
    uint32_t groupTokenMax = entriesPerRank < METADATA_CHUNK_TOKEN_NUM ? entriesPerRank : METADATA_CHUNK_TOKEN_NUM;
    LocalTensor<uint64_t> addressEntries = addressTableBuf_.Get<uint64_t>();
    constexpr uint32_t entryElements = REMOTE_ADDRESS_ENTRY_SIZE / sizeof(uint64_t);
    DataCopyExtParams xCopyParams = {1U, static_cast<uint32_t>(hSize_), 0U, 0U, 0U};
    GM_ADDR localWindowAddr = GetWindowAddr(rankId_);

    for (uint64_t groupStart = scanStart; groupStart < scanEnd; groupStart += groupTokenMax) {
        uint32_t groupTokens =
            static_cast<uint32_t>((groupStart + groupTokenMax > scanEnd) ? scanEnd - groupStart : groupTokenMax);
        PrefetchTokenMetaData(groupStart, groupTokens);
        Duplicate<int32_t>(rankCounts, 0, rankCountElements_);
        SyncFunc<HardEvent::V_S>();

        for (uint32_t i = 0U; i < groupTokens; ++i) {
            uint64_t sourceTokenIndex = groupStart + i;
            ReadTokenMetaDataStruct metaDataStruct;
            ReadTokenMetaDataFromLocal(metaDataStruct, i);
            // Rank range was already validated in Pass 1 over the same [scanStart, scanEnd) window.
            uint64_t tokenSlot = (static_cast<uint64_t>(metaDataStruct.curMicroBatchIds) * axisBS_ +
                                  metaDataStruct.curTokenBatchOffset) *
                                     expertNumPerToken_ +
                                 metaDataStruct.curTokenTopkOffset;
            ascendc_assert(tokenSlot < tokenSlotNum_, "tokenSlot out of range, tokenSlot=%lu, tokenSlotNum=%lu",
                           tokenSlot, tokenSlotNum_);
            uint64_t remoteDataOffset =
                winTokenInfoTableSize_ + tokenSlot * static_cast<uint64_t>(axisHS_) * sizeof(xType);
            uint64_t remoteFlagOffset = tokenSlot * sizeof(int32_t);
            uint32_t dstRank = metaDataStruct.curAttenWorkRank;
            if (dstRank == rankId_) {
                SendToLocal(localWindowAddr + remoteDataOffset, localWindowAddr + remoteFlagOffset, sourceTokenIndex,
                            xCopyParams);
                continue;
            }

            uint32_t localIndex = static_cast<uint32_t>(rankCounts.GetValue(dstRank));
            uint32_t entryOffset = (dstRank * entriesPerRank + localIndex) * entryElements;
            addressEntries.SetValue(entryOffset, tokenSlot);
            addressEntries.SetValue(entryOffset + 1U, sourceTokenIndex);
            rankCounts.SetValue(dstRank, static_cast<int32_t>(localIndex + 1U));
        }

        SyncFunc<HardEvent::S_MTE3>();
        for (uint32_t rank = 0U; rank < worldSize_; ++rank) {
            uint32_t count = static_cast<uint32_t>(rankCounts.GetValue(rank));
            if (count == 0U) {
                continue;
            }
            uint32_t writeStart = static_cast<uint32_t>(rankPrefix.GetValue(rank));
            GlobalTensor<uint64_t> rankAddressTable;
            rankAddressTable.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(GetRankAddressTableAddr(rank)));
            DataCopyExtParams entryCopyParams{1U, count * REMOTE_ADDRESS_ENTRY_SIZE, 0U, 0U, 0U};
            DataCopyPad(rankAddressTable[static_cast<uint64_t>(writeStart) * entryElements],
                        addressEntries[rank * entriesPerRank * entryElements], entryCopyParams);
            rankPrefix.SetValue(rank, static_cast<int32_t>(writeStart + count));
        }
        SyncFunc<HardEvent::MTE3_S>();
    }
    SyncAll<true>();
}

template <TemplateFFNToAttentionUrmaTypeClass>
__aicore__ inline uint32_t FFNToAttentionUrma<TemplateFFNToAttentionUrmaTypeFunc>::GetAddressTableCount(
    uint32_t targetRank)
{
    // Relay-phase only: reads the count row prefetched into rankInfoBuf_'s first segment
    // before the relay loop in Process().
    return rankInfoBuf_.Get<int32_t>().GetValue(targetRank);
}

template <TemplateFFNToAttentionUrmaTypeClass>
template <typename BatchHandleType>
__aicore__ inline void FFNToAttentionUrma<TemplateFFNToAttentionUrmaTypeFunc>::CommitAndDrainChannel(
    BatchHandleType& batchHandle, HcommChannelHandle channel, uint32_t& preparedWqeCount, uint32_t& sqWriteCount)
{
    if (preparedWqeCount != 0U) {
        (void)hcomm_.BatchCommit(batchHandle);
        preparedWqeCount = 0U;
    }
    int32_t ret = hcomm_.Drain(channel);
    ascendc_assert(ret == 0, "Urma drain failed, ret=%d, rankId=%u, channel=%lu", ret, rankId_, channel);
    sqWriteCount = 0U;
}

template <TemplateFFNToAttentionUrmaTypeClass>
template <typename BatchHandleType>
__aicore__ inline void FFNToAttentionUrma<TemplateFFNToAttentionUrmaTypeFunc>::ClearRemoteFlags(
    BatchHandleType& batchHandle, GM_ADDR remoteWindowAddr, HcommChannelHandle channel,
    GlobalTensor<uint64_t> rankAddressTable, uint32_t entryCount, uint32_t channelIdx)
{
    if (entryCount == 0U) {
        return;
    }
    LocalTensor<uint64_t> entriesLocal = metadataBuf_.Get<uint64_t>();
    constexpr uint32_t entryElements = REMOTE_ADDRESS_ENTRY_SIZE / sizeof(uint64_t);
    constexpr uint32_t entryChunkMax = METADATA_CHUNK_BUFFER_SIZE / REMOTE_ADDRESS_ENTRY_SIZE;
    const DataCopyPadExtParams<uint64_t> padParams{false, 0U, 0U, 0UL};
    uint32_t preparedWqeCount = 0U;
    uint32_t sqWriteCount = 0U;
    // This channel owns table entries whose index is congruent to channelIdx (mod channelsPerRank_),
    // the same split as the data/flag phases below.
    uint32_t entryCountChannel =
        (entryCount > channelIdx) ? (entryCount - 1U - channelIdx) / channelsPerRank_ + 1U : 0U;
    const uint32_t strideBytes = (channelsPerRank_ - 1U) * REMOTE_ADDRESS_ENTRY_SIZE;

    // Per-round flag clear: zero the remote flag slots this channel is about to use BEFORE the
    // data/flag phases, so the receiver sees a fresh 0->1 edge on every slot every round. All
    // clears use the ordered config so they land after the previous round's data/flag WQEs on
    // this channel and can never stomp an unconsumed round. No extra Drain is introduced here:
    // the existing full-batch commit / SQ-limit handling is kept as-is.
    for (uint32_t chunkStart = 0U; chunkStart < entryCountChannel; chunkStart += entryChunkMax) {
        uint32_t curEntries =
            (chunkStart + entryChunkMax > entryCountChannel) ? entryCountChannel - chunkStart : entryChunkMax;
        DataCopyExtParams copyParams{static_cast<uint16_t>(curEntries), REMOTE_ADDRESS_ENTRY_SIZE, strideBytes, 0U, 0U};
        SyncFunc<HardEvent::S_MTE2>();
        DataCopyPad<uint64_t, PaddingMode::Compact>(
            entriesLocal,
            rankAddressTable[static_cast<uint64_t>(channelIdx + chunkStart * channelsPerRank_) * entryElements],
            copyParams, padParams);
        SyncFunc<HardEvent::MTE2_S>();
        for (uint32_t i = 0U; i < curEntries; ++i) {
            uint32_t entryOffset = i * entryElements;
            uint64_t remoteFlagOffset = entriesLocal.GetValue(entryOffset) * sizeof(int32_t);
            bool enableCqe = sqWriteCount + 1U >= HCOMM_SQ_MAX_PENDING;
            if (enableCqe) {
                (void)hcomm_.WriteNbi<remoteOrderedFlagCqeWqeConfig>(batchHandle, remoteWindowAddr + remoteFlagOffset,
                                                                     zeroFlagAddr_,
                                                                     static_cast<uint32_t>(sizeof(int32_t)));
            } else {
                (void)hcomm_.WriteNbi<remoteOrderedFlagWqeConfig>(batchHandle, remoteWindowAddr + remoteFlagOffset,
                                                                  zeroFlagAddr_,
                                                                  static_cast<uint32_t>(sizeof(int32_t)));
            }
            ++preparedWqeCount;
            ++sqWriteCount;
            if (enableCqe) {
                CommitAndDrainChannel(batchHandle, channel, preparedWqeCount, sqWriteCount);
            } else if (preparedWqeCount == URMA_BATCH_WQE_CAPACITY) {
                (void)hcomm_.BatchCommit(batchHandle);
                preparedWqeCount = 0U;
            }
        }
    }
    if (preparedWqeCount != 0U) {
        (void)hcomm_.BatchCommit(batchHandle);
    }
}

template <TemplateFFNToAttentionUrmaTypeClass>
__aicore__ inline void FFNToAttentionUrma<TemplateFFNToAttentionUrmaTypeFunc>::SendAddressTable(uint32_t targetRank,
                                                                                                uint32_t entryCount,
                                                                                                uint32_t channelIdx)
{
    if (entryCount == 0U) {
        return;
    }

    GM_ADDR remoteWindowAddr = GetWindowAddr(targetRank);
    HcommChannelHandle channel = GetUrmaCommHandle(targetRank, channelIdx);
    auto batchHandle = hcomm_.MakeBatchHandle(channel, hcommBatchTensor_, URMA_BATCH_BUFFER_SIZE, remoteWindowAddr);
    GlobalTensor<uint64_t> rankAddressTable;
    rankAddressTable.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(GetRankAddressTableAddr(targetRank)));
    LocalTensor<uint64_t> entriesLocal = metadataBuf_.Get<uint64_t>();
    constexpr uint32_t entryElements = REMOTE_ADDRESS_ENTRY_SIZE / sizeof(uint64_t);
    constexpr uint32_t entryChunkMax = METADATA_CHUNK_BUFFER_SIZE / REMOTE_ADDRESS_ENTRY_SIZE;
    const DataCopyPadExtParams<uint64_t> padParams{false, 0U, 0U, 0UL};
    uint32_t preparedWqeCount = 0U;
    uint32_t sqWriteCount = 0U;
    // This channel owns table entries whose index is congruent to channelIdx (mod channelsPerRank_).
    uint32_t entryCountChannel =
        (entryCount > channelIdx) ? (entryCount - 1U - channelIdx) / channelsPerRank_ + 1U : 0U;
    const uint32_t strideBytes = (channelsPerRank_ - 1U) * REMOTE_ADDRESS_ENTRY_SIZE;

    // Per-round flag clear before this round's data/flag phases (see ClearRemoteFlags): the
    // receiver observes a fresh 0->1 edge per round on every flag slot.
    ClearRemoteFlags(batchHandle, remoteWindowAddr, channel, rankAddressTable, entryCount, channelIdx);

    // Data phase: an entry whose index within the rank is congruent to channelIdx is sent on this
    // channel. Strided MTE2 reads fetch only this channel's entries (every channelsPerRank_-th one,
    // starting at channelIdx) instead of the whole table.
    for (uint32_t chunkStart = 0U; chunkStart < entryCountChannel; chunkStart += entryChunkMax) {
        uint32_t curEntries =
            (chunkStart + entryChunkMax > entryCountChannel) ? entryCountChannel - chunkStart : entryChunkMax;
        DataCopyExtParams copyParams{static_cast<uint16_t>(curEntries), REMOTE_ADDRESS_ENTRY_SIZE, strideBytes, 0U, 0U};
        SyncFunc<HardEvent::S_MTE2>();
        DataCopyPad<uint64_t, PaddingMode::Compact>(
            entriesLocal,
            rankAddressTable[static_cast<uint64_t>(channelIdx + chunkStart * channelsPerRank_) * entryElements],
            copyParams, padParams);
        SyncFunc<HardEvent::MTE2_S>();
        for (uint32_t i = 0U; i < curEntries; ++i) {
            uint32_t entryOffset = i * entryElements;
            RemoteAddressEntry addressEntry{entriesLocal.GetValue(entryOffset),
                                            entriesLocal.GetValue(entryOffset + 1U)};
            bool enableCqe = sqWriteCount + 1U >= HCOMM_SQ_MAX_PENDING;
            AppendRemoteDataWrite(batchHandle, remoteWindowAddr, addressEntry, enableCqe);
            ++preparedWqeCount;
            ++sqWriteCount;
            if (enableCqe) {
                CommitAndDrainChannel(batchHandle, channel, preparedWqeCount, sqWriteCount);
            } else if (preparedWqeCount == URMA_BATCH_WQE_CAPACITY) {
                (void)hcomm_.BatchCommit(batchHandle);
                preparedWqeCount = 0U;
            }
        }
    }

    // Flag phase: the same entries on the same channel, appended after the data WQEs. Every flag
    // write uses the ordered config (see AppendRemoteFlagWrite) so each per-token flag lands after
    // this round's clear/data WQEs; the receiver sees a fresh 0->1 edge per round on every slot.
    for (uint32_t chunkStart = 0U; chunkStart < entryCountChannel; chunkStart += entryChunkMax) {
        uint32_t curEntries =
            (chunkStart + entryChunkMax > entryCountChannel) ? entryCountChannel - chunkStart : entryChunkMax;
        DataCopyExtParams copyParams{static_cast<uint16_t>(curEntries), REMOTE_ADDRESS_ENTRY_SIZE, strideBytes, 0U, 0U};
        SyncFunc<HardEvent::S_MTE2>();
        DataCopyPad<uint64_t, PaddingMode::Compact>(
            entriesLocal,
            rankAddressTable[static_cast<uint64_t>(channelIdx + chunkStart * channelsPerRank_) * entryElements],
            copyParams, padParams);
        SyncFunc<HardEvent::MTE2_S>();
        for (uint32_t i = 0U; i < curEntries; ++i) {
            uint32_t entryOffset = i * entryElements;
            RemoteAddressEntry addressEntry{entriesLocal.GetValue(entryOffset),
                                            entriesLocal.GetValue(entryOffset + 1U)};
            bool enableCqe = sqWriteCount + 1U >= HCOMM_SQ_MAX_PENDING;
            AppendRemoteFlagWrite(batchHandle, remoteWindowAddr, addressEntry, enableCqe);
            ++preparedWqeCount;
            ++sqWriteCount;
            if (enableCqe) {
                CommitAndDrainChannel(batchHandle, channel, preparedWqeCount, sqWriteCount);
            } else if (preparedWqeCount == URMA_BATCH_WQE_CAPACITY) {
                (void)hcomm_.BatchCommit(batchHandle);
                preparedWqeCount = 0U;
            }
        }
    }

    if (preparedWqeCount != 0U) {
        (void)hcomm_.BatchCommit(batchHandle);
    }
}

template <TemplateFFNToAttentionUrmaTypeClass>
__aicore__ inline void FFNToAttentionUrma<TemplateFFNToAttentionUrmaTypeFunc>::Process()
{
    if ASCEND_IS_AIV {
        HcommInit();
        InitLocalFlag();
        BuildAddressTable();

        // Prefetch the total rank-count row (written by the last AIV inside BuildAddressTable,
        // published before its trailing SyncAll) into the now-idle first segment of rankInfoBuf_,
        // so the relay loop resolves counts via UB reads instead of per-slot 4B GM copies.
        LocalTensor<int32_t> rankCountRow = rankInfoBuf_.Get<int32_t>();
        GlobalTensor<int32_t> totalRankCounts;
        totalRankCounts.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(totalRankCountWinAddr_));
        SyncFunc<HardEvent::S_MTE2>();
        DataCopyPad(rankCountRow, totalRankCounts,
                    DataCopyExtParams{1U, static_cast<uint32_t>(rankCountElements_ * sizeof(int32_t)), 0U, 0U, 0U},
                    DataCopyPadExtParams<int32_t>{false, 0U, 0U, 0});
        SyncFunc<HardEvent::MTE2_S>();

        // Multi-channel relay phase: every AIV relays the (dstRank, channel) slots assigned to it
        // (slot = dstRank * channelsPerRank_ + channelIdx, slot % aivNum_ == aivId_). An entry's
        // index within the rank decides its channel, and its data WQE and flag WQE share that
        // channel, so per-token flag ordering is preserved while multiple AIVs and channels
        // submit in parallel.
        for (uint32_t dstRank = 0U; dstRank < worldSize_; ++dstRank) {
            if (dstRank == rankId_) {
                continue;
            }
            uint32_t entryCount = GetAddressTableCount(dstRank);
            for (uint32_t channelIdx = 0U; channelIdx < channelsPerRank_; ++channelIdx) {
                uint32_t slot = dstRank * channelsPerRank_ + channelIdx;
                if (slot % aivNum_ != aivId_) {
                    continue;
                }
                SendAddressTable(dstRank, entryCount, channelIdx);
            }
        }

        // Keep all AIVs alive until every AIV finishes submitting remote WQEs.
        SyncAll<true>();
    }
}

#endif

} // namespace FFNToAttentionImpl
#endif // FFN_TO_ATTENTION_URMA_H
