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

#if (ASC_DEVKIT_VERSION_NUM >= 90200000) && (HCOMM_VERSION_NUM >= 90200000)
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

#include "utils/debug/asc_printf.h"

namespace FFNToAttentionImpl {

#if defined(ENABLE_FFN_TO_ATTENTION_V2_KERNEL)

constexpr uint32_t TOKEN_DTYPE_BYTES = 2U;

constexpr uint8_t URMA_BUFFER_NUM = 2U;
constexpr uint32_t URMA_UB_ALIGN = 32U;
constexpr uint64_t URMA_WIN_ADDR_ALIGN = 512UL;

constexpr uint32_t URMA_HCOMM_INIT_SIZE = 512U;
constexpr uint32_t URMA_FLAG_SLOT_SIZE = 32U;
constexpr int32_t URMA_FLAG_VALUE = 1;

constexpr uint32_t HCOMM_WQEBB_BYTES = 64U;
constexpr uint32_t HCOMM_SQE_BYTES = 48U;
constexpr uint32_t HCOMM_SGE_BYTES = 16U;
constexpr uint32_t SGE_PER_FLAG_WQE = 6U;
constexpr uint32_t URMA_BATCH_WQEBB_CAPACITY = 256U;
constexpr uint32_t URMA_BATCH_BUFFER_SIZE = URMA_BATCH_WQEBB_CAPACITY * HCOMM_WQEBB_BYTES;
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
static constexpr AscendC::UrmaWqeEntry remoteOrderedFlagWqeConfig = {
    .odr = 6, .fence = 1, .se = 0, .cqe = 0, .inlineEn = 0};
#define TemplateFFNToAttentionUrmaTypeClass typename xType, bool isInputRankTable
#define TemplateFFNToAttentionUrmaTypeFunc xType, isInputRankTable

using namespace AscendC;

using HcommChannelHandle = AscendC::ChannelHandle;

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
    __aicore__ inline void SelectPayloadSlot(uint32_t microBatchId);
    __aicore__ inline void InitLocalFlag();
    __aicore__ inline void CopyXToPayload();
    template <typename BatchHandleType>
    __aicore__ inline void AppendRemoteDataWrite(BatchHandleType& batchHandle, GM_ADDR remoteDataAddr,
                                                 const BufDesc* sourceDescs, uint32_t sgeCount);
    template <typename BatchHandleType>
    __aicore__ inline void AppendRemoteFlagWrite(BatchHandleType& batchHandle, GM_ADDR remoteFlagAddr,
                                                 const BufDesc* sourceDescs, uint32_t sgeCount);
    __aicore__ inline void SortAddressBucketByTokenSlot(LocalTensor<uint64_t>& addressEntries, uint32_t entryStart,
                                                        uint32_t entryCount);
    __aicore__ inline void BuildAddressTable(uint32_t microBatchId);
    __aicore__ inline uint32_t GetAddressTableCount(uint32_t targetRank);
    __aicore__ inline void SendAddressTable(uint32_t targetRank, uint32_t entryCount, uint32_t channelIdx);
    __aicore__ inline GM_ADDR GetRankAddressTableAddr(uint32_t targetRank)
    {
        return addressTablePayloadAddr_ + static_cast<uint64_t>(targetRank) * addressTableStridePerRank_;
    }

    TPipe* tpipe_{nullptr};
    Hcomm<COMM_PROTOCOL_UBC_CTP> hcomm_;
    __gm__ Mc2Aclnn::AttentionFFNContext* mc2Context_{nullptr};

    GM_ADDR localFlagAddr_{nullptr};
    GM_ADDR tokenBaseAddr_{nullptr};
    GM_ADDR activeMaskWorkspaceAddr_{nullptr};
    GM_ADDR addressTablePayloadAddr_{nullptr};
    GM_ADDR totalRankCountPayloadAddr_{nullptr};
    GM_ADDR perCoreRankCountPayloadAddr_{nullptr};

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
    uint64_t remoteTokenRowBytes_{0};
    uint64_t payloadTokenRowStride_{0};
    uint64_t payloadRequiredSlotBytes_{0};
    uint64_t payloadAddressTableOffset_{0};
    uint64_t payloadCountRowsOffset_{0};
    uint64_t payloadFlagOffset_{0};
    uint32_t activeMicroBatchMask_{0};
    uint64_t winTokenInfoTableSize_{0};
    uint64_t addressTableStridePerRank_{0};
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
    channelsPerRank_ = mc2Context_->channelsPerRank;
    if (worldSize_ == 0U || channelsPerRank_ == 0U || channelsPerRank_ > Mc2Aclnn::HCCL_MAX_RANK_SIZE / worldSize_) {
        channelsPerRank_ = 1U;
    }

    activeMaskWorkspaceAddr_ = workspaceGM;
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
    actualTokenNum_ = static_cast<uint64_t>(actualTokenNumValue);
    hSize_ = static_cast<uint64_t>(axisH_) * sizeof(xType);
    remoteTokenRowBytes_ = static_cast<uint64_t>(axisHS_) * sizeof(xType);
    payloadTokenRowStride_ = tilingData->ffnToAttentionV2Info.payloadTokenRowStride;
    payloadRequiredSlotBytes_ = tilingData->ffnToAttentionV2Info.payloadRequiredSlotBytes;
    payloadAddressTableOffset_ = tilingData->ffnToAttentionV2Info.payloadAddressTableOffset;
    payloadCountRowsOffset_ = tilingData->ffnToAttentionV2Info.payloadCountRowsOffset;
    payloadFlagOffset_ = tilingData->ffnToAttentionV2Info.payloadFlagOffset;
    batchSizeSendCnt_ = static_cast<uint64_t>(axisBS_) * expertNumPerToken_;
    const uint64_t tokenSlotNum = static_cast<uint64_t>(microBatchNum_) * batchSizeSendCnt_;
    tokenSlotNum_ = tokenSlotNum;
    winTokenInfoTableSize_ = Ceil(tokenSlotNum * sizeof(int32_t), URMA_WIN_ADDR_ALIGN) * URMA_WIN_ADDR_ALIGN;

    addressTableStridePerRank_ =
        Ceil(maxTokenNum_ * REMOTE_ADDRESS_ENTRY_SIZE, URMA_WIN_ADDR_ALIGN) * URMA_WIN_ADDR_ALIGN;

    rankCountElements_ = static_cast<uint32_t>(
        Ceil(static_cast<uint64_t>(worldSize_) * sizeof(int32_t), URMA_UB_ALIGN) * URMA_UB_ALIGN / sizeof(int32_t));
    perCoreRankCountStride_ = tilingData->ffnToAttentionV2Info.payloadCountRowStride;

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
__aicore__ inline void FFNToAttentionUrma<TemplateFFNToAttentionUrmaTypeFunc>::SelectPayloadSlot(uint32_t microBatchId)
{
    tokenBaseAddr_ = reinterpret_cast<GM_ADDR>(mc2Context_->payloadBufferAddr) +
                     static_cast<uint64_t>(microBatchId) * mc2Context_->payloadSlotStride;
    addressTablePayloadAddr_ = tokenBaseAddr_ + payloadAddressTableOffset_;
    totalRankCountPayloadAddr_ = tokenBaseAddr_ + payloadCountRowsOffset_;
    perCoreRankCountPayloadAddr_ = totalRankCountPayloadAddr_ + perCoreRankCountStride_;
    localFlagAddr_ = tokenBaseAddr_ + payloadFlagOffset_ + static_cast<uint64_t>(aivId_) * URMA_FLAG_SLOT_SIZE;
}

template <TemplateFFNToAttentionUrmaTypeClass>
__aicore__ inline void FFNToAttentionUrma<TemplateFFNToAttentionUrmaTypeFunc>::InitLocalFlag()
{
    constexpr uint32_t flagSlotElementNum = URMA_FLAG_SLOT_SIZE / sizeof(int32_t);
    Duplicate<int32_t>(statusTensor_, URMA_FLAG_VALUE, flagSlotElementNum);
    SyncFunc<HardEvent::V_MTE3>();
    SyncFunc<HardEvent::S_MTE3>();
    GlobalTensor<int32_t> localFlagGMTensor;
    localFlagGMTensor.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(localFlagAddr_));
    DataCopy(localFlagGMTensor, statusTensor_, flagSlotElementNum);
}

template <TemplateFFNToAttentionUrmaTypeClass>
__aicore__ inline void FFNToAttentionUrma<TemplateFFNToAttentionUrmaTypeFunc>::CopyXToPayload()
{
    const uint64_t tokenStride = payloadTokenRowStride_;
    const uint64_t tokenCapacityBytes = maxTokenNum_ * tokenStride;
    if (unlikely(tokenCapacityBytes > payloadRequiredSlotBytes_)) {
        AscendC::printf("[FfnToAttnUrma][CopyXToPayload] payload token capacity exceeds required slot size, "
                        "tokenCapacityBytes=%lu, payloadRequiredSlotBytes=%lu\n",
                        tokenCapacityBytes, payloadRequiredSlotBytes_);
        AscendC::Trap();
    }
    if (unlikely(payloadRequiredSlotBytes_ > mc2Context_->payloadSlotStride)) {
        AscendC::printf("[FfnToAttnUrma][CopyXToPayload] payload slot is too small for x, required=%lu, stride=%lu\n",
                        payloadRequiredSlotBytes_, mc2Context_->payloadSlotStride);
        AscendC::Trap();
    }
    if (unlikely(mc2Context_->payloadSlotNum > mc2Context_->payloadBufferSize / mc2Context_->payloadSlotStride)) {
        AscendC::printf("[FfnToAttnUrma][CopyXToPayload] payload buffer cannot hold all slots, payloadSlotNum=%u, "
                        "payloadBufferSize=%lu, payloadSlotStride=%lu\n",
                        mc2Context_->payloadSlotNum, mc2Context_->payloadBufferSize, mc2Context_->payloadSlotStride);
        AscendC::Trap();
    }

    const DataCopyExtParams copyParams{1U, static_cast<uint32_t>(hSize_), 0U, 0U, 0U};
    const DataCopyPadExtParams<xType> padParams{false, 0U, 0U, 0U};
    const uint64_t tokensPerCore = actualTokenNum_ / aivNum_;
    const uint64_t remainder = actualTokenNum_ % aivNum_;
    const uint64_t scanStart =
        static_cast<uint64_t>(aivId_) * tokensPerCore + ((aivId_ < remainder) ? aivId_ : remainder);
    const uint64_t scanEnd = scanStart + tokensPerCore + ((aivId_ < remainder) ? 1U : 0U);
    uint32_t localMicroBatchMask = 0U;
    const uint32_t payloadSlotNum = mc2Context_->payloadSlotNum;
    const uint32_t microBatchIdLimit = microBatchNum_ < payloadSlotNum ? microBatchNum_ : payloadSlotNum;
    LocalTensor<int32_t> microBatchLocal = metadataBuf_.Get<int32_t>();
    const DataCopyPadExtParams<int32_t> microBatchPadParams{false, 0U, 0U, 0U};
    for (uint64_t tokenStart = scanStart; tokenStart < scanEnd; tokenStart += METADATA_CHUNK_TOKEN_NUM) {
        const uint32_t tokenNum = static_cast<uint32_t>(
            (tokenStart + METADATA_CHUNK_TOKEN_NUM > scanEnd) ? scanEnd - tokenStart : METADATA_CHUNK_TOKEN_NUM);
        SyncFunc<HardEvent::S_MTE2>();
        DataCopyPad(microBatchLocal, microBatchIdsGMTensor_[tokenStart],
                    DataCopyExtParams{1U, tokenNum * sizeof(int32_t), 0U, 0U, 0U}, microBatchPadParams);
        SyncFunc<HardEvent::MTE2_S>();
        for (uint32_t i = 0U; i < tokenNum; ++i) {
            const int32_t microBatchId = microBatchLocal.GetValue(i);
            const uint32_t slotIdx = static_cast<uint32_t>(microBatchId);
            // 上限不超过 3，负 ID 转为 uint32_t 后也会被此比较拒绝。
            if (unlikely(slotIdx >= microBatchIdLimit)) {
                AscendC::printf("[FfnToAttnUrma][CopyXToPayload] micro batch id exceeds context payload slot count, "
                                "microBatchId=%d, limit=%u\n",
                                microBatchId, microBatchIdLimit);
                AscendC::Trap();
            }
            localMicroBatchMask |= 1U << slotIdx;
            const uint64_t tokenIdx = tokenStart + i;
            GM_ADDR slotBase = reinterpret_cast<GM_ADDR>(mc2Context_->payloadBufferAddr) +
                               static_cast<uint64_t>(slotIdx) * mc2Context_->payloadSlotStride;
            GlobalTensor<xType> tokenPayloadGM;
            tokenPayloadGM.SetGlobalBuffer(reinterpret_cast<__gm__ xType*>(slotBase));
            LocalTensor<xType> tokenLocal = xQueue_.AllocTensor<xType>();
            DataCopyPad(tokenLocal, xGMTensor_[tokenIdx * axisH_], copyParams, padParams);
            xQueue_.EnQue(tokenLocal);
            tokenLocal = xQueue_.DeQue<xType>();
            DataCopyPad(tokenPayloadGM[tokenIdx * tokenStride / sizeof(xType)], tokenLocal, copyParams);
            xQueue_.FreeTensor(tokenLocal);
        }
    }
    SyncFunc<HardEvent::MTE3_S>();

    // Publish one 32-byte mask row per AIV in this launch's private workspace.
    Duplicate<int32_t>(statusTensor_, 0, URMA_FLAG_SLOT_SIZE / sizeof(int32_t));
    SyncFunc<HardEvent::V_S>();
    statusTensor_.SetValue(0U, static_cast<int32_t>(localMicroBatchMask));
    GlobalTensor<int32_t> activeMaskGM;
    activeMaskGM.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(activeMaskWorkspaceAddr_));
    SyncFunc<HardEvent::V_MTE3>();
    SyncFunc<HardEvent::S_MTE3>();
    DataCopy(activeMaskGM[static_cast<uint64_t>(aivId_) * URMA_FLAG_SLOT_SIZE / sizeof(int32_t)], statusTensor_,
             URMA_FLAG_SLOT_SIZE / sizeof(int32_t));
    SyncFunc<HardEvent::MTE3_S>();
    SyncAll<true>();

    LocalTensor<int32_t> maskRows = metadataBuf_.Get<int32_t>();
    constexpr uint32_t maskRowsPerChunk = METADATA_CHUNK_BUFFER_SIZE / URMA_FLAG_SLOT_SIZE;
    for (uint32_t coreStart = 0U; coreStart < aivNum_; coreStart += maskRowsPerChunk) {
        const uint32_t rowNum = (coreStart + maskRowsPerChunk > aivNum_) ? aivNum_ - coreStart : maskRowsPerChunk;
        SyncFunc<HardEvent::S_MTE2>();
        DataCopyPad(maskRows, activeMaskGM[static_cast<uint64_t>(coreStart) * URMA_FLAG_SLOT_SIZE / sizeof(int32_t)],
                    DataCopyExtParams{1U, rowNum * URMA_FLAG_SLOT_SIZE, 0U, 0U, 0U},
                    DataCopyPadExtParams<int32_t>{false, 0U, 0U, 0U});
        SyncFunc<HardEvent::MTE2_S>();
        for (uint32_t row = 0U; row < rowNum; ++row) {
            activeMicroBatchMask_ |= static_cast<uint32_t>(
                maskRows.GetValue(static_cast<uint64_t>(row) * URMA_FLAG_SLOT_SIZE / sizeof(int32_t)));
        }
    }
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
template <typename BatchHandleType>
__aicore__ inline void FFNToAttentionUrma<TemplateFFNToAttentionUrmaTypeFunc>::AppendRemoteDataWrite(
    BatchHandleType& batchHandle, GM_ADDR remoteDataAddr, const BufDesc* sourceDescs, uint32_t sgeCount)
{
    (void)hcomm_.WriteNbi<remoteWqeConfig>(batchHandle, remoteDataAddr, sourceDescs, sgeCount);
}

template <TemplateFFNToAttentionUrmaTypeClass>
template <typename BatchHandleType>
__aicore__ inline void FFNToAttentionUrma<TemplateFFNToAttentionUrmaTypeFunc>::AppendRemoteFlagWrite(
    BatchHandleType& batchHandle, GM_ADDR remoteFlagAddr, const BufDesc* sourceDescs, uint32_t sgeCount)
{
    (void)hcomm_.WriteNbi<remoteOrderedFlagWqeConfig>(batchHandle, remoteFlagAddr, sourceDescs, sgeCount);
}

template <TemplateFFNToAttentionUrmaTypeClass>
__aicore__ inline void FFNToAttentionUrma<TemplateFFNToAttentionUrmaTypeFunc>::SortAddressBucketByTokenSlot(
    LocalTensor<uint64_t>& addressEntries, uint32_t entryStart, uint32_t entryCount)
{
    constexpr uint32_t entryElements = REMOTE_ADDRESS_ENTRY_SIZE / sizeof(uint64_t);
    // Each core sorts its current UB bucket by tokenSlot,
    // moving tokenSlot and sourceTokenIndex together as a pair.
    for (uint32_t gap = entryCount >> 1U; gap > 0U; gap >>= 1U) {
        for (uint32_t i = gap; i < entryCount; ++i) {
            const uint32_t keyOffset = (entryStart + i) * entryElements;
            const uint64_t keyTokenSlot = addressEntries.GetValue(keyOffset);
            const uint64_t keySourceTokenIndex = addressEntries.GetValue(keyOffset + 1U);
            uint32_t j = i;
            while (j >= gap) {
                const uint32_t prevOffset = (entryStart + j - gap) * entryElements;
                const uint64_t prevTokenSlot = addressEntries.GetValue(prevOffset);
                if (prevTokenSlot <= keyTokenSlot) {
                    break;
                }
                const uint32_t writeOffset = (entryStart + j) * entryElements;
                addressEntries.SetValue(writeOffset, prevTokenSlot);
                addressEntries.SetValue(writeOffset + 1U, addressEntries.GetValue(prevOffset + 1U));
                j -= gap;
            }
            const uint32_t writeOffset = (entryStart + j) * entryElements;
            addressEntries.SetValue(writeOffset, keyTokenSlot);
            addressEntries.SetValue(writeOffset + 1U, keySourceTokenIndex);
        }
    }
}

template <TemplateFFNToAttentionUrmaTypeClass>
__aicore__ inline void FFNToAttentionUrma<TemplateFFNToAttentionUrmaTypeFunc>::BuildAddressTable(uint32_t microBatchId)
{
    if constexpr (isInputRankTable) {
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
            if (metaDataStruct.curMicroBatchIds != microBatchId) {
                continue;
            }
            uint32_t dstRank = metaDataStruct.curAttenWorkRank;
            rankCounts.SetValue(dstRank, rankCounts.GetValue(dstRank) + 1);
        }
    }

    GlobalTensor<int32_t> perCoreRankCounts;
    GM_ADDR countRowAddr = perCoreRankCountPayloadAddr_ + static_cast<uint64_t>(aivId_) * perCoreRankCountStride_;
    perCoreRankCounts.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(countRowAddr));
    SyncFunc<HardEvent::S_MTE3>();
    DataCopy(perCoreRankCounts, rankCounts, rankCountElements_);
    SyncFunc<HardEvent::MTE3_S>();
    SyncAll<true>();

    uint32_t rankCountRowBytes = rankCountElements_ * sizeof(int32_t);
    uint32_t prefixRowsPerBatch = ADDRESS_TABLE_BUFFER_SIZE / rankCountRowBytes;
    LocalTensor<int32_t> prefixRows = addressTableBuf_.Get<int32_t>();
    GlobalTensor<int32_t> allPerCoreRankCounts;
    allPerCoreRankCounts.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(perCoreRankCountPayloadAddr_));
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

    if (aivId_ == aivNum_ - 1U) {
        for (uint32_t rank = 0U; rank < worldSize_; ++rank) {
            rankCounts.SetValue(rank, rankPrefix.GetValue(rank) + rankCounts.GetValue(rank));
        }
        GlobalTensor<int32_t> totalRankCounts;
        totalRankCounts.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(totalRankCountPayloadAddr_));
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
            if (metaDataStruct.curMicroBatchIds != microBatchId) {
                continue;
            }
            uint64_t tokenSlot = (static_cast<uint64_t>(metaDataStruct.curMicroBatchIds) * axisBS_ +
                                  metaDataStruct.curTokenBatchOffset) *
                                     expertNumPerToken_ +
                                 metaDataStruct.curTokenTopkOffset;
            uint32_t dstRank = metaDataStruct.curAttenWorkRank;
            uint32_t localIndex = static_cast<uint32_t>(rankCounts.GetValue(dstRank));
            uint32_t entryOffset = (dstRank * entriesPerRank + localIndex) * entryElements;
            addressEntries.SetValue(entryOffset, tokenSlot);
            addressEntries.SetValue(entryOffset + 1U, sourceTokenIndex);
            rankCounts.SetValue(dstRank, static_cast<int32_t>(localIndex + 1U));
        }

        for (uint32_t rank = 0U; rank < worldSize_; ++rank) {
            const uint32_t count = static_cast<uint32_t>(rankCounts.GetValue(rank));
            if (count != 0U) {
                SortAddressBucketByTokenSlot(addressEntries, rank * entriesPerRank, count);
            }
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
    return rankInfoBuf_.Get<int32_t>().GetValue(targetRank);
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
    uint32_t preparedWqebbCount = 0U;
    const uint32_t blockStart =
        static_cast<uint32_t>(static_cast<uint64_t>(channelIdx) * entryCount / channelsPerRank_);
    const uint32_t blockEnd =
        static_cast<uint32_t>(static_cast<uint64_t>(channelIdx + 1U) * entryCount / channelsPerRank_);
    if (blockStart >= blockEnd) {
        return;
    }
    const uint32_t entryCountChannel = blockEnd - blockStart;
    BufDesc dataSourceDesc = {};
    dataSourceDesc.len = static_cast<uint32_t>(hSize_);
    constexpr uint32_t dataWqebbCount =
        (HCOMM_SQE_BYTES + HCOMM_SGE_BYTES + HCOMM_WQEBB_BYTES - 1U) / HCOMM_WQEBB_BYTES;

    // Data phase: each channel owns one contiguous block of the rank table so consecutive remote
    // token slots stay on the same channel. Each token uses one WQE with one source SGE to send
    // only H * sizeof(xType) bytes. Source rows use the aligned HS stride, while remote slots use
    // HS * sizeof(xType) bytes as their stride; the reserved tail of each remote slot is not written.
    for (uint32_t chunkStart = 0U; chunkStart < entryCountChannel; chunkStart += entryChunkMax) {
        uint32_t curEntries =
            (chunkStart + entryChunkMax > entryCountChannel) ? entryCountChannel - chunkStart : entryChunkMax;
        DataCopyExtParams copyParams{1U, curEntries * REMOTE_ADDRESS_ENTRY_SIZE, 0U, 0U, 0U};
        SyncFunc<HardEvent::S_MTE2>();
        DataCopyPad(entriesLocal, rankAddressTable[static_cast<uint64_t>(blockStart + chunkStart) * entryElements],
                    copyParams, padParams);
        SyncFunc<HardEvent::MTE2_S>();
        for (uint32_t i = 0U; i < curEntries; ++i) {
            const uint32_t entryOffset = i * entryElements;
            const uint64_t tokenSlot = entriesLocal.GetValue(entryOffset);
            const uint64_t sourceTokenIndex = entriesLocal.GetValue(entryOffset + 1U);
            if (preparedWqebbCount + dataWqebbCount > URMA_BATCH_WQEBB_CAPACITY) {
                (void)hcomm_.BatchCommit(batchHandle);
                preparedWqebbCount = 0U;
            }
            dataSourceDesc.addr = tokenBaseAddr_ + sourceTokenIndex * payloadTokenRowStride_;
            GM_ADDR remoteDataAddr = remoteWindowAddr + winTokenInfoTableSize_ + tokenSlot * remoteTokenRowBytes_;
            AppendRemoteDataWrite(batchHandle, remoteDataAddr, &dataSourceDesc, 1U);
            preparedWqebbCount += dataWqebbCount;
        }
    }

    // Flag phase: group consecutive slots with up to SGE_PER_FLAG_WQE SGEs. Each SGE reads the
    // same local flag value and writes one int32_t in the contiguous remote flag table.
    // Ordered flag WQEs follow all data WQEs on this channel.
    BufDesc sourceDescs[SGE_PER_FLAG_WQE] = {};
    for (uint32_t i = 0U; i < SGE_PER_FLAG_WQE; ++i) {
        sourceDescs[i].addr = localFlagAddr_;
        sourceDescs[i].len = static_cast<uint32_t>(sizeof(int32_t));
    }
    for (uint32_t chunkStart = 0U; chunkStart < entryCountChannel; chunkStart += entryChunkMax) {
        uint32_t curEntries =
            (chunkStart + entryChunkMax > entryCountChannel) ? entryCountChannel - chunkStart : entryChunkMax;
        DataCopyExtParams copyParams{1U, curEntries * REMOTE_ADDRESS_ENTRY_SIZE, 0U, 0U, 0U};
        SyncFunc<HardEvent::S_MTE2>();
        DataCopyPad(entriesLocal, rankAddressTable[static_cast<uint64_t>(blockStart + chunkStart) * entryElements],
                    copyParams, padParams);
        SyncFunc<HardEvent::MTE2_S>();
        uint32_t i = 0U;
        while (i < curEntries) {
            const uint32_t runFirst = i;
            const uint64_t runTokenSlot = entriesLocal.GetValue(i * entryElements);
            ++i;
            while (i < curEntries && entriesLocal.GetValue(i * entryElements) == runTokenSlot + (i - runFirst)) {
                ++i;
            }

            const uint32_t runLen = i - runFirst;
            uint32_t done = 0U;
            while (done < runLen) {
                const uint32_t remaining = runLen - done;
                const uint32_t sgeCount = remaining > SGE_PER_FLAG_WQE ? SGE_PER_FLAG_WQE : remaining;
                const uint32_t flagWqebbCount =
                    (HCOMM_SQE_BYTES + sgeCount * HCOMM_SGE_BYTES + HCOMM_WQEBB_BYTES - 1U) / HCOMM_WQEBB_BYTES;
                if (preparedWqebbCount + flagWqebbCount > URMA_BATCH_WQEBB_CAPACITY) {
                    (void)hcomm_.BatchCommit(batchHandle);
                    preparedWqebbCount = 0U;
                }
                GM_ADDR remoteFlagAddr = remoteWindowAddr + (runTokenSlot + done) * sizeof(int32_t);
                AppendRemoteFlagWrite(batchHandle, remoteFlagAddr, sourceDescs, sgeCount);
                preparedWqebbCount += flagWqebbCount;
                done += sgeCount;
            }
        }
    }

    if (preparedWqebbCount != 0U) {
        (void)hcomm_.BatchCommit(batchHandle);
    }
}

template <TemplateFFNToAttentionUrmaTypeClass>
__aicore__ inline void FFNToAttentionUrma<TemplateFFNToAttentionUrmaTypeFunc>::Process()
{
    if ASCEND_IS_AIV {
        HcommInit();

        CopyXToPayload();
        for (uint32_t microBatchId = 0U; microBatchId < microBatchNum_; ++microBatchId) {
            if ((activeMicroBatchMask_ & (1U << microBatchId)) == 0U) {
                continue;
            }
            SelectPayloadSlot(microBatchId);
            InitLocalFlag();
            BuildAddressTable(microBatchId);

            LocalTensor<int32_t> rankCountRow = rankInfoBuf_.Get<int32_t>();
            GlobalTensor<int32_t> totalRankCounts;
            totalRankCounts.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(totalRankCountPayloadAddr_));
            SyncFunc<HardEvent::S_MTE2>();
            DataCopyPad(rankCountRow, totalRankCounts,
                        DataCopyExtParams{1U, static_cast<uint32_t>(rankCountElements_ * sizeof(int32_t)), 0U, 0U, 0U},
                        DataCopyPadExtParams<int32_t>{false, 0U, 0U, 0});
            SyncFunc<HardEvent::MTE2_S>();

            const uint32_t totalSlots = worldSize_ * channelsPerRank_;
            for (uint32_t slot = aivId_; slot < totalSlots; slot += aivNum_) {
                uint32_t dstRank = slot / channelsPerRank_;
                uint32_t channelIdx = slot % channelsPerRank_;
                uint32_t entryCount = GetAddressTableCount(dstRank);
                SendAddressTable(dstRank, entryCount, channelIdx);
            }
            SyncAll<true>();
        }
    }
}

#endif

} // namespace FFNToAttentionImpl
#endif // FFN_TO_ATTENTION_URMA_H
