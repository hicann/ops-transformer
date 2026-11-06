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
 * \file moe_ep_combine_clos_only.h
 * \brief Combine kernel implementation for Clos-only communication.
 */
#ifndef MOE_EP_COMBINE_CLOS_ONLY_H
#define MOE_EP_COMBINE_CLOS_ONLY_H

#include <cstddef>

#if __has_include("version/asc_devkit_version.h") && __has_include("version/hcomm_version.h")
#include "version/asc_devkit_version.h"
#include "version/hcomm_version.h"

#if (ASC_DEVKIT_VERSION_NUM >= 90200000) && (HCOMM_VERSION_NUM >= 90200000)
#define ENABLE_MOE_EP_COMBINE_KERNEL
#endif

#endif

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#include "kernel_tiling/kernel_tiling.h"
#include "adv_api/hccl/hccl.h"
#if __has_include("adv_api/hcomm/hcomm.h")
#include "adv_api/hcomm/hcomm.h"
#endif

#include "moe_ep_combine_tiling_key.h"
#include "../../common/op_kernel/moe_distribute_base.h"
#include "../../common/op_kernel/mc2_kernel_utils.h"
#include "../../common/op_kernel/moe_ep_exception_dump_writer.h"
#include "../../common/op_kernel/moe_shared_jetty_context.h"

#include "moe_ep_combine_base.h"
#include "moe_ep_combine_tiling.h"

namespace MoeEpCombineClosOnlyImpl {

#if defined(ENABLE_MOE_EP_COMBINE_KERNEL)

using namespace AscendC;

#define TemplateMoeEpCombineTypeClass typename XType, uint32_t HasTopkWeight
#define TemplateMoeEpCombineTypeFunc XType, HasTopkWeight
#define HCOMM_INIT_SIZE 512UL

static constexpr uint32_t WIN_ADDR_ALIGN = 512;
static constexpr uint32_t RECV_META_FIELDS = 5U;
static constexpr uint32_t META_TOKEN_IDX_OFFSET = 1U;
static constexpr uint32_t META_TOPK_IDX_OFFSET = 2U;
static constexpr uint32_t META_RECV_X_IDX_OFFSET = 4U;
// A peer SQ holds 32768 WQEBBs. The WQE crossing the 32767-pending boundary requests a CQE,
// then that peer channel is drained before more WQEs are issued.
static constexpr uint32_t HCOMM_SQ_MAX_PENDING = 32767U;
// PR 111 requires each committed batch to contain fewer WQEBBs than the SQ depth.
static constexpr uint32_t HCOMM_BATCH_CAPACITY = 256;
static constexpr uint32_t HCOMM_PLAIN_WRITE_WQE_BYTES = 64;
static constexpr uint32_t HCOMM_BATCH_BUFFER_BYTES = HCOMM_BATCH_CAPACITY * HCOMM_PLAIN_WRITE_WQE_BYTES;
constexpr uint64_t UB_ALIGN = 32UL;
// Two 1024-row buffers retain the previous 2048-row maximum metadata UB budget.
static constexpr uint32_t META_CHUNK_TOKEN_MAX = 1024U;
static constexpr struct UrmaWqeEntry DEFAULT_WQE_CONFIG = {.odr = 5, .fence = 1, .se = 0, .cqe = 0, .inlineEn = 0};
static constexpr struct UrmaWqeEntry DEFAULT_CQE_WQE_CONFIG = {.odr = 5, .fence = 1, .se = 0, .cqe = 1, .inlineEn = 0};
// Keep the final flag ordered after payload writes; the epilogue waits for it without a sender-side completion wait.
static constexpr struct UrmaWqeEntry JETTY_FLAG_WQE_CONFIG = {.odr = 6, .fence = 1, .se = 0, .cqe = 0, .inlineEn = 0};
static constexpr struct UrmaWqeEntry JETTY_FLAG_CQE_WQE_CONFIG = {
    .odr = 6, .fence = 1, .se = 0, .cqe = 1, .inlineEn = 0};

template <TemplateMoeEpCombineTypeClass>
class MoeEpCombine {
public:
    __aicore__ inline MoeEpCombine(){};

    __aicore__ inline void Init(GM_ADDR context, GM_ADDR x, GM_ADDR topkIdx, GM_ADDR recvSrcMetadata,
                                GM_ADDR numRecvPerExpert, GM_ADDR topkWeights, GM_ADDR workspace, GM_ADDR /*tilingGM*/,
                                TPipe* pipe, const MoeEpCombineInfo* tilingData);

    __aicore__ inline void Process();

private:
    struct MetadataChunk {
        uint32_t rank;
        uint64_t begin;
        uint32_t count;
        bool last;
    };
    __aicore__ inline void InitContext(GM_ADDR context);
    __aicore__ inline void InitMetadata(GM_ADDR x, GM_ADDR recvSrcMetadata, GM_ADDR topkWeights);
    __aicore__ inline void SendJettyFlag(uint32_t dstRank, uint32_t jettyIndex);
    __aicore__ inline void BeginPreparedWrites(uint32_t dstRank, uint32_t jettyIndex);
    __aicore__ inline void InitFlagSource();
    template <auto const& config, auto const& cqeConfig>
    __aicore__ inline void PrepareWrite(GM_ADDR dst, GM_ADDR src, uint64_t len);
    template <auto const& config>
    __aicore__ inline void PrepareMultiSgeWrite(GM_ADDR dst, const AscendC::BufDesc* srcDescs, uint32_t srcNum);
    __aicore__ inline void FlushPreparedWrites(bool keepHandle = false);
    __aicore__ inline void ReservePreparedWrites(uint32_t wqebbCount);
    __aicore__ inline void DrainActivePeer();
    __aicore__ inline void SplitRange(uint64_t rangeBegin, uint64_t rangeEnd, uint32_t coreCount, uint32_t coreIndex,
                                      uint64_t& coreBegin, uint64_t& coreEnd);
    __aicore__ inline void SendRemoteMetadataSlot(uint32_t recvXIdx, int32_t srcTokenIdx, int32_t srcTopKIdx,
                                                  GM_ADDR remoteDataBase, uint64_t tokenBytes);
    __aicore__ inline bool NextMetadataChunk(uint32_t& peerOffset, uint64_t& begin, uint64_t& end,
                                             MetadataChunk& chunk);
    __aicore__ inline void PrefetchMetadata(const MetadataChunk& chunk, uint32_t buffer, TEventID event);
    __aicore__ inline void SendMetadataChunk(const MetadataChunk& chunk, uint32_t buffer, TEventID event);
    __aicore__ inline void SendMetadataPipeline();
    __aicore__ inline void SendPhaseFromMetadata();

    __aicore__ inline uint64_t GetJettyHandle(uint32_t jettyIndex)
    {
        return mc2Context_->multiChannelHandles[jettyIndex];
    }
    __aicore__ inline GM_ADDR GetUrmaWinAddrByRankId(uint32_t rankId, uint64_t offset)
    {
        return (GM_ADDR)(winRankAddr_[rankId] + offset);
    }
    __aicore__ inline GM_ADDR GetUrmaStateAddrByRankId(uint32_t rankId, uint64_t offset)
    {
        return (GM_ADDR)(winRankAddr_[rankId] + offset);
    }

    TPipe* tpipe_{nullptr};
    const MoeEpCombineInfo* tilingData_{nullptr};
    using ContextType = Mc2Kernel::MoeSharedJettyContext;
    __gm__ ContextType* mc2Context_{nullptr};
    MoeEpExceptionDump::MoeEpCoreDiagWriter diagWriter_;

    uint32_t rankId_{0};
    uint32_t epWorldSize_{0};
    uint32_t combineJettyCount_{1};
    uint32_t numMaxTokensPerRank_{0};
    uint32_t topK_{0};
    uint32_t axisH_{0};
    uint32_t hAlignSize_{0};
    uint64_t combineStateWinOffset_{0};
    uint64_t combineDataWinOffset_{0};

    uint32_t perSlotBytes_{0};
    uint64_t actualA_{0};
    uint64_t recvCapacity_{0};
    uint32_t metadataChunkTokens_{1};

    GlobalTensor<XType> xGm_;
    GlobalTensor<int32_t> recvSrcMetadataGm_;
    GlobalTensor<int32_t> recvRankOffsetsGm_;
    GlobalTensor<float> topkWeightsGm_;

    LocalTensor<uint32_t> statusTensor_;
    LocalTensor<int32_t> rankOffsetsTensor_;
    LocalTensor<uint8_t> hcommTensor_;
    LocalTensor<uint8_t> hcommBatchTensor_;

    TBuf<> readStateBuf_;
    TBuf<> rankOffsetsBuf_;
    TBuf<> hcommBuf_;
    TBuf<TPosition::VECOUT> hcommBatchBuf_;

    TBuf<> metadataBuf_[2]; // One metadata chunk can load while the other builds WQEs.

    AscendC::Hcomm<COMM_PROTOCOL_UBC_CTP> hcomm_; // 通信上下文
    using CommHandle = AscendC::MultiChannelHandle;
    using HcommBatchHandle = AscendC::BatchHandle<CommHandle>;

    GM_ADDR winRankAddr_[Mc2Kernel::MOE_SHARED_JETTY_MAX_RANK_SIZE];
    GM_ADDR flagSourceWinAddr_{nullptr};
    uint32_t aivId_{0};
    HcommBatchHandle activeBatchHandle_{};
    uint64_t activeJettyHandle_{0};
    uint32_t activePeerIndex_{0};
    GM_ADDR activeRemoteWindow_{nullptr};
    uint32_t preparedWriteCount_{0};
    uint32_t activeSqPendingWqebb_{0};
    bool activeBatchInitialized_{false};
};

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::Init(
    GM_ADDR context, GM_ADDR x, GM_ADDR /*topkIdx*/, GM_ADDR recvSrcMetadata, GM_ADDR /*numRecvPerExpert*/,
    GM_ADDR topkWeights, GM_ADDR /*workspace*/, GM_ADDR tilingGM, TPipe* pipe, const MoeEpCombineInfo* tilingData)
{
    tpipe_ = pipe;
    tilingData_ = tilingData;
    aivId_ = GetBlockIdx();
    epWorldSize_ = tilingData_->cfg.epWorldSize;
    numMaxTokensPerRank_ = tilingData_->cfg.numMaxTokensPerRank;
    topK_ = tilingData_->cfg.topK;
    axisH_ = tilingData_->cfg.hidden;
    hAlignSize_ = Ceil(axisH_ * sizeof(XType), UB_ALIGN) * UB_ALIGN;
    perSlotBytes_ = tilingData_->cfg.perSlotBytes;
    recvCapacity_ = tilingData_->recvCapacity;
    tpipe_->InitBuffer(hcommBuf_, HCOMM_INIT_SIZE);
    hcommTensor_ = hcommBuf_.Get<uint8_t>();
    hcomm_.Init(hcommTensor_, HCOMM_INIT_SIZE);
    tpipe_->InitBuffer(hcommBatchBuf_, HCOMM_BATCH_BUFFER_BYTES);
    hcommBatchTensor_ = hcommBatchBuf_.Get<uint8_t>();
    Duplicate<uint8_t>(hcommBatchTensor_, 0U, HCOMM_BATCH_BUFFER_BYTES);
    SyncFunc<AscendC::HardEvent::V_S>();

    InitContext(context);
    constexpr size_t metadataOffset =
        offsetof(MoeEpCombineTilingData, moeEpCombineInfo) + offsetof(MoeEpCombineInfo, dumpMetadata);
    MoeEpExceptionDump::WriteMetadata<ContextType>(context, tilingGM + metadataOffset);
    diagWriter_.Init<ContextType>(context, MOE_EP_CORE_DIAG_COMBINE, tpipe_);

    combineStateWinOffset_ = tilingData->combineStateWinOffset;
    combineDataWinOffset_ = tilingData->combineDataWinOffset;
    flagSourceWinAddr_ = GetUrmaStateAddrByRankId(rankId_, tilingData->combineFlagSourceWinOffset);

    InitMetadata(x, recvSrcMetadata, topkWeights);
    diagWriter_.RunPosRecord(MOE_EP_COMBINE_RUN_POS_INIT_DONE);
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::InitContext(GM_ADDR context)
{
    mc2Context_ = reinterpret_cast<__gm__ ContextType*>(context);
    rankId_ = mc2Context_->epRankId;
    combineJettyCount_ = Mc2Kernel::MOE_SHARED_JETTY_NUM;
    for (uint32_t i = 0; i < epWorldSize_; ++i) {
        winRankAddr_[i] = (GM_ADDR)mc2Context_->epHcclBuffer[i];
    }
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::InitMetadata(GM_ADDR x, GM_ADDR recvSrcMetadata,
                                                                                GM_ADDR topkWeights)
{
    xGm_.SetGlobalBuffer((__gm__ XType*)x);
    recvSrcMetadataGm_.SetGlobalBuffer((__gm__ int32_t*)recvSrcMetadata);
    recvRankOffsetsGm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t*>(recvSrcMetadata + tilingData_->metadataRankOffsetsOffset));

    uint32_t rankOffsetsBytes = Ceil(static_cast<uint64_t>(epWorldSize_ + 1U) * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(rankOffsetsBuf_, rankOffsetsBytes);
    rankOffsetsTensor_ = rankOffsetsBuf_.Get<int32_t>();
    DataCopyExtParams rankOffsetsCopyParams{1U, static_cast<uint32_t>((epWorldSize_ + 1U) * sizeof(int32_t)), 0U, 0U,
                                            0U};
    DataCopyPadExtParams<int32_t> rankOffsetsPadParams{false, 0U, 0U, 0U};
    DataCopyPad(rankOffsetsTensor_, recvRankOffsetsGm_, rankOffsetsCopyParams, rankOffsetsPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    int32_t actualASigned = rankOffsetsTensor_.GetValue(epWorldSize_);

    actualA_ = (actualASigned > 0) ? static_cast<uint64_t>(actualASigned) : 0U;
    if (actualA_ > recvCapacity_) {
        actualA_ = recvCapacity_;
    }
    if constexpr (HasTopkWeight == 1) {
        topkWeightsGm_.SetGlobalBuffer((__gm__ float*)topkWeights);
    }
    metadataChunkTokens_ = (actualA_ < META_CHUNK_TOKEN_MAX) ? static_cast<uint32_t>(actualA_) : META_CHUNK_TOKEN_MAX;
    metadataChunkTokens_ = (metadataChunkTokens_ == 0U) ? 1U : metadataChunkTokens_;
    uint32_t metadataChunkBytes =
        Ceil(static_cast<uint64_t>(metadataChunkTokens_) * RECV_META_FIELDS * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(metadataBuf_[0], metadataChunkBytes);
    tpipe_->InitBuffer(metadataBuf_[1], metadataChunkBytes);
    tpipe_->InitBuffer(readStateBuf_, WIN_ADDR_ALIGN);
    statusTensor_ = readStateBuf_.Get<uint32_t>();
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::FlushPreparedWrites(bool keepHandle)
{
    if (preparedWriteCount_ != 0) {
        hcomm_.BatchCommit(activeBatchHandle_);
        preparedWriteCount_ = 0;
    }
    if (!keepHandle) {
        activeBatchHandle_ = {};
        activeJettyHandle_ = 0;
        activeBatchInitialized_ = false;
    }
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::BeginPreparedWrites(uint32_t dstRank,
                                                                                       uint32_t jettyIndex)
{
    uint32_t peerIndex = dstRank < rankId_ ? dstRank : dstRank - 1U;
    uint64_t commHandle = GetJettyHandle(jettyIndex);
    if (activeBatchInitialized_ && activeJettyHandle_ == commHandle && activePeerIndex_ == peerIndex &&
        activeRemoteWindow_ == winRankAddr_[dstRank]) {
        return;
    }
    if (!activeBatchInitialized_ || activeJettyHandle_ != commHandle) {
        if (activeBatchInitialized_) {
            FlushPreparedWrites();
        }
        activeBatchHandle_ = hcomm_.MakeBatchHandle(static_cast<CommHandle>(commHandle), hcommBatchTensor_,
                                                    HCOMM_BATCH_BUFFER_BYTES, winRankAddr_[dstRank]);
        activeJettyHandle_ = commHandle;
        activeBatchInitialized_ = true;
    }
    activePeerIndex_ = peerIndex;
    activeRemoteWindow_ = winRankAddr_[dstRank];
    // All payload/flag addresses for this peer belong to the same registered MoE window.
    hcomm_.GetHandleRef(activeBatchHandle_, activePeerIndex_, activeRemoteWindow_);
    activeSqPendingWqebb_ = 0U;
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::ReservePreparedWrites(uint32_t wqebbCount)
{
    if (preparedWriteCount_ + wqebbCount > HCOMM_BATCH_CAPACITY) {
        FlushPreparedWrites(true);
    }
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::DrainActivePeer()
{
    FlushPreparedWrites(true);
    auto& handle = hcomm_.GetHandleRef(activeBatchHandle_, activePeerIndex_, activeRemoteWindow_);
    (void)hcomm_.Drain(handle);
    activeSqPendingWqebb_ = 0U;
}

template <TemplateMoeEpCombineTypeClass>
template <auto const& config, auto const& cqeConfig>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::PrepareWrite(GM_ADDR dst, GM_ADDR src, uint64_t len)
{
    // Single-SGE WQE occupies 1 WQEBB (64 bytes).
    ReservePreparedWrites(1U);
    const bool drainNow = activeSqPendingWqebb_ + 1U > HCOMM_SQ_MAX_PENDING;
    if (drainNow) {
        hcomm_.WriteNbi<cqeConfig>(activeBatchHandle_.handle, dst, src, len);
    } else {
        hcomm_.WriteNbi<config>(activeBatchHandle_.handle, dst, src, len);
    }
    ++preparedWriteCount_;
    activeSqPendingWqebb_ += 1U;
    if (drainNow) {
        DrainActivePeer();
    }
}

template <TemplateMoeEpCombineTypeClass>
template <auto const& config>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::PrepareMultiSgeWrite(
    GM_ADDR dst, const AscendC::BufDesc* srcDescs, uint32_t srcNum)
{
    // Multi-SGE WQE: SQE header (48B) + srcNum * SGE (16B each), rounded up to WQEBB boundary (64B).
    constexpr uint32_t sqeHeaderBytes = 48U;
    constexpr uint32_t sgeEntryBytes = 16U;
    constexpr uint32_t wqebbSize = HCOMM_PLAIN_WRITE_WQE_BYTES;
    uint32_t wqeBytes = sqeHeaderBytes + srcNum * sgeEntryBytes;
    uint32_t wqebbCount = (wqeBytes + wqebbSize - 1U) / wqebbSize;
    ReservePreparedWrites(wqebbCount);
    const bool drainNow = activeSqPendingWqebb_ + wqebbCount > HCOMM_SQ_MAX_PENDING;
    if (drainNow) {
        hcomm_.WriteNbi<DEFAULT_CQE_WQE_CONFIG>(activeBatchHandle_.handle, dst, srcDescs, srcNum);
    } else {
        hcomm_.WriteNbi<config>(activeBatchHandle_.handle, dst, srcDescs, srcNum);
    }
    preparedWriteCount_ += wqebbCount;
    activeSqPendingWqebb_ += wqebbCount;
    if (drainNow) {
        DrainActivePeer();
    }
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::SendJettyFlag(uint32_t dstRank, uint32_t jettyIndex)
{
    uint64_t flagIndex = static_cast<uint64_t>(rankId_) * combineJettyCount_ + jettyIndex;
    uint64_t flagOffset =
        static_cast<uint64_t>(numMaxTokensPerRank_) * topK_ * WIN_ADDR_ALIGN + flagIndex * WIN_ADDR_ALIGN;
    GM_ADDR flagAddr = GetUrmaStateAddrByRankId(dstRank, combineStateWinOffset_) + flagOffset;
    if (dstRank == rankId_) {
        GlobalTensor<uint64_t> localFlag;
        localFlag.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(flagAddr));
        localFlag.SetValue(0, 1U);
        DataCacheCleanAndInvalid<uint64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(localFlag);
        return;
    }
    BeginPreparedWrites(dstRank, jettyIndex);
    PrepareWrite<JETTY_FLAG_WQE_CONFIG, JETTY_FLAG_CQE_WQE_CONFIG>(flagAddr, flagSourceWinAddr_, WIN_ADDR_ALIGN);
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::InitFlagSource()
{
    if (aivId_ != 0U) {
        return;
    }
    LocalTensor<uint64_t> flagTensor = statusTensor_.ReinterpretCast<uint64_t>();
    Duplicate<uint64_t>(flagTensor, 1U, WIN_ADDR_ALIGN / sizeof(uint64_t));
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    // Reinitialize to the same constant for standalone combine as well; no receive/clear path touches this slot.
    GlobalTensor<uint64_t> flagSource;
    flagSource.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(flagSourceWinAddr_));
    DataCopy(flagSource, flagTensor, WIN_ADDR_ALIGN / sizeof(uint64_t));
    SyncFunc<AscendC::HardEvent::MTE3_S>();
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::SplitRange(uint64_t rangeBegin, uint64_t rangeEnd,
                                                                              uint32_t coreCount, uint32_t coreIndex,
                                                                              uint64_t& coreBegin, uint64_t& coreEnd)
{
    if (rangeBegin >= rangeEnd || coreCount == 0U || coreIndex >= coreCount) {
        coreBegin = rangeBegin;
        coreEnd = rangeBegin;
        return;
    }
    uint64_t count = rangeEnd - rangeBegin;
    uint64_t base = count / coreCount;
    uint64_t remainder = count % coreCount;
    uint64_t prefix = static_cast<uint64_t>(coreIndex) * base +
                      ((static_cast<uint64_t>(coreIndex) < remainder) ? coreIndex : remainder);
    uint64_t coreCountValue = base + ((static_cast<uint64_t>(coreIndex) < remainder) ? 1U : 0U);
    coreBegin = rangeBegin + prefix;
    coreEnd = coreBegin + coreCountValue;
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::SendRemoteMetadataSlot(
    uint32_t recvXIdx, int32_t srcTokenIdx, int32_t srcTopKIdx, GM_ADDR remoteDataBase, uint64_t tokenBytes)
{
    uint64_t dstSlot =
        static_cast<uint64_t>(static_cast<uint32_t>(srcTokenIdx)) * topK_ + static_cast<uint32_t>(srcTopKIdx);
    GM_ADDR tokenAddr = reinterpret_cast<GM_ADDR>(xGm_.GetPhyAddr(static_cast<uint64_t>(recvXIdx) * axisH_));
    GM_ADDR remoteSlotBase = remoteDataBase + dstSlot * perSlotBytes_;
    if constexpr (HasTopkWeight == 1) {
        GM_ADDR weightAddr = reinterpret_cast<GM_ADDR>(topkWeightsGm_.GetPhyAddr(recvXIdx));
        AscendC::BufDesc srcDescs[2] = {{tokenAddr, hAlignSize_}, {weightAddr, sizeof(float)}};
        PrepareMultiSgeWrite<DEFAULT_WQE_CONFIG>(remoteSlotBase, srcDescs, 2U);
    } else {
        PrepareWrite<DEFAULT_WQE_CONFIG, DEFAULT_CQE_WQE_CONFIG>(remoteSlotBase, tokenAddr, tokenBytes);
    }
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline bool MoeEpCombine<TemplateMoeEpCombineTypeFunc>::NextMetadataChunk(uint32_t& peerOffset,
                                                                                     uint64_t& begin, uint64_t& end,
                                                                                     MetadataChunk& chunk)
{
    if (peerOffset == epWorldSize_) {
        return false;
    }
    uint32_t rank = (aivId_ + peerOffset) % epWorldSize_;
    if (begin == end && rank != rankId_ && actualA_ != 0U) {
        uint64_t rankBegin = static_cast<uint32_t>(rankOffsetsTensor_.GetValue(rank));
        uint64_t rankEnd = static_cast<uint32_t>(rankOffsetsTensor_.GetValue(rank + 1U));
        SplitRange(rankBegin, rankEnd, combineJettyCount_, aivId_, begin, end);
    }
    uint64_t remaining = end - begin;
    uint32_t count = remaining < metadataChunkTokens_ ? static_cast<uint32_t>(remaining) : metadataChunkTokens_;
    chunk = {rank, begin, count, count == remaining};
    begin += count;
    if (chunk.last) {
        ++peerOffset;
        begin = 0U;
        end = 0U;
    }
    return true;
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::PrefetchMetadata(const MetadataChunk& chunk,
                                                                                    uint32_t buffer, TEventID event)
{
    if (chunk.count == 0U) {
        return;
    }
    // The scalar consumer of this buffer has finished; do not overwrite its last UB reads.
    SyncFunc<AscendC::HardEvent::S_MTE2>();
    LocalTensor<int32_t> metadataLocal = metadataBuf_[buffer].Get<int32_t>();
    constexpr uint32_t metaBytesPerToken = RECV_META_FIELDS * sizeof(int32_t);
    DataCopyExtParams copyParams{1U, chunk.count * metaBytesPerToken, 0U, 0U, 0U};
    const DataCopyPadExtParams<int32_t> padParams{false, 0U, 0U, 0U};
    DataCopyPad(metadataLocal, recvSrcMetadataGm_[chunk.begin * RECV_META_FIELDS], copyParams, padParams);
    SetFlag<AscendC::HardEvent::MTE2_S>(event);
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::SendMetadataChunk(const MetadataChunk& chunk,
                                                                                     uint32_t buffer, TEventID event)
{
    if (chunk.count != 0U) {
        BeginPreparedWrites(chunk.rank, aivId_);
        WaitFlag<AscendC::HardEvent::MTE2_S>(event);
        LocalTensor<int32_t> metadataLocal = metadataBuf_[buffer].Get<int32_t>();
        GM_ADDR remoteDataBase = GetUrmaWinAddrByRankId(chunk.rank, combineDataWinOffset_);
        uint64_t tokenBytes = static_cast<uint64_t>(axisH_) * sizeof(XType);
        for (uint32_t i = 0U; i < chunk.count; ++i) {
            uint32_t metaOffset = i * RECV_META_FIELDS;
            int32_t srcTokenIdx = metadataLocal.GetValue(metaOffset + META_TOKEN_IDX_OFFSET);
            int32_t srcTopKIdx = metadataLocal.GetValue(metaOffset + META_TOPK_IDX_OFFSET);
            int32_t recvXIdx = metadataLocal.GetValue(metaOffset + META_RECV_X_IDX_OFFSET);
            SendRemoteMetadataSlot(static_cast<uint32_t>(recvXIdx), srcTokenIdx, srcTopKIdx, remoteDataBase,
                                   tokenBytes);
        }
    }
    if (chunk.last) {
        SendJettyFlag(chunk.rank, aivId_);
    }
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::SendMetadataPipeline()
{
    TEventID ready[2] = {tpipe_->AllocEventID<AscendC::HardEvent::MTE2_S>(),
                         tpipe_->AllocEventID<AscendC::HardEvent::MTE2_S>()};
    MetadataChunk chunks[2];
    uint32_t peerOffset = 0U;
    uint64_t begin = 0U;
    uint64_t end = 0U;
    uint32_t current = 0U;
    bool hasCurrent = NextMetadataChunk(peerOffset, begin, end, chunks[current]);
    if (hasCurrent) {
        PrefetchMetadata(chunks[current], current, ready[current]);
    }
    while (hasCurrent) {
        uint32_t next = current ^ 1U;
        bool hasNext = NextMetadataChunk(peerOffset, begin, end, chunks[next]);
        if (hasNext) {
            PrefetchMetadata(chunks[next], next, ready[next]);
        }
        SendMetadataChunk(chunks[current], current, ready[current]);
        current = next;
        hasCurrent = hasNext;
    }
    // Every nonempty prefetch has been consumed, including the final chunk.
    tpipe_->ReleaseEventID<AscendC::HardEvent::MTE2_S>(ready[0]);
    tpipe_->ReleaseEventID<AscendC::HardEvent::MTE2_S>(ready[1]);
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::SendPhaseFromMetadata()
{
    InitFlagSource();
    SyncAll<true>();
    if (aivId_ >= combineJettyCount_) {
        return;
    }
    activeBatchHandle_ = {};
    activeJettyHandle_ = 0U;
    preparedWriteCount_ = 0U;
    activeSqPendingWqebb_ = 0U;
    activeBatchInitialized_ = false;

    // Prefetch across peer boundaries too; typical peer slices fit in a single metadata chunk.
    SendMetadataPipeline();
    FlushPreparedWrites();
    DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(recvSrcMetadataGm_);
    diagWriter_.RunPosRecord(MOE_EP_COMBINE_RUN_POS_URMA_REQUESTS_ISSUE_DONE);
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::Process()
{
    SendPhaseFromMetadata();
}

#endif
} // namespace MoeEpCombineClosOnlyImpl

#endif // MOE_EP_COMBINE_CLOS_ONLY_H
