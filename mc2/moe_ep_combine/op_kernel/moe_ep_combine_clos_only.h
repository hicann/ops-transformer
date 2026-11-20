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
#include "moe_ep_combine_vf.h"
#include "moe_ep_combine_tiling.h"

namespace MoeEpCombineClosOnlyImpl {

#if defined(ENABLE_MOE_EP_COMBINE_KERNEL)

using namespace AscendC;
using namespace MoeEpCombineLayout;

#define TemplateMoeEpCombineTypeClass typename XType, uint32_t HasTopkWeight
#define TemplateMoeEpCombineTypeFunc XType, HasTopkWeight
#define HCOMM_INIT_SIZE 512UL

static constexpr uint32_t WIN_ADDR_ALIGN = 512;
// PR 111 requires each committed batch to contain fewer WQEBBs than the SQ depth.
static constexpr uint32_t HCOMM_BATCH_CAPACITY = 256;
static constexpr uint32_t HCOMM_PLAIN_WRITE_WQE_BYTES = 64;
static constexpr uint32_t HCOMM_BATCH_BUFFER_BYTES = HCOMM_BATCH_CAPACITY * HCOMM_PLAIN_WRITE_WQE_BYTES;
constexpr uint64_t UB_ALIGN = 32UL;
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
    __aicore__ inline void PrepareWrite(GM_ADDR dst, GM_ADDR src, uint64_t len, bool forceCqe);
    __aicore__ inline void FlushPreparedWrites(bool keepHandle = false);
    __aicore__ inline void ReservePreparedWrites(uint32_t wqebbCount);
    __aicore__ inline void ReservePreparedCqe();
    __aicore__ inline void SplitRange(uint64_t rangeBegin, uint64_t rangeEnd, uint32_t coreCount, uint32_t coreIndex,
                                      uint64_t& coreBegin, uint64_t& coreEnd);
    __aicore__ inline void BuildBatchAddrs(uint32_t batchCount, __ubuf__ uint32_t* metaUb, __ubuf__ uint64_t* addrUb);
    template <auto const& config>
    __aicore__ inline void EmitTokenWrite(uint32_t regionOff, uint32_t tokenIdx, const LocalTensor<uint64_t>& addrUb,
                                          GM_ADDR remoteDataBase, GM_ADDR xBase, GM_ADDR weightBase,
                                          uint64_t tokenBytes);
    __aicore__ inline void ConsumeBatchAddrs(const MetadataChunk& chunk, uint32_t buffer);
    __aicore__ inline bool NextMetadataChunk(uint32_t& peerOffset, uint64_t& begin, uint64_t& end,
                                             MetadataChunk& chunk);
    __aicore__ inline void PrefetchMetadata(const MetadataChunk& chunk, uint32_t buffer);
    __aicore__ inline void PrepareBatchAddrs(const MetadataChunk& chunk, uint32_t buffer);
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
    uint64_t combineStateWinOffset_{0};
    uint64_t combineDataWinOffset_{0};

    static constexpr uint32_t kTokenWqebbCount = (HasTopkWeight == 1) ? 2U : 1U;
    static constexpr uint32_t kPipeAddrElemsPerBatch = METADATA_BATCH_TOKENS * ((HasTopkWeight == 1) ? 3U : 2U);

    uint32_t perSlotBytes_{0};
    uint64_t actualA_{0};
    uint64_t recvCapacity_{0};

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

    TBuf<> metaPipeBuf_;
    TBuf<> addrPipeBuf_;
    event_t evMte2V_[2] = {};
    event_t evVS_[2] = {};

    AscendC::Hcomm<COMM_PROTOCOL_UBC_CTP> hcomm_; // 通信上下文
    using CommHandle = AscendC::MultiChannelHandle;
    using HcommBatchHandle = AscendC::BatchHandle<CommHandle>;

    GM_ADDR winRankAddr_[Mc2Kernel::MOE_SHARED_JETTY_MAX_RANK_SIZE];
    GM_ADDR flagSourceWinAddr_{nullptr};
    uint32_t aivId_{0};
    uint32_t jettyIndex_{0};
    HcommBatchHandle activeBatchHandle_{};
    uint64_t activeJettyHandle_{0};
    uint32_t activePeerIndex_{0};
    uint32_t lastRemoteRank_{0};
    GM_ADDR activeRemoteWindow_{nullptr};
    uint32_t preparedWriteCount_{0};
    uint32_t preparedCqeCount_{0};
    uint32_t batchCapacity_{HCOMM_BATCH_CAPACITY};
    uint32_t cqCapacity_{0};
    uint32_t cqeIntervalLimit_{0};
    uint32_t wqebbSinceCqe_{0};
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
#if defined(ASCENDC_OOM) && ASCENDC_OOM == 1
    // Register indirect GM ranges before the first dump or window access.
    OOMCheckAddrRange<uint8_t>(winRankAddr_[rankId_], tilingData->totalWinSizeEp);
    OOMCheckAddrRange<uint8_t>(tilingGM, sizeof(MoeEpCombineTilingData));
    for (uint32_t rankId = 0; rankId < epWorldSize_; ++rankId) {
        if (rankId != rankId_) {
            OOMCheckAddrRange<uint8_t>(winRankAddr_[rankId], tilingData->totalWinSizeEp);
        }
    }
#endif
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
    tpipe_->InitBuffer(metaPipeBuf_, METADATA_BATCH_ELEMS * sizeof(int32_t) * 2U);
    tpipe_->InitBuffer(addrPipeBuf_, kPipeAddrElemsPerBatch * sizeof(uint64_t) * 2U);
    tpipe_->InitBuffer(readStateBuf_, WIN_ADDR_ALIGN);
    statusTensor_ = readStateBuf_.Get<uint32_t>();
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::FlushPreparedWrites(bool keepHandle)
{
    if (preparedWriteCount_ != 0) {
        hcomm_.BatchCommit(activeBatchHandle_);
        preparedWriteCount_ = 0;
        preparedCqeCount_ = 0;
        batchCapacity_ = HCOMM_BATCH_CAPACITY;
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
        const auto& handle = activeBatchHandle_.handle;
        const uint32_t sqDepth = handle.sqContext.contextInfo.ubJfs.sqDepth;
        cqCapacity_ = handle.cqContext.contextInfo.ubJfc.cqDepth;
        wqebbSinceCqe_ = handle.cursor.sqHead - handle.cursor.sqTail;
        // Leave a full batch of headroom so BatchCommit can poll an already submitted CQE before SQ overflow.
        cqeIntervalLimit_ = sqDepth - HCOMM_BATCH_CAPACITY - kTokenWqebbCount;
        const uint32_t available = sqDepth - wqebbSinceCqe_;
        batchCapacity_ = available < HCOMM_BATCH_CAPACITY ? available : HCOMM_BATCH_CAPACITY;
        activeJettyHandle_ = commHandle;
        activeBatchInitialized_ = true;
    }
    activePeerIndex_ = peerIndex;
    activeRemoteWindow_ = winRankAddr_[dstRank];
    // All payload/flag addresses for this peer belong to the same registered MoE window.
    hcomm_.GetHandleRef(activeBatchHandle_, activePeerIndex_, activeRemoteWindow_);
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::ReservePreparedWrites(uint32_t wqebbCount)
{
    if (preparedWriteCount_ + wqebbCount > batchCapacity_) {
        FlushPreparedWrites(true);
    }
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::ReservePreparedCqe()
{
    // BatchCommit can poll older CQEs, but the current batch itself must fit in the CQ.
    if (preparedCqeCount_ == cqCapacity_) {
        FlushPreparedWrites(true);
    }
}

template <TemplateMoeEpCombineTypeClass>
template <auto const& config, auto const& cqeConfig>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::PrepareWrite(GM_ADDR dst, GM_ADDR src, uint64_t len,
                                                                                bool forceCqe)
{
    // Single-SGE WQE occupies 1 WQEBB (64 bytes).
    ReservePreparedWrites(1U);
    const bool needCqe = forceCqe || wqebbSinceCqe_ + 1U > cqeIntervalLimit_;
    if (needCqe) {
        ReservePreparedCqe();
        hcomm_.WriteNbi<cqeConfig>(activeBatchHandle_.handle, dst, src, len);
        ++preparedCqeCount_;
        wqebbSinceCqe_ = 0U;
    } else {
        hcomm_.WriteNbi<config>(activeBatchHandle_.handle, dst, src, len);
        ++wqebbSinceCqe_;
    }
    ++preparedWriteCount_;
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
    PrepareWrite<JETTY_FLAG_WQE_CONFIG, JETTY_FLAG_CQE_WQE_CONFIG>(flagAddr, flagSourceWinAddr_, WIN_ADDR_ALIGN,
                                                                   dstRank == lastRemoteRank_);
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
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::BuildBatchAddrs(uint32_t batchCount,
                                                                                   __ubuf__ uint32_t* metaUb,
                                                                                   __ubuf__ uint64_t* addrUb)
{
    const uint32_t slotStep = perSlotBytes_;
    const uint32_t xStep = axisH_ * sizeof(XType);
    const uint32_t dstTokenStep = topK_ * slotStep;
    if (batchCount == METADATA_BATCH_TOKENS) {
        asc_vf_call<MoeEpCombineVf::BuildBatchAddrs<HasTopkWeight, true>>(metaUb, addrUb, batchCount, slotStep,
                                                                          dstTokenStep, xStep);
    } else {
        asc_vf_call<MoeEpCombineVf::BuildBatchAddrs<HasTopkWeight, false>>(metaUb, addrUb, batchCount, slotStep,
                                                                           dstTokenStep, xStep);
    }
}

template <TemplateMoeEpCombineTypeClass>
template <auto const& config>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::EmitTokenWrite(uint32_t regionOff, uint32_t tokenIdx,
                                                                                  const LocalTensor<uint64_t>& addrUb,
                                                                                  GM_ADDR remoteDataBase, GM_ADDR xBase,
                                                                                  GM_ADDR weightBase,
                                                                                  uint64_t tokenBytes)
{
    GM_ADDR dst = remoteDataBase + addrUb.GetValue(regionOff + tokenIdx);
    GM_ADDR src = xBase + addrUb.GetValue(regionOff + X_OFFSET_REGION + tokenIdx);
    if constexpr (HasTopkWeight == 1) {
        AscendC::BufDesc srcDescs[2] = {
            {src, static_cast<uint32_t>(tokenBytes)},
            {weightBase + addrUb.GetValue(regionOff + WEIGHT_OFFSET_REGION + tokenIdx), sizeof(float)}};
        (void)hcomm_.WriteNbi<config>(activeBatchHandle_.handle, dst, srcDescs, 2U);
    } else {
        (void)hcomm_.WriteNbi<config>(activeBatchHandle_.handle, dst, src, tokenBytes);
    }
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::ConsumeBatchAddrs(const MetadataChunk& chunk,
                                                                                     uint32_t buffer)
{
    if (chunk.count != 0U) {
        BeginPreparedWrites(chunk.rank, jettyIndex_);
        LocalTensor<uint64_t> addrUb = addrPipeBuf_.Get<uint64_t>();
        const uint32_t regionOff = buffer * kPipeAddrElemsPerBatch;
        GM_ADDR remoteDataBase = GetUrmaWinAddrByRankId(chunk.rank, combineDataWinOffset_);
        GM_ADDR xBase = reinterpret_cast<GM_ADDR>(xGm_.GetPhyAddr(0));
        GM_ADDR weightBase = nullptr;
        if constexpr (HasTopkWeight == 1) {
            weightBase = reinterpret_cast<GM_ADDR>(topkWeightsGm_.GetPhyAddr(0));
        }
        const uint64_t tokenBytes = static_cast<uint64_t>(axisH_) * sizeof(XType);
        uint32_t token = 0U;
        while (token < chunk.count) {
            ReservePreparedWrites(kTokenWqebbCount);
            const uint32_t batchCap = (batchCapacity_ - preparedWriteCount_) / kTokenWqebbCount;
            const uint32_t sqCap =
                wqebbSinceCqe_ < cqeIntervalLimit_ ? (cqeIntervalLimit_ - wqebbSinceCqe_) / kTokenWqebbCount : 0U;
            uint32_t run = chunk.count - token;
            run = run < batchCap ? run : batchCap;
            run = run < sqCap ? run : sqCap;
            if (run != 0U) {
                const uint32_t end = token + run;
                for (; token < end; ++token) {
                    EmitTokenWrite<DEFAULT_WQE_CONFIG>(regionOff, token, addrUb, remoteDataBase, xBase, weightBase,
                                                       tokenBytes);
                }
                const uint32_t used = run * kTokenWqebbCount;
                preparedWriteCount_ += used;
                wqebbSinceCqe_ += used;
            } else {
                ReservePreparedCqe();
                EmitTokenWrite<DEFAULT_CQE_WQE_CONFIG>(regionOff, token, addrUb, remoteDataBase, xBase, weightBase,
                                                       tokenBytes);
                ++token;
                preparedWriteCount_ += kTokenWqebbCount;
                ++preparedCqeCount_;
                wqebbSinceCqe_ = 0U;
            }
        }
    }
    if (chunk.last) {
        SendJettyFlag(chunk.rank, jettyIndex_);
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
    uint32_t rank = (jettyIndex_ + peerOffset) % epWorldSize_;
    if (begin == end && rank != rankId_ && actualA_ != 0U) {
        uint64_t rankBegin = static_cast<uint32_t>(rankOffsetsTensor_.GetValue(rank));
        uint64_t rankEnd = static_cast<uint32_t>(rankOffsetsTensor_.GetValue(rank + 1U));
        SplitRange(rankBegin, rankEnd, combineJettyCount_, jettyIndex_, begin, end);
    }
    uint64_t remaining = end - begin;
    uint32_t count = remaining < METADATA_BATCH_TOKENS ? static_cast<uint32_t>(remaining) : METADATA_BATCH_TOKENS;
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
                                                                                    uint32_t buffer)
{
    if (chunk.count == 0U) {
        return;
    }
    LocalTensor<int32_t> metadataLocal = metaPipeBuf_.Get<int32_t>()[buffer * METADATA_BATCH_ELEMS];
    constexpr uint32_t metaBytesPerToken = RECV_META_FIELDS * sizeof(int32_t);
    DataCopyExtParams copyParams{1U, chunk.count * metaBytesPerToken, 0U, 0U, 0U};
    const DataCopyPadExtParams<int32_t> padParams{false, 0U, 0U, 0U};
    DataCopyPad(metadataLocal, recvSrcMetadataGm_[chunk.begin * RECV_META_FIELDS], copyParams, padParams);
    SetFlag<HardEvent::MTE2_V>(evMte2V_[buffer]);
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::PrepareBatchAddrs(const MetadataChunk& chunk,
                                                                                     uint32_t buffer)
{
    if (chunk.count != 0U) {
        WaitFlag<HardEvent::MTE2_V>(evMte2V_[buffer]);
        LocalTensor<int32_t> metaUb = metaPipeBuf_.Get<int32_t>()[buffer * METADATA_BATCH_ELEMS];
        LocalTensor<uint64_t> addrUb = addrPipeBuf_.Get<uint64_t>()[buffer * kPipeAddrElemsPerBatch];
        BuildBatchAddrs(chunk.count, (__ubuf__ uint32_t*)metaUb.GetPhyAddr(), (__ubuf__ uint64_t*)addrUb.GetPhyAddr());
        SetFlag<HardEvent::V_S>(evVS_[buffer]);
    }
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::SendMetadataPipeline()
{
    evMte2V_[0] = static_cast<event_t>(tpipe_->AllocEventID<HardEvent::MTE2_V>());
    evMte2V_[1] = static_cast<event_t>(tpipe_->AllocEventID<HardEvent::MTE2_V>());
    evVS_[0] = static_cast<event_t>(tpipe_->AllocEventID<HardEvent::V_S>());
    evVS_[1] = static_cast<event_t>(tpipe_->AllocEventID<HardEvent::V_S>());
    MetadataChunk chunks[2] = {};
    bool valid[2] = {};
    uint32_t peerOffset = 0U;
    uint64_t begin = 0U;
    uint64_t end = 0U;
    uint32_t current = 0U;
    valid[0] = NextMetadataChunk(peerOffset, begin, end, chunks[0]);
    if (valid[0]) {
        PrefetchMetadata(chunks[0], 0U);
        valid[1] = NextMetadataChunk(peerOffset, begin, end, chunks[1]);
        if (valid[1]) {
            PrefetchMetadata(chunks[1], 1U);
        }
        PrepareBatchAddrs(chunks[0], 0U);
    }
    while (valid[current]) {
        // Keep the peer descriptor before producing k+2 into its ping-pong slot.
        const MetadataChunk chunk = chunks[current];
        if (chunk.count != 0U) {
            WaitFlag<HardEvent::V_S>(evVS_[current]);
        }
        // VF(k) has finished reading meta[k&1], so MTE2(k+2) may reuse it.
        valid[current] = NextMetadataChunk(peerOffset, begin, end, chunks[current]);
        if (valid[current]) {
            PrefetchMetadata(chunks[current], current);
        }
        uint32_t next = current ^ 1U;
        if (valid[next]) {
            PrepareBatchAddrs(chunks[next], next);
        }
        // VF(k+2) starts in the next iteration, after Scalar(k) has consumed addr[k&1].
        ConsumeBatchAddrs(chunk, current);
        current = next;
    }
    tpipe_->ReleaseEventID<HardEvent::MTE2_V>(evMte2V_[0]);
    tpipe_->ReleaseEventID<HardEvent::MTE2_V>(evMte2V_[1]);
    tpipe_->ReleaseEventID<HardEvent::V_S>(evVS_[0]);
    tpipe_->ReleaseEventID<HardEvent::V_S>(evVS_[1]);
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::SendPhaseFromMetadata()
{
    InitFlagSource();
    SyncAll<true>();
    // Cover every logical Jetty even when the device has fewer AIVs than Jettys.
    for (jettyIndex_ = aivId_; jettyIndex_ < combineJettyCount_; jettyIndex_ += GetBlockNum()) {
        activeBatchHandle_ = {};
        activeJettyHandle_ = 0U;
        preparedWriteCount_ = 0U;
        preparedCqeCount_ = 0U;
        activeBatchInitialized_ = false;
        // The final remote flag covers this Jetty's entire stage, including empty peer slices.
        lastRemoteRank_ = (jettyIndex_ + epWorldSize_ - 1U) % epWorldSize_;
        if (lastRemoteRank_ == rankId_) {
            lastRemoteRank_ = (lastRemoteRank_ + epWorldSize_ - 1U) % epWorldSize_;
        }

        // Prefetch across peer boundaries too; typical peer slices fit in a single metadata chunk.
        SendMetadataPipeline();
        FlushPreparedWrites();
    }
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
