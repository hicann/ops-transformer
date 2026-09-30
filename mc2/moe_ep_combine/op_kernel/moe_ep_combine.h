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
 * \file moe_ep_combine.h
 * \brief MoE Expert-Parallel Combine kernel implementation
 */
#ifndef MOE_EP_COMBINE_H
#define MOE_EP_COMBINE_H

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

#include "moe_ep_combine_base.h"
#include "moe_ep_combine_tiling.h"
#include "moe_ep_combine_vf.h"

namespace MoeEpCombineImpl {

#if defined(ENABLE_MOE_EP_COMBINE_KERNEL)

using namespace AscendC;
using namespace MoeEpCombineLayout;

#define TemplateMoeEpCombineTypeClass typename XType, uint32_t HasTopkWeight
#define TemplateMoeEpCombineTypeFunc XType, HasTopkWeight
#define HCOMM_INIT_SIZE 512UL

static constexpr uint32_t WIN_ADDR_ALIGN = 512;
// SQ holds 32768 WQEBBs. The token that pushes the pending count past 32767 carries a CQE and drains
// (drain fires exactly when the count reaches 32768, for both the 1-WQEBB and 2-WQEBB token paths).
static constexpr uint32_t HCOMM_SQ_MAX_PENDING = 32767U;
// PR 111 requires each committed batch to contain fewer WQEBBs than the SQ depth.
static constexpr uint32_t HCOMM_BATCH_CAPACITY = 256;
static constexpr uint32_t HCOMM_PLAIN_WRITE_WQE_BYTES = 64;
static constexpr uint32_t HCOMM_BATCH_BUFFER_BYTES = HCOMM_BATCH_CAPACITY * HCOMM_PLAIN_WRITE_WQE_BYTES;
constexpr uint64_t UB_ALIGN = 32UL;
static constexpr struct UrmaWqeEntry DEFAULT_WQE_CONFIG = {.odr = 5, .fence = 1, .se = 0, .cqe = 0, .inlineEn = 0};
static constexpr struct UrmaWqeEntry DEFAULT_CQE_WQE_CONFIG = {.odr = 5, .fence = 1, .se = 0, .cqe = 1, .inlineEn = 0};
static constexpr struct UrmaWqeEntry CHANNEL_FLAG_WQE_CONFIG = {.odr = 6, .fence = 1, .se = 0, .cqe = 0, .inlineEn = 0};
template <TemplateMoeEpCombineTypeClass>
class MoeEpCombine {
public:
    __aicore__ inline MoeEpCombine(){};

    __aicore__ inline void Init(GM_ADDR context, GM_ADDR x, GM_ADDR topkIdx, GM_ADDR recvSrcMetadata,
                                GM_ADDR numRecvPerExpert, GM_ADDR topkWeights, GM_ADDR workspace, GM_ADDR tilingGM,
                                TPipe* pipe, const MoeEpCombineInfo* tilingData);

    __aicore__ inline void Process();

private:
    __aicore__ inline void SendChannelFlag(uint32_t dstRank, uint32_t channelIndex);
    __aicore__ inline void EnsureHcommInitialized();
    __aicore__ inline void BeginPreparedWrites(uint32_t dstRank, uint32_t channelIndex);
    __aicore__ inline void InitFlagSource();
    template <auto const& config>
    __aicore__ inline void PrepareWrite(GM_ADDR dst, GM_ADDR src, uint64_t len);
    __aicore__ inline void FlushPreparedWrites(bool keepHandle = false);
    __aicore__ inline void SplitRange(uint64_t rangeBegin, uint64_t rangeEnd, uint32_t coreCount, uint32_t coreIndex,
                                      uint64_t& coreBegin, uint64_t& coreEnd);
    __aicore__ inline void GetCoreAssignment(uint32_t totalBlocks, uint32_t& targetRank, uint32_t& coreIndexInGroup,
                                             uint32_t& groupSize);
    __aicore__ inline void ProcessRemoteMetadataRange(uint32_t targetRank, uint64_t rangeBegin, uint64_t rangeEnd,
                                                      uint32_t channelIndex);
    __aicore__ inline uint32_t PipeBatchCount(uint64_t totalCount, uint32_t batchIdx) const;
    __aicore__ inline void CopyMetaBatch(uint64_t globalBegin, uint32_t batchCount, uint32_t bufIdx, event_t evId);
    __aicore__ inline void BuildBatchAddrs(uint32_t batchCount, __ubuf__ uint32_t* metaUb, __ubuf__ uint64_t* addrUb);
    template <auto const& config>
    __aicore__ inline void EmitTokenWrite(uint32_t regionOff, uint32_t tokenIdx, const LocalTensor<uint64_t>& addrUb,
                                          GM_ADDR remoteDataBase, GM_ADDR xBase, GM_ADDR weightBase,
                                          uint64_t tokenBytes);
    __aicore__ inline void ConsumeBatchAddrs(uint32_t batchCount, uint64_t globalBegin, uint64_t rangeEnd,
                                             uint32_t regionOff, const LocalTensor<uint64_t>& addrUb,
                                             GM_ADDR remoteDataBase, GM_ADDR xBase, GM_ADDR weightBase,
                                             uint64_t tokenBytes, uint32_t& sqBudget, uint32_t& wqebbBudget);
    __aicore__ inline void SendPhaseDirectFromMetadata();

    __aicore__ inline uint64_t GetCommHandle(uint32_t rankId, uint32_t channelIndex)
    {
        return mc2Context_->hcommHandle[rankId * channelsPerRank_ + channelIndex];
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
    __gm__ Mc2Aclnn::MoeCommContext* mc2Context_{nullptr};
    MoeEpExceptionDump::MoeEpCoreDiagWriter diagWriter_;

    uint32_t rankId_{0};
    uint32_t epWorldSize_{0};
    uint32_t channelsPerRank_{1};
    uint32_t combineChannelCount_{1};
    uint32_t numMaxTokensPerRank_{0};
    uint32_t topK_{0};
    uint32_t axisH_{0};
    uint32_t hAlignSize_{0};
    uint64_t combineStateWinOffset_{0};
    uint64_t combineDataWinOffset_{0};

    // WQEBBs occupied per token write: multi-SGE (x + weight) rounds up to 2, single-SGE is 1.
    static constexpr uint32_t kTokenWqebbCount = (HasTopkWeight == 1) ? 2U : 1U;
    static constexpr uint32_t kPipeAddrElemsPerBatch = METADATA_BATCH_TOKENS * ((HasTopkWeight == 1) ? 3U : 2U);

    uint32_t perSlotBytes_{0};
    uint64_t actualA_{0};
    uint64_t recvCapacity_{0};
    uint32_t aivNum_{0};

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

    TBuf<> metaPipeBuf_; // Double-buffered UB batches of recvSrcMetadata.
    TBuf<> addrPipeBuf_; // Double-buffered token send/receive byte offsets generated by the VF.

    AscendC::Hcomm<COMM_PROTOCOL_UBC_CTP> hcomm_; // Communication context.
    using HcommBatchHandle = AscendC::BatchHandle<AscendC::ChannelHandle>;

    GM_ADDR winRankAddr_[Mc2Aclnn::HCCL_MAX_RANK_SIZE];
    GM_ADDR flagSourceWinAddr_{nullptr};
    uint32_t aivId_{0};
    HcommBatchHandle activeBatchHandle_{};
    uint64_t activeBatchChannel_{0};
    uint32_t preparedWriteCount_{0};
    bool hcommInitialized_{false};
    bool activeBatchInitialized_{false};
    // Reserve two event IDs per ring with AllocEventID during Init. FetchEventID does not reserve an ID,
    // so calling it twice could return the same ID.
    event_t evMte2V_[2] = {};
    event_t evVS_[2] = {};
};

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::Init(GM_ADDR context, GM_ADDR x, GM_ADDR topkIdx,
                                                                        GM_ADDR recvSrcMetadata,
                                                                        GM_ADDR numRecvPerExpert, GM_ADDR topkWeights,
                                                                        GM_ADDR workspace, GM_ADDR tilingGM,
                                                                        TPipe* pipe, const MoeEpCombineInfo* tilingData)
{
    tpipe_ = pipe;
    tilingData_ = tilingData;
    aivId_ = GetBlockIdx();
    (void)topkIdx;
    (void)numRecvPerExpert;
    (void)workspace;
    epWorldSize_ = tilingData_->cfg.epWorldSize;
    numMaxTokensPerRank_ = tilingData_->cfg.numMaxTokensPerRank;
    topK_ = tilingData_->cfg.topK;
    axisH_ = tilingData_->cfg.hidden;
    hAlignSize_ = Ceil(axisH_ * sizeof(XType), UB_ALIGN) * UB_ALIGN;
    perSlotBytes_ = tilingData_->cfg.perSlotBytes;
    aivNum_ = tilingData_->aivNum;
    recvCapacity_ = tilingData_->recvCapacity;
    tpipe_->InitBuffer(hcommBuf_, HCOMM_INIT_SIZE);
    tpipe_->InitBuffer(hcommBatchBuf_, HCOMM_BATCH_BUFFER_BYTES);

    mc2Context_ = reinterpret_cast<__gm__ Mc2Aclnn::MoeCommContext*>(context);
    rankId_ = mc2Context_->epRankId;
    constexpr size_t metadataOffset =
        offsetof(MoeEpCombineTilingData, moeEpCombineInfo) + offsetof(MoeEpCombineInfo, dumpMetadata);
    MoeEpExceptionDump::WriteMetadata(context, tilingGM + metadataOffset);
    diagWriter_.Init(context, MOE_EP_CORE_DIAG_COMBINE, tpipe_);
    channelsPerRank_ = mc2Context_->channelsPerRank;
    if (channelsPerRank_ == 0U) {
        channelsPerRank_ = 1U;
    }

    combineChannelCount_ = channelsPerRank_;

    for (uint32_t i = 0; i < epWorldSize_; ++i) {
        winRankAddr_[i] = (GM_ADDR)mc2Context_->epHcclBuffer[i];
    }

    combineStateWinOffset_ = tilingData->combineStateWinOffset;
    combineDataWinOffset_ = tilingData->combineDataWinOffset;
    flagSourceWinAddr_ = GetUrmaStateAddrByRankId(rankId_, tilingData->combineFlagSourceWinOffset);

    xGm_.SetGlobalBuffer((__gm__ XType*)x);
    recvSrcMetadataGm_.SetGlobalBuffer((__gm__ int32_t*)recvSrcMetadata);
    recvRankOffsetsGm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t*>(recvSrcMetadata + tilingData->metadataRankOffsetsOffset));

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
    evMte2V_[0] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
    evMte2V_[1] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
    evVS_[0] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_S>());
    evVS_[1] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_S>());
    tpipe_->InitBuffer(readStateBuf_, WIN_ADDR_ALIGN);
    statusTensor_ = readStateBuf_.Get<uint32_t>();
    diagWriter_.RunPosRecord(MOE_EP_COMBINE_RUN_POS_INIT_DONE);
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::EnsureHcommInitialized()
{
    if (hcommInitialized_) {
        return;
    }
    hcommTensor_ = hcommBuf_.Get<uint8_t>();
    hcomm_.Init(hcommTensor_, HCOMM_INIT_SIZE);
    hcommBatchTensor_ = hcommBatchBuf_.Get<uint8_t>();
    Duplicate<uint8_t>(hcommBatchTensor_, 0U, HCOMM_BATCH_BUFFER_BYTES);
    SyncFunc<AscendC::HardEvent::V_S>();
    hcommInitialized_ = true;
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::FlushPreparedWrites(bool keepHandle)
{
    if (preparedWriteCount_ != 0) {
        (void)hcomm_.BatchCommit(activeBatchHandle_);
        preparedWriteCount_ = 0;
    }
    if (!keepHandle) {
        activeBatchHandle_ = {};
        activeBatchChannel_ = 0;
        activeBatchInitialized_ = false;
    }
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::BeginPreparedWrites(uint32_t dstRank,
                                                                                       uint32_t channelIndex)
{
    EnsureHcommInitialized();
    uint64_t commHandle = GetCommHandle(dstRank, channelIndex);
    if (activeBatchInitialized_ && activeBatchChannel_ == commHandle) {
        return;
    }
    if (activeBatchInitialized_) {
        FlushPreparedWrites();
    }
    activeBatchHandle_ =
        hcomm_.MakeBatchHandle(commHandle, hcommBatchTensor_, HCOMM_BATCH_BUFFER_BYTES, winRankAddr_[dstRank]);
    activeBatchChannel_ = commHandle;
    activeBatchInitialized_ = true;
}

template <TemplateMoeEpCombineTypeClass>
template <auto const& config>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::PrepareWrite(GM_ADDR dst, GM_ADDR src, uint64_t len)
{
    // Single-SGE WQE occupies 1 WQEBB (64 bytes).
    if (preparedWriteCount_ + 1U > HCOMM_BATCH_CAPACITY) {
        FlushPreparedWrites(true);
    }
    (void)hcomm_.WriteNbi<config>(activeBatchHandle_, dst, src, len);
    ++preparedWriteCount_;
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::SendChannelFlag(uint32_t dstRank,
                                                                                   uint32_t channelIndex)
{
    uint64_t flagIndex = static_cast<uint64_t>(rankId_) * combineChannelCount_ + channelIndex;
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
    BeginPreparedWrites(dstRank, channelIndex);
    PrepareWrite<CHANNEL_FLAG_WQE_CONFIG>(flagAddr, flagSourceWinAddr_, WIN_ADDR_ALIGN);
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
__aicore__ inline uint32_t MoeEpCombine<TemplateMoeEpCombineTypeFunc>::PipeBatchCount(uint64_t totalCount,
                                                                                      uint32_t batchIdx) const
{
    uint64_t left = totalCount - static_cast<uint64_t>(batchIdx) * METADATA_BATCH_TOKENS;
    return (left > METADATA_BATCH_TOKENS) ? METADATA_BATCH_TOKENS : static_cast<uint32_t>(left);
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::CopyMetaBatch(uint64_t globalBegin,
                                                                                 uint32_t batchCount, uint32_t bufIdx,
                                                                                 event_t evId)
{
    LocalTensor<int32_t> metaUb = metaPipeBuf_.Get<int32_t>()[bufIdx * METADATA_BATCH_ELEMS];
    DataCopyExtParams copyParams{1U, static_cast<uint32_t>(batchCount * RECV_META_FIELDS * sizeof(int32_t)), 0U, 0U,
                                 0U};
    DataCopyPadExtParams<int32_t> padParams{false, 0U, 0U, 0U};
    DataCopyPad(metaUb, recvSrcMetadataGm_[globalBegin * RECV_META_FIELDS], copyParams, padParams);
    SetFlag<HardEvent::MTE2_V>(evId);
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::BuildBatchAddrs(uint32_t batchCount,
                                                                                   __ubuf__ uint32_t* metaUb,
                                                                                   __ubuf__ uint64_t* addrUb)
{
    // Host limits: topK <= 32, hidden <= 8192, sizeof(XType) == 2, perSlotBytes <= 16896.
    // Scalar strides fit u32; the VF retains u64 token-dependent byte offsets.
    const uint32_t slotStep = perSlotBytes_;
    const uint32_t xStep = axisH_ * sizeof(XType);
    const uint32_t dstTokenStep = topK_ * slotStep;
    // Select once on the scalar side; each batch launches exactly one VF.
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
    // ConsumeBatchAddrs accounts for WQEBBs once per run (or once for a CQE token).
    GM_ADDR dst = remoteDataBase + addrUb.GetValue(regionOff + tokenIdx);
    GM_ADDR src = xBase + addrUb.GetValue(regionOff + X_OFFSET_REGION + tokenIdx);
    if constexpr (HasTopkWeight == 1) {
        AscendC::BufDesc srcDescs[2] = {
            {src, hAlignSize_},
            {weightBase + addrUb.GetValue(regionOff + WEIGHT_OFFSET_REGION + tokenIdx), sizeof(float)}};
        (void)hcomm_.WriteNbi<config>(activeBatchHandle_, dst, srcDescs, 2U);
    } else {
        (void)hcomm_.WriteNbi<config>(activeBatchHandle_, dst, src, tokenBytes);
    }
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::ConsumeBatchAddrs(
    uint32_t batchCount, uint64_t globalBegin, uint64_t rangeEnd, uint32_t regionOff,
    const LocalTensor<uint64_t>& addrUb, GM_ADDR remoteDataBase, GM_ADDR xBase, GM_ADDR weightBase, uint64_t tokenBytes,
    uint32_t& sqBudget, uint32_t& wqebbBudget)
{
    const bool batchIsLast = (globalBegin + batchCount == rangeEnd);
    const uint32_t plainEnd = batchIsLast ? (batchCount - 1U) : batchCount;
    uint32_t t = 0U;
    while (t < batchCount) {
        if (wqebbBudget < kTokenWqebbCount) {
            FlushPreparedWrites(true);
            wqebbBudget = HCOMM_BATCH_CAPACITY;
        }
        const uint32_t sqCap = sqBudget / kTokenWqebbCount;
        const uint32_t batchCap = wqebbBudget / kTokenWqebbCount;
        uint32_t run = plainEnd - t;
        run = (run < sqCap) ? run : sqCap;
        run = (run < batchCap) ? run : batchCap;
        if (run != 0U) {
            // The whole run fits both budgets: no capacity checks or counter updates per token.
            for (uint32_t i = 0; i < run; ++i) {
                EmitTokenWrite<DEFAULT_WQE_CONFIG>(regionOff, t + i, addrUb, remoteDataBase, xBase, weightBase,
                                                   tokenBytes);
            }
            const uint32_t used = run * kTokenWqebbCount;
            preparedWriteCount_ += used;
            wqebbBudget -= used;
            sqBudget -= used;
            t += run;
        } else {
            // Only the range's last token or the token filling the SQ requests a CQE.
            const bool drainNow = (sqBudget < kTokenWqebbCount);
            EmitTokenWrite<DEFAULT_CQE_WQE_CONFIG>(regionOff, t, addrUb, remoteDataBase, xBase, weightBase, tokenBytes);
            preparedWriteCount_ += kTokenWqebbCount;
            wqebbBudget -= kTokenWqebbCount;
            if (drainNow) {
                FlushPreparedWrites(true);
                (void)hcomm_.Drain(activeBatchChannel_);
                sqBudget = HCOMM_SQ_MAX_PENDING;
                wqebbBudget = HCOMM_BATCH_CAPACITY;
            } else {
                sqBudget -= kTokenWqebbCount;
            }
            ++t;
        }
    }
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::ProcessRemoteMetadataRange(uint32_t targetRank,
                                                                                              uint64_t rangeBegin,
                                                                                              uint64_t rangeEnd,
                                                                                              uint32_t channelIndex)
{
    if (rangeBegin >= rangeEnd) {
        return;
    }
    BeginPreparedWrites(targetRank, channelIndex);
    // Three-stage software pipeline: MTE2 copies metadata, the VF builds offsets, and scalar code builds WQEs.
    // Metadata and offsets each use two buffers. MTE2_V signals metadata readiness; V_S signals offset readiness.
    // Each event type uses two IDs selected by batch parity. Scalar program order protects buffer reuse:
    // WaitFlag<V_S>(k) confirms VF(k) has finished reading metadata before MTE2(k+2) overwrites meta[k&1];
    // VF(k+2) is launched only after scalar stage S(k) has finished reading addr[k&1].
    LocalTensor<int32_t> metaAll = metaPipeBuf_.Get<int32_t>();
    LocalTensor<uint64_t> addrAll = addrPipeBuf_.Get<uint64_t>();
    __ubuf__ uint32_t* metaBase = (__ubuf__ uint32_t*)metaAll.GetPhyAddr();
    __ubuf__ uint64_t* addrBase = (__ubuf__ uint64_t*)addrAll.GetPhyAddr();
    GM_ADDR remoteDataBase = GetUrmaWinAddrByRankId(targetRank, combineDataWinOffset_);
    GM_ADDR xBase = reinterpret_cast<GM_ADDR>(xGm_.GetPhyAddr(0));
    GM_ADDR weightBase = nullptr;
    if constexpr (HasTopkWeight == 1) {
        weightBase = reinterpret_cast<GM_ADDR>(topkWeightsGm_.GetPhyAddr(0));
    }
    uint64_t tokenBytes = static_cast<uint64_t>(axisH_) * sizeof(XType);
    uint32_t sqBudget = HCOMM_SQ_MAX_PENDING;
    uint32_t wqebbBudget = HCOMM_BATCH_CAPACITY;

    const uint64_t totalCount = rangeEnd - rangeBegin;
    const uint32_t batchNum = static_cast<uint32_t>((totalCount + METADATA_BATCH_TOKENS - 1U) / METADATA_BATCH_TOKENS);
    // Prime both metadata buffers and launch the first address batch.
    if (batchNum > 0U) {
        CopyMetaBatch(rangeBegin, PipeBatchCount(totalCount, 0U), 0U, evMte2V_[0]);
        if (batchNum > 1U) {
            CopyMetaBatch(rangeBegin + METADATA_BATCH_TOKENS, PipeBatchCount(totalCount, 1U), 1U, evMte2V_[1]);
        }
        WaitFlag<HardEvent::MTE2_V>(evMte2V_[0]);
        BuildBatchAddrs(PipeBatchCount(totalCount, 0U), metaBase, addrBase);
        SetFlag<HardEvent::V_S>(evVS_[0]);
    }
    // Steady state: wait VF(k), prefetch metadata(k+2), launch VF(k+1), then consume addresses(k).
    for (uint32_t k = 0U; k < batchNum; ++k) {
        uint32_t buf = k & 1U;
        uint32_t nextBuf = (k + 1U) & 1U;
        uint32_t cnt = PipeBatchCount(totalCount, k);
        WaitFlag<HardEvent::V_S>(evVS_[buf]);
        if (k + 2U < batchNum) {
            CopyMetaBatch(rangeBegin + (k + 2U) * METADATA_BATCH_TOKENS, PipeBatchCount(totalCount, k + 2U), buf,
                          evMte2V_[buf]);
        }
        if (k + 1U < batchNum) {
            WaitFlag<HardEvent::MTE2_V>(evMte2V_[nextBuf]);
            BuildBatchAddrs(PipeBatchCount(totalCount, k + 1U), metaBase + nextBuf * METADATA_BATCH_ELEMS,
                            addrBase + nextBuf * kPipeAddrElemsPerBatch);
            SetFlag<HardEvent::V_S>(evVS_[nextBuf]);
        }
        ConsumeBatchAddrs(cnt, rangeBegin + k * METADATA_BATCH_TOKENS, rangeEnd, buf * kPipeAddrElemsPerBatch, addrAll,
                          remoteDataBase, xBase, weightBase, tokenBytes, sqBudget, wqebbBudget);
    }
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::GetCoreAssignment(uint32_t totalBlocks,
                                                                                     uint32_t& targetRank,
                                                                                     uint32_t& coreIndexInGroup,
                                                                                     uint32_t& groupSize)
{
    uint32_t maxChannelAivNum = epWorldSize_ * combineChannelCount_;
    bool hasExtraAivs = totalBlocks > maxChannelAivNum;
    if (hasExtraAivs) {
        // Put all remote communication AIVs first; the remaining AIVs only publish local completion flags.
        uint32_t remoteAivNum = (epWorldSize_ - 1U) * combineChannelCount_;
        if (aivId_ < remoteAivNum) {
            uint32_t remoteRankIndex = aivId_ / combineChannelCount_;
            targetRank = remoteRankIndex < rankId_ ? remoteRankIndex : remoteRankIndex + 1U;
            coreIndexInGroup = aivId_ % combineChannelCount_;
            groupSize = combineChannelCount_;
        } else {
            targetRank = rankId_;
            coreIndexInGroup = aivId_ - remoteAivNum;
            groupSize = totalBlocks - remoteAivNum;
        }
        return;
    }

    uint32_t baseGroupSize = totalBlocks / epWorldSize_;
    uint32_t remainder = totalBlocks % epWorldSize_;
    uint32_t accumulated = 0;
    for (uint32_t rank = 0; rank < epWorldSize_; ++rank) {
        uint32_t currentGroupSize = baseGroupSize + ((rank < remainder) ? 1U : 0U);
        if (aivId_ < accumulated + currentGroupSize) {
            targetRank = rank;
            groupSize = currentGroupSize;
            coreIndexInGroup = aivId_ - accumulated;
            return;
        }
        accumulated += currentGroupSize;
    }
    targetRank = epWorldSize_;
    groupSize = 0;
    coreIndexInGroup = 0;
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::SendPhaseDirectFromMetadata()
{
    if (epWorldSize_ == 0U || aivNum_ == 0U) {
        return;
    }

    uint32_t activeAivNum = aivNum_;
    activeBatchHandle_ = {};
    activeBatchChannel_ = 0U;
    preparedWriteCount_ = 0U;
    activeBatchInitialized_ = false;

    // Match upstream's rank/channel owners, including the local flag group and low-AIV rank stride.
    bool splitRankTokens = activeAivNum >= epWorldSize_;
    uint32_t targetRank = epWorldSize_;
    uint32_t coreIndexInGroup = 0U;
    uint32_t groupSize = 1U;
    if (splitRankTokens && aivId_ < activeAivNum) {
        GetCoreAssignment(activeAivNum, targetRank, coreIndexInGroup, groupSize);
    }
    bool sendsTokens = actualA_ != 0U && aivId_ < activeAivNum;
    if (sendsTokens) {
        if (splitRankTokens) {
            // Local payload is consumed directly by combine epilogue; combine only sends remote-rank entries.
            if (targetRank != rankId_) {
                uint64_t rankBegin = static_cast<uint32_t>(rankOffsetsTensor_.GetValue(targetRank));
                uint64_t rankEnd = static_cast<uint32_t>(rankOffsetsTensor_.GetValue(targetRank + 1U));
                uint64_t entryBegin = 0U;
                uint64_t entryEnd = 0U;
                SplitRange(rankBegin, rankEnd, groupSize, coreIndexInGroup, entryBegin, entryEnd);
                ProcessRemoteMetadataRange(targetRank, entryBegin, entryEnd, coreIndexInGroup);
            }
        } else {
            for (uint32_t rank = aivId_; rank < epWorldSize_; rank += activeAivNum) {
                if (rank == rankId_) {
                    continue;
                }
                uint64_t rankBegin = static_cast<uint32_t>(rankOffsetsTensor_.GetValue(rank));
                uint64_t rankEnd = static_cast<uint32_t>(rankOffsetsTensor_.GetValue(rank + 1U));
                ProcessRemoteMetadataRange(rank, rankBegin, rankEnd, 0U);
            }
        }
    }

    InitFlagSource();
    // Publish AIV0's shared constant-source initialization before any remote flag is submitted.
    SyncAll<true>();

    // Match upstream: publish flags after this AIV's payload phase, even for empty rank/channel slices.
    // In the low-AIV branch the same strided rank list is visited again, using channel 0 only.
    bool publishesChannelFlag = aivId_ < activeAivNum && (!splitRankTokens || coreIndexInGroup < combineChannelCount_);
    if (publishesChannelFlag) {
        if (splitRankTokens) {
            SendChannelFlag(targetRank, coreIndexInGroup);
        } else {
            for (uint32_t dstRank = aivId_; dstRank < epWorldSize_; dstRank += activeAivNum) {
                SendChannelFlag(dstRank, 0U);
            }
        }
    }
    // Submit pending writes; epilogue drains the token send completions.
    FlushPreparedWrites();
    // Publish channel CQ counters for epilogue.
    DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(recvSrcMetadataGm_);
    diagWriter_.RunPosRecord(MOE_EP_COMBINE_RUN_POS_URMA_REQUESTS_ISSUE_DONE);
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::Process()
{
    SendPhaseDirectFromMetadata();
}

#endif

} // namespace MoeEpCombineImpl

#endif // MOE_EP_COMBINE_H
