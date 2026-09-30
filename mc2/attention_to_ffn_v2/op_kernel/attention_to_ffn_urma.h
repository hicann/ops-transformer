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
 * \file attention_to_ffn_urma.h
 * \brief AttentionToFfnV2 URMA implementation.
 */
#ifndef ATTENTION_TO_FFN_URMA_H
#define ATTENTION_TO_FFN_URMA_H

#if __has_include("version/asc_devkit_version.h") && __has_include("version/hcomm_version.h")
#include "version/asc_devkit_version.h"
#include "version/hcomm_version.h"

#if (ASC_DEVKIT_MAJOR > 9 || (ASC_DEVKIT_MAJOR == 9 && ASC_DEVKIT_MINOR > 1)) && \
    (HCOMM_MAJOR > 9 || (HCOMM_MAJOR == 9 && HCOMM_MINOR > 1))
#define ENABLE_ATTENTION_TO_FFN_V2_KERNEL
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
#include "adv_api/reduce/sum.h"
#include "adv_api/reduce/reduce.h"
#include "attention_to_ffn_v2_tiling.h"

#include "../../common/op_kernel/attention_ffn_context.h"
#include "../../common/op_kernel/mc2_kernel_utils.h"

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
#include "../../common/op_kernel/quantize_functions.h"
#endif

#ifndef FLOAT_OVERFLOW_MODE_CTRL
#define FLOAT_OVERFLOW_MODE_CTRL 60
#endif

namespace AttentionToFFNImpl {

#if defined(ENABLE_ATTENTION_TO_FFN_V2_KERNEL)

#ifndef ATTN_FFN_URMA_SHARED_CONSTANTS
#define ATTN_FFN_URMA_SHARED_CONSTANTS
constexpr uint8_t BUFFER_NUM = 2;
constexpr uint32_t UB_ALIGN = 32;
constexpr uint8_t WIN_OFFSET_CNT = 2;
constexpr uint32_t SCALE_PARAM_PAD_SIZE = 128;
constexpr uint32_t WIN_ALIGN = 512;
constexpr uint32_t REP_STRIDE = 8;
constexpr uint32_t WORKSPACE_ELEMENT_OFFSET = 128;
constexpr uint32_t DYNAMIC_QUANT = 2;
constexpr uint32_t MX_QUANT = 3;
constexpr uint32_t MX_CLIP_QUANT = 4;
constexpr uint32_t RANK_OFFSET_STRIDE = 2;
constexpr uint32_t TOKEN_INFO_TABLE_RS = 2;
constexpr uint32_t TOKEN_INFO_TABLE_COPY_BLOCK_CNT = 2;
constexpr float INT8_MAX_VALUE = 127.0f;
constexpr uint32_t FP4_ELEMS_PER_BYTE = 2;
constexpr uint32_t MX_BLOCK_SIZE = 32;
constexpr uint32_t PERGROUP_BLOCK_SIZE = 128;

__aicore__ constexpr bool IsMxQuant(uint32_t qm)
{
    return qm == MX_QUANT;
}
__aicore__ constexpr bool IsMxClipQuant(uint32_t qm)
{
    return qm == MX_CLIP_QUANT;
}
__aicore__ constexpr bool IsMxOrMxClipQuant(uint32_t qm)
{
    return qm == MX_QUANT || qm == MX_CLIP_QUANT;
}
#endif // ATTN_FFN_URMA_SHARED_CONSTANTS

constexpr uint32_t URMA_HCOMM_INIT_SIZE = 512U;
constexpr uint32_t URMA_FLAG_SLOT_SIZE = 32U;
constexpr int32_t URMA_FLAG_VALUE = 1;

constexpr uint32_t HCOMM_BATCH_WQE_BYTES = 64U;
constexpr uint32_t ATTN_FFN_HCOMM_BATCH_UB_BYTES = 16U * 1024U;
constexpr uint32_t URMA_BATCH_WQE_CAPACITY = ATTN_FFN_HCOMM_BATCH_UB_BYTES / HCOMM_BATCH_WQE_BYTES;
constexpr uint32_t HCOMM_SQ_MAX_PENDING = 32767U;

// Compact per-rank token tables (counting-sort layout) for the relay phase. rankInfoBuf_ layout:
// [bucket fill counts | running write prefix | per-rank totals | table start], one row each.
constexpr uint32_t RANK_INFO_SEGMENT_NUM = 4U;
constexpr uint32_t RANK_INFO_COUNT_SEG_IDX = 0U;
constexpr uint32_t RANK_INFO_PREFIX_SEG_IDX = 1U;
constexpr uint32_t RANK_INFO_TOTAL_SEG_IDX = 2U;
constexpr uint32_t RANK_INFO_START_SEG_IDX = 3U;
constexpr uint32_t BUCKET_BUFFER_SIZE = 32U * 1024U;
constexpr uint32_t BUCKET_GROUP_TOKEN_MAX = 256U;

static constexpr AscendC::UrmaWqeEntry REMOTE_DATA_WQE_CONFIG = {.odr = 5, .fence = 1, .se = 0, .inlineEn = 0};
static constexpr AscendC::UrmaWqeEntry REMOTE_ORDERED_FLAG_WQE_CONFIG = {.odr = 6, .fence = 1, .se = 1, .inlineEn = 0};
static constexpr AscendC::UrmaWqeEntry REMOTE_DATA_CQE_WQE_CONFIG = {
    .odr = 5, .fence = 1, .se = 0, .cqe = 1, .inlineEn = 0};
static constexpr AscendC::UrmaWqeEntry REMOTE_ORDERED_FLAG_CQE_WQE_CONFIG = {
    .odr = 6, .fence = 1, .se = 1, .cqe = 1, .inlineEn = 0};
constexpr uint32_t FLAG_ELEM_STRIDE = URMA_FLAG_SLOT_SIZE / sizeof(int32_t);
using HcommBatchHandle = AscendC::BatchHandle<AscendC::ChannelHandle>;

struct AttentionToFfnTokenMetaData {
    uint32_t tokenId;
    uint32_t topkId;
    int32_t dstExpertId;
    int32_t toRankId;
    int32_t localExpId;
    GM_ADDR remoteDataAddr;
    GM_ADDR remoteFlagAddr;
    GM_ADDR localDataAddr;
};

#define TemplateAttentionToFfnUrmaTypeClass \
    typename XType, typename XOutType, uint32_t QuantMode, bool isSync, bool isActiveMask
#define TemplateAttentionToFfnUrmaTypeFunc XType, XOutType, QuantMode, isSync, isActiveMask

using namespace AscendC;
template <TemplateAttentionToFfnUrmaTypeClass>
class AttentionToFfnUrma {
public:
    using StorageXOutType = typename std::conditional<(Std::IsSame<XOutType, fp4x2_e2m1_t>::value) ||
                                                          (Std::IsSame<XOutType, fp4x2_e1m2_t>::value),
                                                      uint8_t, XOutType>::type;
    using StorageXInType = typename std::conditional<
        (Std::IsSame<XType, fp4x2_e2m1_t>::value) || (Std::IsSame<XType, fp4x2_e1m2_t>::value), uint8_t, XType>::type;

    __aicore__ inline AttentionToFfnUrma(){};
    __aicore__ inline void Init(GM_ADDR mc2Context, GM_ADDR x, GM_ADDR sessionId, GM_ADDR microBatchId, GM_ADDR layerId,
                                GM_ADDR expertIds, GM_ADDR expertRankTable, GM_ADDR scales, GM_ADDR active_mask,
                                GM_ADDR workspaceGM, TPipe* pipe, const AttentionToFfnV2TilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void HcommInit();
    __aicore__ inline void ReadTokenMetaData(AttentionToFfnTokenMetaData& metaData, uint32_t tokenOffset);
    __aicore__ inline GM_ADDR GetWindowAddr(int32_t rankId);
    __aicore__ inline uint64_t GetUrmaCommHandle(uint32_t dstRank, uint32_t channelIndex = 0U);
    __aicore__ inline void CopyTokenDataToGM(const AttentionToFfnTokenMetaData& metaData, GM_ADDR targetAddr,
                                             const DataCopyExtParams& xCopyParams);
    __aicore__ inline void SendToLocal(const AttentionToFfnTokenMetaData& metaData,
                                       const DataCopyExtParams& xCopyParams);
    __aicore__ inline void StageRemoteData(const AttentionToFfnTokenMetaData& metaData,
                                           const DataCopyExtParams& xCopyParams, uint32_t tokenOffset);
    __aicore__ inline void AppendRemoteDataWrite(HcommBatchHandle& batchHandle, GM_ADDR remoteDataAddr,
                                                 GM_ADDR sourceDataAddr, bool enableCqe);
    __aicore__ inline void AppendRemoteFlagWrite(HcommBatchHandle& batchHandle, GM_ADDR remoteFlagAddr,
                                                 uint32_t sourceTokenOffset, bool enableCqe);
    __aicore__ inline void CommitAndDrainChannel(HcommBatchHandle& batchHandle, uint64_t channel,
                                                 uint32_t& preparedWqeCount, uint32_t& sqWriteCount);
    __aicore__ inline void BuildRankTableOffsets();
    __aicore__ inline void ScatterTokensToRankTables();
    __aicore__ inline void RelayRemoteTokens(uint32_t dstRank, uint32_t channelIdx);
    __aicore__ inline void FindExpertRank(int32_t expertId);
    __aicore__ inline void QuantInit(GM_ADDR scales);
    __aicore__ inline void QuantProcess(uint32_t expertIndex);
    __aicore__ inline void ReduceMaxInplace(const LocalTensor<float>& srcLocal, uint32_t count);
    __aicore__ inline void SplitToCore(uint32_t curSendCnt, uint32_t curUseAivNum, uint32_t& startTokenId,
                                       uint32_t& endTokenId, uint32_t& sendTokenNum);
    __aicore__ inline void SetFlagToFFN();
    __aicore__ inline void SetFlagInAttn();
    __aicore__ inline void ClearAttnActiveFlags();
    __aicore__ inline void ActiveMaskCalCnt();

    TPipe* tpipe_{nullptr};
    Hcomm<COMM_PROTOCOL_UBC_CTP> hcomm_;
    __gm__ Mc2Aclnn::AttentionFFNContext* mc2Context_{nullptr};

    GM_ADDR localFlagAddr_{nullptr};
    GM_ADDR layerFlagAddr_{nullptr};
    GM_ADDR syncStatusWorkspaceGM_{nullptr};
    GM_ADDR stagingBaseAddr_{nullptr};
    uint64_t stagingStride_{0};
    GM_ADDR countMatrixBase_{nullptr};
    GM_ADDR bucketTablesBase_{nullptr};
    uint32_t rankCountElements_{0};
    uint32_t countRowStride_{0};
    uint32_t entriesPerRank_{0};
    uint32_t groupTokenMax_{0};

    GlobalTensor<XType> xGMTensor_;
    GlobalTensor<int32_t> sessionIdGMTensor_;
    GlobalTensor<int32_t> microBatchIdGMTensor_;
    GlobalTensor<int32_t> layerIdGMTensor_;
    GlobalTensor<int32_t> expertIdsGMTensor_;
    GlobalTensor<int32_t> expertRankTableGMTensor_;
    GlobalTensor<float> scalesGMTensor_;
    GlobalTensor<bool> activeMaskGMTensor_;
    GlobalTensor<int32_t> syncStatusGMTensor_;

    LocalTensor<XType> xInTensor_;
    LocalTensor<XType> xTmpTensor_;
    LocalTensor<int32_t> statusTensor_;
    LocalTensor<StorageXOutType> xOutTensor_;
    LocalTensor<int32_t> ffnStatusTensor_;
    LocalTensor<float> smoothScalesTensor_;
    LocalTensor<int32_t> expertIdsTensor_;
    LocalTensor<uint8_t> hcommTensor_;
    LocalTensor<int32_t> flagValueTensor_;
    LocalTensor<uint32_t> flagOffsetTensor_;
    LocalTensor<int32_t> tokenRankScratchTensor_;
    LocalTensor<int32_t> bucketTensor_;

    TBuf<> smoothScalesBuf_;
    TBuf<> expertIdsBuf_;
    TBuf<> statusBuf_;
    TBuf<> receiveDataCastFloatBuf_;
    TBuf<> sumOutBuf_;
    TBuf<> activeMaskBuf_;
    TBuf<> castTempBuf_;
    TBuf<> ffnStatusBuf_;
    TBuf<> hcommBuf_;
    TBuf<> hcommBatchBuf_;
    TBuf<> flagValueBuf_;
    TBuf<> flagOffsetBuf_;
    TBuf<> tokenRankScratchBuf_;
    TBuf<> bucketBuf_;
    TBuf<> rankInfoBuf_;
    LocalTensor<uint8_t> hcommBatchWqeTensor_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> xQueue_;
    TQue<QuePosition::VECIN, 1> xInQueue_;
    TQue<QuePosition::VECOUT, 1> xOutQueue_;

    int32_t dstExpertId_{0};
    int32_t toRankId_{0};
    int32_t localExpId_{0};

    uint32_t aivId_{0};
    uint32_t rankId_{0};
    uint32_t axisX_{0};
    uint32_t axisBS_{0};
    uint32_t axisH_{0};
    uint32_t axisL_{0};
    uint32_t axisK_{0};
    uint32_t expertNum_{0};
    uint32_t moeExpertNum_{0};
    uint32_t sharedExpertNum_{1};
    uint32_t axisHS_{0};
    uint32_t expRankTableM_{0};
    uint32_t microBatchNum_{0};
    uint32_t attentionWorkerNum_{0};
    uint32_t infoTableLastDimNum_{0};
    uint32_t aivNum_{0};
    uint32_t worldSize_{0};
    uint32_t channelsPerRank_{1};
    uint32_t ffnNum_{0};
    uint32_t ffnStartRankId_{0};
    uint32_t sessionId_{0};
    uint32_t microBatchId_{0};
    uint32_t layerId_{0};
    uint32_t expertIdsCnt_{0};
    uint32_t totalSendNum_{0};
    uint32_t remoteFlagCnt_{0};
    uint32_t quantMode_{0};
    uint32_t hOutSizeAlign_{0};
    uint32_t curBsCnt_{0};
    uint32_t axisBsAlignSize_{0};
    uint32_t ffnNumAlignSize_{0};
    uint32_t ffnNumAlignCnt_{0};
    uint32_t aivWorkspaceOffset_{0};
    uint64_t hSize_{0};
    uint64_t hCommuSize_{0};
    uint64_t layIdsExpRankTableOffset_{0};
    uint64_t winTokenDataOffset_{0};
    uint64_t winInfoTableOffset_{0};
    uint64_t winOffset_[WIN_OFFSET_CNT]{0, 0};
    bool isScales_{false};
};

template <TemplateAttentionToFfnUrmaTypeClass>
__aicore__ inline void AttentionToFfnUrma<TemplateAttentionToFfnUrmaTypeFunc>::Init(
    GM_ADDR mc2Context, GM_ADDR x, GM_ADDR sessionId, GM_ADDR microBatchId, GM_ADDR layerId, GM_ADDR expertIds,
    GM_ADDR expertRankTable, GM_ADDR scales, GM_ADDR active_mask, GM_ADDR workspaceGM, TPipe* pipe,
    const AttentionToFfnV2TilingData* tilingData)
{
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
#endif
    tpipe_ = pipe;
    aivId_ = GetBlockIdx();
    mc2Context_ = reinterpret_cast<__gm__ Mc2Aclnn::AttentionFFNContext*>(mc2Context);
    rankId_ = mc2Context_->epRankId;

    axisX_ = tilingData->attentionToFfnV2Info.X;
    axisBS_ = tilingData->attentionToFfnV2Info.BS;
    axisH_ = tilingData->attentionToFfnV2Info.H;
    axisL_ = tilingData->attentionToFfnV2Info.L;
    axisK_ = tilingData->attentionToFfnV2Info.K;
    expertNum_ = tilingData->attentionToFfnV2Info.expertNum;
    moeExpertNum_ = tilingData->attentionToFfnV2Info.moeExpertNum;
    sharedExpertNum_ = tilingData->attentionToFfnV2Info.sharedExpertNum;
    axisHS_ = tilingData->attentionToFfnV2Info.HS;
    expRankTableM_ = tilingData->attentionToFfnV2Info.expRankTableM;
    microBatchNum_ = tilingData->attentionToFfnV2Info.microBatchNum;
    attentionWorkerNum_ = tilingData->attentionToFfnV2Info.attentionWorkerNum;
    infoTableLastDimNum_ = tilingData->attentionToFfnV2Info.infoTableLastDimNum;
    aivNum_ = tilingData->attentionToFfnV2Info.aivNum;
    worldSize_ = tilingData->attentionToFfnV2Info.worldSize;
    channelsPerRank_ = mc2Context_->channelsPerRank;
    if (worldSize_ == 0U || channelsPerRank_ == 0U || channelsPerRank_ > Mc2Aclnn::HCCL_MAX_RANK_SIZE / worldSize_) {
        channelsPerRank_ = 1U;
    }
    quantMode_ = tilingData->attentionToFfnV2Info.quantMode;
    isScales_ = tilingData->attentionToFfnV2Info.isScales;
    ffnStartRankId_ = tilingData->attentionToFfnV2Info.ffnStartRankId;
    ffnNum_ = worldSize_ - attentionWorkerNum_;
    curBsCnt_ = axisBS_;
    hSize_ = static_cast<uint64_t>(axisH_) * sizeof(XType);

    xGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ XType*>(x));
    sessionIdGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(sessionId));
    microBatchIdGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(microBatchId));
    layerIdGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(layerId));
    expertIdsGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(expertIds));
    expertRankTableGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(expertRankTable));
    activeMaskGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ bool*>(active_mask));

    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(sessionIdGMTensor_);
    sessionId_ = sessionIdGMTensor_.GetValue(0);
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(microBatchIdGMTensor_);
    microBatchId_ = microBatchIdGMTensor_.GetValue(0);
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(layerIdGMTensor_);
    layerId_ = layerIdGMTensor_.GetValue(0);
    ascendc_assert(layerId_ < axisL_, "layerId %u must be less than L %u", layerId_, axisL_);

    expertIdsCnt_ = axisX_ * axisBS_ * (axisK_ + sharedExpertNum_);
    const uint64_t expertRankTableCnt = static_cast<uint64_t>(expertNum_) * expRankTableM_;
    layIdsExpRankTableOffset_ = static_cast<uint64_t>(layerId_) * expertRankTableCnt;
    ffnNumAlignSize_ = Ceil(ffnNum_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    ffnNumAlignCnt_ = ffnNumAlignSize_ / sizeof(int32_t);
    aivWorkspaceOffset_ = Ceil(ffnNum_ * sizeof(int32_t), WORKSPACE_ELEMENT_OFFSET) * WORKSPACE_ELEMENT_OFFSET;
    axisBsAlignSize_ = Ceil(axisBS_ * sizeof(bool), UB_ALIGN) * UB_ALIGN;

    uint32_t expertIdsAlign = Ceil(expertIdsCnt_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(expertIdsBuf_, expertIdsAlign);
    expertIdsTensor_ = expertIdsBuf_.Get<int32_t>();
    tpipe_->InitBuffer(statusBuf_, UB_ALIGN);
    statusTensor_ = statusBuf_.Get<int32_t>();

    uint32_t maxTotalSendNum = axisX_ * axisBS_ * (axisK_ + sharedExpertNum_);
    uint32_t maxTokensPerAiv = Ceil(maxTotalSendNum, aivNum_);
    uint32_t flagValueBufSize = Ceil(maxTokensPerAiv * URMA_FLAG_SLOT_SIZE, UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(flagValueBuf_, flagValueBufSize);
    flagValueTensor_ = flagValueBuf_.Get<int32_t>();
    uint32_t flagOffsetBufSize = Ceil(maxTokensPerAiv * sizeof(uint32_t), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(flagOffsetBuf_, flagOffsetBufSize);
    flagOffsetTensor_ = flagOffsetBuf_.Get<uint32_t>();

    // The rank stash only feeds this AIV's scatter pass; the relay phase reads the compact
    // per-rank token tables in GM instead of the full cross-AIV scratch.
    uint32_t tokenRankStashBytes = Ceil(maxTokensPerAiv * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(tokenRankScratchBuf_, tokenRankStashBytes);
    tokenRankScratchTensor_ = tokenRankScratchBuf_.Get<int32_t>();
    // bucketBuf_ is reused sequentially: prefix rows (BuildRankTableOffsets) -> per-rank scatter
    // buckets (ScatterTokensToRankTables) -> relay entry chunks (RelayRemoteTokens).
    tpipe_->InitBuffer(bucketBuf_, BUCKET_BUFFER_SIZE);
    bucketTensor_ = bucketBuf_.Get<int32_t>();
    rankCountElements_ =
        static_cast<uint32_t>(Ceil(worldSize_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN / sizeof(int32_t));
    tpipe_->InitBuffer(rankInfoBuf_, RANK_INFO_SEGMENT_NUM * rankCountElements_ * sizeof(int32_t));

    if constexpr (QuantMode != 0) {
        QuantInit(scales);
        castTempBuf_ = receiveDataCastFloatBuf_;
        sumOutBuf_ = smoothScalesBuf_;
    } else {
        hCommuSize_ = hSize_;
        tpipe_->InitBuffer(xQueue_, BUFFER_NUM, hSize_);
    }

    if constexpr (isSync) {
        tpipe_->InitBuffer(ffnStatusBuf_, ffnNumAlignSize_);
        ffnStatusTensor_ = ffnStatusBuf_.Get<int32_t>();
        syncStatusWorkspaceGM_ = workspaceGM;
    }
    if constexpr (isActiveMask) {
        tpipe_->InitBuffer(activeMaskBuf_, axisBsAlignSize_);
        if constexpr (QuantMode == 0) {
            uint32_t bsAlignHalf = Ceil(axisBS_ * sizeof(half), UB_ALIGN) * UB_ALIGN;
            tpipe_->InitBuffer(castTempBuf_, bsAlignHalf);
            tpipe_->InitBuffer(sumOutBuf_, bsAlignHalf);
        }
    }

    winOffset_[0] = 0;
    winOffset_[1] =
        Ceil(attentionWorkerNum_ * microBatchNum_ * infoTableLastDimNum_ * sizeof(int32_t), WIN_ALIGN) * WIN_ALIGN;
    winInfoTableOffset_ =
        (sessionId_ * microBatchNum_ * infoTableLastDimNum_ + microBatchId_ * infoTableLastDimNum_) * sizeof(int32_t);
    winTokenDataOffset_ =
        (static_cast<uint64_t>(sessionId_) * microBatchNum_ * axisBS_ * (axisK_ + sharedExpertNum_) * axisHS_) +
        (static_cast<uint64_t>(microBatchId_) * axisBS_ * (axisK_ + sharedExpertNum_) * axisHS_);

    const uint64_t elementSize = (QuantMode != 0) ? sizeof(XOutType) : sizeof(XType);
    const uint64_t dataRegionSize = static_cast<uint64_t>(attentionWorkerNum_) * microBatchNum_ * axisBS_ *
                                    (axisK_ + sharedExpertNum_) * axisHS_ * elementSize;
    GM_ADDR selfWinAddr = GetWindowAddr(static_cast<int32_t>(rankId_));
    const uint64_t dataRegionAlignSize = Ceil(dataRegionSize, static_cast<uint64_t>(WIN_ALIGN)) * WIN_ALIGN;
    stagingStride_ = Ceil(hCommuSize_, static_cast<uint64_t>(UB_ALIGN)) * UB_ALIGN;
    const uint64_t stagingRegionSize = static_cast<uint64_t>(maxTotalSendNum) * stagingStride_;
    const uint64_t stagingRegionAlignSize = Ceil(stagingRegionSize, static_cast<uint64_t>(WIN_ALIGN)) * WIN_ALIGN;
    stagingBaseAddr_ = selfWinAddr + winOffset_[1] + dataRegionAlignSize;
    localFlagAddr_ = stagingBaseAddr_ + stagingRegionAlignSize;
    layerFlagAddr_ = localFlagAddr_ + static_cast<uint64_t>(expertIdsCnt_) * URMA_FLAG_SLOT_SIZE;

    const uint64_t countMatrixOffset = static_cast<uint64_t>(aivNum_) * aivWorkspaceOffset_;
    countMatrixBase_ = workspaceGM + countMatrixOffset;
    countRowStride_ = static_cast<uint32_t>(Ceil(rankCountElements_ * sizeof(int32_t), WORKSPACE_ELEMENT_OFFSET) *
                                            WORKSPACE_ELEMENT_OFFSET);
    const uint64_t countMatrixBytes =
        Ceil(static_cast<uint64_t>(aivNum_) * countRowStride_, WORKSPACE_ELEMENT_OFFSET) * WORKSPACE_ELEMENT_OFFSET;
    bucketTablesBase_ = countMatrixBase_ + countMatrixBytes;
    const uint32_t bucketRankNum = (worldSize_ == 0U) ? 1U : worldSize_;
    const uint32_t bucketBytes = BUCKET_BUFFER_SIZE / bucketRankNum / UB_ALIGN * UB_ALIGN;
    entriesPerRank_ = (bucketBytes / sizeof(int32_t) == 0U) ? 1U : bucketBytes / sizeof(int32_t);
    groupTokenMax_ = (entriesPerRank_ < BUCKET_GROUP_TOKEN_MAX) ? entriesPerRank_ : BUCKET_GROUP_TOKEN_MAX;
}

template <TemplateAttentionToFfnUrmaTypeClass>
__aicore__ inline void AttentionToFfnUrma<TemplateAttentionToFfnUrmaTypeFunc>::QuantInit(GM_ADDR scales)
{
    uint32_t scaleParamPad = 0;
    if constexpr (QuantMode == DYNAMIC_QUANT) {
        scaleParamPad = SCALE_PARAM_PAD_SIZE;
        hCommuSize_ = axisH_ * sizeof(int8_t) + scaleParamPad;
        hOutSizeAlign_ = Ceil(axisH_ * sizeof(int8_t), UB_ALIGN) * UB_ALIGN;
        tpipe_->InitBuffer(xInQueue_, BUFFER_NUM, hSize_);
        tpipe_->InitBuffer(xOutQueue_, BUFFER_NUM, hCommuSize_);
    } else if constexpr (IsMxOrMxClipQuant(QuantMode)) {
        uint32_t outDataBytes;
        if constexpr (Std::IsSame<XOutType, fp4x2_e2m1_t>::value) {
            outDataBytes = Ceil(axisH_, FP4_ELEMS_PER_BYTE);
        } else {
            outDataBytes = axisH_ * sizeof(XOutType);
        }
        hOutSizeAlign_ = Ceil(outDataBytes, 256U) * 256U;
        uint32_t mxScaleNum = Ceil(Ceil(axisH_, MX_BLOCK_SIZE), 2U) * 2U;
        scaleParamPad = mxScaleNum;
        hCommuSize_ = hOutSizeAlign_ + scaleParamPad;
        uint32_t hInSize = Ceil(axisH_, PERGROUP_BLOCK_SIZE) * PERGROUP_BLOCK_SIZE * sizeof(XType);
        tpipe_->InitBuffer(xInQueue_, BUFFER_NUM, hInSize);
        tpipe_->InitBuffer(xOutQueue_, BUFFER_NUM, hCommuSize_);
    } else {
        scaleParamPad = SCALE_PARAM_PAD_SIZE;
        hCommuSize_ = axisH_ * sizeof(XOutType) + scaleParamPad;
        hOutSizeAlign_ = Ceil(axisH_ * sizeof(XOutType), UB_ALIGN) * UB_ALIGN;
        tpipe_->InitBuffer(xInQueue_, BUFFER_NUM, hSize_);
        tpipe_->InitBuffer(xOutQueue_, BUFFER_NUM, hCommuSize_);
    }
    if (isScales_) {
        scalesGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scales));
    }
    uint32_t hFp32Size = axisH_ * sizeof(float);
    tpipe_->InitBuffer(receiveDataCastFloatBuf_, hFp32Size);
    tpipe_->InitBuffer(smoothScalesBuf_, hFp32Size);
}

template <TemplateAttentionToFfnUrmaTypeClass>
__aicore__ inline void AttentionToFfnUrma<TemplateAttentionToFfnUrmaTypeFunc>::ReduceMaxInplace(
    const LocalTensor<float>& srcLocal, uint32_t count)
{
    uint64_t repsFp32 = count >> 6;
    uint64_t offsetsFp32 = repsFp32 << 6;
    uint64_t remsFp32 = count & 0x3f;
    const uint64_t elemPerRefFp32 = 64UL;
    if (likely(repsFp32 > 1)) {
        Max(srcLocal, srcLocal[elemPerRefFp32], srcLocal, elemPerRefFp32, repsFp32 - 1, {1, 1, 1, 0, REP_STRIDE, 0});
        PipeBarrier<PIPE_V>();
    }
    if (unlikely(remsFp32 > 0) && unlikely(offsetsFp32 > 0)) {
        Max(srcLocal, srcLocal[offsetsFp32], srcLocal, remsFp32, 1, {1, 1, 1, 0, REP_STRIDE, 0});
        PipeBarrier<PIPE_V>();
    }
    uint32_t mask = (repsFp32 > 0) ? elemPerRefFp32 : count;
    WholeReduceMax(srcLocal, srcLocal, mask, 1, REP_STRIDE, 1, REP_STRIDE);
}

template <TemplateAttentionToFfnUrmaTypeClass>
__aicore__ inline void AttentionToFfnUrma<TemplateAttentionToFfnUrmaTypeFunc>::QuantProcess(uint32_t expertIndex)
{
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    if constexpr (QuantMode == DYNAMIC_QUANT) {
        float dynamicScale = 0.0;
        LocalTensor<float> floatLocalTemp;
        floatLocalTemp = receiveDataCastFloatBuf_.Get<float>();
        Cast(floatLocalTemp, xInTensor_, RoundMode::CAST_NONE, axisH_);
        PipeBarrier<PIPE_V>();
        xInQueue_.FreeTensor<XType>(xInTensor_);

        if (isScales_) {
            smoothScalesTensor_ = smoothScalesBuf_.Get<float>();
            DataCopyExtParams scalesCopyInParams{1U, static_cast<uint32_t>(axisH_ * sizeof(float)), 0U, 0U, 0U};
            DataCopyPadExtParams<float> copyPadExtParams{false, 0U, 0U, 0U};
            DataCopyPad(smoothScalesTensor_,
                        scalesGMTensor_[(static_cast<uint64_t>(layerId_) * expertNum_ + expertIndex) * axisH_],
                        scalesCopyInParams, copyPadExtParams);
            SyncFunc<AscendC::HardEvent::MTE2_V>();
            Mul(floatLocalTemp, floatLocalTemp, smoothScalesTensor_, axisH_);
            PipeBarrier<PIPE_V>();
        }

        LocalTensor<float> floatLocalAbsTemp = smoothScalesBuf_.Get<float>();
        Abs(floatLocalAbsTemp, floatLocalTemp, axisH_);
        PipeBarrier<PIPE_V>();
        ReduceMaxInplace(floatLocalAbsTemp, axisH_);
        SyncFunc<AscendC::HardEvent::V_S>();
        dynamicScale = float(INT8_MAX_VALUE) / floatLocalAbsTemp.GetValue(0);
        SyncFunc<AscendC::HardEvent::S_V>();
        Muls(floatLocalTemp, floatLocalTemp, dynamicScale, axisH_);
        PipeBarrier<PIPE_V>();

        LocalTensor<half> halfLocalTemp = floatLocalTemp.ReinterpretCast<half>();
        LocalTensor<int32_t> int32LocalTemp = floatLocalTemp.ReinterpretCast<int32_t>();
        Cast(int32LocalTemp, floatLocalTemp, RoundMode::CAST_RINT, axisH_);
        SetDeqScale((half)1.000000e+00f);
        PipeBarrier<PIPE_V>();
        Cast(halfLocalTemp, int32LocalTemp, RoundMode::CAST_ROUND, axisH_);
        PipeBarrier<PIPE_V>();
        Cast(xOutTensor_, halfLocalTemp, RoundMode::CAST_TRUNC, axisH_);

        floatLocalTemp = xOutTensor_.template ReinterpretCast<float>();
        floatLocalTemp.SetValue(hOutSizeAlign_ / sizeof(float), float(1.0) / dynamicScale);
        SyncFunc<HardEvent::S_MTE3>();
    } else if constexpr (IsMxQuant(QuantMode)) {
        uint32_t mxScaleNum = Ceil(Ceil(axisH_, MX_BLOCK_SIZE), 2U) * 2U;
        LocalTensor<float> receiveDataCastFloat = receiveDataCastFloatBuf_.Get<float>();
        __ubuf__ StorageXInType* srcAddr = (__ubuf__ StorageXInType*)xInTensor_.GetPhyAddr();
        __ubuf__ uint16_t* maxExpAddr = (__ubuf__ uint16_t*)receiveDataCastFloat.GetPhyAddr();
        __ubuf__ uint16_t* halfScaleLocalAddr =
            (__ubuf__ uint16_t*)receiveDataCastFloat[Ceil(mxScaleNum, 32U) * 32U].GetPhyAddr();
        __ubuf__ int8_t* outLocalAddr = (__ubuf__ int8_t*)xOutTensor_.GetPhyAddr();
        __ubuf__ uint16_t* mxScaleLocalAddr;
        if constexpr (Std::IsSame<XOutType, fp4x2_e2m1_t>::value) {
            mxScaleLocalAddr = (__ubuf__ uint16_t*)xOutTensor_[Ceil(axisH_, FP4_ELEMS_PER_BYTE) * 1U].GetPhyAddr();
        } else {
            mxScaleLocalAddr = (__ubuf__ uint16_t*)xOutTensor_[axisH_ * 1U].GetPhyAddr();
        }
        Quant::ComputeMaxExp(srcAddr, maxExpAddr, axisH_);
        Quant::ComputeScale<XOutType>(maxExpAddr, mxScaleLocalAddr, halfScaleLocalAddr, mxScaleNum);
        if constexpr (Std::IsSame<XOutType, fp8_e4m3fn_t>::value || Std::IsSame<XOutType, fp8_e5m2_t>::value) {
            Quant::ComputeFp8Data<StorageXInType, XOutType, AscendC::RoundMode::CAST_TRUNC,
                                  AscendC::RoundMode::CAST_RINT>(srcAddr, halfScaleLocalAddr, outLocalAddr, axisH_);
        } else {
            Quant::ComputeFp4Data<StorageXInType, XOutType, AscendC::RoundMode::CAST_TRUNC,
                                  AscendC::RoundMode::CAST_RINT>(srcAddr, halfScaleLocalAddr, outLocalAddr, axisH_);
        }
        PipeBarrier<PIPE_V>();
        xInQueue_.FreeTensor<XType>(xInTensor_);
    } else if constexpr (IsMxClipQuant(QuantMode)) {
        uint32_t mxScaleNum = Ceil(Ceil(axisH_, MX_BLOCK_SIZE), 2U) * 2U;
        LocalTensor<float> receiveDataCastFloat = receiveDataCastFloatBuf_.Get<float>();
        __ubuf__ StorageXInType* srcAddr = (__ubuf__ StorageXInType*)xInTensor_.GetPhyAddr();
        __ubuf__ uint16_t* maxExpAddr = (__ubuf__ uint16_t*)receiveDataCastFloat.GetPhyAddr();
        __ubuf__ uint16_t* halfScaleLocalAddr =
            (__ubuf__ uint16_t*)receiveDataCastFloat[Ceil(mxScaleNum, 32U) * 32U].GetPhyAddr();
        __ubuf__ int8_t* outLocalAddr = (__ubuf__ int8_t*)xOutTensor_.GetPhyAddr();
        __ubuf__ uint16_t* mxScaleLocalAddr = (__ubuf__ uint16_t*)xOutTensor_[axisH_ * 1U].GetPhyAddr();
        Quant::ComputeMaxExpClip(srcAddr, maxExpAddr, axisH_);
        Quant::ComputeScaleClip<XOutType, StorageXInType>(maxExpAddr, mxScaleLocalAddr, halfScaleLocalAddr, mxScaleNum);
        Quant::ComputeFp8Data<StorageXInType, XOutType, AscendC::RoundMode::CAST_TRUNC, AscendC::RoundMode::CAST_RINT>(
            srcAddr, halfScaleLocalAddr, outLocalAddr, axisH_);
        PipeBarrier<PIPE_V>();
        xInQueue_.FreeTensor<XType>(xInTensor_);
    }
#else
    float dynamicScale = 0.0;
    uint32_t hOutSizeAlign = Ceil(axisH_ * sizeof(int8_t), UB_ALIGN) * UB_ALIGN;
    LocalTensor<float> floatLocalTemp;
    floatLocalTemp = receiveDataCastFloatBuf_.Get<float>();
    Cast(floatLocalTemp, xInTensor_, RoundMode::CAST_NONE, axisH_);
    PipeBarrier<PIPE_V>();
    xInQueue_.FreeTensor<XType>(xInTensor_);

    if (isScales_) {
        smoothScalesTensor_ = smoothScalesBuf_.Get<float>();
        DataCopyExtParams scalesCopyInParams{1U, static_cast<uint32_t>(axisH_ * sizeof(float)), 0U, 0U, 0U};
        DataCopyPadExtParams<float> copyPadExtParams{false, 0U, 0U, 0U};
        DataCopyPad(smoothScalesTensor_,
                    scalesGMTensor_[(static_cast<uint64_t>(layerId_) * expertNum_ + expertIndex) * axisH_],
                    scalesCopyInParams, copyPadExtParams);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        Mul(floatLocalTemp, floatLocalTemp, smoothScalesTensor_, axisH_);
        PipeBarrier<PIPE_V>();
    }

    if (quantMode_ == DYNAMIC_QUANT) {
        LocalTensor<float> floatLocalAbsTemp = smoothScalesBuf_.Get<float>();
        Abs(floatLocalAbsTemp, floatLocalTemp, axisH_);
        PipeBarrier<PIPE_V>();
        ReduceMaxInplace(floatLocalAbsTemp, axisH_);
        SyncFunc<AscendC::HardEvent::V_S>();
        float maxAbs = floatLocalAbsTemp.GetValue(0);
        dynamicScale = (maxAbs == 0.0f) ? 1.0f : float(INT8_MAX_VALUE) / maxAbs;
        SyncFunc<AscendC::HardEvent::S_V>();
        Muls(floatLocalTemp, floatLocalTemp, dynamicScale, axisH_);
        PipeBarrier<PIPE_V>();
    }
    LocalTensor<half> halfLocalTemp = floatLocalTemp.ReinterpretCast<half>();
    LocalTensor<int32_t> int32LocalTemp = floatLocalTemp.ReinterpretCast<int32_t>();
    Cast(int32LocalTemp, floatLocalTemp, RoundMode::CAST_RINT, axisH_);
    SetDeqScale((half)1.000000e+00f);
    PipeBarrier<PIPE_V>();
    Cast(halfLocalTemp, int32LocalTemp, RoundMode::CAST_ROUND, axisH_);
    PipeBarrier<PIPE_V>();
    Cast(xOutTensor_, halfLocalTemp, RoundMode::CAST_TRUNC, axisH_);
    floatLocalTemp = xOutTensor_.template ReinterpretCast<float>();
    floatLocalTemp.SetValue(hOutSizeAlign / sizeof(float), float(1.0) / dynamicScale);
    SyncFunc<HardEvent::S_MTE3>();
#endif
}

template <TemplateAttentionToFfnUrmaTypeClass>
__aicore__ inline void AttentionToFfnUrma<TemplateAttentionToFfnUrmaTypeFunc>::FindExpertRank(int32_t expertId)
{
    ascendc_assert(expertId >= 0 && static_cast<uint32_t>(expertId) < expertNum_,
                   "expertId %d must be in [0, expertNum %u)", expertId, expertNum_);
    uint64_t expRankTableOffset = static_cast<uint64_t>(expertId) * expRankTableM_ + layIdsExpRankTableOffset_;
    uint32_t rankCnt = expertRankTableGMTensor_.GetValue(expRankTableOffset);
    if (rankCnt == 0) {
        return;
    }
    ascendc_assert(static_cast<uint64_t>(rankCnt) * RANK_OFFSET_STRIDE + 1 <= expRankTableM_,
                   "rankCnt %u exceeds expert_rank_table row capacity M %u", rankCnt, expRankTableM_);
    uint32_t rankOffset = (sessionId_ % rankCnt) * RANK_OFFSET_STRIDE + 1;
    toRankId_ = expertRankTableGMTensor_.GetValue(expRankTableOffset + rankOffset);
    localExpId_ = expertRankTableGMTensor_.GetValue(expRankTableOffset + rankOffset + 1);
}

template <TemplateAttentionToFfnUrmaTypeClass>
__aicore__ inline void AttentionToFfnUrma<TemplateAttentionToFfnUrmaTypeFunc>::SplitToCore(
    uint32_t curSendCnt, uint32_t curUseAivNum, uint32_t& startTokenId, uint32_t& endTokenId, uint32_t& sendTokenNum)
{
    sendTokenNum = curSendCnt / curUseAivNum;
    uint32_t remainderTokenNum = curSendCnt % curUseAivNum;
    startTokenId = sendTokenNum * aivId_;
    if (aivId_ < remainderTokenNum) {
        sendTokenNum += 1;
        startTokenId += aivId_;
    } else {
        startTokenId += remainderTokenNum;
    }
    endTokenId = startTokenId + sendTokenNum;
}

template <TemplateAttentionToFfnUrmaTypeClass>
__aicore__ inline void AttentionToFfnUrma<TemplateAttentionToFfnUrmaTypeFunc>::HcommInit()
{
    tpipe_->InitBuffer(hcommBuf_, URMA_HCOMM_INIT_SIZE);
    hcommTensor_ = hcommBuf_.Get<uint8_t>();
    hcomm_.Init(hcommTensor_, URMA_HCOMM_INIT_SIZE);
    tpipe_->InitBuffer(hcommBatchBuf_, ATTN_FFN_HCOMM_BATCH_UB_BYTES);
    hcommBatchWqeTensor_ = hcommBatchBuf_.Get<uint8_t>();
}

template <TemplateAttentionToFfnUrmaTypeClass>
__aicore__ inline GM_ADDR AttentionToFfnUrma<TemplateAttentionToFfnUrmaTypeFunc>::GetWindowAddr(int32_t rankId)
{
    ascendc_assert(rankId >= 0 && static_cast<uint32_t>(rankId) < worldSize_, "rankId %d must be in [0, worldSize %u)",
                   rankId, worldSize_);
    return (GM_ADDR)mc2Context_->epHcclBuffer_[rankId];
}

template <TemplateAttentionToFfnUrmaTypeClass>
__aicore__ inline uint64_t AttentionToFfnUrma<TemplateAttentionToFfnUrmaTypeFunc>::GetUrmaCommHandle(
    uint32_t dstRank, uint32_t channelIndex)
{
    return mc2Context_->hcommHandle_[dstRank * channelsPerRank_ + channelIndex];
}

template <TemplateAttentionToFfnUrmaTypeClass>
__aicore__ inline void AttentionToFfnUrma<TemplateAttentionToFfnUrmaTypeFunc>::ReadTokenMetaData(
    AttentionToFfnTokenMetaData& metaData, uint32_t tokenOffset)
{
    metaData.tokenId = tokenOffset / (axisK_ + sharedExpertNum_);
    metaData.topkId = tokenOffset % (axisK_ + sharedExpertNum_);

    if (metaData.topkId < axisK_) {
        metaData.dstExpertId = expertIdsTensor_.GetValue(metaData.tokenId * axisK_ + metaData.topkId);
    } else {
        metaData.dstExpertId = static_cast<int32_t>(moeExpertNum_ + (metaData.topkId - axisK_));
    }

    dstExpertId_ = metaData.dstExpertId;
    FindExpertRank(metaData.dstExpertId);
    metaData.toRankId = toRankId_;
    metaData.localExpId = localExpId_;

    GM_ADDR toRankAddr = GetWindowAddr(metaData.toRankId);
    uint64_t elementSize = (QuantMode != 0) ? sizeof(XOutType) : sizeof(XType);
    uint64_t tokenDataOffset = winTokenDataOffset_ +
                               (static_cast<uint64_t>(metaData.tokenId) * (axisK_ + sharedExpertNum_) * axisHS_) +
                               (static_cast<uint64_t>(metaData.topkId) * axisHS_);
    metaData.remoteDataAddr = toRankAddr + winOffset_[1] + tokenDataOffset * elementSize;
    metaData.remoteFlagAddr = toRankAddr + winInfoTableOffset_ + (TOKEN_INFO_TABLE_RS + tokenOffset) * sizeof(int32_t);
    metaData.localDataAddr = stagingBaseAddr_ + static_cast<uint64_t>(tokenOffset) * stagingStride_;
}

template <TemplateAttentionToFfnUrmaTypeClass>
__aicore__ inline void AttentionToFfnUrma<TemplateAttentionToFfnUrmaTypeFunc>::CopyTokenDataToGM(
    const AttentionToFfnTokenMetaData& metaData, GM_ADDR targetAddr, const DataCopyExtParams& xCopyParams)
{
    DataCopyPadExtParams<XType> copyPadExtParams{false, 0U, 0U, 0U};
    DataCopyExtParams hCommuCopyParams = {1U, static_cast<uint32_t>(hCommuSize_), 0U, 0U, 0U};

    if constexpr (QuantMode == 0) {
        xTmpTensor_ = xQueue_.AllocTensor<XType>();
        DataCopyPad(xTmpTensor_, xGMTensor_[metaData.tokenId * axisH_], xCopyParams, copyPadExtParams);
        xQueue_.EnQue(xTmpTensor_);
        xTmpTensor_ = xQueue_.DeQue<XType>();
        GlobalTensor<XType> targetGMTensor;
        targetGMTensor.SetGlobalBuffer(reinterpret_cast<__gm__ XType*>(targetAddr));
        DataCopyPad(targetGMTensor, xTmpTensor_, xCopyParams);
        xQueue_.FreeTensor<XType>(xTmpTensor_);
    } else {
        xInTensor_ = xInQueue_.AllocTensor<XType>();
        DataCopyPad(xInTensor_, xGMTensor_[metaData.tokenId * axisH_], xCopyParams, copyPadExtParams);
        xInQueue_.EnQue(xInTensor_);
        xInTensor_ = xInQueue_.DeQue<XType>();
        xOutTensor_ = xOutQueue_.AllocTensor<StorageXOutType>();
        QuantProcess(dstExpertId_);
        GlobalTensor<uint8_t> targetGMTensor;
        targetGMTensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(targetAddr));
        xOutQueue_.EnQue(xOutTensor_);
        xOutTensor_ = xOutQueue_.DeQue<StorageXOutType>();
        auto xOutBytesTensor = xOutTensor_.template ReinterpretCast<uint8_t>();
        DataCopyPad(targetGMTensor, xOutBytesTensor, hCommuCopyParams);
        xOutQueue_.FreeTensor<StorageXOutType>(xOutTensor_);
    }
}

template <TemplateAttentionToFfnUrmaTypeClass>
__aicore__ inline void AttentionToFfnUrma<TemplateAttentionToFfnUrmaTypeFunc>::SendToLocal(
    const AttentionToFfnTokenMetaData& metaData, const DataCopyExtParams& xCopyParams)
{
    CopyTokenDataToGM(metaData, metaData.remoteDataAddr, xCopyParams);
    PipeBarrier<PIPE_MTE3>();

    statusTensor_.SetValue(0, metaData.localExpId);
    SyncFunc<HardEvent::S_MTE3>();
    DataCopyExtParams flagCopyParams = {1U, sizeof(int32_t), 0U, 0U, 0U};
    GlobalTensor<int32_t> tokenInfoTableGMTensor;
    tokenInfoTableGMTensor.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(metaData.remoteFlagAddr));
    DataCopyPad(tokenInfoTableGMTensor, statusTensor_, flagCopyParams);
}

template <TemplateAttentionToFfnUrmaTypeClass>
__aicore__ inline void AttentionToFfnUrma<TemplateAttentionToFfnUrmaTypeFunc>::StageRemoteData(
    const AttentionToFfnTokenMetaData& metaData, const DataCopyExtParams& xCopyParams, uint32_t tokenOffset)
{
    CopyTokenDataToGM(metaData, metaData.localDataAddr, xCopyParams);
    flagValueTensor_.SetValue(remoteFlagCnt_ * FLAG_ELEM_STRIDE, metaData.localExpId);
    flagOffsetTensor_.SetValue(remoteFlagCnt_, tokenOffset);
    ++remoteFlagCnt_;
}

template <TemplateAttentionToFfnUrmaTypeClass>
__aicore__ inline void AttentionToFfnUrma<TemplateAttentionToFfnUrmaTypeFunc>::AppendRemoteDataWrite(
    HcommBatchHandle& batchHandle, GM_ADDR remoteDataAddr, GM_ADDR sourceDataAddr, bool enableCqe)
{
    if (enableCqe) {
        hcomm_.WriteNbi<REMOTE_DATA_CQE_WQE_CONFIG>(batchHandle, remoteDataAddr, sourceDataAddr,
                                                    static_cast<int64_t>(hCommuSize_));
    } else {
        hcomm_.WriteNbi<REMOTE_DATA_WQE_CONFIG>(batchHandle, remoteDataAddr, sourceDataAddr,
                                                static_cast<int64_t>(hCommuSize_));
    }
}

template <TemplateAttentionToFfnUrmaTypeClass>
__aicore__ inline void AttentionToFfnUrma<TemplateAttentionToFfnUrmaTypeFunc>::AppendRemoteFlagWrite(
    HcommBatchHandle& batchHandle, GM_ADDR remoteFlagAddr, uint32_t sourceTokenOffset, bool enableCqe)
{
    GM_ADDR sourceFlagAddr = localFlagAddr_ + static_cast<uint64_t>(sourceTokenOffset) * URMA_FLAG_SLOT_SIZE;
    if (enableCqe) {
        hcomm_.WriteNbi<REMOTE_ORDERED_FLAG_CQE_WQE_CONFIG>(batchHandle, remoteFlagAddr, sourceFlagAddr,
                                                            static_cast<int64_t>(sizeof(int32_t)));
    } else {
        hcomm_.WriteNbi<REMOTE_ORDERED_FLAG_WQE_CONFIG>(batchHandle, remoteFlagAddr, sourceFlagAddr,
                                                        static_cast<int64_t>(sizeof(int32_t)));
    }
}

template <TemplateAttentionToFfnUrmaTypeClass>
__aicore__ inline void AttentionToFfnUrma<TemplateAttentionToFfnUrmaTypeFunc>::CommitAndDrainChannel(
    HcommBatchHandle& batchHandle, uint64_t channel, uint32_t& preparedWqeCount, uint32_t& sqWriteCount)
{
    if (preparedWqeCount != 0U) {
        hcomm_.BatchCommit(batchHandle);
        preparedWqeCount = 0U;
    }
    int32_t ret = hcomm_.Drain(channel);
    ascendc_assert(ret == 0, "Urma drain failed, ret=%d, rankId=%u, channel=%lu", ret, rankId_, channel);
    sqWriteCount = 0U;
}

template <TemplateAttentionToFfnUrmaTypeClass>
__aicore__ inline void AttentionToFfnUrma<TemplateAttentionToFfnUrmaTypeFunc>::BuildRankTableOffsets()
{
    LocalTensor<int32_t> rankInfo = rankInfoBuf_.Get<int32_t>();
    // The count segment was already published to GM, so it is reused as the ReduceSum scratch.
    LocalTensor<int32_t> reduceSums = rankInfo[RANK_INFO_COUNT_SEG_IDX * rankCountElements_];
    LocalTensor<int32_t> writePrefix = rankInfo[RANK_INFO_PREFIX_SEG_IDX * rankCountElements_];
    LocalTensor<int32_t> totalCounts = rankInfo[RANK_INFO_TOTAL_SEG_IDX * rankCountElements_];
    LocalTensor<int32_t> tableStart = rankInfo[RANK_INFO_START_SEG_IDX * rankCountElements_];
    Duplicate<int32_t>(writePrefix, 0, rankCountElements_);
    Duplicate<int32_t>(totalCounts, 0, rankCountElements_);
    SyncFunc<HardEvent::V_S>();

    GlobalTensor<int32_t> countMatrixGM;
    countMatrixGM.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(countMatrixBase_));
    const uint32_t rowBytes = rankCountElements_ * sizeof(int32_t);
    LocalTensor<int32_t> prefixRows = bucketTensor_;
    const uint32_t rowsPerBatch = BUCKET_BUFFER_SIZE / rowBytes;
    const DataCopyPadExtParams<int32_t> padParams{false, 0U, 0U, 0U};

    // Exclusive prefix over the preceding AIV rows: where this AIV's bucket flushes land.
    for (uint32_t rowStart = 0U; rowStart < aivId_; rowStart += rowsPerBatch) {
        uint32_t copyRows = (rowStart + rowsPerBatch > aivId_) ? aivId_ - rowStart : rowsPerBatch;
        DataCopyExtParams copyParams{static_cast<uint16_t>(copyRows), rowBytes,
                                     static_cast<uint32_t>(countRowStride_ - rowBytes), 0U, 0U};
        DataCopyPad(prefixRows, countMatrixGM[static_cast<uint64_t>(rowStart) * countRowStride_ / sizeof(int32_t)],
                    copyParams, padParams);
        SyncFunc<HardEvent::MTE2_V>();
        const uint32_t prefixShape[] = {copyRows, rankCountElements_};
        ReduceSum<int32_t, AscendC::Pattern::Reduce::RA, true>(reduceSums, prefixRows, prefixShape, false);
        Add(writePrefix, writePrefix, reduceSums, rankCountElements_);
        if (rowStart + copyRows < aivId_) {
            SyncFunc<HardEvent::V_MTE2>();
        }
    }
    SyncFunc<HardEvent::V_MTE2>();
    // Every AIV also sums all rows so it locally knows each rank's total entry count.
    for (uint32_t rowStart = 0U; rowStart < aivNum_; rowStart += rowsPerBatch) {
        uint32_t copyRows = (rowStart + rowsPerBatch > aivNum_) ? aivNum_ - rowStart : rowsPerBatch;
        DataCopyExtParams copyParams{static_cast<uint16_t>(copyRows), rowBytes,
                                     static_cast<uint32_t>(countRowStride_ - rowBytes), 0U, 0U};
        DataCopyPad(prefixRows, countMatrixGM[static_cast<uint64_t>(rowStart) * countRowStride_ / sizeof(int32_t)],
                    copyParams, padParams);
        SyncFunc<HardEvent::MTE2_V>();
        const uint32_t totalShape[] = {copyRows, rankCountElements_};
        ReduceSum<int32_t, AscendC::Pattern::Reduce::RA, true>(reduceSums, prefixRows, totalShape, false);
        Add(totalCounts, totalCounts, reduceSums, rankCountElements_);
        if (rowStart + copyRows < aivNum_) {
            SyncFunc<HardEvent::V_MTE2>();
        }
    }
    SyncFunc<HardEvent::V_S>();

    // Compact layout: rank r's table starts right after the tables of all preceding ranks.
    tableStart.SetValue(0, 0);
    for (uint32_t rank = 1U; rank < worldSize_; ++rank) {
        tableStart.SetValue(rank, tableStart.GetValue(rank - 1U) + totalCounts.GetValue(rank - 1U));
    }
}

template <TemplateAttentionToFfnUrmaTypeClass>
__aicore__ inline void AttentionToFfnUrma<TemplateAttentionToFfnUrmaTypeFunc>::ScatterTokensToRankTables()
{
    LocalTensor<int32_t> rankInfo = rankInfoBuf_.Get<int32_t>();
    LocalTensor<int32_t> rankCounts = rankInfo[RANK_INFO_COUNT_SEG_IDX * rankCountElements_];
    LocalTensor<int32_t> writePrefix = rankInfo[RANK_INFO_PREFIX_SEG_IDX * rankCountElements_];
    LocalTensor<int32_t> tableStart = rankInfo[RANK_INFO_START_SEG_IDX * rankCountElements_];
    LocalTensor<int32_t> bucketEntries = bucketTensor_;
    GlobalTensor<int32_t> bucketTablesGM;
    bucketTablesGM.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(bucketTablesBase_));
    const uint64_t roundNum = Ceil(static_cast<uint64_t>(totalSendNum_), static_cast<uint64_t>(aivNum_));

    for (uint64_t roundStart = 0U; roundStart < roundNum; roundStart += groupTokenMax_) {
        uint64_t roundEnd = (roundStart + groupTokenMax_ > roundNum) ? roundNum : roundStart + groupTokenMax_;
        Duplicate<int32_t>(rankCounts, 0, rankCountElements_);
        SyncFunc<HardEvent::V_S>();
        for (uint64_t round = roundStart; round < roundEnd; ++round) {
            uint32_t tokenOffset = static_cast<uint32_t>(static_cast<uint64_t>(aivId_) + round * aivNum_);
            if (tokenOffset >= totalSendNum_) {
                continue;
            }
            int32_t dstRank = tokenRankScratchTensor_.GetValue(static_cast<int32_t>(round));
            if (dstRank == static_cast<int32_t>(rankId_)) {
                continue; // Local tokens were already delivered by SendToLocal in Phase 1.
            }
            uint32_t localIndex = static_cast<uint32_t>(rankCounts.GetValue(dstRank));
            bucketEntries.SetValue(static_cast<uint32_t>(dstRank) * entriesPerRank_ + localIndex,
                                   static_cast<int32_t>(tokenOffset));
            rankCounts.SetValue(dstRank, static_cast<int32_t>(localIndex + 1U));
        }

        SyncFunc<HardEvent::S_MTE3>();
        for (uint32_t rank = 0U; rank < worldSize_; ++rank) {
            uint32_t count = static_cast<uint32_t>(rankCounts.GetValue(rank));
            if (count == 0U) {
                continue;
            }
            uint32_t writeStart =
                static_cast<uint32_t>(tableStart.GetValue(rank)) + static_cast<uint32_t>(writePrefix.GetValue(rank));
            DataCopyExtParams entryCopyParams{1U, static_cast<uint32_t>(count * sizeof(int32_t)), 0U, 0U, 0U};
            DataCopyPad(bucketTablesGM[writeStart], bucketEntries[rank * entriesPerRank_], entryCopyParams);
            writePrefix.SetValue(rank, static_cast<int32_t>(writePrefix.GetValue(rank) + count));
        }
        SyncFunc<HardEvent::MTE3_S>();
    }
}

template <TemplateAttentionToFfnUrmaTypeClass>
__aicore__ inline void AttentionToFfnUrma<TemplateAttentionToFfnUrmaTypeFunc>::RelayRemoteTokens(uint32_t dstRank,
                                                                                                 uint32_t channelIdx)
{
    LocalTensor<int32_t> rankInfo = rankInfoBuf_.Get<int32_t>();
    LocalTensor<int32_t> totalCounts = rankInfo[RANK_INFO_TOTAL_SEG_IDX * rankCountElements_];
    LocalTensor<int32_t> tableStart = rankInfo[RANK_INFO_START_SEG_IDX * rankCountElements_];
    uint32_t entryCount = static_cast<uint32_t>(totalCounts.GetValue(dstRank));
    if (entryCount == 0U) {
        return;
    }
    uint32_t baseEntry = static_cast<uint32_t>(tableStart.GetValue(dstRank));

    GM_ADDR toRankAddr = GetWindowAddr(static_cast<int32_t>(dstRank));
    uint64_t channel = GetUrmaCommHandle(dstRank, channelIdx);
    HcommBatchHandle batchHandle =
        hcomm_.MakeBatchHandle(channel, hcommBatchWqeTensor_, ATTN_FFN_HCOMM_BATCH_UB_BYTES, toRankAddr);
    uint32_t preparedWqeCount = 0U;
    uint32_t sqWriteCount = 0U;
    const uint64_t elementSize = (QuantMode != 0) ? sizeof(XOutType) : sizeof(XType);
    LocalTensor<int32_t> entriesLocal = bucketTensor_;
    constexpr uint32_t entryChunkMax = BUCKET_BUFFER_SIZE / sizeof(int32_t);
    const DataCopyPadExtParams<int32_t> padParams{false, 0U, 0U, 0U};
    GlobalTensor<int32_t> bucketTablesGM;
    bucketTablesGM.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(bucketTablesBase_));

    for (uint32_t chunkStart = 0U; chunkStart < entryCount; chunkStart += entryChunkMax) {
        uint32_t curEntries = (chunkStart + entryChunkMax > entryCount) ? entryCount - chunkStart : entryChunkMax;
        DataCopyExtParams copyParams{1U, static_cast<uint32_t>(curEntries * sizeof(int32_t)), 0U, 0U, 0U};
        SyncFunc<HardEvent::S_MTE2>();
        DataCopyPad(entriesLocal, bucketTablesGM[baseEntry + chunkStart], copyParams, padParams);
        SyncFunc<HardEvent::MTE2_S>();
        for (uint32_t i = 0U; i < curEntries; ++i) {
            if ((chunkStart + i) % channelsPerRank_ != channelIdx) {
                continue;
            }
            uint32_t tokenOffset = static_cast<uint32_t>(entriesLocal.GetValue(i));
            uint32_t tokenId = tokenOffset / (axisK_ + sharedExpertNum_);
            uint32_t topkId = tokenOffset % (axisK_ + sharedExpertNum_);
            uint64_t tokenDataOffset = winTokenDataOffset_ +
                                       (static_cast<uint64_t>(tokenId) * (axisK_ + sharedExpertNum_) * axisHS_) +
                                       (static_cast<uint64_t>(topkId) * axisHS_);
            GM_ADDR remoteDataAddr = toRankAddr + winOffset_[1] + tokenDataOffset * elementSize;
            GM_ADDR localDataAddr = stagingBaseAddr_ + static_cast<uint64_t>(tokenOffset) * stagingStride_;
            const bool enableCqe = sqWriteCount + 1U >= HCOMM_SQ_MAX_PENDING;
            AppendRemoteDataWrite(batchHandle, remoteDataAddr, localDataAddr, enableCqe);
            ++preparedWqeCount;
            ++sqWriteCount;
            if (enableCqe) {
                CommitAndDrainChannel(batchHandle, channel, preparedWqeCount, sqWriteCount);
            } else if (preparedWqeCount == URMA_BATCH_WQE_CAPACITY) {
                hcomm_.BatchCommit(batchHandle);
                preparedWqeCount = 0U;
            }
        }
    }

    for (uint32_t chunkStart = 0U; chunkStart < entryCount; chunkStart += entryChunkMax) {
        uint32_t curEntries = (chunkStart + entryChunkMax > entryCount) ? entryCount - chunkStart : entryChunkMax;
        DataCopyExtParams copyParams{1U, static_cast<uint32_t>(curEntries * sizeof(int32_t)), 0U, 0U, 0U};
        SyncFunc<HardEvent::S_MTE2>();
        DataCopyPad(entriesLocal, bucketTablesGM[baseEntry + chunkStart], copyParams, padParams);
        SyncFunc<HardEvent::MTE2_S>();
        for (uint32_t i = 0U; i < curEntries; ++i) {
            if ((chunkStart + i) % channelsPerRank_ != channelIdx) {
                continue;
            }
            uint32_t tokenOffset = static_cast<uint32_t>(entriesLocal.GetValue(i));
            GM_ADDR remoteFlagAddr =
                toRankAddr + winInfoTableOffset_ + (TOKEN_INFO_TABLE_RS + tokenOffset) * sizeof(int32_t);
            const bool enableCqe = sqWriteCount + 1U >= HCOMM_SQ_MAX_PENDING;
            AppendRemoteFlagWrite(batchHandle, remoteFlagAddr, tokenOffset, enableCqe);
            ++preparedWqeCount;
            ++sqWriteCount;
            if (enableCqe) {
                CommitAndDrainChannel(batchHandle, channel, preparedWqeCount, sqWriteCount);
            } else if (preparedWqeCount == URMA_BATCH_WQE_CAPACITY) {
                hcomm_.BatchCommit(batchHandle);
                preparedWqeCount = 0U;
            }
        }
    }
    if (preparedWqeCount != 0U) {
        hcomm_.BatchCommit(batchHandle);
    }
}

template <TemplateAttentionToFfnUrmaTypeClass>
__aicore__ inline void AttentionToFfnUrma<TemplateAttentionToFfnUrmaTypeFunc>::ActiveMaskCalCnt()
{
    LocalTensor<bool> activeMaskTensor = activeMaskBuf_.Get<bool>();
    LocalTensor<half> tempTensor = castTempBuf_.Get<half>();
    LocalTensor<half> sumOutTensor = sumOutBuf_.Get<half>();
    DataCopyExtParams activeMaskParams = {1U, static_cast<uint32_t>(axisBS_ * sizeof(bool)), 0U, 0U, 0U};
    DataCopyPadExtParams<bool> activeMaskCopyPadParams{false, 0U, 0U, 0U};
    DataCopyPad(activeMaskTensor, activeMaskGMTensor_, activeMaskParams, activeMaskCopyPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    LocalTensor<int8_t> activeMaskInt8Tensor = activeMaskTensor.ReinterpretCast<int8_t>();
    Cast(tempTensor, activeMaskInt8Tensor, RoundMode::CAST_NONE, axisBS_);
    PipeBarrier<PIPE_V>();
    SumParams params{1, axisBsAlignSize_, axisBS_};
    Sum(sumOutTensor, tempTensor, params);
    SyncFunc<AscendC::HardEvent::V_S>();
    curBsCnt_ = static_cast<int32_t>(sumOutTensor.GetValue(0));
}

template <TemplateAttentionToFfnUrmaTypeClass>
__aicore__ inline void AttentionToFfnUrma<TemplateAttentionToFfnUrmaTypeFunc>::SetFlagInAttn()
{
    uint32_t startId = 0;
    uint32_t endId = 0;
    uint32_t sendNum = 0;
    uint32_t totalNum = axisX_ * (axisBS_ - curBsCnt_) * (axisK_ + sharedExpertNum_);
    if (totalNum == 0) {
        return;
    }
    SplitToCore(totalNum, aivNum_, startId, endId, sendNum);
    if (startId >= totalNum) {
        return;
    }

    uint64_t sendMaskTokenCnt = static_cast<uint64_t>(curBsCnt_) * (axisK_ + sharedExpertNum_);
    uint64_t attnTokenInfoTableOffset =
        (static_cast<uint64_t>(microBatchId_) * axisBS_ * (axisK_ + sharedExpertNum_) + sendMaskTokenCnt) *
        sizeof(int32_t);
    GM_ADDR selfRankAddr = GetWindowAddr(static_cast<int32_t>(rankId_));
    GM_ADDR attnTokenInfoTableGM = selfRankAddr + attnTokenInfoTableOffset;
    GlobalTensor<int32_t> attnTableGMTensor;
    attnTableGMTensor.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(attnTokenInfoTableGM));
    DataCopyExtParams dataCopyParams = {1U, static_cast<uint32_t>(sendNum * sizeof(int32_t)), 0U, 0U, 0U};
    LocalTensor<int32_t> tempTensor = expertIdsBuf_.Get<int32_t>();
    Duplicate<int32_t>(tempTensor, static_cast<int32_t>(1), sendNum);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopyPad(attnTableGMTensor[startId], tempTensor, dataCopyParams);
}

template <TemplateAttentionToFfnUrmaTypeClass>
__aicore__ inline void AttentionToFfnUrma<TemplateAttentionToFfnUrmaTypeFunc>::ClearAttnActiveFlags()
{
    uint32_t startId = 0;
    uint32_t endId = 0;
    uint32_t clearNum = 0;
    uint32_t clearTotal = axisX_ * curBsCnt_ * (axisK_ + sharedExpertNum_);
    if (clearTotal == 0) {
        return;
    }
    SplitToCore(clearTotal, aivNum_, startId, endId, clearNum);
    if (startId >= clearTotal) {
        return;
    }

    uint64_t segmentOffset = static_cast<uint64_t>(microBatchId_) * axisBS_ * (axisK_ + sharedExpertNum_);
    GM_ADDR selfRankAddr = GetWindowAddr(static_cast<int32_t>(rankId_));
    GM_ADDR attnTableGM = selfRankAddr + segmentOffset * sizeof(int32_t);
    GlobalTensor<int32_t> attnTableGMTensor;
    attnTableGMTensor.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(attnTableGM));
    DataCopyExtParams clearParams = {1U, static_cast<uint32_t>(clearNum * sizeof(int32_t)), 0U, 0U, 0U};
    LocalTensor<int32_t> clearTensor = expertIdsBuf_.Get<int32_t>();
    Duplicate<int32_t>(clearTensor, static_cast<int32_t>(0), clearNum);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopyPad(attnTableGMTensor[startId], clearTensor, clearParams);
    PipeBarrier<PIPE_MTE3>();
}

template <TemplateAttentionToFfnUrmaTypeClass>
__aicore__ inline void AttentionToFfnUrma<TemplateAttentionToFfnUrmaTypeFunc>::SetFlagToFFN()
{
    if (aivId_ != 0U) {
        return;
    }

    statusTensor_.SetValue(0, URMA_FLAG_VALUE);
    statusTensor_.SetValue(1, static_cast<int32_t>(layerId_));
    SyncFunc<HardEvent::S_MTE3>();
    DataCopyExtParams statusParams = {1U, static_cast<uint32_t>(sizeof(int32_t) * TOKEN_INFO_TABLE_COPY_BLOCK_CNT), 0U,
                                      0U, 0U};

    if constexpr (isSync) {
        uint32_t sentFFNNumAlignSize = Ceil(ffnNum_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
        DataCopyExtParams syncStatusParams = {
            static_cast<uint16_t>(aivNum_), static_cast<uint32_t>(ffnNum_ * sizeof(int32_t)),
            static_cast<uint32_t>(aivWorkspaceOffset_ - ffnNum_ * sizeof(int32_t)), 0U, 0U};
        DataCopyPadExtParams<int32_t> copyPadParams{false, 0U, 0U, 0U};
        TBuf<> syncStatusWorkspaceBuf;
        tpipe_->InitBuffer(syncStatusWorkspaceBuf, aivNum_ * sentFFNNumAlignSize);
        LocalTensor<int32_t> syncStatusWorkspaceTensor = syncStatusWorkspaceBuf.Get<int32_t>();
        DataCopyPad(syncStatusWorkspaceTensor, syncStatusGMTensor_[0], syncStatusParams, copyPadParams);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        LocalTensor<float> syncStatusWorkspaceTensorFloat = syncStatusWorkspaceTensor.ReinterpretCast<float>();
        LocalTensor<float> ffnStatusTensorFloat = ffnStatusTensor_.ReinterpretCast<float>();
        const uint32_t shape[] = {aivNum_, static_cast<uint32_t>(sentFFNNumAlignSize / sizeof(float))};
        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::RA, true>(ffnStatusTensorFloat,
                                                                      syncStatusWorkspaceTensorFloat, shape, true);
        SyncFunc<AscendC::HardEvent::V_S>();
    }

    GlobalTensor<int32_t> localFlagGMTensor;
    localFlagGMTensor.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(layerFlagAddr_));
    DataCopyPad(localFlagGMTensor, statusTensor_, statusParams);
    SyncFunc<HardEvent::MTE3_S>();

    for (uint32_t ffnIdx = ffnStartRankId_; ffnIdx < ffnStartRankId_ + ffnNum_; ++ffnIdx) {
        if constexpr (isSync) {
            if (ffnStatusTensor_.GetValue(ffnIdx - ffnStartRankId_) == 0) {
                continue;
            }
        }
        GM_ADDR toRankAddr = GetWindowAddr(static_cast<int32_t>(ffnIdx));
        GM_ADDR tableFlagGM = toRankAddr + winInfoTableOffset_;
        if (ffnIdx == rankId_) {
            GlobalTensor<int32_t> tableFlagGMTensor;
            tableFlagGMTensor.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(tableFlagGM));
            DataCopyPad(tableFlagGMTensor, statusTensor_, statusParams);
            PipeBarrier<PIPE_MTE3>();
        } else {
            for (uint32_t channelIdx = 1U; channelIdx < channelsPerRank_; ++channelIdx) {
                int32_t drainRet = hcomm_.Drain(GetUrmaCommHandle(ffnIdx, channelIdx));
                ascendc_assert(drainRet == 0, "Urma drain before layer flag failed, ret=%d, rankId=%u, dstRank=%u",
                               drainRet, rankId_, ffnIdx);
            }
            uint64_t channel = GetUrmaCommHandle(ffnIdx);
            HcommBatchHandle batchHandle =
                hcomm_.MakeBatchHandle(channel, hcommBatchWqeTensor_, ATTN_FFN_HCOMM_BATCH_UB_BYTES,
                                       GetWindowAddr(static_cast<int32_t>(ffnIdx)));
            hcomm_.WriteNbi<REMOTE_ORDERED_FLAG_WQE_CONFIG>(
                batchHandle, tableFlagGM, layerFlagAddr_,
                static_cast<int64_t>(sizeof(int32_t) * TOKEN_INFO_TABLE_COPY_BLOCK_CNT));
            hcomm_.BatchCommit(batchHandle);
        }
    }
}

template <TemplateAttentionToFfnUrmaTypeClass>
__aicore__ inline void AttentionToFfnUrma<TemplateAttentionToFfnUrmaTypeFunc>::Process()
{
    if ASCEND_IS_AIV {
        HcommInit();

        if constexpr (isActiveMask) {
            ActiveMaskCalCnt();
        }

        totalSendNum_ = axisX_ * curBsCnt_ * (axisK_ + sharedExpertNum_);
        DataCopyExtParams expertIdsCntParams = {1U, static_cast<uint32_t>(expertIdsCnt_ * sizeof(int32_t)), 0U, 0U, 0U};
        DataCopyPadExtParams<int32_t> expertIdsCopyPadParams{false, 0U, 0U, 0U};
        DataCopyPad(expertIdsTensor_, expertIdsGMTensor_, expertIdsCntParams, expertIdsCopyPadParams);
        SyncFunc<AscendC::HardEvent::MTE2_S>();

        if constexpr (isSync) {
            Duplicate<int32_t>(ffnStatusTensor_, static_cast<int32_t>(0), ffnNumAlignCnt_);
        }

        LocalTensor<int32_t> rankCounts = rankInfoBuf_.Get<int32_t>();
        Duplicate<int32_t>(rankCounts, 0, rankCountElements_);
        SyncFunc<AscendC::HardEvent::V_S>();

        const uint64_t roundNum = Ceil(static_cast<uint64_t>(totalSendNum_), static_cast<uint64_t>(aivNum_));
        DataCopyExtParams xCopyParams = {1U, static_cast<uint32_t>(hSize_), 0U, 0U, 0U};
        remoteFlagCnt_ = 0;

        // Phase 1: metadata + staging. Each AIV also counts its remote records per rank and
        // keeps its own toRankId stash in UB for the scatter pass.
        for (uint64_t round = 0; round < roundNum; ++round) {
            const uint32_t tokenOffset = static_cast<uint32_t>(static_cast<uint64_t>(aivId_) + round * aivNum_);
            if (tokenOffset < totalSendNum_) {
                AttentionToFfnTokenMetaData metaData;
                ReadTokenMetaData(metaData, tokenOffset);
                tokenRankScratchTensor_.SetValue(round, metaData.toRankId);

                if constexpr (isSync) {
                    if (round == 0) {
                        SyncFunc<AscendC::HardEvent::V_S>();
                    }
                    int32_t ffnIdx = metaData.toRankId - static_cast<int32_t>(ffnStartRankId_);
                    if (ffnIdx >= 0 && static_cast<uint32_t>(ffnIdx) < ffnNum_) {
                        ffnStatusTensor_.SetValue(ffnIdx, 1);
                    }
                }

                if (metaData.toRankId == static_cast<int32_t>(rankId_)) {
                    SendToLocal(metaData, xCopyParams);
                } else {
                    StageRemoteData(metaData, xCopyParams, tokenOffset);
                    rankCounts.SetValue(metaData.toRankId, rankCounts.GetValue(metaData.toRankId) + 1);
                }
            }
        }

        if constexpr (isSync) {
            SyncFunc<AscendC::HardEvent::V_MTE3>();
            syncStatusGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(syncStatusWorkspaceGM_));
            uint32_t aivWorkspaceStride = aivWorkspaceOffset_ / sizeof(int32_t);
            DataCopy(syncStatusGMTensor_[aivId_ * aivWorkspaceStride], ffnStatusTensor_, ffnNumAlignCnt_);
        }

        SyncFunc<HardEvent::S_MTE3>();
        {
            DataCopyExtParams flagCopyParams = {1U, sizeof(int32_t), 0U, 0U, 0U};
            for (uint32_t i = 0; i < remoteFlagCnt_; ++i) {
                uint32_t tokOffset = flagOffsetTensor_.GetValue(i);
                const uint64_t flagBufOffset = static_cast<uint64_t>(tokOffset) * URMA_FLAG_SLOT_SIZE;
                GlobalTensor<int32_t> localFlagGMTensor;
                localFlagGMTensor.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(localFlagAddr_ + flagBufOffset));
                DataCopyPad(localFlagGMTensor, flagValueTensor_[i * FLAG_ELEM_STRIDE], flagCopyParams);
            }
        }
        PipeBarrier<PIPE_MTE3>();

        {
            GlobalTensor<int32_t> countRowGM;
            GM_ADDR countRowAddr = countMatrixBase_ + static_cast<uint64_t>(aivId_) * countRowStride_;
            countRowGM.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(countRowAddr));
            SyncFunc<HardEvent::S_MTE3>();
            DataCopy(countRowGM, rankCounts, rankCountElements_);
            SyncFunc<HardEvent::MTE3_S>();
        }
        SyncAll<true>();

        BuildRankTableOffsets();
        ScatterTokensToRankTables();
        SyncAll<true>();

        for (uint32_t dstRank = 0U; dstRank < worldSize_; ++dstRank) {
            if (dstRank == rankId_) {
                continue;
            }
            for (uint32_t channelIdx = 0U; channelIdx < channelsPerRank_; ++channelIdx) {
                uint32_t slot = dstRank * channelsPerRank_ + channelIdx;
                if (slot % aivNum_ != aivId_) {
                    continue;
                }
                RelayRemoteTokens(dstRank, channelIdx);
            }
        }

        SyncAll<true>();
        ClearAttnActiveFlags();
        SyncAll<true>();
        SetFlagToFFN();
        if constexpr (isActiveMask) {
            SetFlagInAttn();
        }
    }
}

#endif

} // namespace AttentionToFFNImpl
#endif // ATTENTION_TO_FFN_URMA_H
