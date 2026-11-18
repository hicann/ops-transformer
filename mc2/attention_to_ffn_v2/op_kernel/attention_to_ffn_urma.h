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

#if (ASC_DEVKIT_VERSION_NUM >= 90200000) && (HCOMM_VERSION_NUM >= 90200000)
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
#include "adv_api/index/arithprogression.h"
#include "attention_to_ffn_v2_tiling.h"

#include "../../common/op_kernel/attention_ffn_context.h"
#include "../../common/op_kernel/mc2_kernel_utils.h"

#include "utils/debug/asc_printf.h"
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
constexpr uint32_t SCALE_PARAM_PAD_SIZE = 128;
constexpr uint32_t WIN_ALIGN = 512;
constexpr uint32_t REP_STRIDE = 8;
constexpr uint32_t WORKSPACE_ELEMENT_OFFSET = 128;
constexpr uint32_t DYNAMIC_QUANT = 2;
constexpr uint32_t MX_QUANT = 3;
constexpr uint32_t MX_CLIP_QUANT = 4;
constexpr uint32_t RANK_OFFSET_STRIDE = 2;
// (toRankId, localExpId) pair stride, in int32 elements, inside expertRankResolveTensor_.
constexpr uint32_t RANK_RESOLVE_STRIDE = 2;
constexpr uint32_t TOKEN_INFO_TABLE_RS = 2;
constexpr uint32_t TOKEN_INFO_TABLE_COPY_BLOCK_CNT = 2;
constexpr float INT8_MAX_VALUE = 127.0f;
constexpr uint32_t FP4_ELEMS_PER_BYTE = 2;
constexpr uint32_t MX_BLOCK_SIZE = 32;
constexpr uint32_t MX_INPUT_READ_BYTES = 512U;

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
constexpr int32_t URMA_FLAG_VALUE = 1;

constexpr uint32_t HCOMM_BATCH_WQE_BYTES = 64U; // one WQEBB
constexpr uint32_t ATTN_FFN_HCOMM_BATCH_UB_BYTES = 16U * 1024U;
constexpr uint32_t URMA_BATCH_WQEBB_CAPACITY = ATTN_FFN_HCOMM_BATCH_UB_BYTES / HCOMM_BATCH_WQE_BYTES;

constexpr uint32_t RANK_INFO_SEGMENT_NUM = 2U;
constexpr uint32_t RANK_INFO_COUNT_SEG_IDX = 0U;
constexpr uint32_t RANK_INFO_TOTAL_SEG_IDX = 1U;
constexpr uint32_t BUCKET_BUFFER_SIZE = 32U * 1024U;
constexpr uint32_t RANK_SCAN_CHUNK_ELEMS = 2048U;
constexpr uint32_t RANK_SCAN_CMP_ALIGN = 256U;
constexpr int32_t ATTN_FFN_FLAG_SENTINEL = 0x40000000;

static constexpr AscendC::UrmaWqeEntry REMOTE_DATA_WQE_CONFIG = {.odr = 5, .fence = 1, .se = 0, .inlineEn = 0};
static constexpr AscendC::UrmaWqeEntry REMOTE_ORDERED_FLAG_WQE_CONFIG = {.odr = 6, .fence = 1, .se = 1, .inlineEn = 0};
constexpr uint64_t RELAY_WQE_MAX_BYTES = 2U * 1024U * 1024U;
static constexpr uint64_t ATTN_FFN_DATA_HOLE_MERGE_SLOTS = 1U;
using HcommBatchHandle = AscendC::BatchHandle<AscendC::ChannelHandle>;

struct AttentionToFfnTokenMetaData {
    int32_t toRankId;
    int32_t localExpId;
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
    __aicore__ inline int32_t GetDstExpertId(uint32_t tokenId, uint32_t topkId);
    __aicore__ inline void ReadTokenMetaData(AttentionToFfnTokenMetaData& metaData, uint32_t tokenId, uint32_t topkId);
    __aicore__ inline GM_ADDR GetWindowAddr(int32_t rankId);
    __aicore__ inline uint64_t GetUrmaCommHandle(uint32_t dstRank);
    __aicore__ inline void PrefetchXRow(uint32_t tokenId, const DataCopyExtParams& xCopyParams,
                                        uint32_t& mxZeroRemaining);
    __aicore__ inline void ConsumeXRow();
    __aicore__ inline void PackRowEntry(GM_ADDR packedAddr, int32_t dstExpertId, const DataCopyExtParams& xCopyParams,
                                        const DataCopyExtParams& hCommuCopyParams, bool rowQuantShared,
                                        bool& rowPayloadReady);
    __aicore__ inline void ReleaseRowPayload(bool rowQuantShared);
    __aicore__ inline void AppendRemoteDataWrite(HcommBatchHandle& batchHandle, GM_ADDR remoteDataAddr,
                                                 GM_ADDR sourceDataAddr, uint64_t byteLen);
    __aicore__ inline void AppendRemoteFlagWrite(HcommBatchHandle& batchHandle, GM_ADDR remoteFlagAddr,
                                                 GM_ADDR sourceFlagAddr, uint64_t byteLen);
    __aicore__ inline void RelayRemoteDataRuns(HcommBatchHandle& batchHandle, GM_ADDR toRankAddr, uint32_t dstRank,
                                               uint32_t& preparedWqeBb);
    __aicore__ inline void EmitDataRun(HcommBatchHandle& batchHandle, GM_ADDR toRankAddr, int64_t runStart,
                                       int64_t runLast, uint32_t& preparedWqeBb);
    __aicore__ inline void BuildRankTableOffsets();
    __aicore__ inline void BuildFlagSpans();
    __aicore__ inline void PackRows();
    __aicore__ inline void RelayRemoteTokens(uint32_t dstRank);
    __aicore__ inline void ResolveExpertRankTable();
    __aicore__ inline void FindExpertRank(int32_t expertId);
    __aicore__ inline void QuantInit(GM_ADDR scales);
    __aicore__ inline void QuantProcess(uint32_t expertIndex);
    __aicore__ inline void ReduceMaxInplace(const LocalTensor<float>& srcLocal, uint32_t count);
    __aicore__ inline float ComputeDynamicScale(const LocalTensor<float>& srcLocal);
    __aicore__ inline void SplitToCore(uint32_t curSendCnt, uint32_t curUseAivNum, uint32_t& startTokenId,
                                       uint32_t& sendTokenNum);
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
    GM_ADDR payloadBufferAddr_{nullptr};
    uint64_t payloadSlotStride_{0};
    GM_ADDR packedBaseAddr_{nullptr};
    GM_ADDR countMatrixBase_{nullptr};
    uint32_t rankCountElements_{0};
    uint32_t countRowStride_{0};
    GM_ADDR tokenRankTableBase_{nullptr};
    uint64_t flagSpanStride_{0};
    uint32_t relayDataSegTokenMax_{1};
    uint32_t relayFlagSegTokenMax_{1};

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
    LocalTensor<int32_t> syncStatusWorkspaceTensor_;
    LocalTensor<float> smoothScalesTensor_;
    LocalTensor<int32_t> expertIdsTensor_;
    LocalTensor<int32_t> expertRankResolveTensor_;
    LocalTensor<uint8_t> hcommTensor_;
    LocalTensor<int32_t> flagValueTensor_;
    LocalTensor<int32_t> tokenRankScratchTensor_;
    LocalTensor<int32_t> bucketTensor_;

    TBuf<> smoothScalesBuf_;
    TBuf<> expertIdsBuf_;
    TBuf<> expertRankResolveBuf_;
    TBuf<> statusBuf_;
    TBuf<> receiveDataCastFloatBuf_;
    TBuf<> sumOutBuf_;
    TBuf<> activeMaskBuf_;
    TBuf<> castTempBuf_;
    TBuf<> ffnStatusBuf_;
    TBuf<> syncStatusWorkspaceBuf_;
    TBuf<> hcommBuf_;
    TBuf<> hcommBatchBuf_;
    TBuf<> flagValueBuf_;
    TBuf<> tokenRankScratchBuf_;
    TBuf<> bucketBuf_;
    TBuf<> rankInfoBuf_;
    // Per-rank URMA window base addresses cached in UB by Init (epHcclBuffer_ is GM-resident and
    // stable for the whole kernel), so the relay/flag lookups read UB instead of GM.
    TBuf<> rankWinBaseBuf_;
    LocalTensor<uint64_t> rankWinBaseTensor_;
    LocalTensor<uint8_t> hcommBatchWqeTensor_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, BUFFER_NUM> xQueue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> xInQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> xOutQueue_;

    int32_t toRankId_{0};
    int32_t localExpId_{0};

    uint32_t aivId_{0};
    uint32_t rankId_{0};
    uint32_t axisX_{0};
    uint32_t axisBS_{0};
    uint32_t axisH_{0};
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
    uint32_t ctxChannelsPerRank_{1};
    uint32_t ffnNum_{0};
    uint32_t ffnStartRankId_{0};
    uint32_t sessionId_{0};
    uint32_t microBatchId_{0};
    uint32_t layerId_{0};
    uint32_t expertIdsCnt_{0};
    uint32_t totalSendNum_{0};
    uint32_t remoteFlagCnt_{0};
    uint32_t sendStartTokenId_{0};
    uint32_t sendTokenNum_{0};
    // First token row of this AIV's expert_ids slice; expert_ids reads subtract it.
    uint32_t expertIdsBaseTokenId_{0};
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
    uint64_t winDataBaseByte_{0};
    uint64_t winFlagBaseByte_{0};
    uint64_t remoteSlotBytes_{0};
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
    ctxChannelsPerRank_ = mc2Context_->channelsPerRank;
    if (worldSize_ == 0U || ctxChannelsPerRank_ == 0U ||
        ctxChannelsPerRank_ > Mc2Aclnn::HCCL_MAX_RANK_SIZE / worldSize_) {
        ctxChannelsPerRank_ = 1U;
    }
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

    expertIdsCnt_ = axisX_ * axisBS_ * (axisK_ + sharedExpertNum_);
    const uint64_t expertRankTableCnt = static_cast<uint64_t>(expertNum_) * expRankTableM_;
    layIdsExpRankTableOffset_ = static_cast<uint64_t>(layerId_) * expertRankTableCnt;
    ffnNumAlignSize_ = Ceil(ffnNum_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    ffnNumAlignCnt_ = ffnNumAlignSize_ / sizeof(int32_t);
    aivWorkspaceOffset_ = Ceil(ffnNum_ * sizeof(int32_t), WORKSPACE_ELEMENT_OFFSET) * WORKSPACE_ELEMENT_OFFSET;
    axisBsAlignSize_ = Ceil(axisBS_ * sizeof(bool), UB_ALIGN) * UB_ALIGN;

    uint32_t maxTokensPerAiv = Ceil(expertIdsCnt_, aivNum_);
    uint32_t maxRowsPerAiv = Ceil(maxTokensPerAiv, axisK_ + sharedExpertNum_) + 1U;
    uint32_t expertIdsSliceElems = maxRowsPerAiv * axisK_;
    uint32_t expertIdsElems = (expertIdsSliceElems > maxTokensPerAiv) ? expertIdsSliceElems : maxTokensPerAiv;
    uint32_t expertIdsAlign = Ceil(expertIdsElems * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(expertIdsBuf_, expertIdsAlign);
    expertIdsTensor_ = expertIdsBuf_.Get<int32_t>();
    tpipe_->InitBuffer(statusBuf_, UB_ALIGN);
    statusTensor_ = statusBuf_.Get<int32_t>();
    {
        GlobalTensor<uint64_t> ctxTailGM;
        ctxTailGM.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(&mc2Context_->payloadBufferAddr));
        DataCopyExtParams tailCopyParams{1U, sizeof(uint64_t) * 3U, 0U, 0U, 0U};
        DataCopyPadExtParams<uint64_t> tailPadParams{false, 0U, 0U, 0U};
        LocalTensor<uint64_t> tailTensor = statusTensor_.ReinterpretCast<uint64_t>();
        DataCopyPad(tailTensor, ctxTailGM, tailCopyParams, tailPadParams);
        SyncFunc<HardEvent::MTE2_S>();
        payloadBufferAddr_ = reinterpret_cast<GM_ADDR>(tailTensor.GetValue(0));
        payloadSlotStride_ = tailTensor.GetValue(2);
    }
    uint32_t expertRankResolveBytes = Ceil(RANK_RESOLVE_STRIDE * expertNum_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(expertRankResolveBuf_, expertRankResolveBytes);
    expertRankResolveTensor_ = expertRankResolveBuf_.Get<int32_t>();

    uint32_t flagValueBufSize = Ceil(maxTokensPerAiv * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(flagValueBuf_, flagValueBufSize);
    flagValueTensor_ = flagValueBuf_.Get<int32_t>();

    uint32_t tokenRankStashBytes = Ceil(maxTokensPerAiv * sizeof(int32_t), RANK_SCAN_CMP_ALIGN) * RANK_SCAN_CMP_ALIGN;
    tpipe_->InitBuffer(tokenRankScratchBuf_, tokenRankStashBytes);
    tokenRankScratchTensor_ = tokenRankScratchBuf_.Get<int32_t>();
    tpipe_->InitBuffer(bucketBuf_, BUCKET_BUFFER_SIZE);
    bucketTensor_ = bucketBuf_.Get<int32_t>();
    rankCountElements_ =
        static_cast<uint32_t>(Ceil(worldSize_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN / sizeof(int32_t));
    tpipe_->InitBuffer(rankInfoBuf_, RANK_INFO_SEGMENT_NUM * rankCountElements_ * sizeof(int32_t));
    const uint32_t winBaseRankNum = (worldSize_ == 0U) ? 1U : worldSize_;
    tpipe_->InitBuffer(rankWinBaseBuf_, Ceil(winBaseRankNum * sizeof(uint64_t), UB_ALIGN) * UB_ALIGN);
    rankWinBaseTensor_ = rankWinBaseBuf_.Get<uint64_t>();
    if (worldSize_ > 0U) {
        GlobalTensor<uint64_t> winBaseGM;
        winBaseGM.SetGlobalBuffer((__gm__ uint64_t*)&mc2Context_->epHcclBuffer_[0]);
        DataCopyExtParams winBaseCopyParams{1U, worldSize_ * static_cast<uint32_t>(sizeof(uint64_t)), 0U, 0U, 0U};
        DataCopyPadExtParams<uint64_t> winBasePadParams{false, 0U, 0U, 0U};
        DataCopyPad(rankWinBaseTensor_, winBaseGM, winBaseCopyParams, winBasePadParams);
        SyncFunc<HardEvent::MTE2_S>();
    }

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
        tpipe_->InitBuffer(syncStatusWorkspaceBuf_, aivNum_ * ffnNumAlignSize_);
        syncStatusWorkspaceTensor_ = syncStatusWorkspaceBuf_.Get<int32_t>();
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

    const uint64_t winTokenDataRegionBytes =
        Ceil(attentionWorkerNum_ * microBatchNum_ * infoTableLastDimNum_ * sizeof(int32_t), WIN_ALIGN) * WIN_ALIGN;
    winInfoTableOffset_ =
        (sessionId_ * microBatchNum_ * infoTableLastDimNum_ + microBatchId_ * infoTableLastDimNum_) * sizeof(int32_t);
    winTokenDataOffset_ =
        (static_cast<uint64_t>(sessionId_) * microBatchNum_ * axisBS_ * (axisK_ + sharedExpertNum_) * axisHS_) +
        (static_cast<uint64_t>(microBatchId_) * axisBS_ * (axisK_ + sharedExpertNum_) * axisHS_);

    const uint64_t elementSize = (QuantMode != 0) ? sizeof(XOutType) : sizeof(XType);

    remoteSlotBytes_ = static_cast<uint64_t>(axisHS_) * elementSize;
    winDataBaseByte_ = winTokenDataRegionBytes + winTokenDataOffset_ * elementSize;
    winFlagBaseByte_ = winInfoTableOffset_ + TOKEN_INFO_TABLE_RS * sizeof(int32_t);
    const uint64_t dataSegByBytes = RELAY_WQE_MAX_BYTES / remoteSlotBytes_;
    relayDataSegTokenMax_ = static_cast<uint32_t>(dataSegByBytes == 0U ? 1U : dataSegByBytes);
    relayFlagSegTokenMax_ = static_cast<uint32_t>(RELAY_WQE_MAX_BYTES / sizeof(int32_t));

    const uint64_t countMatrixOffset = static_cast<uint64_t>(aivNum_) * aivWorkspaceOffset_;
    countMatrixBase_ = workspaceGM + countMatrixOffset;
    countRowStride_ = static_cast<uint32_t>(Ceil(rankCountElements_ * sizeof(int32_t), WORKSPACE_ELEMENT_OFFSET) *
                                            WORKSPACE_ELEMENT_OFFSET);
    tokenRankTableBase_ = countMatrixBase_ + static_cast<uint64_t>(aivNum_) * countRowStride_;
    const uint64_t packedRegionSize = static_cast<uint64_t>(expertIdsCnt_) * remoteSlotBytes_;
    const uint64_t packedRegionAlignSize = Ceil(packedRegionSize, static_cast<uint64_t>(WIN_ALIGN)) * WIN_ALIGN;
    flagSpanStride_ =
        Ceil(static_cast<uint64_t>(expertIdsCnt_) * sizeof(int32_t), static_cast<uint64_t>(UB_ALIGN)) * UB_ALIGN;
    const uint64_t flagSpansBytes = static_cast<uint64_t>(worldSize_) * flagSpanStride_;
    const uint64_t slotInnerBytes =
        packedRegionAlignSize + flagSpansBytes + sizeof(int32_t) * TOKEN_INFO_TABLE_COPY_BLOCK_CNT;
    const uint64_t requiredSlotStride = Ceil(slotInnerBytes, static_cast<uint64_t>(WIN_ALIGN)) * WIN_ALIGN;
    if (unlikely(microBatchId_ >= microBatchNum_)) {
        AscendC::printf("[AttentionToFfnUrma][Init] microBatchId %u must be in [0, microBatchNum %u)\n", microBatchId_,
                        microBatchNum_);
        AscendC::Trap();
    } else if (unlikely(requiredSlotStride > payloadSlotStride_)) {
        AscendC::printf("[AttentionToFfnUrma][Init] context payloadSlotStride %llu is smaller than the required slot "
                        "stride %llu; rebuild the buffer for these shapes\n",
                        payloadSlotStride_, requiredSlotStride);
        AscendC::Trap();
    } else {
        GM_ADDR slotBaseAddr = payloadBufferAddr_ + static_cast<uint64_t>(microBatchId_) * payloadSlotStride_;
        packedBaseAddr_ = slotBaseAddr;
        localFlagAddr_ = slotBaseAddr + packedRegionAlignSize;
        layerFlagAddr_ = localFlagAddr_ + flagSpansBytes;
    }
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
        uint32_t hInSize = Ceil(axisH_ * sizeof(XType), MX_INPUT_READ_BYTES) * MX_INPUT_READ_BYTES;
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
__aicore__ inline float AttentionToFfnUrma<TemplateAttentionToFfnUrmaTypeFunc>::ComputeDynamicScale(
    const LocalTensor<float>& srcLocal)
{
    LocalTensor<float> floatLocalAbsTemp = smoothScalesBuf_.Get<float>();
    Abs(floatLocalAbsTemp, srcLocal, axisH_);
    PipeBarrier<PIPE_V>();
    ReduceMaxInplace(floatLocalAbsTemp, axisH_);
    SyncFunc<AscendC::HardEvent::V_S>();
    const float maxAbs = floatLocalAbsTemp.GetValue(0);
    const float dynamicScale = (maxAbs == 0.0f) ? 1.0f : float(INT8_MAX_VALUE) / maxAbs;
    SyncFunc<AscendC::HardEvent::S_V>();
    return dynamicScale;
}

template <TemplateAttentionToFfnUrmaTypeClass>
__aicore__ inline void AttentionToFfnUrma<TemplateAttentionToFfnUrmaTypeFunc>::QuantProcess(uint32_t expertIndex)
{
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    if constexpr (QuantMode == DYNAMIC_QUANT) {
        LocalTensor<float> floatLocalTemp;
        floatLocalTemp = receiveDataCastFloatBuf_.Get<float>();
        Cast(floatLocalTemp, xInTensor_, RoundMode::CAST_NONE, axisH_);
        PipeBarrier<PIPE_V>();

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

        float dynamicScale = ComputeDynamicScale(floatLocalTemp);
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
        __ubuf__ uint16_t* mxScaleLocalAddr = (__ubuf__ uint16_t*)xOutTensor_[hOutSizeAlign_].GetPhyAddr();
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
    } else if constexpr (IsMxClipQuant(QuantMode)) {
        uint32_t mxScaleNum = Ceil(Ceil(axisH_, MX_BLOCK_SIZE), 2U) * 2U;
        LocalTensor<float> receiveDataCastFloat = receiveDataCastFloatBuf_.Get<float>();
        __ubuf__ StorageXInType* srcAddr = (__ubuf__ StorageXInType*)xInTensor_.GetPhyAddr();
        __ubuf__ uint16_t* maxExpAddr = (__ubuf__ uint16_t*)receiveDataCastFloat.GetPhyAddr();
        __ubuf__ uint16_t* halfScaleLocalAddr =
            (__ubuf__ uint16_t*)receiveDataCastFloat[Ceil(mxScaleNum, 32U) * 32U].GetPhyAddr();
        __ubuf__ int8_t* outLocalAddr = (__ubuf__ int8_t*)xOutTensor_.GetPhyAddr();
        __ubuf__ uint16_t* mxScaleLocalAddr = (__ubuf__ uint16_t*)xOutTensor_[hOutSizeAlign_].GetPhyAddr();
        Quant::ComputeMaxExpClip(srcAddr, maxExpAddr, axisH_);
        Quant::ComputeScaleClip<XOutType, StorageXInType>(maxExpAddr, mxScaleLocalAddr, halfScaleLocalAddr, mxScaleNum);
        Quant::ComputeFp8Data<StorageXInType, XOutType, AscendC::RoundMode::CAST_TRUNC, AscendC::RoundMode::CAST_RINT>(
            srcAddr, halfScaleLocalAddr, outLocalAddr, axisH_);
        PipeBarrier<PIPE_V>();
    }
#endif
}

template <TemplateAttentionToFfnUrmaTypeClass>
__aicore__ inline void AttentionToFfnUrma<TemplateAttentionToFfnUrmaTypeFunc>::ResolveExpertRankTable()
{
    const uint32_t rowBytes = expRankTableM_ * sizeof(int32_t);
    const uint32_t expertsPerChunk = BUCKET_BUFFER_SIZE / rowBytes;
    GlobalTensor<int32_t> expertRankTableGM = expertRankTableGMTensor_[layIdsExpRankTableOffset_];
    LocalTensor<int32_t> chunkTensor = bucketTensor_;
    const DataCopyPadExtParams<int32_t> padParams{false, 0U, 0U, 0U};
    for (uint32_t start = 0U; start < expertNum_; start += expertsPerChunk) {
        const uint32_t expertCnt = (start + expertsPerChunk > expertNum_) ? expertNum_ - start : expertsPerChunk;
        DataCopyExtParams copyParams{1U, expertCnt * rowBytes, 0U, 0U, 0U};
        DataCopyPad(chunkTensor, expertRankTableGM[static_cast<uint64_t>(start) * expRankTableM_], copyParams,
                    padParams);
        SyncFunc<HardEvent::MTE2_S>();
        for (uint32_t e = start; e < start + expertCnt; ++e) {
            const uint32_t rowBase = (e - start) * expRankTableM_;
            const uint32_t rankCnt = static_cast<uint32_t>(chunkTensor.GetValue(rowBase));
            int32_t toRankId = -1;
            int32_t localExpId = -1;
            if (rankCnt != 0U) {
                if (unlikely(static_cast<uint64_t>(rankCnt) * RANK_OFFSET_STRIDE + 1 > expRankTableM_)) {
                    AscendC::printf("[AttentionToFfnUrma][ResolveExpertRankTable] rankCnt %u exceeds "
                                    "expert_rank_table row capacity M %u\n",
                                    rankCnt, expRankTableM_);
                    AscendC::Trap();
                } else {
                    const uint32_t rankOffset = (sessionId_ % rankCnt) * RANK_OFFSET_STRIDE + 1;
                    toRankId = chunkTensor.GetValue(rowBase + rankOffset);
                    localExpId = chunkTensor.GetValue(rowBase + rankOffset + 1U);
                }
            }
            expertRankResolveTensor_.SetValue(e * RANK_RESOLVE_STRIDE, toRankId);
            expertRankResolveTensor_.SetValue(e * RANK_RESOLVE_STRIDE + 1U, localExpId);
        }
        if (start + expertCnt < expertNum_) {
            SyncFunc<HardEvent::S_MTE2>(); // Scalar reads must drain before the next chunk overwrite.
        }
    }
}

template <TemplateAttentionToFfnUrmaTypeClass>
__aicore__ inline void AttentionToFfnUrma<TemplateAttentionToFfnUrmaTypeFunc>::FindExpertRank(int32_t expertId)
{
    // The (toRankId, localExpId) pair was resolved into UB by ResolveExpertRankTable; -1 marks an
    // expert without a rank placement, which keeps the previous mapping exactly like the GM path.
    const int32_t toRankId = expertRankResolveTensor_.GetValue(static_cast<uint32_t>(expertId) * RANK_RESOLVE_STRIDE);
    if (toRankId < 0) {
        return;
    }
    toRankId_ = toRankId;
    localExpId_ = expertRankResolveTensor_.GetValue(static_cast<uint32_t>(expertId) * RANK_RESOLVE_STRIDE + 1U);
}

template <TemplateAttentionToFfnUrmaTypeClass>
__aicore__ inline void AttentionToFfnUrma<TemplateAttentionToFfnUrmaTypeFunc>::SplitToCore(uint32_t curSendCnt,
                                                                                           uint32_t curUseAivNum,
                                                                                           uint32_t& startTokenId,
                                                                                           uint32_t& sendTokenNum)
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
    return (GM_ADDR)rankWinBaseTensor_(static_cast<uint32_t>(rankId));
}

template <TemplateAttentionToFfnUrmaTypeClass>
__aicore__ inline uint64_t AttentionToFfnUrma<TemplateAttentionToFfnUrmaTypeFunc>::GetUrmaCommHandle(uint32_t dstRank)
{
    return mc2Context_->hcommHandle_[dstRank * ctxChannelsPerRank_];
}

template <TemplateAttentionToFfnUrmaTypeClass>
__aicore__ inline int32_t AttentionToFfnUrma<TemplateAttentionToFfnUrmaTypeFunc>::GetDstExpertId(uint32_t tokenId,
                                                                                                 uint32_t topkId)
{
    if (topkId < axisK_) {
        return expertIdsTensor_.GetValue((tokenId - expertIdsBaseTokenId_) * axisK_ + topkId);
    }
    return static_cast<int32_t>(moeExpertNum_ + (topkId - axisK_));
}

template <TemplateAttentionToFfnUrmaTypeClass>
__aicore__ inline void AttentionToFfnUrma<TemplateAttentionToFfnUrmaTypeFunc>::ReadTokenMetaData(
    AttentionToFfnTokenMetaData& metaData, uint32_t tokenId, uint32_t topkId)
{
    FindExpertRank(GetDstExpertId(tokenId, topkId));
    metaData.toRankId = toRankId_;
    metaData.localExpId = localExpId_;
}

template <TemplateAttentionToFfnUrmaTypeClass>
__aicore__ inline void AttentionToFfnUrma<TemplateAttentionToFfnUrmaTypeFunc>::PrefetchXRow(
    uint32_t tokenId, const DataCopyExtParams& xCopyParams, uint32_t& mxZeroRemaining)
{
    DataCopyPadExtParams<XType> copyPadExtParams{false, 0U, 0U, 0U};
    if constexpr (QuantMode == 0) {
        LocalTensor<XType> xTmp = xQueue_.AllocTensor<XType>();
        DataCopyPad(xTmp, xGMTensor_[static_cast<uint64_t>(tokenId) * axisH_], xCopyParams, copyPadExtParams);
        xQueue_.EnQue(xTmp);
    } else {
        LocalTensor<XType> xIn = xInQueue_.AllocTensor<XType>();
        if constexpr (IsMxOrMxClipQuant(QuantMode)) {
            if (mxZeroRemaining > 0U) {
                --mxZeroRemaining;
                LocalTensor<uint8_t> singleByteTok = xIn.template ReinterpretCast<uint8_t>();
                Duplicate(singleByteTok, static_cast<uint8_t>(0),
                          Ceil(axisH_ * sizeof(XType), MX_INPUT_READ_BYTES) * MX_INPUT_READ_BYTES);
                SyncFunc<HardEvent::V_MTE2>();
            }
            const uint32_t tailPadElems = (UB_ALIGN - axisH_ * sizeof(XType) % UB_ALIGN) % UB_ALIGN / sizeof(XType);
            copyPadExtParams =
                DataCopyPadExtParams<XType>{true, 0U, static_cast<uint8_t>(tailPadElems), static_cast<XType>(0)};
        }
        DataCopyPad(xIn, xGMTensor_[static_cast<uint64_t>(tokenId) * axisH_], xCopyParams, copyPadExtParams);
        xInQueue_.EnQue(xIn);
    }
}

template <TemplateAttentionToFfnUrmaTypeClass>
__aicore__ inline void AttentionToFfnUrma<TemplateAttentionToFfnUrmaTypeFunc>::ConsumeXRow()
{
    if constexpr (QuantMode == 0) {
        xTmpTensor_ = xQueue_.DeQue<XType>();
    } else {
        xInTensor_ = xInQueue_.DeQue<XType>();
    }
}

// Return the row payload buffers once the row's entries are all packed.
template <TemplateAttentionToFfnUrmaTypeClass>
__aicore__ inline void AttentionToFfnUrma<TemplateAttentionToFfnUrmaTypeFunc>::ReleaseRowPayload(bool rowQuantShared)
{
    if constexpr (QuantMode == 0) {
        xQueue_.FreeTensor<XType>(xTmpTensor_);
    } else {
        if (rowQuantShared) {
            xOutQueue_.FreeTensor<StorageXOutType>(xOutTensor_);
        }
        xInQueue_.FreeTensor<XType>(xInTensor_);
    }
}

template <TemplateAttentionToFfnUrmaTypeClass>
__aicore__ inline void AttentionToFfnUrma<TemplateAttentionToFfnUrmaTypeFunc>::PackRowEntry(
    GM_ADDR packedAddr, int32_t dstExpertId, const DataCopyExtParams& xCopyParams,
    const DataCopyExtParams& hCommuCopyParams, bool rowQuantShared, bool& rowPayloadReady)
{
    if constexpr (QuantMode == 0) {
        GlobalTensor<XType> targetGMTensor;
        targetGMTensor.SetGlobalBuffer(reinterpret_cast<__gm__ XType*>(packedAddr));
        DataCopyPad(targetGMTensor, xTmpTensor_, xCopyParams);
    } else {
        if (!rowQuantShared || !rowPayloadReady) {
            xOutTensor_ = xOutQueue_.AllocTensor<StorageXOutType>();
            QuantProcess(dstExpertId);
            xOutQueue_.EnQue(xOutTensor_);
            xOutTensor_ = xOutQueue_.DeQue<StorageXOutType>();
            rowPayloadReady = true;
        }
        GlobalTensor<uint8_t> targetGMTensor;
        targetGMTensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(packedAddr));
        auto xOutBytesTensor = xOutTensor_.template ReinterpretCast<uint8_t>();
        DataCopyPad(targetGMTensor, xOutBytesTensor, hCommuCopyParams);
        if (!rowQuantShared) {
            xOutQueue_.FreeTensor<StorageXOutType>(xOutTensor_);
        }
    }
}

template <TemplateAttentionToFfnUrmaTypeClass>
__aicore__ inline void AttentionToFfnUrma<TemplateAttentionToFfnUrmaTypeFunc>::AppendRemoteDataWrite(
    HcommBatchHandle& batchHandle, GM_ADDR remoteDataAddr, GM_ADDR sourceDataAddr, uint64_t byteLen)
{
    hcomm_.WriteNbi<REMOTE_DATA_WQE_CONFIG>(batchHandle, remoteDataAddr, sourceDataAddr, static_cast<int64_t>(byteLen));
}

template <TemplateAttentionToFfnUrmaTypeClass>
__aicore__ inline void AttentionToFfnUrma<TemplateAttentionToFfnUrmaTypeFunc>::AppendRemoteFlagWrite(
    HcommBatchHandle& batchHandle, GM_ADDR remoteFlagAddr, GM_ADDR sourceFlagAddr, uint64_t byteLen)
{
    hcomm_.WriteNbi<REMOTE_ORDERED_FLAG_WQE_CONFIG>(batchHandle, remoteFlagAddr, sourceFlagAddr,
                                                    static_cast<int64_t>(byteLen));
}

template <TemplateAttentionToFfnUrmaTypeClass>
__aicore__ inline void AttentionToFfnUrma<TemplateAttentionToFfnUrmaTypeFunc>::EmitDataRun(
    HcommBatchHandle& batchHandle, GM_ADDR toRankAddr, int64_t runStart, int64_t runLast, uint32_t& preparedWqeBb)
{
    uint64_t segStart = static_cast<uint64_t>(runStart);
    const uint64_t runEnd = static_cast<uint64_t>(runLast) + 1U;
    while (segStart < runEnd) {
        const uint64_t remaining = runEnd - segStart;
        const uint64_t segLen = (remaining > relayDataSegTokenMax_) ? relayDataSegTokenMax_ : remaining;
        GM_ADDR remoteDataAddr = toRankAddr + winDataBaseByte_ + segStart * remoteSlotBytes_;
        GM_ADDR sourceDataAddr = packedBaseAddr_ + segStart * remoteSlotBytes_;
        AppendRemoteDataWrite(batchHandle, remoteDataAddr, sourceDataAddr, segLen * remoteSlotBytes_);
        ++preparedWqeBb;
        if (preparedWqeBb == URMA_BATCH_WQEBB_CAPACITY) {
            hcomm_.BatchCommit(batchHandle);
            preparedWqeBb = 0U;
        }
        segStart += segLen;
    }
}

template <TemplateAttentionToFfnUrmaTypeClass>
__aicore__ inline void AttentionToFfnUrma<TemplateAttentionToFfnUrmaTypeFunc>::RelayRemoteDataRuns(
    HcommBatchHandle& batchHandle, GM_ADDR toRankAddr, uint32_t dstRank, uint32_t& preparedWqeBb)
{
    LocalTensor<int32_t> staging = bucketTensor_;
    LocalTensor<int32_t> posList = bucketTensor_[RANK_SCAN_CHUNK_ELEMS];
    LocalTensor<uint8_t> scanMask =
        bucketTensor_.ReinterpretCast<uint8_t>()[2 * RANK_SCAN_CHUNK_ELEMS * sizeof(int32_t)];
    GlobalTensor<int32_t> tokenRankGM;
    tokenRankGM.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(tokenRankTableBase_));
    const DataCopyPadExtParams<int32_t> padParams{false, 0U, 0U, 0U};

    constexpr int64_t RUN_CLOSED = -1;
    int64_t runStart = RUN_CLOSED;
    int64_t runLast = RUN_CLOSED;
    for (uint32_t chunkStart = 0U; chunkStart < totalSendNum_; chunkStart += RANK_SCAN_CHUNK_ELEMS) {
        const uint32_t chunkLen =
            (chunkStart + RANK_SCAN_CHUNK_ELEMS > totalSendNum_) ? totalSendNum_ - chunkStart : RANK_SCAN_CHUNK_ELEMS;
        const DataCopyExtParams copyParams{1U, static_cast<uint32_t>(chunkLen * sizeof(int32_t)), 0U, 0U, 0U};
        DataCopyPad(staging, tokenRankGM[chunkStart], copyParams, padParams);
        SyncFunc<HardEvent::MTE2_V>();
        ArithProgression<int32_t>(posList, 0, 1, chunkLen);
        CompareScalar(scanMask, staging, static_cast<int32_t>(dstRank), AscendC::CMPMODE::EQ,
                      Ceil(chunkLen * sizeof(int32_t), RANK_SCAN_CMP_ALIGN) * RANK_SCAN_CMP_ALIGN / sizeof(int32_t));
        PipeBarrier<PIPE_V>();
        uint64_t rsvdCnt = 0;
        GatherMask(posList, posList, scanMask.ReinterpretCast<uint32_t>(), true, chunkLen, {1, 1, 0, 0}, rsvdCnt);
        SyncFunc<HardEvent::V_S>();
        for (uint32_t k = 0U; k < static_cast<uint32_t>(rsvdCnt); ++k) {
            const int64_t pos = static_cast<int64_t>(chunkStart) + static_cast<int64_t>(posList.GetValue(k));
            if (runStart >= 0) {
                const uint64_t holeSlots = static_cast<uint64_t>(pos - runLast - 1);
                if (holeSlots > ATTN_FFN_DATA_HOLE_MERGE_SLOTS) {
                    EmitDataRun(batchHandle, toRankAddr, runStart, runLast, preparedWqeBb);
                    runStart = RUN_CLOSED;
                }
            }
            if (runStart < 0) {
                runStart = pos;
            }
            runLast = pos;
        }
        SyncFunc<HardEvent::V_MTE2>(); // the next MTE2 reuses staging after the vector reads drain
        SyncFunc<HardEvent::S_V>();    // the next chunk's vector writes wait for the scalar reads
    }
    if (runStart >= 0) {
        EmitDataRun(batchHandle, toRankAddr, runStart, runLast, preparedWqeBb);
    }
}

template <TemplateAttentionToFfnUrmaTypeClass>
__aicore__ inline void AttentionToFfnUrma<TemplateAttentionToFfnUrmaTypeFunc>::BuildRankTableOffsets()
{
    LocalTensor<int32_t> rankInfo = rankInfoBuf_.Get<int32_t>();
    // The count segment was already published to GM, so it is reused as the ReduceSum scratch.
    LocalTensor<int32_t> reduceSums = rankInfo[RANK_INFO_COUNT_SEG_IDX * rankCountElements_];
    LocalTensor<int32_t> totalCounts = rankInfo[RANK_INFO_TOTAL_SEG_IDX * rankCountElements_];
    Duplicate<int32_t>(totalCounts, 0, rankCountElements_);
    SyncFunc<HardEvent::V_S>();

    GlobalTensor<int32_t> countMatrixGM;
    countMatrixGM.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(countMatrixBase_));
    const uint32_t rowBytes = rankCountElements_ * sizeof(int32_t);
    LocalTensor<int32_t> countRows = bucketTensor_;
    const uint32_t rowsPerBatch = BUCKET_BUFFER_SIZE / rowBytes;
    const DataCopyPadExtParams<int32_t> padParams{false, 0U, 0U, 0U};

    for (uint32_t rowStart = 0U; rowStart < aivNum_;) {
        const uint32_t copyRows = (rowsPerBatch < aivNum_ - rowStart) ? rowsPerBatch : aivNum_ - rowStart;
        DataCopyExtParams copyParams{static_cast<uint16_t>(copyRows), rowBytes,
                                     static_cast<uint32_t>(countRowStride_ - rowBytes), 0U, 0U};
        DataCopyPad(countRows, countMatrixGM[static_cast<uint64_t>(rowStart) * countRowStride_ / sizeof(int32_t)],
                    copyParams, padParams);
        SyncFunc<HardEvent::MTE2_V>();
        const uint32_t batchShape[] = {copyRows, rankCountElements_};
        ReduceSum<int32_t, AscendC::Pattern::Reduce::RA, true>(reduceSums, countRows, batchShape, false);
        Add(totalCounts, totalCounts, reduceSums, rankCountElements_);
        rowStart += copyRows;
        if (rowStart < aivNum_) {
            SyncFunc<HardEvent::V_MTE2>();
        }
    }
    SyncFunc<HardEvent::V_S>();
}

template <TemplateAttentionToFfnUrmaTypeClass>
__aicore__ inline void AttentionToFfnUrma<TemplateAttentionToFfnUrmaTypeFunc>::BuildFlagSpans()
{
    if (sendTokenNum_ == 0U) {
        return;
    }
    LocalTensor<int32_t> rankInfo = rankInfoBuf_.Get<int32_t>();
    LocalTensor<int32_t> totalCounts = rankInfo[RANK_INFO_TOTAL_SEG_IDX * rankCountElements_];
    LocalTensor<int32_t> staging = bucketTensor_;
    LocalTensor<int32_t> posList = bucketTensor_[RANK_SCAN_CHUNK_ELEMS];
    LocalTensor<uint8_t> scanMask =
        bucketTensor_.ReinterpretCast<uint8_t>()[2 * RANK_SCAN_CHUNK_ELEMS * sizeof(int32_t)];
    SyncFunc<HardEvent::S_V>(); // the scratch entries were produced by scalar writes in pass 1

    for (uint32_t rank = 0U; rank < worldSize_; ++rank) {
        if (totalCounts.GetValue(rank) == 0) {
            continue;
        }
        GM_ADDR spanBase = localFlagAddr_ + static_cast<uint64_t>(rank) * flagSpanStride_;
        GlobalTensor<int32_t> spanGM;
        spanGM.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(spanBase));
        for (uint32_t chunkStart = 0U; chunkStart < sendTokenNum_; chunkStart += RANK_SCAN_CHUNK_ELEMS) {
            const uint32_t chunkLen = (chunkStart + RANK_SCAN_CHUNK_ELEMS > sendTokenNum_) ?
                                          sendTokenNum_ - chunkStart :
                                          RANK_SCAN_CHUNK_ELEMS;
            Duplicate<int32_t>(staging, ATTN_FFN_FLAG_SENTINEL, chunkLen);
            ArithProgression<int32_t>(posList, 0, 1, chunkLen);
            LocalTensor<int32_t> scratchChunk = tokenRankScratchTensor_[chunkStart];
            CompareScalar(
                scanMask, scratchChunk, static_cast<int32_t>(rank), AscendC::CMPMODE::EQ,
                Ceil(chunkLen * sizeof(int32_t), RANK_SCAN_CMP_ALIGN) * RANK_SCAN_CMP_ALIGN / sizeof(int32_t));
            PipeBarrier<PIPE_V>();
            uint64_t rsvdCnt = 0;
            GatherMask(posList, posList, scanMask.ReinterpretCast<uint32_t>(), true, chunkLen, {1, 1, 0, 0}, rsvdCnt);
            SyncFunc<HardEvent::V_S>();
            for (uint32_t k = 0U; k < static_cast<uint32_t>(rsvdCnt); ++k) {
                const uint32_t pos = static_cast<uint32_t>(posList.GetValue(k));
                staging.SetValue(pos, flagValueTensor_.GetValue(chunkStart + pos));
            }
            SyncFunc<HardEvent::S_MTE3>();
            DataCopyExtParams copyParams{1U, static_cast<uint32_t>(chunkLen * sizeof(int32_t)), 0U, 0U, 0U};
            DataCopyPad(spanGM[sendStartTokenId_ + chunkStart], staging, copyParams);
            SyncFunc<HardEvent::MTE3_V>(); // staging is reused by the next chunk
            SyncFunc<HardEvent::S_V>();    // the next chunk's vector writes wait for the scalar reads
        }
    }
    SyncFunc<HardEvent::MTE3_S>();
}

template <TemplateAttentionToFfnUrmaTypeClass>
__aicore__ inline void AttentionToFfnUrma<TemplateAttentionToFfnUrmaTypeFunc>::PackRows()
{
    if (sendTokenNum_ == 0U) {
        return;
    }
    const DataCopyExtParams xCopyParams = {1U, static_cast<uint32_t>(hSize_), 0U, 0U, 0U};
    const DataCopyExtParams hCommuCopyParams = {1U, static_cast<uint32_t>(hCommuSize_), 0U, 0U, 0U};
    const uint32_t tokensPerRow = axisK_ + sharedExpertNum_;
    const uint32_t endTokenId = sendStartTokenId_ + sendTokenNum_;
    uint32_t tokenOffset = sendStartTokenId_;
    uint32_t tokenId = 0U;
    uint32_t topkId = 0U;
    if (tokensPerRow > 0U) {
        tokenId = sendStartTokenId_ / tokensPerRow;
        topkId = sendStartTokenId_ % tokensPerRow;
    }
    bool rowQuantShared = true;
    if constexpr (QuantMode == DYNAMIC_QUANT) {
        rowQuantShared = !isScales_;
    }
    uint32_t mxZeroRemaining = BUFFER_NUM;
    PrefetchXRow(tokenId, xCopyParams, mxZeroRemaining);
    while (tokenOffset < endTokenId) {
        const uint32_t rowRemain = tokensPerRow - topkId;
        const uint32_t rangeRemain = endTokenId - tokenOffset;
        const uint32_t rowSegLen = (rowRemain < rangeRemain) ? rowRemain : rangeRemain;
        const bool rowComplete = (rowSegLen == rowRemain);
        if (rowComplete && tokenOffset + rowSegLen < endTokenId) {
            PrefetchXRow(tokenId + 1U, xCopyParams, mxZeroRemaining);
        }
        ConsumeXRow();
        bool rowPayloadReady = false;
        for (uint32_t j = 0U; j < rowSegLen; ++j) {
            const int32_t dstExpertId = GetDstExpertId(tokenId, topkId);
            PackRowEntry(packedBaseAddr_ + static_cast<uint64_t>(tokenOffset) * remoteSlotBytes_, dstExpertId,
                         xCopyParams, hCommuCopyParams, rowQuantShared, rowPayloadReady);
            ++tokenOffset;
            ++topkId;
        }
        ReleaseRowPayload(rowQuantShared);
        if (topkId == tokensPerRow) {
            topkId = 0U;
            ++tokenId;
        }
    }
    SyncFunc<HardEvent::MTE3_S>();
}

template <TemplateAttentionToFfnUrmaTypeClass>
__aicore__ inline void AttentionToFfnUrma<TemplateAttentionToFfnUrmaTypeFunc>::RelayRemoteTokens(uint32_t dstRank)
{
    LocalTensor<int32_t> rankInfo = rankInfoBuf_.Get<int32_t>();
    LocalTensor<int32_t> totalCounts = rankInfo[RANK_INFO_TOTAL_SEG_IDX * rankCountElements_];
    if (totalCounts.GetValue(dstRank) == 0) {
        return;
    }
    GM_ADDR toRankAddr = GetWindowAddr(static_cast<int32_t>(dstRank));
    uint64_t channel = GetUrmaCommHandle(dstRank);
    HcommBatchHandle batchHandle =
        hcomm_.MakeBatchHandle(channel, hcommBatchWqeTensor_, ATTN_FFN_HCOMM_BATCH_UB_BYTES, toRankAddr);
    uint32_t preparedWqeBb = 0U;

    RelayRemoteDataRuns(batchHandle, toRankAddr, dstRank, preparedWqeBb);

    GM_ADDR flagSpanBase = localFlagAddr_ + static_cast<uint64_t>(dstRank) * flagSpanStride_;
    for (uint32_t segStart = 0U; segStart < totalSendNum_; segStart += relayFlagSegTokenMax_) {
        const uint32_t remaining = totalSendNum_ - segStart;
        const uint32_t segLen = (remaining > relayFlagSegTokenMax_) ? relayFlagSegTokenMax_ : remaining;
        GM_ADDR remoteFlagAddr = toRankAddr + winFlagBaseByte_ + static_cast<uint64_t>(segStart) * sizeof(int32_t);
        GM_ADDR sourceFlagAddr = flagSpanBase + static_cast<uint64_t>(segStart) * sizeof(int32_t);
        AppendRemoteFlagWrite(batchHandle, remoteFlagAddr, sourceFlagAddr,
                              static_cast<uint64_t>(segLen) * sizeof(int32_t));
        ++preparedWqeBb;
        if (preparedWqeBb == URMA_BATCH_WQEBB_CAPACITY) {
            hcomm_.BatchCommit(batchHandle);
            preparedWqeBb = 0U;
        }
    }
    if (preparedWqeBb != 0U) {
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
    uint32_t sendNum = 0;
    uint32_t totalNum = axisX_ * (axisBS_ - curBsCnt_) * (axisK_ + sharedExpertNum_);
    if (totalNum == 0) {
        return;
    }
    SplitToCore(totalNum, aivNum_, startId, sendNum);
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
    uint32_t clearNum = 0;
    uint32_t clearTotal = axisX_ * curBsCnt_ * (axisK_ + sharedExpertNum_);
    if (clearTotal == 0) {
        return;
    }
    SplitToCore(clearTotal, aivNum_, startId, clearNum);
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
    const uint32_t endRank = ffnStartRankId_ + ffnNum_;
    const uint32_t firstOwned = ffnStartRankId_ + (aivId_ + aivNum_ - ffnStartRankId_ % aivNum_) % aivNum_;
    if (aivId_ >= aivNum_ || firstOwned >= endRank) {
        return;
    }

    statusTensor_.SetValue(0, URMA_FLAG_VALUE);
    statusTensor_.SetValue(1, static_cast<int32_t>(layerId_));
    SyncFunc<HardEvent::S_MTE3>();
    DataCopyExtParams statusParams = {1U, static_cast<uint32_t>(sizeof(int32_t) * TOKEN_INFO_TABLE_COPY_BLOCK_CNT), 0U,
                                      0U, 0U};

    if constexpr (isSync) {
        DataCopyExtParams syncStatusParams = {
            static_cast<uint16_t>(aivNum_), static_cast<uint32_t>(ffnNum_ * sizeof(int32_t)),
            static_cast<uint32_t>(aivWorkspaceOffset_ - ffnNum_ * sizeof(int32_t)), 0U, 0U};
        DataCopyPadExtParams<int32_t> copyPadParams{false, 0U, 0U, 0U};
        DataCopyPad(syncStatusWorkspaceTensor_, syncStatusGMTensor_[0], syncStatusParams, copyPadParams);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        LocalTensor<float> syncStatusWorkspaceTensorFloat = syncStatusWorkspaceTensor_.ReinterpretCast<float>();
        LocalTensor<float> ffnStatusTensorFloat = ffnStatusTensor_.ReinterpretCast<float>();
        const uint32_t shape[] = {aivNum_, ffnNumAlignCnt_};
        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::RA, true>(ffnStatusTensorFloat,
                                                                      syncStatusWorkspaceTensorFloat, shape, true);
        SyncFunc<AscendC::HardEvent::V_S>();
    }

    GlobalTensor<int32_t> localFlagGMTensor;
    localFlagGMTensor.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(layerFlagAddr_));
    DataCopyPad(localFlagGMTensor, statusTensor_, statusParams);
    SyncFunc<HardEvent::MTE3_S>();

    for (uint32_t ffnIdx = firstOwned; ffnIdx < endRank; ffnIdx += aivNum_) {
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
            uint64_t channel = GetUrmaCommHandle(ffnIdx);
            HcommBatchHandle batchHandle =
                hcomm_.MakeBatchHandle(channel, hcommBatchWqeTensor_, ATTN_FFN_HCOMM_BATCH_UB_BYTES, toRankAddr);
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
        uint32_t startTokenId = 0;
        uint32_t sendTokenNum = 0;
        SplitToCore(totalSendNum_, aivNum_, startTokenId, sendTokenNum);
        const uint32_t endTokenId = startTokenId + sendTokenNum;
        sendStartTokenId_ = startTokenId;
        sendTokenNum_ = sendTokenNum;
        DataCopyPadExtParams<int32_t> expertIdsCopyPadParams{false, 0U, 0U, 0U};
        if (sendTokenNum > 0U && axisK_ > 0U) {
            expertIdsBaseTokenId_ = startTokenId / (axisK_ + sharedExpertNum_);
            const uint32_t lastTokenId = (endTokenId - 1U) / (axisK_ + sharedExpertNum_);
            const uint32_t sliceElemCnt = (lastTokenId - expertIdsBaseTokenId_ + 1U) * axisK_;
            DataCopyExtParams expertIdsSliceParams = {1U, static_cast<uint32_t>(sliceElemCnt * sizeof(int32_t)), 0U, 0U,
                                                      0U};
            DataCopyPad(expertIdsTensor_, expertIdsGMTensor_[expertIdsBaseTokenId_ * axisK_], expertIdsSliceParams,
                        expertIdsCopyPadParams);
        }
        SyncFunc<AscendC::HardEvent::MTE2_S>();
        ResolveExpertRankTable();

        if constexpr (isSync) {
            Duplicate<int32_t>(ffnStatusTensor_, static_cast<int32_t>(0), ffnNumAlignCnt_);
        }

        LocalTensor<int32_t> rankCounts = rankInfoBuf_.Get<int32_t>();
        Duplicate<int32_t>(rankCounts, 0, rankCountElements_);
        SyncFunc<AscendC::HardEvent::V_S>();

        remoteFlagCnt_ = 0;
        const uint32_t tokensPerRow = axisK_ + sharedExpertNum_;
        uint32_t tokenOffset = startTokenId;
        uint32_t tokenId = 0U;
        uint32_t topkId = 0U;
        if (tokensPerRow > 0U) {
            tokenId = startTokenId / tokensPerRow;
            topkId = startTokenId % tokensPerRow;
        }
        while (tokenOffset < endTokenId) {
            const uint32_t rowRemain = tokensPerRow - topkId;
            const uint32_t rangeRemain = endTokenId - tokenOffset;
            const uint32_t rowSegLen = (rowRemain < rangeRemain) ? rowRemain : rangeRemain;
            for (uint32_t j = 0U; j < rowSegLen; ++j) {
                AttentionToFfnTokenMetaData metaData;
                ReadTokenMetaData(metaData, tokenId, topkId);
                tokenRankScratchTensor_.SetValue(tokenOffset - startTokenId, metaData.toRankId);

                if constexpr (isSync) {
                    int32_t ffnIdx = metaData.toRankId - static_cast<int32_t>(ffnStartRankId_);
                    if (ffnIdx >= 0 && static_cast<uint32_t>(ffnIdx) < ffnNum_) {
                        ffnStatusTensor_.SetValue(ffnIdx, 1);
                    }
                }
                flagValueTensor_.SetValue(remoteFlagCnt_, metaData.localExpId);
                ++remoteFlagCnt_;
                ++tokenOffset;
                ++topkId;
            }
            if (topkId == tokensPerRow) {
                topkId = 0U;
                ++tokenId;
            }
        }
        if (sendTokenNum_ > 0U) {
            SyncFunc<HardEvent::S_V>(); // the scratch was filled by the scalar writes above
            LocalTensor<int32_t> gatherDst = bucketTensor_;
            LocalTensor<uint8_t> scanMask =
                bucketTensor_.ReinterpretCast<uint8_t>()[2 * RANK_SCAN_CHUNK_ELEMS * sizeof(int32_t)];
            for (uint32_t rank = 0U; rank < worldSize_; ++rank) {
                uint32_t rankTokenCnt = 0U;
                for (uint32_t chunkStart = 0U; chunkStart < sendTokenNum_; chunkStart += RANK_SCAN_CHUNK_ELEMS) {
                    const uint32_t chunkLen = (chunkStart + RANK_SCAN_CHUNK_ELEMS > sendTokenNum_) ?
                                                  sendTokenNum_ - chunkStart :
                                                  RANK_SCAN_CHUNK_ELEMS;
                    LocalTensor<int32_t> scratchChunk = tokenRankScratchTensor_[chunkStart];
                    CompareScalar(
                        scanMask, scratchChunk, static_cast<int32_t>(rank), AscendC::CMPMODE::EQ,
                        Ceil(chunkLen * sizeof(int32_t), RANK_SCAN_CMP_ALIGN) * RANK_SCAN_CMP_ALIGN / sizeof(int32_t));
                    PipeBarrier<PIPE_V>();
                    uint64_t rsvdCnt = 0;
                    GatherMask(gatherDst, gatherDst, scanMask.ReinterpretCast<uint32_t>(), true, chunkLen, {1, 1, 0, 0},
                               rsvdCnt);
                    SyncFunc<HardEvent::V_S>();
                    rankTokenCnt += static_cast<uint32_t>(rsvdCnt);
                }
                rankCounts.SetValue(rank, static_cast<int32_t>(rankTokenCnt));
            }
        }

        if constexpr (isSync) {
            SyncFunc<AscendC::HardEvent::S_MTE3>();
            syncStatusGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(syncStatusWorkspaceGM_));
            uint32_t aivWorkspaceStride = aivWorkspaceOffset_ / sizeof(int32_t);
            DataCopy(syncStatusGMTensor_[aivId_ * aivWorkspaceStride], ffnStatusTensor_, ffnNumAlignCnt_);
        }
        SyncFunc<HardEvent::S_MTE3>();
        {
            GlobalTensor<int32_t> countRowGM;
            GM_ADDR countRowAddr = countMatrixBase_ + static_cast<uint64_t>(aivId_) * countRowStride_;
            countRowGM.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(countRowAddr));
            DataCopy(countRowGM, rankCounts, rankCountElements_);
            SyncFunc<HardEvent::MTE3_S>();
        }
        if (sendTokenNum_ > 0U) {
            GlobalTensor<int32_t> tokenRankTableGM;
            tokenRankTableGM.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(tokenRankTableBase_));
            const DataCopyExtParams tokenRankCopyParams{1U, static_cast<uint32_t>(sendTokenNum_ * sizeof(int32_t)), 0U,
                                                        0U, 0U};
            DataCopyPad(tokenRankTableGM[sendStartTokenId_], tokenRankScratchTensor_, tokenRankCopyParams);
            SyncFunc<HardEvent::MTE3_S>();
        }
        SyncAll<true>();

        BuildRankTableOffsets();
        BuildFlagSpans();

        PackRows();
        SyncAll<true>();

        for (uint32_t dstRank = aivId_; dstRank < worldSize_; dstRank += aivNum_) {
            RelayRemoteTokens(dstRank);
        }
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
