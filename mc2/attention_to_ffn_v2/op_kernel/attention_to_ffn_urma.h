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

constexpr uint32_t HCOMM_BATCH_WQE_BYTES = 64U;
constexpr uint32_t URMA_BATCH_WQE_CAPACITY = 16U * 1024U / HCOMM_BATCH_WQE_BYTES;
// Exactly capacity WQEBBs: the hcomm runtime writes no CQE scratch beyond the WQEs themselves
// (same sizing as moe_ep_combine's cqe=1 batches and ffn_to_attention_v2's batch buffer).
constexpr uint32_t ATTN_FFN_HCOMM_BATCH_UB_BYTES = URMA_BATCH_WQE_CAPACITY * HCOMM_BATCH_WQE_BYTES;

constexpr uint32_t ATTN_FFN_STATUS_TABLE_SLOTS = 4U;
constexpr uint32_t RANK_INFO_SEGMENT_NUM = 4U;
constexpr uint32_t RANK_INFO_COUNT_SEG_IDX = 0U;
constexpr uint32_t RANK_INFO_PREFIX_SEG_IDX = 1U;
constexpr uint32_t RANK_INFO_TOTAL_SEG_IDX = 2U;
constexpr uint32_t RANK_INFO_START_SEG_IDX = 3U;
constexpr uint32_t BUCKET_BUFFER_SIZE = 32U * 1024U;
constexpr uint32_t BUCKET_GROUP_TOKEN_MAX = 256U;

static constexpr AscendC::UrmaWqeEntry REMOTE_DATA_WQE_CONFIG = {.odr = 5, .fence = 1, .se = 0, .inlineEn = 0};
static constexpr AscendC::UrmaWqeEntry REMOTE_ORDERED_FLAG_WQE_CONFIG = {.odr = 6, .fence = 1, .se = 1, .inlineEn = 0};
static constexpr AscendC::UrmaWqeEntry REMOTE_ORDERED_FLAG_CQE_WQE_CONFIG = {
    .odr = 6, .fence = 1, .se = 1, .cqe = 1, .inlineEn = 0};
constexpr uint64_t RELAY_MERGE_MAX_BYTES = 2U * 1024U * 1024U;
using HcommBatchHandle = AscendC::BatchHandle<AscendC::ChannelHandle>;

struct AttentionToFfnTokenMetaData {
    uint32_t tokenId;
    uint32_t topkId;
    int32_t dstExpertId;
    int32_t toRankId;
    int32_t localExpId;
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
    __aicore__ inline void ReadTokenMetaData(AttentionToFfnTokenMetaData& metaData, uint32_t tokenOffset,
                                             uint32_t tokenId, uint32_t topkId);
    __aicore__ inline GM_ADDR GetWindowAddr(int32_t rankId);
    __aicore__ inline uint64_t GetUrmaCommHandle(uint32_t dstRank, uint32_t channelIndex = 0U);
    __aicore__ inline void LoadXRow(uint32_t tokenId, const DataCopyExtParams& xCopyParams);
    __aicore__ inline void StageRowEntry(const AttentionToFfnTokenMetaData& metaData,
                                         const DataCopyExtParams& xCopyParams,
                                         const DataCopyExtParams& hCommuCopyParams, bool rowQuantShared,
                                         bool& rowPayloadReady);
    __aicore__ inline void ReleaseRowPayload(bool rowQuantShared);
    __aicore__ inline void AppendRemoteDataWrite(HcommBatchHandle& batchHandle, GM_ADDR remoteDataAddr,
                                                 GM_ADDR sourceDataAddr, uint64_t byteLen);
    __aicore__ inline void AppendRemoteFlagWrite(HcommBatchHandle& batchHandle, GM_ADDR remoteFlagAddr,
                                                 uint32_t sourceTokenOffset, uint32_t tokenCount);
    __aicore__ inline void RelayRunDataWrites(HcommBatchHandle& batchHandle, GM_ADDR toRankAddr,
                                              uint32_t runTokenOffset, uint32_t runLen, uint32_t& preparedWqeCount);
    __aicore__ inline void RelayRunFlagWrites(HcommBatchHandle& batchHandle, GM_ADDR toRankAddr,
                                              uint32_t runTokenOffset, uint32_t runLen, uint32_t& preparedWqeCount);
    __aicore__ inline void BuildRankTableOffsets();
    __aicore__ inline void ScatterTokensToRankTables();
    __aicore__ inline void RelayRemoteTokens(uint32_t dstRank, uint32_t channelIdx);
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
    __aicore__ inline bool CheckPreviousSends();
    __aicore__ inline void ReportSendCompletion();

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
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> xQueue_;
    TQue<QuePosition::VECIN, 1> xInQueue_;
    TQue<QuePosition::VECOUT, 1> xOutQueue_;

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
    uint32_t sendStartTokenId_{0};
    uint32_t sendTokenNum_{0};
    // First token row of this AIV's expert_ids slice; expert_ids reads subtract it.
    uint32_t expertIdsBaseTokenId_{0};
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
    uint64_t winDataBaseByte_{0};
    uint64_t winFlagBaseByte_{0};
    uint64_t remoteSlotBytes_{0};
    bool isScales_{false};
    // ---- Tensor保活延迟释放（异步发送完成状态表）----
    // statusTableGM_为context下发的已注册（设备可写）Host状态表设备地址，nullptr表示未启用；
    // completedRounds_为上一个kernel发布的"已完成轮次"P（本kernel检查完前序WQE后发布P+1；
    // 首个kernel读到host初始化的-2，发布-1）。
    GM_ADDR statusTableGM_{nullptr};
    uint64_t completedRounds_{0};
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

    uint32_t maxTotalSendNum = expertIdsCnt_;
    uint32_t maxTokensPerAiv = Ceil(maxTotalSendNum, aivNum_);
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
        ctxTailGM.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(&mc2Context_->hostStatusTableAddr));
        DataCopyExtParams tailCopyParams{1U, sizeof(uint64_t) * 2U, 0U, 0U, 0U};
        DataCopyPadExtParams<uint64_t> tailPadParams{false, 0U, 0U, 0U};
        LocalTensor<uint64_t> tailTensor = statusTensor_.ReinterpretCast<uint64_t>();
        DataCopyPad(tailTensor, ctxTailGM, tailCopyParams, tailPadParams);
        SyncFunc<HardEvent::MTE2_S>();
        statusTableGM_ = reinterpret_cast<GM_ADDR>(tailTensor.GetValue(0));
        completedRounds_ = tailTensor.GetValue(1);
    }
    uint32_t expertRankResolveBytes = Ceil(RANK_RESOLVE_STRIDE * expertNum_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(expertRankResolveBuf_, expertRankResolveBytes);
    expertRankResolveTensor_ = expertRankResolveBuf_.Get<int32_t>();

    uint32_t flagValueBufSize = Ceil(maxTokensPerAiv * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(flagValueBuf_, flagValueBufSize);
    flagValueTensor_ = flagValueBuf_.Get<int32_t>();

    uint32_t tokenRankStashBytes = Ceil(maxTokensPerAiv * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
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
    stagingStride_ = Ceil(hCommuSize_, static_cast<uint64_t>(UB_ALIGN)) * UB_ALIGN;

    remoteSlotBytes_ = static_cast<uint64_t>(axisHS_) * elementSize;
    winDataBaseByte_ = winTokenDataRegionBytes + winTokenDataOffset_ * elementSize;
    winFlagBaseByte_ = winInfoTableOffset_ + TOKEN_INFO_TABLE_RS * sizeof(int32_t);
    const bool relayDataMergeEligible = (stagingStride_ == remoteSlotBytes_);
    const uint64_t dataSegByBytes = RELAY_MERGE_MAX_BYTES / remoteSlotBytes_;
    relayDataSegTokenMax_ =
        relayDataMergeEligible ? static_cast<uint32_t>(dataSegByBytes == 0U ? 1U : dataSegByBytes) : 1U;
    relayFlagSegTokenMax_ = static_cast<uint32_t>(RELAY_MERGE_MAX_BYTES / sizeof(int32_t));

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

    const uint64_t bucketTableBytes =
        Ceil(static_cast<uint64_t>(maxTotalSendNum) * sizeof(int32_t), WORKSPACE_ELEMENT_OFFSET) *
        WORKSPACE_ELEMENT_OFFSET;
    stagingBaseAddr_ = bucketTablesBase_ + bucketTableBytes;
    const uint64_t stagingRegionSize = static_cast<uint64_t>(maxTotalSendNum) * stagingStride_;
    const uint64_t stagingRegionAlignSize = Ceil(stagingRegionSize, static_cast<uint64_t>(WIN_ALIGN)) * WIN_ALIGN;
    const uint64_t flagEntryRegionBytes =
        Ceil(static_cast<uint64_t>(expertIdsCnt_) * sizeof(int32_t), static_cast<uint64_t>(UB_ALIGN)) * UB_ALIGN;
    localFlagAddr_ = stagingBaseAddr_ + stagingRegionAlignSize;
    layerFlagAddr_ = localFlagAddr_ + flagEntryRegionBytes;
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
#else
    float dynamicScale = 0.0;
    uint32_t hOutSizeAlign = Ceil(axisH_ * sizeof(int8_t), UB_ALIGN) * UB_ALIGN;
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

    if (quantMode_ == DYNAMIC_QUANT) {
        dynamicScale = ComputeDynamicScale(floatLocalTemp);
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
__aicore__ inline void AttentionToFfnUrma<TemplateAttentionToFfnUrmaTypeFunc>::ResolveExpertRankTable()
{
    ascendc_assert(expRankTableM_ > 0U, "expert_rank_table row capacity M %u must be positive", expRankTableM_);
    const uint32_t rowBytes = expRankTableM_ * sizeof(int32_t);
    const uint32_t expertsPerChunk = BUCKET_BUFFER_SIZE / rowBytes;
    ascendc_assert(expertsPerChunk > 0U, "expert_rank_table row bytes %u exceed bucket buffer %u", rowBytes,
                   BUCKET_BUFFER_SIZE);
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
                ascendc_assert(static_cast<uint64_t>(rankCnt) * RANK_OFFSET_STRIDE + 1 <= expRankTableM_,
                               "rankCnt %u exceeds expert_rank_table row capacity M %u", rankCnt, expRankTableM_);
                const uint32_t rankOffset = (sessionId_ % rankCnt) * RANK_OFFSET_STRIDE + 1;
                toRankId = chunkTensor.GetValue(rowBase + rankOffset);
                localExpId = chunkTensor.GetValue(rowBase + rankOffset + 1U);
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
    ascendc_assert(expertId >= 0 && static_cast<uint32_t>(expertId) < expertNum_,
                   "expertId %d must be in [0, expertNum %u)", expertId, expertNum_);
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
    ascendc_assert(rankId >= 0 && static_cast<uint32_t>(rankId) < worldSize_, "rankId %d must be in [0, worldSize %u)",
                   rankId, worldSize_);
    return (GM_ADDR)rankWinBaseTensor_(static_cast<uint32_t>(rankId));
}

template <TemplateAttentionToFfnUrmaTypeClass>
__aicore__ inline uint64_t AttentionToFfnUrma<TemplateAttentionToFfnUrmaTypeFunc>::GetUrmaCommHandle(
    uint32_t dstRank, uint32_t channelIndex)
{
    return mc2Context_->hcommHandle_[dstRank * channelsPerRank_ + channelIndex];
}

template <TemplateAttentionToFfnUrmaTypeClass>
__aicore__ inline void AttentionToFfnUrma<TemplateAttentionToFfnUrmaTypeFunc>::ReadTokenMetaData(
    AttentionToFfnTokenMetaData& metaData, uint32_t tokenOffset, uint32_t tokenId, uint32_t topkId)
{
    metaData.tokenId = tokenId;
    metaData.topkId = topkId;

    if (metaData.topkId < axisK_) {
        // expert_ids is staged as this AIV's row slice [expertIdsBaseTokenId_, ...], so the
        // read subtracts the slice base; the covered GM offsets are unchanged.
        metaData.dstExpertId =
            expertIdsTensor_.GetValue((metaData.tokenId - expertIdsBaseTokenId_) * axisK_ + metaData.topkId);
    } else {
        metaData.dstExpertId = static_cast<int32_t>(moeExpertNum_ + (metaData.topkId - axisK_));
    }

    FindExpertRank(metaData.dstExpertId);
    metaData.toRankId = toRankId_;
    metaData.localExpId = localExpId_;

    ascendc_assert(metaData.toRankId >= 0 && static_cast<uint32_t>(metaData.toRankId) < worldSize_,
                   "toRankId %d must be in [0, worldSize %u)", metaData.toRankId, worldSize_);
    metaData.localDataAddr = stagingBaseAddr_ + static_cast<uint64_t>(tokenOffset) * stagingStride_;
}

template <TemplateAttentionToFfnUrmaTypeClass>
__aicore__ inline void AttentionToFfnUrma<TemplateAttentionToFfnUrmaTypeFunc>::LoadXRow(
    uint32_t tokenId, const DataCopyExtParams& xCopyParams)
{
    DataCopyPadExtParams<XType> copyPadExtParams{false, 0U, 0U, 0U};
    if constexpr (QuantMode == 0) {
        xTmpTensor_ = xQueue_.AllocTensor<XType>();
        DataCopyPad(xTmpTensor_, xGMTensor_[static_cast<uint64_t>(tokenId) * axisH_], xCopyParams, copyPadExtParams);
        xQueue_.EnQue(xTmpTensor_);
        xTmpTensor_ = xQueue_.DeQue<XType>();
    } else {
        xInTensor_ = xInQueue_.AllocTensor<XType>();
        if constexpr (IsMxOrMxClipQuant(QuantMode)) {
            LocalTensor<uint8_t> singleByteTok = xInTensor_.template ReinterpretCast<uint8_t>();
            Duplicate(singleByteTok, static_cast<uint8_t>(0),
                      Ceil(axisH_ * sizeof(XType), MX_INPUT_READ_BYTES) * MX_INPUT_READ_BYTES);
            SyncFunc<HardEvent::V_MTE2>();
            const uint32_t tailPadElems = (UB_ALIGN - axisH_ * sizeof(XType) % UB_ALIGN) % UB_ALIGN / sizeof(XType);
            copyPadExtParams =
                DataCopyPadExtParams<XType>{true, 0U, static_cast<uint8_t>(tailPadElems), static_cast<XType>(0)};
        }
        DataCopyPad(xInTensor_, xGMTensor_[static_cast<uint64_t>(tokenId) * axisH_], xCopyParams, copyPadExtParams);
        xInQueue_.EnQue(xInTensor_);
        xInTensor_ = xInQueue_.DeQue<XType>();
    }
}

// Return the row payload buffers once the row's entries are all staged.
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
__aicore__ inline void AttentionToFfnUrma<TemplateAttentionToFfnUrmaTypeFunc>::StageRowEntry(
    const AttentionToFfnTokenMetaData& metaData, const DataCopyExtParams& xCopyParams,
    const DataCopyExtParams& hCommuCopyParams, bool rowQuantShared, bool& rowPayloadReady)
{
    if constexpr (QuantMode == 0) {
        GlobalTensor<XType> targetGMTensor;
        targetGMTensor.SetGlobalBuffer(reinterpret_cast<__gm__ XType*>(metaData.localDataAddr));
        DataCopyPad(targetGMTensor, xTmpTensor_, xCopyParams);
    } else {
        if (!rowQuantShared || !rowPayloadReady) {
            xOutTensor_ = xOutQueue_.AllocTensor<StorageXOutType>();
            QuantProcess(metaData.dstExpertId);
            xOutQueue_.EnQue(xOutTensor_);
            xOutTensor_ = xOutQueue_.DeQue<StorageXOutType>();
            rowPayloadReady = true;
        }
        GlobalTensor<uint8_t> targetGMTensor;
        targetGMTensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(metaData.localDataAddr));
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
    HcommBatchHandle& batchHandle, GM_ADDR remoteFlagAddr, uint32_t sourceTokenOffset, uint32_t tokenCount)
{
    // The local flag region holds one compact 4B entry per tokenOffset, mirroring the remote info
    // table layout, so a whole consecutive run is one contiguous span on both sides.
    GM_ADDR sourceFlagAddr = localFlagAddr_ + static_cast<uint64_t>(sourceTokenOffset) * sizeof(int32_t);
    hcomm_.WriteNbi<REMOTE_ORDERED_FLAG_WQE_CONFIG>(batchHandle, remoteFlagAddr, sourceFlagAddr,
                                                    static_cast<int64_t>(tokenCount) * sizeof(int32_t));
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

    for (uint32_t rowStart = 0U; rowStart < aivNum_;) {
        uint32_t maxRows = aivNum_ - rowStart;
        if (rowStart < aivId_) {
            const uint32_t prefixRowsLeft = aivId_ - rowStart;
            maxRows = (maxRows < prefixRowsLeft) ? maxRows : prefixRowsLeft;
        }
        const uint32_t copyRows = (rowsPerBatch < maxRows) ? rowsPerBatch : maxRows;
        DataCopyExtParams copyParams{static_cast<uint16_t>(copyRows), rowBytes,
                                     static_cast<uint32_t>(countRowStride_ - rowBytes), 0U, 0U};
        DataCopyPad(prefixRows, countMatrixGM[static_cast<uint64_t>(rowStart) * countRowStride_ / sizeof(int32_t)],
                    copyParams, padParams);
        SyncFunc<HardEvent::MTE2_V>();
        const uint32_t batchShape[] = {copyRows, rankCountElements_};
        ReduceSum<int32_t, AscendC::Pattern::Reduce::RA, true>(reduceSums, prefixRows, batchShape, false);
        Add(totalCounts, totalCounts, reduceSums, rankCountElements_);
        if (rowStart + copyRows <= aivId_) {
            Add(writePrefix, writePrefix, reduceSums, rankCountElements_);
        }
        rowStart += copyRows;
        if (rowStart < aivNum_) {
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
    // This AIV's contiguous token range only: the rank stash is indexed by the token's offset
    // inside [sendStartTokenId_, sendStartTokenId_ + sendTokenNum_).
    const uint64_t scatterCnt = sendTokenNum_;

    for (uint64_t groupStart = 0U; groupStart < scatterCnt; groupStart += groupTokenMax_) {
        uint64_t groupEnd = (groupStart + groupTokenMax_ > scatterCnt) ? scatterCnt : groupStart + groupTokenMax_;
        Duplicate<int32_t>(rankCounts, 0, rankCountElements_);
        SyncFunc<HardEvent::V_S>();
        for (uint64_t idx = groupStart; idx < groupEnd; ++idx) {
            uint32_t tokenOffset = sendStartTokenId_ + static_cast<uint32_t>(idx);
            int32_t dstRank = tokenRankScratchTensor_.GetValue(static_cast<int32_t>(idx));
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
__aicore__ inline void AttentionToFfnUrma<TemplateAttentionToFfnUrmaTypeFunc>::RelayRunDataWrites(
    HcommBatchHandle& batchHandle, GM_ADDR toRankAddr, uint32_t runTokenOffset, uint32_t runLen,
    uint32_t& preparedWqeCount)
{
    uint32_t segStart = runTokenOffset;
    uint32_t remaining = runLen;
    while (remaining > 0U) {
        uint32_t segLen = (remaining > relayDataSegTokenMax_) ? relayDataSegTokenMax_ : remaining;
        GM_ADDR remoteDataAddr = toRankAddr + winDataBaseByte_ + static_cast<uint64_t>(segStart) * remoteSlotBytes_;
        GM_ADDR localDataAddr = stagingBaseAddr_ + static_cast<uint64_t>(segStart) * stagingStride_;
        uint64_t byteLen = (segLen == 1U) ? hCommuSize_ : static_cast<uint64_t>(segLen) * stagingStride_;
        AppendRemoteDataWrite(batchHandle, remoteDataAddr, localDataAddr, byteLen);
        ++preparedWqeCount;
        if (preparedWqeCount == URMA_BATCH_WQE_CAPACITY) {
            hcomm_.BatchCommit(batchHandle);
            preparedWqeCount = 0U;
        }
        segStart += segLen;
        remaining -= segLen;
    }
}

// Emit the flag WQEs of one run: the local flag region and the remote info table both hold one
// compact 4B entry per tokenOffset, so each segment is a single contiguous WQE.
template <TemplateAttentionToFfnUrmaTypeClass>
__aicore__ inline void AttentionToFfnUrma<TemplateAttentionToFfnUrmaTypeFunc>::RelayRunFlagWrites(
    HcommBatchHandle& batchHandle, GM_ADDR toRankAddr, uint32_t runTokenOffset, uint32_t runLen,
    uint32_t& preparedWqeCount)
{
    uint32_t segStart = runTokenOffset;
    uint32_t remaining = runLen;
    while (remaining > 0U) {
        uint32_t segLen = (remaining > relayFlagSegTokenMax_) ? relayFlagSegTokenMax_ : remaining;
        GM_ADDR remoteFlagAddr = toRankAddr + winFlagBaseByte_ + static_cast<uint64_t>(segStart) * sizeof(int32_t);
        AppendRemoteFlagWrite(batchHandle, remoteFlagAddr, segStart, segLen);
        ++preparedWqeCount;
        if (preparedWqeCount == URMA_BATCH_WQE_CAPACITY) {
            hcomm_.BatchCommit(batchHandle);
            preparedWqeCount = 0U;
        }
        segStart += segLen;
        remaining -= segLen;
    }
}

template <TemplateAttentionToFfnUrmaTypeClass>
__aicore__ inline void AttentionToFfnUrma<TemplateAttentionToFfnUrmaTypeFunc>::RelayRemoteTokens(uint32_t dstRank,
                                                                                                 uint32_t channelIdx)
{
    LocalTensor<int32_t> rankInfo = rankInfoBuf_.Get<int32_t>();
    LocalTensor<int32_t> totalCounts = rankInfo[RANK_INFO_TOTAL_SEG_IDX * rankCountElements_];
    LocalTensor<int32_t> tableStart = rankInfo[RANK_INFO_START_SEG_IDX * rankCountElements_];
    const uint32_t entryCount = static_cast<uint32_t>(totalCounts.GetValue(dstRank));
    if (entryCount == 0U) {
        return;
    }

    const uint32_t blockStart =
        static_cast<uint32_t>(static_cast<uint64_t>(channelIdx) * entryCount / channelsPerRank_);
    const uint32_t blockEnd =
        static_cast<uint32_t>(static_cast<uint64_t>(channelIdx + 1U) * entryCount / channelsPerRank_);
    if (blockStart >= blockEnd) {
        return;
    }
    const uint32_t blockEntries = blockEnd - blockStart;
    uint32_t baseEntry = static_cast<uint32_t>(tableStart.GetValue(dstRank)) + blockStart;

    GM_ADDR toRankAddr = GetWindowAddr(static_cast<int32_t>(dstRank));
    uint64_t channel = GetUrmaCommHandle(dstRank, channelIdx);
    HcommBatchHandle batchHandle =
        hcomm_.MakeBatchHandle(channel, hcommBatchWqeTensor_, ATTN_FFN_HCOMM_BATCH_UB_BYTES, toRankAddr);
    uint32_t preparedWqeCount = 0U;
    LocalTensor<int32_t> entriesLocal = bucketTensor_;
    constexpr uint32_t entryChunkMax = BUCKET_BUFFER_SIZE / sizeof(int32_t);
    const DataCopyPadExtParams<int32_t> padParams{false, 0U, 0U, 0U};
    GlobalTensor<int32_t> bucketTablesGM;
    bucketTablesGM.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(bucketTablesBase_));

    // Ordering: every flag WQE keeps REMOTE_ORDERED_FLAG_WQE_CONFIG (fence=1, odr=6), i.e. it
    // lands only after every earlier WQE on the channel - in particular its run's data WQE(s).
    for (uint32_t chunkStart = 0U; chunkStart < blockEntries; chunkStart += entryChunkMax) {
        uint32_t curEntries = (chunkStart + entryChunkMax > blockEntries) ? blockEntries - chunkStart : entryChunkMax;
        DataCopyExtParams copyParams{1U, static_cast<uint32_t>(curEntries * sizeof(int32_t)), 0U, 0U, 0U};
        SyncFunc<HardEvent::S_MTE2>();
        DataCopyPad(entriesLocal, bucketTablesGM[baseEntry + chunkStart], copyParams, padParams);
        SyncFunc<HardEvent::MTE2_S>();
        uint32_t i = 0U;
        while (i < curEntries) {
            const uint32_t runFirstPos = i;
            const uint32_t runTokenOffset = static_cast<uint32_t>(entriesLocal.GetValue(i));
            ++i;
            // Extend across position-adjacent entries with consecutive tokenOffsets; the whole
            // chunk lies inside this channel's block, so no channel filter is needed here.
            while (i < curEntries &&
                   static_cast<uint32_t>(entriesLocal.GetValue(i)) == runTokenOffset + (i - runFirstPos)) {
                ++i;
            }
            const uint32_t runLen = i - runFirstPos;
            RelayRunDataWrites(batchHandle, toRankAddr, runTokenOffset, runLen, preparedWqeCount);
            RelayRunFlagWrites(batchHandle, toRankAddr, runTokenOffset, runLen, preparedWqeCount);
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
    bool ownsFfnRank = false;
    for (uint32_t ffnIdx = ffnStartRankId_; ffnIdx < ffnStartRankId_ + ffnNum_; ++ffnIdx) {
        if ((ffnIdx * channelsPerRank_) % aivNum_ == aivId_) {
            ownsFfnRank = true;
            break;
        }
    }
    if (!ownsFfnRank) {
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

    for (uint32_t ffnIdx = ffnStartRankId_; ffnIdx < ffnStartRankId_ + ffnNum_; ++ffnIdx) {
        if ((ffnIdx * channelsPerRank_) % aivNum_ != aivId_) {
            continue;
        }
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
                hcomm_.MakeBatchHandle(channel, hcommBatchWqeTensor_, ATTN_FFN_HCOMM_BATCH_UB_BYTES, toRankAddr);
            // The layer flag is the channel's last WQE. Request a CQE on it only when the keepalive
            // mechanism is enabled (statusTableGM_ set): the next launch's CheckPreviousSends Drain
            // consumes that CQE and retires the whole posted set. Without the status table there is
            // no consumer, and a CQ nobody drains could fill up and stall the channel, so the flag
            // keeps the plain ordered config - behavior identical to the mechanism being absent.
            if (statusTableGM_ != nullptr) {
                hcomm_.WriteNbi<REMOTE_ORDERED_FLAG_CQE_WQE_CONFIG>(
                    batchHandle, tableFlagGM, layerFlagAddr_,
                    static_cast<int64_t>(sizeof(int32_t) * TOKEN_INFO_TABLE_COPY_BLOCK_CNT));
            } else {
                hcomm_.WriteNbi<REMOTE_ORDERED_FLAG_WQE_CONFIG>(
                    batchHandle, tableFlagGM, layerFlagAddr_,
                    static_cast<int64_t>(sizeof(int32_t) * TOKEN_INFO_TABLE_COPY_BLOCK_CNT));
            }
            hcomm_.BatchCommit(batchHandle);
        }
    }
}

template <TemplateAttentionToFfnUrmaTypeClass>
__aicore__ inline bool AttentionToFfnUrma<TemplateAttentionToFfnUrmaTypeFunc>::CheckPreviousSends()
{
    bool localDrainFailed = false;
    const uint32_t totalSlots = worldSize_ * channelsPerRank_;
    for (uint32_t slot = aivId_; slot < totalSlots; slot += aivNum_) {
        const uint32_t dstRank = slot / channelsPerRank_;
        const uint32_t channelIdx = slot % channelsPerRank_;
        if (dstRank == rankId_) {
            continue;
        }

        const int32_t drainRet = hcomm_.Drain(GetUrmaCommHandle(dstRank, channelIdx));
        localDrainFailed = localDrainFailed || (drainRet != 0);
    }

    constexpr uint32_t statusBlockElements = UB_ALIGN / sizeof(int32_t);
    const uint32_t countRowStrideElements = countRowStride_ / sizeof(int32_t);
    Duplicate<int32_t>(statusTensor_, localDrainFailed ? 1 : 0, statusBlockElements);
    SyncFunc<HardEvent::V_MTE3>();

    // The count matrix is not populated until later in Process, so its per-AIV rows can carry
    // the drain result temporarily. Each AIV owns one aligned row and writes its first 32B block.
    GlobalTensor<int32_t> drainStatusGM;
    drainStatusGM.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(countMatrixBase_));
    DataCopy(drainStatusGM[aivId_ * countRowStrideElements], statusTensor_, statusBlockElements);
    SyncFunc<HardEvent::MTE3_S>();
    SyncAll<true>();

    if (aivId_ != 0U) {
        return false;
    }

    DataCopyExtParams statusCopyParams{static_cast<uint16_t>(aivNum_), UB_ALIGN, countRowStride_ - UB_ALIGN, 0U, 0U};
    DataCopyPadExtParams<int32_t> statusPadParams{false, 0U, 0U, 0U};
    DataCopyPad(bucketTensor_, drainStatusGM, statusCopyParams, statusPadParams);
    SyncFunc<HardEvent::MTE2_S>();

    for (uint32_t coreIdx = 0U; coreIdx < aivNum_; ++coreIdx) {
        if (bucketTensor_.GetValue(coreIdx * statusBlockElements) != 0) {
            return false;
        }
    }
    return true;
}

template <TemplateAttentionToFfnUrmaTypeClass>
__aicore__ inline void AttentionToFfnUrma<TemplateAttentionToFfnUrmaTypeFunc>::ReportSendCompletion()
{
    if (aivId_ != 0U) {
        return;
    }
    const uint64_t publishedRounds = completedRounds_ + 1U;
    LocalTensor<uint64_t> statusU64 = statusTensor_.ReinterpretCast<uint64_t>();
    Duplicate<uint64_t>(statusU64, publishedRounds, ATTN_FFN_STATUS_TABLE_SLOTS);
    SyncFunc<HardEvent::V_MTE3>();
    DataCopyExtParams qwordParams{1U, sizeof(uint64_t), 0U, 0U, 0U};
    // Context round counter := publishedRounds for the next kernel.
    GlobalTensor<uint64_t> ctxTailGM;
    ctxTailGM.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(&mc2Context_->completedRounds));
    DataCopyPad(ctxTailGM, statusU64, qwordParams);
    // Host status table slot 0 := publishedRounds (the whole 32B block carries the same value).
    GlobalTensor<uint64_t> tableGM;
    tableGM.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(statusTableGM_));
    DataCopy(tableGM, statusU64, ATTN_FFN_STATUS_TABLE_SLOTS);
    SyncFunc<HardEvent::MTE3_S>();
}

template <TemplateAttentionToFfnUrmaTypeClass>
__aicore__ inline void AttentionToFfnUrma<TemplateAttentionToFfnUrmaTypeFunc>::Process()
{
    if ASCEND_IS_AIV {
        HcommInit();

        if (statusTableGM_ != nullptr) {
            const bool previousSendsCompleted = CheckPreviousSends();
            if (previousSendsCompleted) {
                ReportSendCompletion();
            }
            // AIV0 must finish reading the temporary drain-status rows before other AIVs can
            // repurpose the count matrix below.
            SyncAll<true>();
        }

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

        DataCopyExtParams xCopyParams = {1U, static_cast<uint32_t>(hSize_), 0U, 0U, 0U};
        DataCopyExtParams hCommuCopyParams = {1U, static_cast<uint32_t>(hCommuSize_), 0U, 0U, 0U};
        remoteFlagCnt_ = 0;
        const uint32_t tokensPerRow = axisK_ + sharedExpertNum_;
        uint32_t tokenOffset = startTokenId;
        uint32_t tokenId = 0U;
        uint32_t topkId = 0U;
        if (tokensPerRow > 0U) {
            tokenId = startTokenId / tokensPerRow;
            topkId = startTokenId % tokensPerRow;
        }
        // Dynamic quant + per-expert smooth scales is the only expert-dependent payload; every
        // other mode reuses one quantized row across the whole token row.
        bool rowQuantShared = true;
        if constexpr (QuantMode == DYNAMIC_QUANT) {
            rowQuantShared = !isScales_;
        }
        while (tokenOffset < endTokenId) {
            const uint32_t rowRemain = tokensPerRow - topkId;
            const uint32_t rangeRemain = endTokenId - tokenOffset;
            const uint32_t rowSegLen = (rowRemain < rangeRemain) ? rowRemain : rangeRemain;
            LoadXRow(tokenId, xCopyParams);
            bool rowPayloadReady = false;
            for (uint32_t j = 0U; j < rowSegLen; ++j) {
                AttentionToFfnTokenMetaData metaData;
                ReadTokenMetaData(metaData, tokenOffset, tokenId, topkId);
                tokenRankScratchTensor_.SetValue(tokenOffset - startTokenId, metaData.toRankId);

                if constexpr (isSync) {
                    int32_t ffnIdx = metaData.toRankId - static_cast<int32_t>(ffnStartRankId_);
                    if (ffnIdx >= 0 && static_cast<uint32_t>(ffnIdx) < ffnNum_) {
                        ffnStatusTensor_.SetValue(ffnIdx, 1);
                    }
                }
                StageRowEntry(metaData, xCopyParams, hCommuCopyParams, rowQuantShared, rowPayloadReady);
                // Compact 4B flag entry (the local expert id) per staged token, flushed as one
                // contiguous block after the loop.
                flagValueTensor_.SetValue(remoteFlagCnt_, metaData.localExpId);
                ++remoteFlagCnt_;
                rankCounts.SetValue(metaData.toRankId, rankCounts.GetValue(metaData.toRankId) + 1);
                ++tokenOffset;
                ++topkId;
            }
            ReleaseRowPayload(rowQuantShared);
            if (topkId == tokensPerRow) {
                topkId = 0U;
                ++tokenId;
            }
        }

        if constexpr (isSync) {
            SyncFunc<AscendC::HardEvent::S_MTE3>();
            syncStatusGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(syncStatusWorkspaceGM_));
            uint32_t aivWorkspaceStride = aivWorkspaceOffset_ / sizeof(int32_t);
            DataCopy(syncStatusGMTensor_[aivId_ * aivWorkspaceStride], ffnStatusTensor_, ffnNumAlignCnt_);
        }

        SyncFunc<HardEvent::S_MTE3>();
        if (remoteFlagCnt_ > 0U) {
            GlobalTensor<int32_t> localFlagGMTensor;
            localFlagGMTensor.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(
                localFlagAddr_ + static_cast<uint64_t>(sendStartTokenId_) * sizeof(int32_t)));
            DataCopyExtParams flagCopyParams{1U, static_cast<uint32_t>(remoteFlagCnt_ * sizeof(int32_t)), 0U, 0U, 0U};
            DataCopyPad(localFlagGMTensor, flagValueTensor_, flagCopyParams);
        }

        {
            GlobalTensor<int32_t> countRowGM;
            GM_ADDR countRowAddr = countMatrixBase_ + static_cast<uint64_t>(aivId_) * countRowStride_;
            countRowGM.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(countRowAddr));
            DataCopy(countRowGM, rankCounts, rankCountElements_);
            SyncFunc<HardEvent::MTE3_S>();
        }
        SyncAll<true>();

        BuildRankTableOffsets();
        ScatterTokensToRankTables();
        SyncAll<true>();

        const uint32_t totalSlots = worldSize_ * channelsPerRank_;
        for (uint32_t slot = aivId_; slot < totalSlots; slot += aivNum_) {
            uint32_t dstRank = slot / channelsPerRank_;
            uint32_t channelIdx = slot % channelsPerRank_;
            RelayRemoteTokens(dstRank, channelIdx);
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
