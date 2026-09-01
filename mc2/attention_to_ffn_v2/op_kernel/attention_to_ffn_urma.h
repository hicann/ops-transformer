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
 *
 * Communication pattern (mirrors ffn_to_attention_urma.h for token data relay):
 *   - All cores initialise Hcomm (needed for completion-flag WriteNbi in SetFlagToFFN).
 *   - Token data relay: worker cores stage token data (with optional quantization)
 *     into per-core workspace slots and publish a per-token flag (local expert ID)
 *     into a per-core flag slot.  After SyncAll, core 0 iterates over every
 *     worker's staged slot and relays the data + flag to the remote rank window
 *     via WriteNbi/Drain.  A second SyncAll ensures core 0 has finished relaying
 *     before workers overwrite their slots in the next round.
 *   - Completion flags (SetFlagToFFN / SetFlagInAttn): each core sends directly
 *     via its own Hcomm handle, following the V1 attention_to_ffn pattern.
 */
#ifndef ATTENTION_TO_FFN_URMA_H
#define ATTENTION_TO_FFN_URMA_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "kernel_tiling/kernel_tiling.h"
#include "adv_api/hcomm/hcomm.h"
#include "adv_api/reduce/sum.h"
#include "adv_api/reduce/reduce.h"
#include "attention_to_ffn_v2_tiling.h"

#if __has_include("../common/attention_ffn_context.h")
#include "../common/attention_ffn_context.h"
#include "../common/mc2_kernel_utils.h"
#else
#include "../../common/op_kernel/attention_ffn_context.h"
#include "../../common/op_kernel/mc2_kernel_utils.h"
#endif

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
#if __has_include("../common/quantize_functions.h")
#include "../common/quantize_functions.h"
#else
#include "../../common/op_kernel/quantize_functions.h"
#endif
#endif

#ifndef FLOAT_OVERFLOW_MODE_CTRL
#define FLOAT_OVERFLOW_MODE_CTRL 60
#endif

namespace AttentionToFFNImpl {

#ifndef ATTN_FFN_URMA_SHARED_CONSTANTS
#define ATTN_FFN_URMA_SHARED_CONSTANTS
constexpr uint8_t BUFFER_NUM = 2;
constexpr uint32_t UB_ALIGN = 32;
constexpr uint8_t WIN_OFFSET_CNT = 2;
constexpr uint32_t SCALE_PARAM_PAD_SIZE = 128;
constexpr uint32_t WIN_ALIGN = 512;
constexpr uint32_t REP_STRIDE = 8;
constexpr uint32_t EXPERT_TABLE_REP_STRIDE = 16;
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

struct AttentionToFfnTokenMetaData {
    uint32_t tokenId;
    uint32_t topkId;
    int32_t dstExpertId;
    int32_t toRankId;
    int32_t localExpId;
    GM_ADDR remoteDataAddr;
    GM_ADDR remoteFlagAddr;
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
                                GM_ADDR workspaceGM, TPipe *pipe, const AttentionToFfnV2TilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void HcommInit();
    __aicore__ inline void InitLocalFlag();
    __aicore__ inline void ReadTokenMetaData(AttentionToFfnTokenMetaData &metaData, uint32_t tokenOffset);
    __aicore__ inline GM_ADDR GetWindowAddr(int32_t rankId);
    __aicore__ inline uint64_t GetUrmaCommHandle(uint32_t dstRank);
    __aicore__ inline void SendToLocal(const AttentionToFfnTokenMetaData &metaData,
                                       const DataCopyExtParams &xCopyParams);
    __aicore__ inline void StageRemoteData(const AttentionToFfnTokenMetaData &metaData,
                                           const DataCopyExtParams &xCopyParams);
    __aicore__ inline void SendToRemote(uint32_t sourceAivId, const AttentionToFfnTokenMetaData &metaData);
    __aicore__ inline void FindExpertRank(int32_t expertId);
    __aicore__ inline void QuantInit(GM_ADDR scales);
    __aicore__ inline void QuantProcess(uint32_t expertIndex);
    __aicore__ inline void ReduceMaxInplace(const LocalTensor<float> &srcLocal, uint32_t count);
    __aicore__ inline void SplitToCore(uint32_t curSendCnt, uint32_t curUseAivNum, uint32_t &startTokenId,
                                       uint32_t &endTokenId, uint32_t &sendTokenNum);
    __aicore__ inline void SetFlagToFFN();
    __aicore__ inline void SetFlagInAttn();
    __aicore__ inline void ActiveMaskCalCnt();

    TPipe *tpipe_{nullptr};
    Hcomm<COMM_PROTOCOL_UBC_CTP> hcomm_;
    __gm__ Mc2Aclnn::AttentionFFNContext *mc2Context_{nullptr};

    GM_ADDR localDataAddr_{nullptr};
    GM_ADDR localFlagAddr_{nullptr};
    GM_ADDR syncStatusWorkspaceGM_{nullptr};

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
    LocalTensor<int32_t> ffnFlagTensor_;
    LocalTensor<float> smoothScalesTensor_;
    LocalTensor<int32_t> expertIdsTensor_;
    LocalTensor<uint8_t> hcommTensor_;

    TBuf<> smoothScalesBuf_;
    TBuf<> expertIdsBuf_;
    TBuf<> statusBuf_;
    TBuf<> receiveDataCastFloatBuf_;
    TBuf<> sumOutBuf_;
    TBuf<> activeMaskBuf_;
    TBuf<> castTempBuf_;
    TBuf<> ffnFlagBuf_;
    TBuf<> ffnStatusBuf_;
    TBuf<> syncStatusWorkspaceBuf_;
    TBuf<> hcommBuf_;
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
    uint32_t ffnNum_{0};
    uint32_t ffnStartRankId_{0};
    uint32_t sessionId_{0};
    uint32_t microBatchId_{0};
    uint32_t layerId_{0};
    uint32_t expertIdsCnt_{0};
    uint32_t expertRankTableCnt_{0};
    uint32_t totalSendNum_{0};
    uint32_t sendNum_{0};
    uint32_t quantMode_{0};
    uint32_t scaleParamPad_{0};
    uint32_t hOutSizeAlign_{0};
    uint32_t curBsCnt_{0};
    uint32_t axisBsAlignSize_{0};
    uint32_t ffnNumAlignSize_{0};
    uint32_t ffnNumAlignCnt_{0};
    uint32_t aivWorkspaceOffset_{0};
    uint64_t hSize_{0};
    uint64_t hCommuSize_{0};
    uint64_t dataWorkspaceStride_{0};
    uint64_t layIdsExpRankTableOffset_{0};
    uint64_t winTokenDataOffset_{0};
    uint64_t winInfoTableOffset_{0};
    uint64_t winOffset_[WIN_OFFSET_CNT]{0, 0};
    bool isScales_{false};
};

template <TemplateAttentionToFfnUrmaTypeClass>
__aicore__ inline void AttentionToFfnUrma<TemplateAttentionToFfnUrmaTypeFunc>::Init(
    GM_ADDR mc2Context, GM_ADDR x, GM_ADDR sessionId, GM_ADDR microBatchId, GM_ADDR layerId, GM_ADDR expertIds,
    GM_ADDR expertRankTable, GM_ADDR scales, GM_ADDR active_mask, GM_ADDR workspaceGM, TPipe *pipe,
    const AttentionToFfnV2TilingData *tilingData)
{
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
#endif
    tpipe_ = pipe;
    aivId_ = GetBlockIdx();
    mc2Context_ = reinterpret_cast<__gm__ Mc2Aclnn::AttentionFFNContext *>(mc2Context);
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
    quantMode_ = tilingData->attentionToFfnV2Info.quantMode;
    isScales_ = tilingData->attentionToFfnV2Info.isScales;
    ffnStartRankId_ = tilingData->attentionToFfnV2Info.ffnStartRankId;
    ffnNum_ = worldSize_ - attentionWorkerNum_;
    curBsCnt_ = axisBS_;
    hSize_ = static_cast<uint64_t>(axisH_) * sizeof(XType);

    xGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ XType *>(x));
    sessionIdGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(sessionId));
    microBatchIdGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(microBatchId));
    layerIdGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(layerId));
    expertIdsGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(expertIds));
    expertRankTableGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(expertRankTable));
    activeMaskGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ bool *>(active_mask));

    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(sessionIdGMTensor_);
    sessionId_ = sessionIdGMTensor_.GetValue(0);
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(microBatchIdGMTensor_);
    microBatchId_ = microBatchIdGMTensor_.GetValue(0);
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(layerIdGMTensor_);
    layerId_ = layerIdGMTensor_.GetValue(0);

    expertIdsCnt_ = axisX_ * axisBS_ * (axisK_ + sharedExpertNum_);
    expertRankTableCnt_ = expertNum_ * expRankTableM_;
    layIdsExpRankTableOffset_ = layerId_ * expertRankTableCnt_;
    ffnNumAlignSize_ = Ceil(ffnNum_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    ffnNumAlignCnt_ = ffnNumAlignSize_ / sizeof(int32_t);
    aivWorkspaceOffset_ = Ceil(ffnNum_ * sizeof(int32_t), WORKSPACE_ELEMENT_OFFSET) * WORKSPACE_ELEMENT_OFFSET;
    axisBsAlignSize_ = Ceil(axisBS_ * sizeof(bool), UB_ALIGN) * UB_ALIGN;

    uint32_t expertIdsAlign = Ceil(expertIdsCnt_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(expertIdsBuf_, expertIdsAlign);
    tpipe_->InitBuffer(ffnFlagBuf_, UB_ALIGN);
    expertIdsTensor_ = expertIdsBuf_.Get<int32_t>();
    ffnFlagTensor_ = ffnFlagBuf_.Get<int32_t>();
    tpipe_->InitBuffer(statusBuf_, UB_ALIGN);
    statusTensor_ = statusBuf_.Get<int32_t>();

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

    dataWorkspaceStride_ = Ceil(hCommuSize_, static_cast<uint64_t>(URMA_FLAG_SLOT_SIZE)) * URMA_FLAG_SLOT_SIZE;
    const uint64_t urmaWorkspaceOffset = tilingData->attentionToFfnV2Info.urmaWorkspaceOffset;
    localDataAddr_ = workspaceGM + urmaWorkspaceOffset + static_cast<uint64_t>(aivId_) * dataWorkspaceStride_;
    localFlagAddr_ = workspaceGM + urmaWorkspaceOffset + static_cast<uint64_t>(aivNum_) * dataWorkspaceStride_ +
                     static_cast<uint64_t>(aivId_) * URMA_FLAG_SLOT_SIZE;
}

template <TemplateAttentionToFfnUrmaTypeClass>
__aicore__ inline void AttentionToFfnUrma<TemplateAttentionToFfnUrmaTypeFunc>::QuantInit(GM_ADDR scales)
{
    if constexpr (QuantMode == DYNAMIC_QUANT) {
        scaleParamPad_ = SCALE_PARAM_PAD_SIZE;
        hCommuSize_ = axisH_ * sizeof(int8_t) + scaleParamPad_;
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
        scaleParamPad_ = mxScaleNum;
        hCommuSize_ = hOutSizeAlign_ + scaleParamPad_;
        uint32_t hInSize = Ceil(axisH_, PERGROUP_BLOCK_SIZE) * PERGROUP_BLOCK_SIZE * sizeof(XType);
        tpipe_->InitBuffer(xInQueue_, BUFFER_NUM, hInSize);
        tpipe_->InitBuffer(xOutQueue_, BUFFER_NUM, hCommuSize_);
    } else {
        scaleParamPad_ = SCALE_PARAM_PAD_SIZE;
        hCommuSize_ = axisH_ * sizeof(XOutType) + scaleParamPad_;
        hOutSizeAlign_ = Ceil(axisH_ * sizeof(XOutType), UB_ALIGN) * UB_ALIGN;
        tpipe_->InitBuffer(xInQueue_, BUFFER_NUM, hSize_);
        tpipe_->InitBuffer(xOutQueue_, BUFFER_NUM, hCommuSize_);
    }
    if (isScales_) {
        scalesGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(scales));
    }
    uint32_t hFp32Size = axisH_ * sizeof(float);
    tpipe_->InitBuffer(receiveDataCastFloatBuf_, hFp32Size);
    tpipe_->InitBuffer(smoothScalesBuf_, hFp32Size);
}

template <TemplateAttentionToFfnUrmaTypeClass>
__aicore__ inline void AttentionToFfnUrma<TemplateAttentionToFfnUrmaTypeFunc>::ReduceMaxInplace(
    const LocalTensor<float> &srcLocal, uint32_t count)
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
            DataCopyPad(smoothScalesTensor_, scalesGMTensor_[expertIndex * axisH_], scalesCopyInParams,
                        copyPadExtParams);
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
        PipeBarrier<PIPE_V>();
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
        __ubuf__ StorageXInType *srcAddr = (__ubuf__ StorageXInType *)xInTensor_.GetPhyAddr();
        __ubuf__ uint16_t *maxExpAddr = (__ubuf__ uint16_t *)receiveDataCastFloat.GetPhyAddr();
        __ubuf__ uint16_t *halfScaleLocalAddr =
            (__ubuf__ uint16_t *)receiveDataCastFloat[Ceil(mxScaleNum, 32U) * 32U].GetPhyAddr();
        __ubuf__ int8_t *outLocalAddr = (__ubuf__ int8_t *)xOutTensor_.GetPhyAddr();
        __ubuf__ uint16_t *mxScaleLocalAddr;
        if constexpr (Std::IsSame<XOutType, fp4x2_e2m1_t>::value) {
            mxScaleLocalAddr = (__ubuf__ uint16_t *)xOutTensor_[Ceil(axisH_, FP4_ELEMS_PER_BYTE) * 1U].GetPhyAddr();
        } else {
            mxScaleLocalAddr = (__ubuf__ uint16_t *)xOutTensor_[axisH_ * 1U].GetPhyAddr();
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
        __ubuf__ StorageXInType *srcAddr = (__ubuf__ StorageXInType *)xInTensor_.GetPhyAddr();
        __ubuf__ uint16_t *maxExpAddr = (__ubuf__ uint16_t *)receiveDataCastFloat.GetPhyAddr();
        __ubuf__ uint16_t *halfScaleLocalAddr =
            (__ubuf__ uint16_t *)receiveDataCastFloat[Ceil(mxScaleNum, 32U) * 32U].GetPhyAddr();
        __ubuf__ int8_t *outLocalAddr = (__ubuf__ int8_t *)xOutTensor_.GetPhyAddr();
        __ubuf__ uint16_t *mxScaleLocalAddr = (__ubuf__ uint16_t *)xOutTensor_[axisH_ * 1U].GetPhyAddr();
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
        DataCopyPad(smoothScalesTensor_, scalesGMTensor_[expertIndex * axisH_], scalesCopyInParams, copyPadExtParams);
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
    PipeBarrier<PIPE_V>();
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
    uint64_t expRankTableOffset = expertId * expRankTableM_ + layIdsExpRankTableOffset_;
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
        expertRankTableGMTensor_[expRankTableOffset]);
    uint32_t rankCnt = expertRankTableGMTensor_.GetValue(expRankTableOffset);
    if (rankCnt == 0) {
        return;
    }
    uint32_t rankOffset = (sessionId_ % rankCnt) * RANK_OFFSET_STRIDE + 1;
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
        expertRankTableGMTensor_[expRankTableOffset + rankOffset]);
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
        expertRankTableGMTensor_[expRankTableOffset + rankOffset + 1]);
    toRankId_ = expertRankTableGMTensor_.GetValue(expRankTableOffset + rankOffset);
    localExpId_ = expertRankTableGMTensor_.GetValue(expRankTableOffset + rankOffset + 1);
}

template <TemplateAttentionToFfnUrmaTypeClass>
__aicore__ inline void AttentionToFfnUrma<TemplateAttentionToFfnUrmaTypeFunc>::SplitToCore(
    uint32_t curSendCnt, uint32_t curUseAivNum, uint32_t &startTokenId, uint32_t &endTokenId, uint32_t &sendTokenNum)
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
}

template <TemplateAttentionToFfnUrmaTypeClass>
__aicore__ inline void AttentionToFfnUrma<TemplateAttentionToFfnUrmaTypeFunc>::InitLocalFlag()
{
    constexpr uint32_t flagSlotElementNum = URMA_FLAG_SLOT_SIZE / sizeof(int32_t);
    Duplicate<int32_t>(statusTensor_, URMA_FLAG_VALUE, flagSlotElementNum);
    SyncFunc<HardEvent::V_MTE3>();
    GlobalTensor<int32_t> localFlagGMTensor;
    localFlagGMTensor.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(localFlagAddr_));
    DataCopy(localFlagGMTensor, statusTensor_, flagSlotElementNum);
    PipeBarrier<PIPE_MTE3>();
}

template <TemplateAttentionToFfnUrmaTypeClass>
__aicore__ inline GM_ADDR AttentionToFfnUrma<TemplateAttentionToFfnUrmaTypeFunc>::GetWindowAddr(int32_t rankId)
{
    return (GM_ADDR)mc2Context_->epHcclBuffer_[rankId];
}

template <TemplateAttentionToFfnUrmaTypeClass>
__aicore__ inline uint64_t AttentionToFfnUrma<TemplateAttentionToFfnUrmaTypeFunc>::GetUrmaCommHandle(uint32_t dstRank)
{
    uint32_t index = dstRank > rankId_ ? dstRank - 1U : dstRank;
    return mc2Context_->hcommHandle_[index];
}

template <TemplateAttentionToFfnUrmaTypeClass>
__aicore__ inline void AttentionToFfnUrma<TemplateAttentionToFfnUrmaTypeFunc>::ReadTokenMetaData(
    AttentionToFfnTokenMetaData &metaData, uint32_t tokenOffset)
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
}

template <TemplateAttentionToFfnUrmaTypeClass>
__aicore__ inline void AttentionToFfnUrma<TemplateAttentionToFfnUrmaTypeFunc>::SendToLocal(
    const AttentionToFfnTokenMetaData &metaData, const DataCopyExtParams &xCopyParams)
{
    DataCopyPadExtParams<XType> copyPadExtParams{false, 0U, 0U, 0U};
    DataCopyExtParams hCommuCopyParams = {1U, static_cast<uint32_t>(hCommuSize_), 0U, 0U, 0U};
    DataCopyExtParams flagCopyParams = {1U, sizeof(int32_t), 0U, 0U, 0U};

    if constexpr (QuantMode == 0) {
        xTmpTensor_ = xQueue_.AllocTensor<XType>();
        DataCopyPad(xTmpTensor_, xGMTensor_[metaData.tokenId * axisH_], xCopyParams, copyPadExtParams);
        xQueue_.EnQue(xTmpTensor_);
        xTmpTensor_ = xQueue_.DeQue<XType>();
        GlobalTensor<XType> tokenDataGMTensor;
        tokenDataGMTensor.SetGlobalBuffer(reinterpret_cast<__gm__ XType *>(metaData.remoteDataAddr));
        DataCopyPad(tokenDataGMTensor, xTmpTensor_, xCopyParams);
        xQueue_.FreeTensor<XType>(xTmpTensor_);
    } else {
        xInTensor_ = xInQueue_.AllocTensor<XType>();
        DataCopyPad(xInTensor_, xGMTensor_[metaData.tokenId * axisH_], xCopyParams, copyPadExtParams);
        xInQueue_.EnQue(xInTensor_);
        xInTensor_ = xInQueue_.DeQue<XType>();
        xOutTensor_ = xOutQueue_.AllocTensor<StorageXOutType>();
        QuantProcess(dstExpertId_);
        GlobalTensor<uint8_t> tokenDataGMTensor;
        tokenDataGMTensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(metaData.remoteDataAddr));
        xOutQueue_.EnQue(xOutTensor_);
        xOutTensor_ = xOutQueue_.DeQue<StorageXOutType>();
        auto xOutBytesTensor = xOutTensor_.template ReinterpretCast<uint8_t>();
        DataCopyPad(tokenDataGMTensor, xOutBytesTensor, hCommuCopyParams);
        xOutQueue_.FreeTensor<StorageXOutType>(xOutTensor_);
    }
    PipeBarrier<PIPE_MTE3>();

    statusTensor_.SetValue(0, metaData.localExpId);
    SyncFunc<HardEvent::S_MTE3>();
    GlobalTensor<int32_t> tokenInfoTableGMTensor;
    tokenInfoTableGMTensor.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(metaData.remoteFlagAddr));
    DataCopyPad(tokenInfoTableGMTensor, statusTensor_, flagCopyParams);
}

template <TemplateAttentionToFfnUrmaTypeClass>
__aicore__ inline void AttentionToFfnUrma<TemplateAttentionToFfnUrmaTypeFunc>::StageRemoteData(
    const AttentionToFfnTokenMetaData &metaData, const DataCopyExtParams &xCopyParams)
{
    DataCopyPadExtParams<XType> copyPadExtParams{false, 0U, 0U, 0U};
    DataCopyExtParams hCommuCopyParams = {1U, static_cast<uint32_t>(hCommuSize_), 0U, 0U, 0U};
    DataCopyExtParams flagCopyParams = {1U, sizeof(int32_t), 0U, 0U, 0U};

    if constexpr (QuantMode == 0) {
        xTmpTensor_ = xQueue_.AllocTensor<XType>();
        DataCopyPad(xTmpTensor_, xGMTensor_[metaData.tokenId * axisH_], xCopyParams, copyPadExtParams);
        xQueue_.EnQue(xTmpTensor_);
        xTmpTensor_ = xQueue_.DeQue<XType>();
        GlobalTensor<XType> localDataGMTensor;
        localDataGMTensor.SetGlobalBuffer(reinterpret_cast<__gm__ XType *>(localDataAddr_));
        DataCopyPad(localDataGMTensor, xTmpTensor_, xCopyParams);
        xQueue_.FreeTensor<XType>(xTmpTensor_);
    } else {
        xInTensor_ = xInQueue_.AllocTensor<XType>();
        DataCopyPad(xInTensor_, xGMTensor_[metaData.tokenId * axisH_], xCopyParams, copyPadExtParams);
        xInQueue_.EnQue(xInTensor_);
        xInTensor_ = xInQueue_.DeQue<XType>();
        xOutTensor_ = xOutQueue_.AllocTensor<StorageXOutType>();
        QuantProcess(dstExpertId_);
        GlobalTensor<uint8_t> localDataGMTensor;
        localDataGMTensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(localDataAddr_));
        xOutQueue_.EnQue(xOutTensor_);
        xOutTensor_ = xOutQueue_.DeQue<StorageXOutType>();
        auto xOutBytesTensor = xOutTensor_.template ReinterpretCast<uint8_t>();
        DataCopyPad(localDataGMTensor, xOutBytesTensor, hCommuCopyParams);
        xOutQueue_.FreeTensor<StorageXOutType>(xOutTensor_);
    }
    SyncFunc<HardEvent::MTE3_S>();

    statusTensor_.SetValue(0, metaData.localExpId);
    SyncFunc<HardEvent::S_MTE3>();
    GlobalTensor<int32_t> localFlagGMTensor;
    localFlagGMTensor.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(localFlagAddr_));
    DataCopyPad(localFlagGMTensor, statusTensor_, flagCopyParams);
    PipeBarrier<PIPE_MTE3>();
}

template <TemplateAttentionToFfnUrmaTypeClass>
__aicore__ inline void AttentionToFfnUrma<TemplateAttentionToFfnUrmaTypeFunc>::SendToRemote(
    uint32_t sourceAivId, const AttentionToFfnTokenMetaData &metaData)
{
    GM_ADDR sourceDataAddr = localDataAddr_ + static_cast<uint64_t>(sourceAivId) * dataWorkspaceStride_;
    GM_ADDR sourceFlagAddr = localFlagAddr_ + static_cast<uint64_t>(sourceAivId) * URMA_FLAG_SLOT_SIZE;
    uint64_t channel = GetUrmaCommHandle(static_cast<uint32_t>(metaData.toRankId));

    int32_t ret = hcomm_.WriteNbi(channel, metaData.remoteDataAddr, sourceDataAddr, static_cast<int64_t>(hCommuSize_));
    ret = hcomm_.Drain(channel);

    ret = hcomm_.WriteNbi(channel, metaData.remoteFlagAddr, sourceFlagAddr, static_cast<int64_t>(sizeof(int32_t)));
    ret = hcomm_.Drain(channel);
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
    attnTableGMTensor.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(attnTokenInfoTableGM));
    DataCopyExtParams dataCopyParams = {1U, static_cast<uint32_t>(sendNum * sizeof(int32_t)), 0U, 0U, 0U};
    LocalTensor<int32_t> tempTensor = expertIdsBuf_.Get<int32_t>();
    Duplicate<int32_t>(tempTensor, static_cast<int32_t>(1), sendNum);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopyPad(attnTableGMTensor[startId], tempTensor, dataCopyParams);
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
        tpipe_->InitBuffer(syncStatusWorkspaceBuf_, aivNum_ * sentFFNNumAlignSize);
        syncStatusWorkspaceTensor_ = syncStatusWorkspaceBuf_.Get<int32_t>();
        syncStatusGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(syncStatusWorkspaceGM_));
        DataCopyPad(syncStatusWorkspaceTensor_, syncStatusGMTensor_[0], syncStatusParams, copyPadParams);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        LocalTensor<float> syncStatusWorkspaceTensorFloat = syncStatusWorkspaceTensor_.ReinterpretCast<float>();
        LocalTensor<float> ffnStatusTensorFloat = ffnStatusTensor_.ReinterpretCast<float>();
        const uint32_t shape[] = {aivNum_, static_cast<uint32_t>(sentFFNNumAlignSize / sizeof(float))};
        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::RA, true>(ffnStatusTensorFloat,
                                                                      syncStatusWorkspaceTensorFloat, shape, true);
        SyncFunc<AscendC::HardEvent::V_S>();
    }

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
            tableFlagGMTensor.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(tableFlagGM));
            DataCopyPad(tableFlagGMTensor, statusTensor_, statusParams);
            PipeBarrier<PIPE_MTE3>();
        } else {
            GlobalTensor<int32_t> localFlagGMTensor;
            localFlagGMTensor.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(localFlagAddr_));
            DataCopyPad(localFlagGMTensor, statusTensor_, statusParams);
            PipeBarrier<PIPE_MTE3>();
            uint64_t channel = GetUrmaCommHandle(ffnIdx);
            int32_t ret = hcomm_.WriteNbi(channel, tableFlagGM, localFlagAddr_,
                                          static_cast<int64_t>(sizeof(int32_t) * TOKEN_INFO_TABLE_COPY_BLOCK_CNT));
            ret = hcomm_.Drain(channel);
        }
    }
}

template <TemplateAttentionToFfnUrmaTypeClass>
__aicore__ inline void AttentionToFfnUrma<TemplateAttentionToFfnUrmaTypeFunc>::Process()
{
    if ASCEND_IS_AIV {
        HcommInit();
        InitLocalFlag();

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

        for (uint32_t idx = 0; idx < Ceil(expertRankTableCnt_, EXPERT_TABLE_REP_STRIDE); ++idx) {
            DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
                expertRankTableGMTensor_[layIdsExpRankTableOffset_ + idx * EXPERT_TABLE_REP_STRIDE]);
        }

        const uint64_t roundNum = Ceil(static_cast<uint64_t>(totalSendNum_), static_cast<uint64_t>(aivNum_));
        DataCopyExtParams xCopyParams = {1U, static_cast<uint32_t>(hSize_), 0U, 0U, 0U};

        for (uint64_t round = 0; round < roundNum; ++round) {
            const uint32_t tokenOffset = static_cast<uint32_t>(static_cast<uint64_t>(aivId_) + round * aivNum_);
            if (tokenOffset < totalSendNum_) {
                AttentionToFfnTokenMetaData metaData;
                ReadTokenMetaData(metaData, tokenOffset);

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
                    StageRemoteData(metaData, xCopyParams);
                }
            }

            SyncAll<true>();

            if (aivId_ == 0U) {
                for (uint32_t sourceAivId = 0U; sourceAivId < aivNum_; ++sourceAivId) {
                    const uint32_t sourceTokenOffset =
                        static_cast<uint32_t>(static_cast<uint64_t>(sourceAivId) + round * aivNum_);
                    if (sourceTokenOffset >= totalSendNum_) {
                        break;
                    }
                    AttentionToFfnTokenMetaData metaData;
                    ReadTokenMetaData(metaData, sourceTokenOffset);
                    if (metaData.toRankId == static_cast<int32_t>(rankId_)) {
                        continue;
                    }
                    SendToRemote(sourceAivId, metaData);
                }
            }

            SyncAll<true>();
        }

        if constexpr (isSync) {
            SyncFunc<AscendC::HardEvent::V_MTE3>();
            syncStatusGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(syncStatusWorkspaceGM_));
            uint32_t aivWorkspaceStride = aivWorkspaceOffset_ / sizeof(int32_t);
            DataCopy(syncStatusGMTensor_[aivId_ * aivWorkspaceStride], ffnStatusTensor_, ffnNumAlignCnt_);
        }

        PipeBarrier<PIPE_ALL>();
        SyncAll<true>();
        SetFlagToFFN();
        if constexpr (isActiveMask) {
            SetFlagInAttn();
        }
    }
}

} // namespace AttentionToFFNImpl
#endif // ATTENTION_TO_FFN_URMA_H
