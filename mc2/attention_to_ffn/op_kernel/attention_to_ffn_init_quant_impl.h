/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Initialization and quantization member definitions.
#pragma once
#include "attention_to_ffn.h"

namespace AttentionToFFNImpl {

template <TemplateAttentionToFFNTypeClass>
__aicore__ inline void AttentionToFFN<TemplateAttentionToFFNTypeFunc>::QuantInit(GM_ADDR scales)
{
    scaleParamPad_ = SCALE_PARAM_PAD_SIZE; // 预留128B给量化参数
    hCommuSize_ = axisH_ * sizeof(int8_t) + scaleParamPad_;
    tpipe_->InitBuffer(xInQueue_, BUFFER_NUM, hSize_);       // 14K *2
    tpipe_->InitBuffer(xOutQueue_, BUFFER_NUM, hCommuSize_); // 7K *2 + 256B
    if (isScales_) {                                         // 输入smoothScale参数
        scalesGMTensor_.SetGlobalBuffer((__gm__ float *)(scales));
    }
    uint32_t hFp32Size = axisH_ * sizeof(float);
    tpipe_->InitBuffer(receiveDataCastFloatBuf_, hFp32Size); // H * 4
    tpipe_->InitBuffer(smoothScalesBuf_, hFp32Size);
}

template <TemplateAttentionToFFNTypeClass>
__aicore__ inline void AttentionToFFN<TemplateAttentionToFFNTypeFunc>::InitByTinglingData(
    const AttentionToFFNTilingData *tilingData)
{
    aivId_ = GetBlockIdx();
    auto contextGM0 = AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
    winContext_ = (__gm__ HcclOpResParam *)AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
    rankId_ = winContext_->localUsrRankId;
    axisX_ = tilingData->attentionToFFNInfo.X;
    axisBS_ = tilingData->attentionToFFNInfo.BS;
    axisH_ = tilingData->attentionToFFNInfo.H;
    axisL_ = tilingData->attentionToFFNInfo.L;
    axisK_ = tilingData->attentionToFFNInfo.K;
    expertNum_ = tilingData->attentionToFFNInfo.expertNum; // 所有专家数：共享专家+所有的moe专家
    moeExpertNum_ = tilingData->attentionToFFNInfo.moeExpertNum;
    expRankTableM_ = tilingData->attentionToFFNInfo.expRankTableM;
    microBatchNum_ = tilingData->attentionToFFNInfo.microBatchNum;
    attentionWorkerNum_ = tilingData->attentionToFFNInfo.attentionWorkerNum;
    infoTableLastDimNum_ = tilingData->attentionToFFNInfo.infoTableLastDimNum;
    axisHS_ = tilingData->attentionToFFNInfo.HS;
    aivNum_ = tilingData->attentionToFFNInfo.aivNum;
    worldSize_ = tilingData->attentionToFFNInfo.worldSize;
    quantMode_ = tilingData->attentionToFFNInfo.quantMode;
    isScales_ = tilingData->attentionToFFNInfo.isScales;
    ffnStartRankId_ = tilingData->attentionToFFNInfo.ffnStartRankId;
    windowType_ = tilingData->attentionToFFNInfo.windowType;
    sharedExpertNum_ = tilingData->attentionToFFNInfo.sharedExpertNum;
    ffnNum_ = worldSize_ - attentionWorkerNum_;
    curBsCnt_ = axisBS_;
    hSize_ = axisH_ * sizeof(XType);
    expertIdsCnt_ = axisX_ * axisBS_ * axisK_;
    expertRankTableCnt_ = expertNum_ * expRankTableM_;
    ffnNumAlignSize_ = Ceil(ffnNum_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    axisBsAlignSize_ = Ceil(axisBS_ * sizeof(bool), UB_ALIGN) * UB_ALIGN;
    aivWorkspaceOffset_ = Ceil(ffnNum_ * sizeof(int32_t), WORKSPACE_ELEMENT_OFFSET) * WORKSPACE_ELEMENT_OFFSET;
}

template <TemplateAttentionToFFNTypeClass>
__aicore__ inline void AttentionToFFN<TemplateAttentionToFFNTypeFunc>::ReadSessionMetadata()
{
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(sessionIdGMTensor_);
    sessionId_ = sessionIdGMTensor_.GetValue(0); // 当前x=1，session_ids直接从gm上读取第一个值
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(microBatchIdGMTensor_);
    microBatchId_ = microBatchIdGMTensor_.GetValue(0); // 当前x=1，microBatchId_直接从gm上读取第一个值
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(layerIdGMTensor_);
    layerId_ = layerIdGMTensor_.GetValue(0); // 当前x=1，layerId_直接从gm上读取第一个值
    layIdsExpRankTableOffset_ = layerId_ * expertRankTableCnt_;
}

template <TemplateAttentionToFFNTypeClass>
__aicore__ inline void AttentionToFFN<TemplateAttentionToFFNTypeFunc>::Init(
    GM_ADDR x, GM_ADDR sessionId, GM_ADDR microBatchId, GM_ADDR layerId, GM_ADDR expertIds, GM_ADDR expertRankTable,
    GM_ADDR scales, GM_ADDR activeMask, GM_ADDR workspaceGM, TPipe *pipe, const AttentionToFFNTilingData *tilingData)
{
    tpipe_ = pipe;
    InitByTinglingData(tilingData);

    xGMTensor_.SetGlobalBuffer((__gm__ XType *)x);
    sessionIdGMTensor_.SetGlobalBuffer((__gm__ int32_t *)sessionId);
    microBatchIdGMTensor_.SetGlobalBuffer((__gm__ int32_t *)microBatchId);
    layerIdGMTensor_.SetGlobalBuffer((__gm__ int32_t *)layerId);
    expertIdsGMTensor_.SetGlobalBuffer((__gm__ int32_t *)expertIds);
    expertRankTableGMTensor_.SetGlobalBuffer((__gm__ int32_t *)expertRankTable);
    activeMaskGMTensor_.SetGlobalBuffer((__gm__ bool *)activeMask);
    ReadSessionMetadata();

    uint32_t expertIdsAlign = Ceil(expertIdsCnt_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN; // 约束32对齐
    tpipe_->InitBuffer(expertIdsBuf_, expertIdsAlign);                                    // 对齐32B
    tpipe_->InitBuffer(ffnFlagBuf_, UB_ALIGN);                                            // 对齐32B
    expertIdsTensor_ = expertIdsBuf_.Get<int32_t>();
    ffnFlagTensor_ = ffnFlagBuf_.Get<int32_t>();
    if constexpr (isQuant) {
        QuantInit(scales);
        castTempBuf_ = receiveDataCastFloatBuf_;
        sumOutBuf_ = smoothScalesBuf_;
    } else {
        tpipe_->InitBuffer(xQueue_, BUFFER_NUM, hSize_); // H * 2
    }
    if constexpr (isSync) {
        tpipe_->InitBuffer(ffnStatusBuf_, ffnNumAlignSize_); // ffnNum_
        ffnStatusTensor_ = ffnStatusBuf_.Get<int32_t>();
        syncStatusWorkspaceGM_ = workspaceGM;
    }
    if constexpr (isActiveMask) {
        tpipe_->InitBuffer(activeMaskBuf_, axisBsAlignSize_); // BS
        if constexpr (!isQuant) {
            uint32_t bsAlignHalf = Ceil(axisBS_ * sizeof(half), UB_ALIGN) * UB_ALIGN; // 约束32对齐
            tpipe_->InitBuffer(castTempBuf_, bsAlignHalf);                            // BS * 2
            tpipe_->InitBuffer(sumOutBuf_, bsAlignHalf);                              // BS * 2
        }
        attnStatusBuf_ = expertIdsBuf_;
    }

    winOffset_[0] = 0;
    winOffset_[1] = Ceil(attentionWorkerNum_ * microBatchNum_ * infoTableLastDimNum_ * sizeof(int32_t), WIN_ALIGN) *
                    WIN_ALIGN; // token_info_table大小 偏移向上取整
    winInfoTableOffset_ = (sessionId_ * microBatchNum_ * infoTableLastDimNum_ + microBatchId_ * infoTableLastDimNum_) *
                          sizeof(int32_t); // tokenInfoTable上当前attnWorkId以及microBatchId偏移
    winTokenDataOffset_ =
        (sessionId_ * microBatchNum_ * axisBS_ * (axisK_ + sharedExpertNum_) * axisHS_) +
        (microBatchId_ * axisBS_ * (axisK_ + sharedExpertNum_) * axisHS_); // tokenData上attnWorkId以及microBatchId偏移
}

template <TemplateAttentionToFFNTypeClass>
__aicore__ inline void AttentionToFFN<TemplateAttentionToFFNTypeFunc>::ActiveMaskCalCnt()
{ // 搬运x_active_mask, 当前仅用于计算有效token总数
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

template <TemplateAttentionToFFNTypeClass>
__aicore__ inline void AttentionToFFN<TemplateAttentionToFFNTypeFunc>::ReduceMaxInplace(
    const LocalTensor<float> &srcLocal, uint32_t count)
{
    uint64_t repsFp32 = count >> 6;       // 6 is count / elemPerRefFp32
    uint64_t offsetsFp32 = repsFp32 << 6; // 6 is repsFp32 * elemPerRefFp32
    uint64_t remsFp32 = count & 0x3f;     // 0x3f 63, count % elemPerRefFp32
    const uint64_t elemPerRefFp32 = 64UL; // 256 bit / sizeof(float)
    if (likely(repsFp32 > 1)) {           // 8 is rep stride
        Max(srcLocal, srcLocal[elemPerRefFp32], srcLocal, elemPerRefFp32, repsFp32 - 1, {1, 1, 1, 0, REP_STRIDE, 0});
        PipeBarrier<PIPE_V>();
    }
    if (unlikely(remsFp32 > 0) && unlikely(offsetsFp32 > 0)) {
        Max(srcLocal, srcLocal[offsetsFp32], srcLocal, remsFp32, 1, {1, 1, 1, 0, REP_STRIDE, 0});
        PipeBarrier<PIPE_V>();
    }
    uint32_t mask = (repsFp32 > 0) ? elemPerRefFp32 : count;
    // 8 is rep stride
    WholeReduceMax(srcLocal, srcLocal, mask, 1, REP_STRIDE, 1, REP_STRIDE);
}

template <TemplateAttentionToFFNTypeClass>
__aicore__ inline void AttentionToFFN<TemplateAttentionToFFNTypeFunc>::QuantProcess(uint32_t expertIndex)
{
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
        dynamicScale = float(INT8_MAX_VALUE) / floatLocalAbsTemp.GetValue(0);
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
    floatLocalTemp.SetValue(hOutSizeAlign / sizeof(float), float(1.0) / dynamicScale); // int8->float32
}

} // namespace AttentionToFFNImpl
