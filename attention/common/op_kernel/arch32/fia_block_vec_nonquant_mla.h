/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fia_block_vec_nonquant_mla.h
 * \brief
 */
#ifndef FIA_BLOCK_VEC_NONQUANT_MLA_H
#define FIA_BLOCK_VEC_NONQUANT_MLA_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "../fia_public_define.h"
#include "../vector_common.h"

using namespace AttentionCommon;
using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;

template <typename FIAT> 
class FiaBlockVecNonQuantMla {
public:
    // =================================类型定义区=================================
    // 中间计算数据类型为float，高精度模式
    using T = float;

    using Q_T = typename FIAT::queryType;
    using KV_T = typename FIAT::kvType;
    using OUT_T = typename FIAT::outputType;
    using ORIGIN_T = typename FIAT::orginalType;
    static constexpr bool PAGE_ATTENTION = FIAT::pageAttention;
    static constexpr bool FLASH_DECODE = FIAT::flashDecode;
    static constexpr FIA_LAYOUT LAYOUT_T = FIAT::layout;
    static constexpr FIA_LAYOUT KV_LAYOUT_T = FIAT::kvLayout;

    using UPDATE_T = T;
    using TMP_T = T;
    using MM1_OUT_T = float;
    using MM2_OUT_T = float;

    __aicore__ inline FiaBlockVecNonQuantMla(){};
    __aicore__ inline void ProcessVec1L(const AttentionCommon::RunInfo &info);
    __aicore__ inline void ProcessVec2L(const AttentionCommon::RunInfo &info);
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void InitParams(const struct AttentionCommon::ConstInfo &constInfo,
                                      const optiling::FusedInferAttentionScoreTilingData *__restrict tilingData);
    __aicore__ inline void InitMm2ResInt32GmGlobalTensor(GlobalTensor<int32_t> mm2ResInt32Gm);
    __aicore__ inline void InitVec1GlobalTensor(GlobalTensor<MM1_OUT_T> mm1ResGm, GlobalTensor<KV_T> vec1ResGm,
                                                GlobalTensor<bool> attenMaskBoolGm,
                                                GlobalTensor<uint64_t> actualSeqLengthsGmQ,
                                                GlobalTensor<uint64_t> actualSeqLengthsGm, GlobalTensor<T> lseMaxFdGm,
                                                GlobalTensor<T> lseSumFdGm);
    __aicore__ inline void InitVec2GlobalTensor(GlobalTensor<T> accumOutGm, GlobalTensor<UPDATE_T> vec2ResGm,
                                                GlobalTensor<MM2_OUT_T> mm2ResGm, GlobalTensor<OUT_T> attentionOutGm);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    __aicore__ inline void InitSoftmaxDefaultBuffer();
    // ================================Vector1==========================================
    __aicore__ inline void ProcessVec1SingleBuf(const AttentionCommon::RunInfo &info, const MSplitInfo &mSplitInfo);
    __aicore__ inline void DealBmm1ResBaseBlock(const AttentionCommon::RunInfo &info, const MSplitInfo &mSplitInfo, uint32_t startRow,
                                                uint32_t dealRowCount, uint32_t columnCount,
                                                uint32_t actualColumnCount);
    __aicore__ inline void SoftmaxFlashV2Compute(const AttentionCommon::RunInfo &info, const MSplitInfo &mSplitInfo,
                                                 LocalTensor<T> &mmResUb, LocalTensor<uint8_t> &softmaxTmpUb,
                                                 uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount,
                                                 uint32_t actualColumnCount);
    __aicore__ inline void AmlaVecCompute(const AttentionCommon::RunInfo &info, const MSplitInfo &mSplitInfo, LocalTensor<T> &mmResUb,
                                          LocalTensor<uint8_t> &softmaxTmpUb, uint32_t startRow, uint32_t dealRowCount,
                                          uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void ElewiseCompute(const AttentionCommon::RunInfo &info, const MSplitInfo &mSplitInfo, LocalTensor<T> &mmResUb,
                                          TBuf<> &tmpBuf, uint32_t startRow, uint32_t dealRowCount,
                                          uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void AttentionMaskCompute(const AttentionCommon::RunInfo &info, const MSplitInfo &mSplitInfo,
                                                LocalTensor<T> &mmResUb, TBuf<> &tmpBuf, uint32_t startRow,
                                                uint32_t dealRowCount, uint32_t columnCount,
                                                uint32_t actualColumnCount);
    __aicore__ inline void AttenMaskCopyDirectly(const AttentionCommon::RunInfo &info, LocalTensor<bool> &attenMaskUb,
                                                 uint32_t dealRowCount, uint32_t actualColumnCount);
    __aicore__ inline bool IsSkipAttenMask(const AttentionCommon::RunInfo &info, const MSplitInfo &mSplitInfo, uint32_t startRow,
                                           uint32_t dealRowCount);
    __aicore__ inline void AttenMaskCopyForSplitG(const AttentionCommon::RunInfo &info, const MSplitInfo &mSplitInfo,
                                                  LocalTensor<bool> &attenMaskUb, uint32_t startRow,
                                                  uint32_t dealRowCount, bool &selectNext, bool &selectPre);
    __aicore__ inline void AttenMaskCopyForBNSD(const AttentionCommon::RunInfo &info, const MSplitInfo &mSplitInfo,
                                                LocalTensor<bool> &attenMaskUb, uint32_t startRow,
                                                uint32_t dealRowCount);
    __aicore__ inline void ProcessAmlaNupdate(const AttentionCommon::RunInfo &info, const MSplitInfo &mSplitInfo);
    __aicore__ inline void ComputeLogSumExpAndCopyToGm(const AttentionCommon::RunInfo &info, const MSplitInfo &mSplitInfo,
                                                       LocalTensor<T> &softmaxSumUb, LocalTensor<T> &softmaxMaxUb);
    // ================================Vecotr2==========================================
    __aicore__ inline void ProcessVec2SingleBuf(const AttentionCommon::RunInfo &info, const MSplitInfo &mSplitInfo);
    __aicore__ inline void DealBmm2ResBaseBlock(const AttentionCommon::RunInfo &info, const MSplitInfo &mSplitInfo, uint32_t startRow,
                                                uint32_t dealRowCount, uint32_t columnCount,
                                                uint32_t actualColumnCount);
    __aicore__ inline void ProcessVec2Inner(const AttentionCommon::RunInfo &info, const MSplitInfo &mSplitInfo, uint32_t mStartRow,
                                            uint32_t mDealSize);
    __aicore__ inline void Bmm2DataCopyOutTrans(const AttentionCommon::RunInfo &info, LocalTensor<OUT_T> &attenOutUb, uint32_t wsMStart,
                                                uint32_t dealRowCount, uint32_t columnCount,
                                                uint32_t actualColumnCount);
    __aicore__ inline void Bmm2ResCopyOut(const AttentionCommon::RunInfo &info, LocalTensor<T> &bmm2ResUb, uint32_t wsMStart,
                                          uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void Bmm2CastAndCopyOut(const AttentionCommon::RunInfo &info, LocalTensor<T> &bmm2ResUb, uint32_t wsMStart,
                                              uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void Bmm2FDDataCopyOut(const AttentionCommon::RunInfo &info, LocalTensor<T> &bmm2ResUb, uint32_t wsMStart,
                                             uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline uint64_t CalcAccumOffset(uint32_t bN2Idx, uint32_t gS1Idx);
    template <typename RT>
    __aicore__ inline void DealInvalidRows(const AttentionCommon::RunInfo &info, LocalTensor<RT> &attenOutUb, uint32_t wsMStart,
                                           uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);
    template <typename RT>
    __aicore__ inline void DealInvalidRowsBelow(const AttentionCommon::RunInfo &info, LocalTensor<RT> &attenOutUb,
                                                uint32_t wsMStart, uint32_t dealRowCount, uint32_t columnCount);
    __aicore__ inline void GetConfusionTransposeTiling(int64_t numR, int64_t numC, const uint32_t stackBufferSize,
                                                       const uint32_t typeSize, ConfusionTransposeTiling &tiling);

protected:
    uint32_t pingpongFlag = 0U;
    GlobalTensor<int32_t> mm2ResInt32Gm;
    GlobalTensor<MM1_OUT_T> mm1ResGm; // 存放S
    GlobalTensor<KV_T> vec1ResGm;     // 存放A1, A2
    GlobalTensor<T> lseSumFdGm;       // no
    GlobalTensor<T> lseMaxFdGm;

    GlobalTensor<bool> attenMaskBoolGm;
    GlobalTensor<uint64_t> actualSeqLengthsGmQ;
    GlobalTensor<uint64_t> actualSeqLengthsGm;

    GlobalTensor<UPDATE_T> vec2ResGm;
    GlobalTensor<MM2_OUT_T> mm2ResGm;

    GlobalTensor<T> accumOutGm;
    GlobalTensor<OUT_T> attentionOutGm;

    // =================================常量区=================================
    static constexpr uint64_t SYNC_INPUT_BUF1_FLAG = 2;
    static constexpr uint64_t SYNC_INPUT_BUF1_PONG_FLAG = 3;
    static constexpr uint64_t SYNC_INPUT_BUF2_FLAG = 4;
    static constexpr uint64_t SYNC_INPUT_BUF2_PONG_FLAG = 5;
    static constexpr uint64_t SYNC_OUTPUT_BUF1_FLAG = 4;
    static constexpr uint64_t SYNC_OUTPUT_BUF2_FLAG = 5;
    static constexpr uint32_t INPUT1_BUFFER_OFFSET = AttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_32K;
    static constexpr uint32_t INPUT2_BUFFER_OFFSET = AttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_8K;
    static constexpr uint32_t SOFTMAX_TMP_BUFFER_OFFSET = AttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_1K;

    static constexpr T BOOL_ATTEN_MASK_SCALAR_VALUE = -1000000000000.0; // 用于mask为bool类型
    static constexpr uint64_t kvHeadNum = 1ULL;
    static constexpr uint32_t BASE_BLOCK_MAX_ELEMENT_NUM = AttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_32K / sizeof(T); // 32768/4=8096
    static constexpr uint32_t BLOCK_ELEMENT_NUM = fa_base_vector::BYTE_BLOCK / sizeof(T); // 32/4=8
    static constexpr T FLOAT_E_SCALAR = 8388608;
    static constexpr T LN2 = 0.6931471805599453094172;
    static constexpr T RECIP_OF_LN2 = 1 / LN2;
    AttentionCommon::ConstInfo constInfo = {};

    static constexpr T SOFTMAX_MIN_NUM = -2e38;
    static constexpr uint64_t headDim = 512ULL;
    static constexpr uint64_t headDimAlign = 512ULL;
    static constexpr uint64_t headDimRope = 64ULL;
    static constexpr uint64_t headDimAll = 576ULL;

private:
    // ================================Local Buffer区====================================
    // queue
    TBuf<> inputBuff1;  // 32K
    TBuf<> inputBuff2;  // 16K
    TBuf<> outputBuff1; // 32K
    TBuf<> outputBuff2; // 4K

    // 临时tbuf
    TBuf<> tmpBuff1;         // 32K
    TBuf<> attenMaskTmpBuff; // 8K

    TBuf<> nValueBuff;
    TBuf<> cofValueBuff;
    TBuf<> aMlaSumBuff;
    TBuf<> softmaxMaxBuff;        // PRE_LOAD_NUM * 2K
    TBuf<> softmaxExpBuff;        // PRE_LOAD_NUM * 2K
    TBuf<> softmaxSumBuff;        // PRE_LOAD_NUM * 2K
    TBuf<> softmaxMaxDefaultBuff; // 2K
    TBuf<> softmaxSumDefaultBuff; // 2K

    uint64_t mSizeVector = 0ULL;
    uint64_t mSizeVStart = 0ULL;

    LocalTensor<T> softmaxMaxDefaultUb;
    LocalTensor<T> softmaxSumDefaultUb;

    LocalTensor<T> nValueUb;
    LocalTensor<T> cofValueUb;
    LocalTensor<T> aMlaSumUb;
    LocalTensor<T> softmaxMaxUb;
    LocalTensor<T> softmaxSumUb;
    LocalTensor<T> softmaxExpUb;

    // attention mask
    uint32_t attenMaskSizeAlign = 0U;

    const optiling::FusedInferAttentionScoreTilingData *__restrict tilingData = nullptr;
};

template <typename FIAT> __aicore__ inline void FiaBlockVecNonQuantMla<FIAT>::InitBuffers(TPipe *pipe)
{
    // queue
    pipe->InitBuffer(inputBuff1, AttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_32K * 2); // 2:pingpong
    pipe->InitBuffer(inputBuff2, AttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_8K * 2);  // 2:pingpong
    pipe->InitBuffer(outputBuff1, AttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(outputBuff2, AttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_4K);

    // tmpBuff
    pipe->InitBuffer(tmpBuff1, AttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(attenMaskTmpBuff, AttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_8K);

    // M_MAX = 512/2vector = 256, 256 * sizeof(T) * N_Buffer
    pipe->InitBuffer(nValueBuff, AttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_1K * constInfo.preLoadNum);
    pipe->InitBuffer(cofValueBuff, AttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_1K * constInfo.preLoadNum);
    pipe->InitBuffer(aMlaSumBuff, AttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_1K * constInfo.preLoadNum);

    pipe->InitBuffer(softmaxMaxBuff, AttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_1K * constInfo.preLoadNum);
    pipe->InitBuffer(softmaxExpBuff, AttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_1K * constInfo.preLoadNum);
    pipe->InitBuffer(softmaxSumBuff, AttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_1K * constInfo.preLoadNum);

    pipe->InitBuffer(softmaxMaxDefaultBuff, AttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_1K);
    pipe->InitBuffer(softmaxSumDefaultBuff, AttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_1K);

    nValueUb = nValueBuff.Get<T>();
    cofValueUb = cofValueBuff.Get<T>();
    aMlaSumUb = aMlaSumBuff.Get<T>();

    softmaxMaxUb = softmaxMaxBuff.Get<T>();
    softmaxSumUb = softmaxSumBuff.Get<T>();
    softmaxExpUb = softmaxExpBuff.Get<T>();

    softmaxMaxDefaultUb = softmaxMaxDefaultBuff.Get<T>();
    softmaxSumDefaultUb = softmaxSumDefaultBuff.Get<T>();
}

template <typename FIAT>
__aicore__ inline void
FiaBlockVecNonQuantMla<FIAT>::InitParams(const struct AttentionCommon::ConstInfo &constInfo,
                                                 const optiling::FusedInferAttentionScoreTilingData *__restrict tilingData)
{
    this->constInfo = constInfo;
    this->tilingData = tilingData;
}

template <typename FIAT>
__aicore__ inline void
FiaBlockVecNonQuantMla<FIAT>::InitMm2ResInt32GmGlobalTensor(GlobalTensor<int32_t> mm2ResInt32Gm)
{
    this->mm2ResInt32Gm = mm2ResInt32Gm;
}

template <typename FIAT>
__aicore__ inline void FiaBlockVecNonQuantMla<FIAT>::InitVec1GlobalTensor(
    GlobalTensor<MM1_OUT_T> mm1ResGm, GlobalTensor<KV_T> vec1ResGm, GlobalTensor<bool> attenMaskBoolGm,
    GlobalTensor<uint64_t> actualSeqLengthsGmQ, GlobalTensor<uint64_t> actualSeqLengthsGm, GlobalTensor<T> lseMaxFdGm,
    GlobalTensor<T> lseSumFdGm)
{
    this->mm1ResGm = mm1ResGm;
    this->vec1ResGm = vec1ResGm;
    this->attenMaskBoolGm = attenMaskBoolGm;
    this->actualSeqLengthsGmQ = actualSeqLengthsGmQ;
    this->actualSeqLengthsGm = actualSeqLengthsGm;
    this->lseMaxFdGm = lseMaxFdGm;
    this->lseSumFdGm = lseSumFdGm;
}


template <typename FIAT>
__aicore__ inline void FiaBlockVecNonQuantMla<FIAT>::InitVec2GlobalTensor(GlobalTensor<T> accumOutGm,
                                                                                  GlobalTensor<UPDATE_T> vec2ResGm,
                                                                                  GlobalTensor<MM2_OUT_T> mm2ResGm,
                                                                                  GlobalTensor<OUT_T> attentionOutGm)
{
    this->accumOutGm = accumOutGm;
    this->vec2ResGm = vec2ResGm;
    this->mm2ResGm = mm2ResGm;
    this->attentionOutGm = attentionOutGm;
}

template <typename FIAT> __aicore__ inline void FiaBlockVecNonQuantMla<FIAT>::AllocEventID()
{
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG);
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_PONG_FLAG);
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF2_FLAG);
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF2_PONG_FLAG);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
}

template <typename FIAT> __aicore__ inline void FiaBlockVecNonQuantMla<FIAT>::FreeEventID()
{
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_PONG_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF2_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF2_PONG_FLAG);
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
}

template <typename FIAT> __aicore__ inline void FiaBlockVecNonQuantMla<FIAT>::InitSoftmaxDefaultBuffer()
{
    Duplicate(softmaxMaxDefaultUb, SOFTMAX_MIN_NUM, SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T));
    Duplicate(softmaxSumDefaultUb, AttentionCommon::ConstInfo::FLOAT_ZERO, SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T));
}

template <typename FIAT>
__aicore__ inline void
FiaBlockVecNonQuantMla<FIAT>::AttenMaskCopyDirectly(const AttentionCommon::RunInfo &info, LocalTensor<bool> &attenMaskUb,
                                                            uint32_t dealRowCount, uint32_t actualColumnCount)
{
    LocalTensor<bool> maskUb = inputBuff2.Get<bool>();
    maskUb = maskUb[pingpongFlag * INPUT2_BUFFER_OFFSET / sizeof(bool)];
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF2_FLAG + pingpongFlag);
    attenMaskSizeAlign = Align(actualColumnCount, 32U);
#if (__CCE_AICORE__ > 200)
    if (actualColumnCount % 32 == 0) {
        DataCopy(maskUb, attenMaskBoolGm[info.attenMaskOffset], attenMaskSizeAlign);
    } else {
        uint32_t typeElementSize = fa_base_vector::BYTE_BLOCK / sizeof(bool);
        DataCopyExtParams intriParams;
        intriParams.blockLen = actualColumnCount * sizeof(bool);
        intriParams.blockCount = 1;
        intriParams.dstStride = (attenMaskSizeAlign - actualColumnCount) / typeElementSize;
        intriParams.srcStride = 0;
        DataCopyPadExtParams<bool> padParams;
        padParams.isPad = true;
        padParams.leftPadding = 0;
        padParams.rightPadding = (attenMaskSizeAlign - actualColumnCount) % typeElementSize;
        padParams.paddingValue = 0;
        DataCopyPad(maskUb, attenMaskBoolGm[info.attenMaskOffset], intriParams, padParams);
    }
#else
    DataCopy(maskUb, attenMaskBoolGm[info.attenMaskOffset], attenMaskSizeAlign);
#endif

    SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF2_FLAG);
    WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF2_FLAG);
    for (int i = 1; i < dealRowCount; i++) {
        DataCopy(maskUb[i * attenMaskSizeAlign], maskUb, attenMaskSizeAlign);
    }
}

template <typename FIAT>
__aicore__ inline void FiaBlockVecNonQuantMla<FIAT>::ComputeLogSumExpAndCopyToGm(const AttentionCommon::RunInfo &info,
                                                                                         const MSplitInfo &mSplitInfo,
                                                                                         LocalTensor<T> &softmaxSumUb,
                                                                                         LocalTensor<T> &softmaxMaxUb)
{
    if (mSplitInfo.vecDealM == 0) {
        return;
    }
    // workspace同步修改
    //  src-Shape  { gsizeV, S1, fa_base_vector::FP32_BLOCK_ELEMENT_NUM }
    //  dst-Shape  { B  N2, splitKV s1, G, fa_base_vector::FP32_BLOCK_ELEMENT_NUM}
    // 这里的offset计算，后续FD切G改切M时，同步改掉
    uint64_t baseOffset = mSplitInfo.nBufferStartM / 2;
    size_t size = mSplitInfo.vecDealM * fa_base_vector::FP32_BLOCK_ELEMENT_NUM;
    uint64_t accumTmpOutNum = CalcAccumOffset(info.bIdx, info.gS1Idx);
    uint64_t offset = (accumTmpOutNum * kvHeadNum * constInfo.mBaseSize +              // taskoffset
                       info.tndCoreStartKVSplitPos * kvHeadNum * constInfo.mBaseSize + // 份数offset
                       mSplitInfo.nBufferStartM + mSplitInfo.vecStartM) *
                      fa_base_vector::FP32_BLOCK_ELEMENT_NUM; // m轴offset

    LocalTensor<T> tmp = outputBuff2.Get<T>();
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
    Brcb(tmp, softmaxSumUb[baseOffset], (mSplitInfo.vecDealM + 7) / 8, {1, 8});
    SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
    DataCopy(lseSumFdGm[offset], tmp, size);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);

    tmp = outputBuff2.Get<T>();
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
    Brcb(tmp, softmaxMaxUb[baseOffset], (mSplitInfo.vecDealM + 7) / 8, {1, 8});
    SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
    DataCopy(lseMaxFdGm[offset], tmp, size);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
}

template <typename FIAT>
__aicore__ inline void
FiaBlockVecNonQuantMla<FIAT>::AttenMaskCopyForBNSD(const AttentionCommon::RunInfo &info, const MSplitInfo &mSplitInfo,
                                                           LocalTensor<bool> &attenMaskUb, uint32_t startRow,
                                                           uint32_t dealRowCount)
{
    uint32_t gS1StartIdx = info.gS1Idx + mSplitInfo.nBufferStartM + mSplitInfo.vecStartM + startRow;
    uint32_t s1StartIdx = gS1StartIdx % info.actS1Size;

#define ATTENMASK_STRIDE 2048U
    uint32_t offset;
    uint32_t s1StartCopyIdx = 0; // 第一次拷贝从s1=0开始，拷贝整个attentionmask，尚未支持s1泛化，只支持s1<=16场景
    int32_t delta = s1StartCopyIdx - info.s2Idx * constInfo.s2BaseSize + info.actS2Size - info.actS1Size;
    if (delta < 0) {
        offset = (-delta) < (int32_t)info.actS1Size ? (-delta) : info.actS1Size; // min (-delta, s1Size)
    } else {
        offset = (delta < (int32_t)constInfo.s2BaseSize ? delta : constInfo.s2BaseSize) *
                 ATTENMASK_STRIDE; // min(delta, s2inner)
    }

    attenMaskSizeAlign = Align(info.actualSingleProcessSInnerSize, 32U);

    DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = info.actS1Size;
    dataCopyParams.blockLen = attenMaskSizeAlign * sizeof(bool) / 32;
    dataCopyParams.srcStride = (ATTENMASK_STRIDE - attenMaskSizeAlign) * sizeof(bool) / 32;
    dataCopyParams.dstStride = 0;

    attenMaskUb = inputBuff2.Get<bool>();
    attenMaskUb = attenMaskUb[pingpongFlag * INPUT2_BUFFER_OFFSET / sizeof(bool)];
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF2_FLAG + pingpongFlag);
    DataCopy(attenMaskUb, attenMaskBoolGm[offset], dataCopyParams);

    LocalTensor<bool> attenMaskUbDst = attenMaskTmpBuff.Get<bool>();
    SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF2_FLAG);
    WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF2_FLAG);

    uint32_t headS1Count = 0;
    if (s1StartIdx + dealRowCount > info.actS1Size) {
        headS1Count = info.actS1Size - s1StartIdx;

    } else {
        headS1Count = dealRowCount;
    }

    // head
    DataCopy(attenMaskUbDst, attenMaskUb[s1StartIdx * attenMaskSizeAlign], headS1Count * attenMaskSizeAlign);
    // mid
    uint32_t reminRowCount = dealRowCount - headS1Count;
    uint32_t midGCount = reminRowCount / info.actS1Size;
    uint32_t tailS1Size = reminRowCount % info.actS1Size;
    for (uint32_t i = 0; i < midGCount; i++) {
        DataCopy(attenMaskUbDst[(headS1Count + i * info.actS1Size) * attenMaskSizeAlign], attenMaskUb,
                 info.actS1Size * attenMaskSizeAlign);
    }
    // tail
    if (tailS1Size > 0) {
        DataCopy(attenMaskUbDst[(headS1Count + midGCount * info.actS1Size) * attenMaskSizeAlign], attenMaskUb,
                 tailS1Size * attenMaskSizeAlign);
    }
    attenMaskUb = attenMaskUbDst;
}

template <typename FIAT>
__aicore__ inline void
FiaBlockVecNonQuantMla<FIAT>::AttenMaskCopyForSplitG(const AttentionCommon::RunInfo &info, const MSplitInfo &mSplitInfo,
                                                             LocalTensor<bool> &attenMaskUb, uint32_t startRow,
                                                             uint32_t dealRowCount, bool &selectNext, bool &selectPre)
{
    uint32_t gS1StartIdx = info.gS1Idx + mSplitInfo.nBufferStartM + mSplitInfo.vecStartM + startRow;
    uint32_t gStartIdx, s1StartIdx;
    GetGS1Idx<LAYOUT_T>(gS1StartIdx, gStartIdx, s1StartIdx, constInfo);
    uint32_t gS1EndIdx = gS1StartIdx + dealRowCount - 1;
    uint32_t s1EndIdx = gS1EndIdx / constInfo.gSize;
    uint32_t s1Count = s1EndIdx - s1StartIdx + 1;

#define ATTENMASK_STRIDE 2048U

    uint32_t offset;
    uint32_t offsetPre = 0;
    int32_t delta = s1StartIdx - info.s2Idx * constInfo.s2BaseSize + info.nextTokensPerBatch;
    if (delta < 0) {
        offset = (-delta) < (int32_t)s1Count ? (-delta) : s1Count;
    } else {
        offset = (delta < (int32_t)constInfo.s2BaseSize ? delta : constInfo.s2BaseSize) * ATTENMASK_STRIDE;
    }

    if (constInfo.slidingFlag) {
        int32_t sOuterOffset = offset / ATTENMASK_STRIDE;
        int32_t sInnerOffset = offset % ATTENMASK_STRIDE;
        selectNext = (sOuterOffset < sInnerOffset + constInfo.s2BaseSize);

        delta = s1StartIdx - info.s2Idx * constInfo.s2BaseSize - info.preTokensPerBatch -1;
        if (delta < 0) {
            offsetPre = (-delta) < (int32_t)s1Count ? (-delta) : s1Count;
        } else {
            offsetPre = (delta < (int32_t)constInfo.s2BaseSize ? delta : constInfo.s2BaseSize) * ATTENMASK_STRIDE;
        }
        sOuterOffset = offsetPre / ATTENMASK_STRIDE;
        sInnerOffset = offsetPre % ATTENMASK_STRIDE;
        selectPre = (sOuterOffset > sInnerOffset - (int32_t)s1Count);
        if (!selectPre && !selectNext) {
            return;
        }
    }

    attenMaskSizeAlign = Align(info.actualSingleProcessSInnerSize, 32U);

    DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = s1Count;
    dataCopyParams.blockLen = attenMaskSizeAlign * sizeof(bool) / 32;
    dataCopyParams.srcStride = (ATTENMASK_STRIDE - attenMaskSizeAlign) * sizeof(bool) / 32;
    dataCopyParams.dstStride = 0;

    attenMaskUb = inputBuff2.Get<bool>();
    attenMaskUb = attenMaskUb[pingpongFlag * INPUT2_BUFFER_OFFSET / sizeof(bool)];
    if (selectPre && selectNext) { // preToken与nextToken对应的mask串行搬运了，待优化
        WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF2_FLAG + pingpongFlag);
        DataCopy(attenMaskUb, attenMaskBoolGm[offsetPre], dataCopyParams);
        SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF2_FLAG);
        WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF2_FLAG);
        LocalTensor<bool> attenMaskUbTmp = attenMaskTmpBuff.Get<bool>();
        DataCopy(attenMaskUbTmp, attenMaskUb, s1Count * attenMaskSizeAlign / sizeof(bool));
        SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF2_FLAG + pingpongFlag);
    }
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF2_FLAG + pingpongFlag);
    if (selectNext) {
        DataCopy(attenMaskUb, attenMaskBoolGm[offset], dataCopyParams);
    }
    if (selectPre && !selectNext) {
        DataCopy(attenMaskUb, attenMaskBoolGm[offsetPre], dataCopyParams);
    }

    LocalTensor<int16_t> mask16 = attenMaskUb.template ReinterpretCast<int16_t>();
    LocalTensor<int16_t> attenMaskUbDst = attenMaskTmpBuff.Get<int16_t>();

    SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF2_FLAG);
    WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF2_FLAG);
    if (selectPre && selectNext) {
        AscendC::Xor(mask16, mask16, attenMaskUbDst, s1Count * attenMaskSizeAlign / sizeof(int16_t));
        AscendC::PipeBarrier<PIPE_V>();
    }

    uint32_t headGCount = s1Count > 1 ? (constInfo.gSize - gStartIdx) : dealRowCount;
    uint32_t dstMaskOffset = 0;
    uint32_t srcMaskBaseOffset = 0;
    // head
    SetMaskCount();
    SetVectorMask<int16_t, MaskMode::COUNTER>(attenMaskSizeAlign / 2);
    Copy<int16_t, false>(attenMaskUbDst[dstMaskOffset], mask16[srcMaskBaseOffset], AscendC::MASK_PLACEHOLDER,
                         headGCount, {1, 1, static_cast<uint16_t>(attenMaskSizeAlign / 32), 0});
    dstMaskOffset += headGCount * attenMaskSizeAlign / 2;
    srcMaskBaseOffset += attenMaskSizeAlign / 2;
    // mid
    uint32_t reminRowCount = dealRowCount - headGCount;
    uint32_t midS1Count = reminRowCount / constInfo.gSize;
    uint32_t tailGSize = reminRowCount % constInfo.gSize;
    for (uint32_t midIdx = 0; midIdx < midS1Count; midIdx++) {
        Copy<int16_t, false>(attenMaskUbDst[dstMaskOffset], mask16[srcMaskBaseOffset], AscendC::MASK_PLACEHOLDER,
                             constInfo.gSize, {1, 1, static_cast<uint16_t>(attenMaskSizeAlign / 32), 0});
        dstMaskOffset += constInfo.gSize * attenMaskSizeAlign / 2;
        srcMaskBaseOffset += attenMaskSizeAlign / 2;
    }
    // tail
    if (tailGSize > 0) {
        Copy<int16_t, false>(attenMaskUbDst[dstMaskOffset], mask16[srcMaskBaseOffset], AscendC::MASK_PLACEHOLDER,
                             tailGSize, {1, 1, static_cast<uint16_t>(attenMaskSizeAlign / 32), 0});
    }
    SetMaskNorm();
    ResetMask();
    attenMaskUb = attenMaskUbDst.template ReinterpretCast<bool>();
}

template <typename FIAT>
__aicore__ inline bool FiaBlockVecNonQuantMla<FIAT>::IsSkipAttenMask(const AttentionCommon::RunInfo &info,
                                                                             const MSplitInfo &mSplitInfo,
                                                                             uint32_t startRow, uint32_t dealRowCount)
{
    if (constInfo.slidingFlag) {
        return false;
    }
    uint32_t actualSeqQ = constInfo.qSeqSize;
    if constexpr (LAYOUT_T == FIA_LAYOUT::TND) {
        actualSeqQ = info.actS1Size;
    }

    // s2<s1时，必然走mask
    if (info.actS2Size < actualSeqQ) {
        return false;
    }

    // 当前的s2位置不超过需要打标记的位置时，不需要mask
    uint32_t s2EndPos = info.s2Idx * constInfo.s2BaseSize + info.actualSingleProcessSInnerSize;
    if (s2EndPos <= (info.actS2Size - actualSeqQ + 1)) {
        return true;
    }

    // BSH/BSND/TND格式切G，最后一个s1不需要mask
    if constexpr (LAYOUT_T != FIA_LAYOUT::BNSD) {
        uint32_t gS1StartIdx = info.gS1Idx + mSplitInfo.nBufferStartM + mSplitInfo.vecStartM + startRow;
        uint32_t gS1EndIdx = gS1StartIdx + dealRowCount - 1;
        uint32_t s1StartIdx = gS1StartIdx / constInfo.gSize;
        uint32_t s1EndIdx = gS1EndIdx / constInfo.gSize;
        if (s1StartIdx == s1EndIdx && s1StartIdx == actualSeqQ - 1) {
            return true;
        }
    }
    return false;
}

template <typename FIAT>
__aicore__ inline void FiaBlockVecNonQuantMla<FIAT>::ElewiseCompute(
    const AttentionCommon::RunInfo &info, const MSplitInfo &mSplitInfo, LocalTensor<T> &mmResUb, TBuf<> &tmpBuf, uint32_t startRow,
    uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    Muls(mmResUb, mmResUb, static_cast<T>(tilingData->baseParams.scaleValue), dealRowCount * columnCount);

    if (constInfo.attenMaskFlag == 1) {
        AscendC::PipeBarrier<PIPE_V>();
        AttentionMaskCompute(info, mSplitInfo, mmResUb, tmpBuf, startRow, dealRowCount, columnCount, actualColumnCount);
    }
}


template <typename FIAT>
__aicore__ inline void FiaBlockVecNonQuantMla<FIAT>::AttentionMaskCompute(
    const AttentionCommon::RunInfo &info, const MSplitInfo &mSplitInfo, LocalTensor<T> &mmResUb, TBuf<> &tmpBuf, uint32_t startRow,
    uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    LocalTensor<bool> attenMaskUb;
    bool selectNext = true;
    bool selectPre = false;
    if ((!constInfo.slidingFlag) && (constInfo.qSeqSize == 1) && (LAYOUT_T != FIA_LAYOUT::TND)) {
        AttenMaskCopyDirectly(info, attenMaskUb, dealRowCount, actualColumnCount);
    } else {
        if (IsSkipAttenMask(info, mSplitInfo, startRow, dealRowCount)) {
            return;
        }
        if constexpr (LAYOUT_T == FIA_LAYOUT::BNSD) {
            AttenMaskCopyForBNSD(info, mSplitInfo, attenMaskUb, startRow, dealRowCount);
        } else { // BSH/BSND/TND
            AttenMaskCopyForSplitG(info, mSplitInfo, attenMaskUb, startRow, dealRowCount, selectNext, selectPre);
        }
    }
    if (!selectPre && !selectNext) {
        return;
    }
    AscendC::PipeBarrier<PIPE_V>();
    LocalTensor<uint8_t> ubWorkSpace = tmpBuf.Get<uint8_t>();
    SelectWithBytesMaskShapeInfo selectWithBytesMaskShapeInfo;
    selectWithBytesMaskShapeInfo.firstAxis = dealRowCount;
    selectWithBytesMaskShapeInfo.srcLastAxis = columnCount;
    selectWithBytesMaskShapeInfo.maskLastAxis = attenMaskSizeAlign;
    attenMaskUb.SetSize(dealRowCount * attenMaskSizeAlign); // Select接口要求mask size与参数匹配
    mmResUb.SetSize(dealRowCount * columnCount);            // Select接口要求src size与参数匹配
    if (selectPre) {
        SelectWithBytesMask(mmResUb, BOOL_ATTEN_MASK_SCALAR_VALUE, mmResUb, attenMaskUb, ubWorkSpace,
                            selectWithBytesMaskShapeInfo);
    } else {
        SelectWithBytesMask(mmResUb, mmResUb, BOOL_ATTEN_MASK_SCALAR_VALUE, attenMaskUb, ubWorkSpace,
                            selectWithBytesMaskShapeInfo);
    }
    mmResUb.SetSize(AttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_32K / sizeof(T)); // mmResUb Size复原,mask不用复原,与原来一致
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF2_FLAG + pingpongFlag);
}

template <typename FIAT>
__aicore__ inline void FiaBlockVecNonQuantMla<FIAT>::SoftmaxFlashV2Compute(
    const AttentionCommon::RunInfo &info, const MSplitInfo &mSplitInfo, LocalTensor<T> &mmResUb, LocalTensor<uint8_t> &softmaxTmpUb,
    uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    SoftMaxShapeInfo srcShape{dealRowCount, columnCount, dealRowCount, actualColumnCount};
    SoftMaxTiling newTiling =
        SoftMaxFlashV2TilingFunc(srcShape, sizeof(T), sizeof(T), softmaxTmpUb.GetSize(), true, false);

    LocalTensor<T> inSumTensor;
    LocalTensor<T> inMaxTensor;
    uint32_t baseOffset = mSplitInfo.nBufferStartM / 2 + startRow;
    uint32_t outIdx = info.loop % (constInfo.preLoadNum);
    uint32_t softmaxOutOffset = outIdx * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T) + baseOffset;
    if (info.isFirstSInnerLoop) {
        inMaxTensor = softmaxMaxDefaultUb;
        inSumTensor = softmaxSumDefaultUb;
    } else {
        uint32_t inIdx = (info.loop - 1) % (constInfo.preLoadNum);
        inMaxTensor = softmaxMaxUb[inIdx * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T) + baseOffset];
        inSumTensor = softmaxSumUb[inIdx * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T) + baseOffset];
    }
    SoftmaxFlashV2<T, true, true, false, false, FIA_SOFTMAX_FLASHV2_CFG_WITHOUT_BRC>(
        mmResUb, softmaxSumUb[softmaxOutOffset], softmaxMaxUb[softmaxOutOffset], mmResUb,
        softmaxExpUb[softmaxOutOffset], inSumTensor, inMaxTensor, softmaxTmpUb, newTiling, srcShape);
}

template <typename FIAT>
__aicore__ inline void FiaBlockVecNonQuantMla<FIAT>::AmlaVecCompute(
    const AttentionCommon::RunInfo &info, const MSplitInfo &mSplitInfo, LocalTensor<T> &mmResUb, LocalTensor<uint8_t> &softmaxTmpUb,
    uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    uint32_t baseOffset = mSplitInfo.nBufferStartM / 2 + startRow;
    uint32_t calCount = dealRowCount;
    uint32_t outIdx = info.loop % (constInfo.preLoadNum);
    uint32_t softmaxOutOffset = outIdx * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T) + baseOffset;
    // compute n(i)
    LocalTensor<T> nTmp = softmaxTmpUb.template ReinterpretCast<T>();
    LocalTensor<T> nUpdateTmp = nTmp[SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T)];
    Muls(nTmp, softmaxMaxUb[softmaxOutOffset], ((T)(-1.0)) * RECIP_OF_LN2, calCount);

    AscendC::PipeBarrier<PIPE_V>();
    Cast(nTmp, nTmp, RoundMode::CAST_ROUND, calCount);
    AscendC::PipeBarrier<PIPE_V>();

    uint32_t prOutIdx = (info.loop - 1) % (constInfo.preLoadNum);
    uint32_t PreSoftmaxOutOffset = prOutIdx * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T) + baseOffset;
    // n(i) - n(i-1)
    if (info.isFirstSInnerLoop) {
        Duplicate(nUpdateTmp, AttentionCommon::ConstInfo::FLOAT_ZERO, calCount); // n1=n0
    } else {
        Sub(nUpdateTmp, nTmp, nValueUb[PreSoftmaxOutOffset], calCount);
    }
    AscendC::PipeBarrier<PIPE_V>();
    // update n(i), DataCopy not support when calCount is not align 32B, so use Adds
    Adds(nValueUb[softmaxOutOffset], nTmp, AttentionCommon::ConstInfo::FLOAT_ZERO, calCount);
    AscendC::PipeBarrier<PIPE_V>();

    // update softmax res
    LocalTensor<T> nUpdateTmp2 = nTmp[2 * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T)];
    LocalTensor<KV_T> nTmp_KvT = nTmp[3 * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T)].template ReinterpretCast<KV_T>();
    LocalTensor<T> tmpCofUb = nTmp[4 * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T)];
    LocalTensor<T> epsUb = nTmp[5 * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T)];
    Muls(nUpdateTmp2, softmaxMaxUb[softmaxOutOffset], RECIP_OF_LN2, calCount);
    AscendC::PipeBarrier<PIPE_V>();
    Add(nTmp, nUpdateTmp2, nTmp, calCount);
    AscendC::PipeBarrier<PIPE_V>();
    Muls(nTmp, nTmp, LN2, calCount);
    AscendC::PipeBarrier<PIPE_V>();
    Exp(nTmp, nTmp, calCount);
    AscendC::PipeBarrier<PIPE_V>();
    Cast(nTmp_KvT, nTmp, RoundMode::CAST_ROUND, calCount); // fp32->fp16/bf16
    AscendC::PipeBarrier<PIPE_V>();
    Cast(nUpdateTmp2, nTmp_KvT, RoundMode::CAST_NONE, calCount); // fp16/bf16->fp32
    AscendC::PipeBarrier<PIPE_V>();
    if (info.s2Idx + 1 == info.curSInnerLoopTimes) {
        Mul(aMlaSumUb[softmaxOutOffset], softmaxSumUb[softmaxOutOffset], nUpdateTmp2, calCount);
    }

    LocalTensor<T> nTmp3 = nTmp[6 * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T)];
    Brcb(nTmp3, nUpdateTmp2, (dealRowCount + 7) / 8, {1, 8});
    AscendC::PipeBarrier<PIPE_V>();
    fa_base_vector::RowMuls(mmResUb, mmResUb, nTmp3, dealRowCount, columnCount, actualColumnCount);

    Div(tmpCofUb, nTmp, nUpdateTmp2, calCount); // cof(i)=tmpS32/tmpS16
    if (info.isFirstSInnerLoop) {
        Duplicate(cofValueUb[softmaxOutOffset], (T)1.0, calCount); // cof_0=1
        AscendC::PipeBarrier<PIPE_V>();
        Div(epsUb, cofValueUb[softmaxOutOffset], tmpCofUb, calCount); // 1 / cof(i)
    } else {
        AscendC::PipeBarrier<PIPE_V>();
        Div(epsUb, cofValueUb[PreSoftmaxOutOffset], tmpCofUb, calCount); // cof(i - 1) / cof(i)
    }
    AscendC::PipeBarrier<PIPE_V>();

    Adds(cofValueUb[softmaxOutOffset], tmpCofUb, AttentionCommon::ConstInfo::FLOAT_ZERO, calCount); // store cof(i)
    Adds(epsUb, epsUb, (T)(-1.0), calCount); // cof(i - 1) / cof(i) - 1
    AscendC::PipeBarrier<PIPE_V>();
    Muls(epsUb, epsUb, (T)1.5, calCount); // (cof(i - 1) - cof(i)) / cof(i) * 1.5

    Maxs(nUpdateTmp, nUpdateTmp, (T)(-30.0), calCount); // N = max(n(i) - n(i-1), -30)
    AscendC::PipeBarrier<PIPE_V>();
    Adds(epsUb, epsUb, (T)(0.000001), calCount);
    AscendC::PipeBarrier<PIPE_V>();
    Add(nUpdateTmp, nUpdateTmp, epsUb, calCount);
    AscendC::PipeBarrier<PIPE_V>();
    Muls(nUpdateTmp, nUpdateTmp, FLOAT_E_SCALAR, calCount); // N = N * pow(2, 23)
    AscendC::PipeBarrier<PIPE_V>();

    // nUpdate int32 out
    LocalTensor<int32_t> tmQue = outputBuff2.Get<int32_t>();
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
    LocalTensor<int32_t> nInt32Out = tmQue[startRow]; // 缓存nUpdate

    Cast(nInt32Out, nUpdateTmp, RoundMode::CAST_ROUND, dealRowCount);
    AscendC::PipeBarrier<PIPE_V>();

    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
}

template <typename FIAT>
template <typename RT>
__aicore__ inline void
FiaBlockVecNonQuantMla<FIAT>::DealInvalidRowsBelow(const AttentionCommon::RunInfo &info, LocalTensor<RT> &attenOutUb,
                                                           uint32_t wsMStart, uint32_t dealRowCount, uint32_t columnCount)
{
    // BSH and TND
    uint32_t s1Tok = info.actS2Size + info.preTokensPerBatch;
    uint32_t s1 = (info.gS1Idx + wsMStart) / constInfo.gSize;
    uint32_t gIdx = (info.gS1Idx + wsMStart) % constInfo.gSize;
    for (uint32_t i = 0; i < dealRowCount;) {
        if (s1 >= s1Tok) {
            uint32_t gNum = constInfo.gSize - gIdx;
            if (i + gNum > dealRowCount) {
                gNum = dealRowCount - i;
            }
            Duplicate(attenOutUb[i * columnCount], static_cast<RT>(AttentionCommon::ConstInfo::FLOAT_ZERO), columnCount * gNum);
            i += gNum;
            s1++;
            gIdx = 0;
            continue;
        }
        break;
    }
}

template <typename FIAT>
template <typename RT>
__aicore__ inline void
FiaBlockVecNonQuantMla<FIAT>::DealInvalidRows(const AttentionCommon::RunInfo &info, LocalTensor<RT> &attenOutUb,
                                                      uint32_t wsMStart, uint32_t dealRowCount, uint32_t columnCount,
                                                      uint32_t actualColumnCount)
{
    if (!constInfo.attenMaskFlag) {
        return;
    }
    AscendC::PipeBarrier<PIPE_V>();
    if (constInfo.slidingFlag && (info.preTokensPerBatch < (info.actS1Size - info.actS2Size))) { // 下方存在行无效
        DealInvalidRowsBelow(info, attenOutUb, wsMStart, dealRowCount, columnCount);
    }
    if (info.nextTokensPerBatch >= 0) {
        return;
    }
    uint32_t s1Tok = -info.nextTokensPerBatch;
    if constexpr (LAYOUT_T == FIA_LAYOUT::BNSD) {
        uint32_t s1 = (info.gS1Idx + wsMStart) % info.actS1Size;
        for (uint32_t i = 0; i < dealRowCount;) {
            if (s1 < s1Tok) {
                uint32_t s1Num = s1Tok - s1;
                if (i + s1Num > dealRowCount) {
                    s1Num = dealRowCount - i;
                }
                Duplicate(attenOutUb[i * columnCount], static_cast<RT>(AttentionCommon::ConstInfo::FLOAT_ZERO), columnCount * s1Num);
            }
            i += info.actS1Size - s1;
            s1 = 0;
        }
        return;
    }
    // BSH and TND
    uint32_t s1 = (info.gS1Idx + wsMStart) / constInfo.gSize;
    uint32_t gIdx = (info.gS1Idx + wsMStart) % constInfo.gSize;
    for (uint32_t i = 0; i < dealRowCount;) {
        if (s1 < s1Tok) {
            uint32_t gNum = constInfo.gSize - gIdx;
            if (i + gNum > dealRowCount) {
                gNum = dealRowCount - i;
            }
            Duplicate(attenOutUb[i * columnCount], static_cast<RT>(AttentionCommon::ConstInfo::FLOAT_ZERO), columnCount * gNum);
            i += gNum;
            s1++;
            gIdx = 0;
            continue;
        }
        break;
    }
}


template <typename FIAT>
__aicore__ inline void
FiaBlockVecNonQuantMla<FIAT>::DealBmm1ResBaseBlock(const AttentionCommon::RunInfo &info, const MSplitInfo &mSplitInfo,
                                                           uint32_t startRow, uint32_t dealRowCount,
                                                           uint32_t columnCount, uint32_t actualColumnCount)
{
    uint32_t computeSize = dealRowCount * columnCount;
    uint64_t inOutGmOffset = (info.loop % constInfo.preLoadNum) * constInfo.mmResUbSize +
                             (mSplitInfo.nBufferStartM + mSplitInfo.vecStartM + startRow) * columnCount;
    LocalTensor<MM1_OUT_T> mmResUb = inputBuff1.Get<MM1_OUT_T>();
    mmResUb = mmResUb[pingpongFlag * INPUT1_BUFFER_OFFSET / sizeof(MM1_OUT_T)];
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG + pingpongFlag);

    DataCopy(mmResUb, mm1ResGm[inOutGmOffset], computeSize);
    SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF1_FLAG);

    ElewiseCompute(info, mSplitInfo, mmResUb, tmpBuff1, startRow, dealRowCount, columnCount, actualColumnCount);

    AscendC::PipeBarrier<PIPE_V>();
    LocalTensor<T> tmpAFloorUb = tmpBuff1.Get<T>();
    LocalTensor<uint8_t> softmaxTmpUb = tmpAFloorUb.template ReinterpretCast<uint8_t>();
    SoftmaxFlashV2Compute(info, mSplitInfo, mmResUb, softmaxTmpUb, startRow, dealRowCount, columnCount,
                          actualColumnCount);

    AscendC::PipeBarrier<PIPE_V>();
    AmlaVecCompute(info, mSplitInfo, mmResUb, softmaxTmpUb, startRow, dealRowCount, columnCount, actualColumnCount);

    AscendC::PipeBarrier<PIPE_V>();
    LocalTensor<KV_T> tmpMMResCastTensor = outputBuff1.Get<KV_T>();
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);

    Cast(tmpMMResCastTensor, mmResUb, AscendC::RoundMode::CAST_ROUND, computeSize);
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG + pingpongFlag);

    SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    DataCopy(vec1ResGm[inOutGmOffset], tmpMMResCastTensor, computeSize);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
}

template <typename FIAT>
__aicore__ inline void FiaBlockVecNonQuantMla<FIAT>::ProcessAmlaNupdate(const AttentionCommon::RunInfo &info,
                                                                                const MSplitInfo &mSplitInfo)
{
    if (mSplitInfo.vecDealM == 0) {
        return;
    }
    if (info.isFirstSInnerLoop) {
        return;
    }

    LocalTensor<int32_t> nUpdateTensor = outputBuff2.Get<int32_t>(); // shape:1/2*s1*g
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
    SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);

    constexpr uint32_t dGroupSize = 128U;
    constexpr uint32_t mSplitSize = 64U; // tmpQue size 32KB，一次只能处理64个N，最大保存的数据大小：64*128*sizeof(int32)
    constexpr uint32_t ONE_BLOCK_SIZE = 32U; // 32B
    uint32_t subMSize = Align(mSplitInfo.vecDealM, 16U);
    uint16_t elementPerBlock = ONE_BLOCK_SIZE / sizeof(int32_t); // 单个datablock的元素数，int32_t类型的为32/4=8
    
    uint32_t loopCount = (subMSize + mSplitSize - 1) / mSplitSize;
    uint32_t tailSplitSize = subMSize - (loopCount - 1) * mSplitSize; // 尾块
    for (uint32_t loop = 0, processMSize = mSplitSize; loop < loopCount; loop++) {
        if (loop == (loopCount - 1)) {
            processMSize = tailSplitSize;
        }

        LocalTensor<int32_t> tmpQue = outputBuff1.Get<int32_t>();
        WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
        // (m,1)单次brcb扩充成(m,8), 重复16次, 扩充为(m,128)
        for (uint32_t i = 0; i < dGroupSize / elementPerBlock; i++) {
            Brcb(tmpQue[i * elementPerBlock],
                 nUpdateTensor[loop * mSplitSize], 
                 static_cast<uint8_t>((processMSize + elementPerBlock - 1) / elementPerBlock),
                 {static_cast<uint16_t>(dGroupSize / elementPerBlock), // 单次迭代内，目的操作数不同datablock间地址步长,单位为datablock
                  static_cast<uint16_t>(dGroupSize)}); // 相邻迭代间，目的操作数相同datablock地址步长
        }

        SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
        WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);

        uint64_t baseoffset = (info.bn2IdxInCurCore % constInfo.preLoadNum) * constInfo.bmm2ResUbSize +
                              (mSplitInfo.nBufferStartM + mSplitInfo.vecStartM + loop * mSplitSize) * headDim;

        SetAtomicAdd<int32_t>();
        DataCopyParams dataCopyParams;
        dataCopyParams.blockCount = static_cast<uint16_t>(processMSize);
        dataCopyParams.blockLen = dGroupSize * sizeof(int32_t) / ONE_BLOCK_SIZE; // 每个block是128个元素，单位为32B
        dataCopyParams.srcStride = 0; // 前面一个数据块的尾与后面数据块的头的间隔
        dataCopyParams.dstStride = static_cast<uint16_t>((headDim - dGroupSize) * sizeof(int32_t) / ONE_BLOCK_SIZE); // 单位为32B
        for (uint32_t i = 0; i < headDim / dGroupSize; i++) { // 4=512/128
            DataCopy(mm2ResInt32Gm[baseoffset + i * dGroupSize] ,tmpQue, dataCopyParams);
        }

        SetAtomicNone();
        SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    }
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
}

template <typename FIAT>
__aicore__ inline void FiaBlockVecNonQuantMla<FIAT>::ProcessVec1SingleBuf(const AttentionCommon::RunInfo &info,
                                                                                  const MSplitInfo &mSplitInfo)
{
    if (mSplitInfo.vecDealM == 0) {
        return;
    }
    uint32_t mSplitSize = BASE_BLOCK_MAX_ELEMENT_NUM / info.actualSingleProcessSInnerSizeAlign;
    // 1. 向下8对齐是因为UB操作至少32B
    // 2. info.actualSingleProcessSInnerSizeAlign最大512, mSplitSize可以确保最小为16
    mSplitSize = mSplitSize / 8 * 8;

    if (mSplitSize > mSplitInfo.vecDealM) {
        mSplitSize = mSplitInfo.vecDealM;
    }
    uint32_t loopCount = (mSplitInfo.vecDealM + mSplitSize - 1) / mSplitSize;
    uint32_t tailSplitSize = mSplitInfo.vecDealM - (loopCount - 1) * mSplitSize;
    for (uint32_t i = 0, dealSize = mSplitSize; i < loopCount; i++) {
        if (i == (loopCount - 1)) {
            dealSize = tailSplitSize;
        }
        DealBmm1ResBaseBlock(info, mSplitInfo, i * mSplitSize, dealSize, info.actualSingleProcessSInnerSizeAlign,
                             info.actualSingleProcessSInnerSize);
        pingpongFlag ^= 1; // pingpong 0 1切换
    }
}

template <typename FIAT> __aicore__ inline void FiaBlockVecNonQuantMla<FIAT>::ProcessVec1L(const AttentionCommon::RunInfo &info)
{
    uint32_t nBufferLoopTimes = (info.actMBaseSize + constInfo.nBufferMBaseSize - 1) / constInfo.nBufferMBaseSize;
    uint32_t nBufferTail = info.actMBaseSize - (nBufferLoopTimes - 1) * constInfo.nBufferMBaseSize;
    for (uint32_t i = 0; i < nBufferLoopTimes; i++) {
        MSplitInfo mSplitInfo;
        mSplitInfo.nBufferIdx = i;
        mSplitInfo.nBufferStartM = i * constInfo.nBufferMBaseSize;
        mSplitInfo.nBufferDealM = (i + 1 != nBufferLoopTimes) ? constInfo.nBufferMBaseSize : nBufferTail;

        mSplitInfo.vecDealM = (mSplitInfo.nBufferDealM <= 16) ? mSplitInfo.nBufferDealM :
                                                                (((mSplitInfo.nBufferDealM + 15) / 16 + 1) / 2 * 16);
        mSplitInfo.vecStartM = 0;
        if (GetBlockIdx() % 2 == 1) {
            mSplitInfo.vecStartM = mSplitInfo.vecDealM;
            mSplitInfo.vecDealM = mSplitInfo.nBufferDealM - mSplitInfo.vecDealM;
        }

        CrossCoreWaitFlag(constInfo.syncC1V1);
        // vec1 compute
        ProcessVec1SingleBuf(info, mSplitInfo);
        CrossCoreSetFlag<AttentionCommon::ConstInfo::FIA_SYNC_MODE2, PIPE_MTE3>(constInfo.syncV1C2);
        CrossCoreWaitFlag(constInfo.syncC2V1);
        // add nUpdate to mm2ResGm
        ProcessAmlaNupdate(info, mSplitInfo);
        CrossCoreSetFlag<AttentionCommon::ConstInfo::FIA_SYNC_MODE2, PIPE_MTE3>(constInfo.syncV1NupdateC2);
        // move lse for flash decode
        if (info.s2Idx == info.curSInnerLoopTimes - 1) {
            if (info.tndIsS2SplitCore) {
                if constexpr (FLASH_DECODE) {
                    uint32_t outIdx = info.loop % (constInfo.preLoadNum);
                    auto sumTensor = softmaxSumUb[outIdx * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T)];
                    auto maxTensor = softmaxMaxUb[outIdx * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T)];
                    ComputeLogSumExpAndCopyToGm(info, mSplitInfo, sumTensor, maxTensor);
                }
            }
        }
    }
}

template <typename FIAT>
__aicore__ inline uint64_t FiaBlockVecNonQuantMla<FIAT>::CalcAccumOffset(uint32_t bN2Idx, uint32_t gS1Idx)
{
#ifdef ASCENDC_CPU_DEBUG
    const uint32_t *bN2IdxOfFdHead = tilingData->fdParams.bN2IdxOfFdHead;
    const uint32_t *gS1IdxOfFdHead = tilingData->fdParams.gS1IdxOfFdHead;
    const uint32_t *s2SplitNumOfFdHead = tilingData->fdParams.s2SplitNumOfFdHead;
#else
    uint32_t bN2IdxOfFdHead[ARRAY_SIZE(tilingData->fdParams.bN2IdxOfFdHead)];
    uint32_t gS1IdxOfFdHead[ARRAY_SIZE(tilingData->fdParams.gS1IdxOfFdHead)];
    uint32_t s2SplitNumOfFdHead[ARRAY_SIZE(tilingData->fdParams.s2SplitNumOfFdHead)];
    copy_data_align64((uint8_t *)bN2IdxOfFdHead, (uint8_t *)(tilingData->fdParams.bN2IdxOfFdHead),
                      sizeof(bN2IdxOfFdHead));
    copy_data_align64((uint8_t *)gS1IdxOfFdHead, (uint8_t *)(tilingData->fdParams.gS1IdxOfFdHead),
                      sizeof(gS1IdxOfFdHead));
    copy_data_align64((uint8_t *)s2SplitNumOfFdHead, (uint8_t *)(tilingData->fdParams.s2SplitNumOfFdHead),
                      sizeof(s2SplitNumOfFdHead));
#endif
    uint64_t accumTmpOutNum = 0;
    int taskId = 0;
    while (bN2IdxOfFdHead[taskId] != bN2Idx || gS1IdxOfFdHead[taskId] * constInfo.mBaseSize != gS1Idx) {
        accumTmpOutNum += s2SplitNumOfFdHead[taskId]; // 计算前面的workspace数
        taskId++;
    }
    return accumTmpOutNum;
}


template <typename FIAT>
__aicore__ inline void FiaBlockVecNonQuantMla<FIAT>::ProcessVec2SingleBuf(const AttentionCommon::RunInfo &info,
                                                                                  const MSplitInfo &mSplitInfo)
{
    if (info.s2Idx + 1 != info.curSInnerLoopTimes) {
        return;
    }
    if (mSplitInfo.vecDealM == 0) {
        return;
    }

    ProcessVec2Inner(info, mSplitInfo, 0, mSplitInfo.vecDealM);
}

template <typename FIAT> __aicore__ inline void FiaBlockVecNonQuantMla<FIAT>::ProcessVec2L(const AttentionCommon::RunInfo &info)
{
    uint32_t nBufferLoopTimes = (info.actMBaseSize + constInfo.nBufferMBaseSize - 1) / constInfo.nBufferMBaseSize;
    uint32_t nBufferTail = info.actMBaseSize - (nBufferLoopTimes - 1) * constInfo.nBufferMBaseSize;
    for (uint32_t i = 0; i < nBufferLoopTimes; i++) {
        MSplitInfo mSplitInfo;
        mSplitInfo.nBufferIdx = i;
        mSplitInfo.nBufferStartM = i * constInfo.nBufferMBaseSize;
        mSplitInfo.nBufferDealM = (i + 1 != nBufferLoopTimes) ? constInfo.nBufferMBaseSize : nBufferTail;

        mSplitInfo.vecDealM = (mSplitInfo.nBufferDealM <= 16) ? mSplitInfo.nBufferDealM :
                                                                (((mSplitInfo.nBufferDealM + 15) / 16 + 1) / 2 * 16);
        mSplitInfo.vecStartM = 0;
        if (GetBlockIdx() % 2 == 1) {
            mSplitInfo.vecStartM = mSplitInfo.vecDealM;
            mSplitInfo.vecDealM = mSplitInfo.nBufferDealM - mSplitInfo.vecDealM;
        }
        CrossCoreWaitFlag(constInfo.syncC2V2);
        ProcessVec2SingleBuf(info, mSplitInfo);
    }
}

template <typename FIAT>
__aicore__ inline void FiaBlockVecNonQuantMla<FIAT>::ProcessVec2Inner(const AttentionCommon::RunInfo &info,
                                                                              const MSplitInfo &mSplitInfo,
                                                                              uint32_t mStartRow, uint32_t mDealSize)
{
    uint32_t mSplitSize = BASE_BLOCK_MAX_ELEMENT_NUM / headDimAlign;
    if (mSplitSize > mDealSize) {
        mSplitSize = mDealSize;
    }

    uint32_t loopCount = (mDealSize + mSplitSize - 1) / mSplitSize;
    uint32_t tailSplitSize = mDealSize - (loopCount - 1) * mSplitSize;
    for (uint32_t i = 0, dealSize = mSplitSize; i < loopCount; i++) {
        if (i == (loopCount - 1)) {
            dealSize = tailSplitSize;
        }
        DealBmm2ResBaseBlock(info, mSplitInfo, i * mSplitSize + mStartRow, dealSize, headDimAlign, headDim);
        pingpongFlag ^= 1; // pingpong 0 1切换
    }
}


template <typename FIAT>
__aicore__ inline void FiaBlockVecNonQuantMla<FIAT>::GetConfusionTransposeTiling(
    int64_t numR, int64_t numC, const uint32_t stackBufferSize, const uint32_t typeSize,
    ConfusionTransposeTiling &tiling)
{
    (void)stackBufferSize;
    uint32_t blockSize = ONE_BLK_SIZE / typeSize;
    uint32_t height = numC;
    uint32_t width = numR;
    uint32_t highBlock = height / BLOCK_CUBE;
    uint32_t stride = height * blockSize * typeSize / ONE_BLK_SIZE;
    uint32_t repeat = width / blockSize;

    tiling.param0 = blockSize;
    tiling.param1 = height;
    tiling.param2 = width;
    tiling.param3 = highBlock;
    tiling.param4 = stride;
    tiling.param5 = repeat;
}

template <typename FIAT>
__aicore__ inline void
FiaBlockVecNonQuantMla<FIAT>::Bmm2FDDataCopyOut(const AttentionCommon::RunInfo &info, LocalTensor<T> &bmm2ResUb,
                                                        uint32_t wsMStart, uint32_t dealRowCount, uint32_t columnCount,
                                                        uint32_t actualColumnCount)
{
    DealInvalidRows(info, bmm2ResUb, wsMStart, dealRowCount, columnCount, actualColumnCount);
    LocalTensor<T> tmp = outputBuff1.Get<T>();
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    DataCopy(tmp, bmm2ResUb, columnCount * dealRowCount);
    SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    uint64_t accumTmpOutNum = CalcAccumOffset(info.bIdx, info.gS1Idx);
    uint64_t offset = accumTmpOutNum * kvHeadNum * constInfo.mBaseSize * headDim +              // taskoffset
                      info.tndCoreStartKVSplitPos * kvHeadNum * constInfo.mBaseSize * headDim + // 份数offset
                      wsMStart * actualColumnCount;                                             // m轴offset
    GlobalTensor<T> dst = accumOutGm[offset];
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = dealRowCount;
    dataCopyParams.blockLen = actualColumnCount * sizeof(T);
    dataCopyParams.srcStride = (columnCount - actualColumnCount) / (fa_base_vector::BYTE_BLOCK / sizeof(T));
    dataCopyParams.dstStride = 0;
    DataCopyPad(dst, tmp, dataCopyParams);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
}

template <typename FIAT>
__aicore__ inline void
FiaBlockVecNonQuantMla<FIAT>::Bmm2DataCopyOutTrans(const AttentionCommon::RunInfo &info, LocalTensor<OUT_T> &attenOutUb,
                                                           uint32_t wsMStart, uint32_t dealRowCount,
                                                           uint32_t columnCount, uint32_t actualColumnCount)
{
    if (constInfo.outputLayout == FIA_LAYOUT::NBSD || constInfo.outputLayout == FIA_LAYOUT::NTD) {
        FusedTransposeInfo transInfo;
        transInfo.n2Idx = info.n2Idx;
        transInfo.bIdx = info.bIdx;
        auto gS1StartIdx = info.gS1Idx + wsMStart;
        auto gS1EndIdx = gS1StartIdx + dealRowCount - 1;
        GetGS1Idx<LAYOUT_T>(gS1StartIdx, transInfo.gStartIdx, transInfo.s1StartIdx, constInfo);
        GetGS1Idx<LAYOUT_T>(gS1EndIdx, transInfo.gEndIdx, transInfo.s1EndIdx, constInfo);
        if constexpr (LAYOUT_T == FIA_LAYOUT::BNSD) {
            transInfo.gCount = transInfo.gEndIdx - transInfo.gStartIdx + 1;
            fa_base_vector::Bmm2DataCopyOutNBSDGTiling(attenOutUb, transInfo, constInfo, attentionOutGm);
        } else {
            transInfo.s1Count = transInfo.s1EndIdx - transInfo.s1StartIdx + 1;
            transInfo.gCount = dealRowCount;
            fa_base_vector::Bmm2DataCopyOutNBSDMTiling<LAYOUT_T, OUT_T>(attenOutUb, transInfo, constInfo,
                                                                        actualSeqLengthsGmQ, attentionOutGm);
        }
        return;
    }
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = dealRowCount;
    dataCopyParams.blockLen = actualColumnCount * sizeof(OUT_T);
    dataCopyParams.srcStride = (columnCount - actualColumnCount) / (fa_base_vector::BYTE_BLOCK / sizeof(OUT_T));
    dataCopyParams.dstStride = 0;
    DataCopyPad(attentionOutGm[info.attenOutOffset + wsMStart * actualColumnCount], attenOutUb, dataCopyParams);
    return;
}

template <typename FIAT>
__aicore__ inline void
FiaBlockVecNonQuantMla<FIAT>::Bmm2CastAndCopyOut(const AttentionCommon::RunInfo &info, LocalTensor<T> &bmm2ResUb,
                                                         uint32_t wsMStart, uint32_t dealRowCount, uint32_t columnCount,
                                                         uint32_t actualColumnCount)
{
    LocalTensor<OUT_T> tmpBmm2ResCastTensor = outputBuff1.Get<OUT_T>();
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    if constexpr (IsSameType<OUT_T, bfloat16_t>::value) { // bf16 采取四舍六入五成双模式
        Cast(tmpBmm2ResCastTensor, bmm2ResUb, AscendC::RoundMode::CAST_RINT, dealRowCount * columnCount);
    } else {
        Cast(tmpBmm2ResCastTensor, bmm2ResUb, AscendC::RoundMode::CAST_ROUND, dealRowCount * columnCount);
    }

    DealInvalidRows(info, tmpBmm2ResCastTensor, wsMStart, dealRowCount, columnCount, actualColumnCount);
    SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    Bmm2DataCopyOutTrans(info, tmpBmm2ResCastTensor, wsMStart, dealRowCount, columnCount, actualColumnCount);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
}

template <typename FIAT>
__aicore__ inline void
FiaBlockVecNonQuantMla<FIAT>::Bmm2ResCopyOut(const AttentionCommon::RunInfo &info, LocalTensor<T> &bmm2ResUb, uint32_t wsMStart,
                                                     uint32_t dealRowCount, uint32_t columnCount,
                                                     uint32_t actualColumnCount)
{
    if constexpr (FLASH_DECODE) {
        if (info.tndIsS2SplitCore) {
            Bmm2FDDataCopyOut(info, bmm2ResUb, wsMStart, dealRowCount, columnCount, actualColumnCount);
        } else {
            Bmm2CastAndCopyOut(info, bmm2ResUb, wsMStart, dealRowCount, columnCount, actualColumnCount);
        }
    } else {
        Bmm2CastAndCopyOut(info, bmm2ResUb, wsMStart, dealRowCount, columnCount, actualColumnCount);
    }
}

template <typename FIAT>
__aicore__ inline void
FiaBlockVecNonQuantMla<FIAT>::DealBmm2ResBaseBlock(const AttentionCommon::RunInfo &info, const MSplitInfo &mSplitInfo,
                                                           uint32_t startRow, uint32_t dealRowCount,
                                                           uint32_t columnCount, uint32_t actualColumnCount)
{
    uint32_t vec2ComputeSize = dealRowCount * columnCount;
    uint32_t mStart = mSplitInfo.nBufferStartM + mSplitInfo.vecStartM + startRow;
    uint64_t srcGmOffset = (info.bn2IdxInCurCore % constInfo.preLoadNum) * constInfo.bmm2ResUbSize + mStart * columnCount;
    LocalTensor<MM2_OUT_T> tmpBmm2ResUb = inputBuff1.Get<MM2_OUT_T>();
    tmpBmm2ResUb = tmpBmm2ResUb[pingpongFlag * INPUT1_BUFFER_OFFSET / sizeof(MM2_OUT_T)];
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG + pingpongFlag);
    DataCopy(tmpBmm2ResUb, mm2ResGm[srcGmOffset], vec2ComputeSize);

    SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF1_FLAG);

    // 将绝对值大于1e10的数置为0
    LocalTensor<T> bmm2ResUb = tmpBuff1.Get<T>();
    bmm2ResUb.SetSize(vec2ComputeSize);
    LocalTensor<T> absBmm2ResUb = bmm2ResUb.template ReinterpretCast<T>();
    Abs(absBmm2ResUb, tmpBmm2ResUb, vec2ComputeSize);
    AscendC::PipeBarrier<PIPE_V>();
    LocalTensor<uint8_t> cmpMaskUb = absBmm2ResUb.template ReinterpretCast<uint8_t>();
    CompareScalar(cmpMaskUb, absBmm2ResUb, (T)1e10, CMPMODE::LE, vec2ComputeSize);
    AscendC::PipeBarrier<PIPE_V>();
    Select(tmpBmm2ResUb, cmpMaskUb, tmpBmm2ResUb, AttentionCommon::ConstInfo::FLOAT_ZERO, SELMODE::VSEL_TENSOR_SCALAR_MODE, vec2ComputeSize);
    AscendC::PipeBarrier<PIPE_V>();

    uint32_t baseOffset = mSplitInfo.nBufferStartM / 2 + startRow;
    uint32_t idx = info.loop % (constInfo.preLoadNum);
    LocalTensor<T> tmpSumUb = attenMaskTmpBuff.Get<T>(); // sumUb用临时内存 16 * 32B  = 512B
    Brcb(tmpSumUb, aMlaSumUb[idx * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T) + baseOffset], (dealRowCount + 7) / 8, {1, 8});
    AscendC::PipeBarrier<PIPE_V>();
    fa_base_vector::RowDivs(bmm2ResUb, tmpBmm2ResUb, tmpSumUb, dealRowCount, columnCount, actualColumnCount);
    AscendC::PipeBarrier<PIPE_V>();

    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG + pingpongFlag);
    Bmm2ResCopyOut(info, bmm2ResUb, mStart, dealRowCount, columnCount, actualColumnCount);
}

#endif
