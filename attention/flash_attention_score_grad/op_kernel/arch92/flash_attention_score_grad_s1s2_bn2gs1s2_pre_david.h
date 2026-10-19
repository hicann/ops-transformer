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
 * \file flash_attention_score_grad_s1s2_bn2gs1s2_pre_david.h
 * \brief
 */
#ifndef _FLASH_ATTENTION_SCORE_GRAD_S1S2_BNGS1S2_PRE_KERNEL_DAVID_H_
#define _FLASH_ATTENTION_SCORE_GRAD_S1S2_BNGS1S2_PRE_KERNEL_DAVID_H_
#include "kernel_operator.h"

using namespace AscendC;

#define FAG_PRE_CLASS_TEMPLATE \
    template <typename T1, typename T2, DTemplateType dTemplateType = DTemplateType::Aligned128, \
              const int8_t IS_D_NO_EQUAL = 0, const uint32_t IS_TND = 0, const uint8_t SPLIT_AXIS = 0, \
              const uint8_t DETER_SPARSE_TYPE = 0, const bool IS_ATTEN_MASK = 0>

#define FAG_PRE_FUNCTION_TEMPLATE \
    template <typename T1, typename T2, DTemplateType dTemplateType, const int8_t IS_D_NO_EQUAL, \
              const uint32_t IS_TND, const uint8_t SPLIT_AXIS, const uint8_t DETER_SPARSE_TYPE, \
              const bool IS_ATTEN_MASK>

#define FAG_PRE_FUNCTION_PARAMS_TEMPLATE \
    T1, T2, dTemplateType, IS_D_NO_EQUAL, IS_TND, SPLIT_AXIS, DETER_SPARSE_TYPE, IS_ATTEN_MASK

FAG_PRE_CLASS_TEMPLATE
class FlashAttentionScoreGradS1S2BNGS1S2PreDavid {
public:
    __aicore__ inline FlashAttentionScoreGradS1S2BNGS1S2PreDavid(){};
    __aicore__ inline void Init(__gm__ uint8_t* dq, __gm__ uint8_t* dk, __gm__ uint8_t* dv, __gm__ uint8_t* dx,
                                __gm__ uint8_t* y, __gm__ uint8_t* actual_seq_qlen, __gm__ uint8_t* actual_seq_kvlen,
                                __gm__ uint8_t* drop_mask, __gm__ uint8_t* deqScaleDy, __gm__ uint8_t* deqScaleO,
                                __gm__ uint8_t* workspace, FagTilingType ordTilingData, TPipe* pipe_in);
    __aicore__ inline void Process();
    __aicore__ inline void SyncALLCores();
    __aicore__ inline void CalTempDimAlign();
    __aicore__ inline void InitIndex(int64_t startIdx, int64_t& curS);
    __aicore__ inline void CopyInSfmg(int64_t leftNburst, int64_t& curS);
    __aicore__ inline void DoCopyIn(int64_t curS, int64_t curNBurst, int64_t dstOffset);
    __aicore__ inline void CalculateSoftmaxGrad(int64_t taskId, int64_t sfmgOutputOffset, int64_t curNBurst,
                                                int64_t deqScaleIdx = 0);
    __aicore__ inline void DoSoftmaxGrad();
    __aicore__ inline void DoSoftmaxGradFp8();

    TPipe* pipe;
    TQue<QuePosition::VECIN, 1> input1Que[2];
    TQue<QuePosition::VECIN, 1> input2Que[2];
    TQue<QuePosition::VECOUT, 1> out1Que;

    GlobalTensor<float> dqWorkSpaceGm, dkWorkSpaceGm, dvWorkSpaceGm, deqScaleDyGm, deqScaleOGm, sfmgWorkspaceGm,
        dsinkWorkSpaceGm;
    GlobalTensor<T1> dqGm, dkGm, dvGm, dxGm, yGm;

    LocalTensor<T1> input1Buf;
    LocalTensor<T1> input2Buf;
    LocalTensor<T2> outputBuf;

    FagTilingType TilingData;
    constexpr static uint32_t ADDR_ALIGN_SIZE = 512;
    constexpr static uint32_t HELP_LEN = 256;
    constexpr static uint32_t BIT8 = 8;
    constexpr static uint32_t NUMBER_8 = 8;
    constexpr static uint32_t B16_VECTOR_MASK = 128;
    constexpr static uint32_t S1S2_TND = 3;
    constexpr static uint32_t INPUT_FROAMT_BN2GS2D = 3;                              // BNSD
    constexpr static uint32_t INPUT_FROAMT_S2BN2GD = 2;                              // SBH
    constexpr static uint32_t INPUT_FROAMT_BS2N2GD = 1;                              // BSH  BSND
    constexpr static uint32_t HEAD_DIM_ALIGN = static_cast<uint32_t>(dTemplateType); // BSH  BSND
    constexpr static uint32_t TEMP_DIM_ALIGN_320 = 320;
    constexpr static uint32_t TEMP_DIM_ALIGN_384 = 384;
    constexpr static uint32_t TEMP_DIM_ALIGN_448 = 448;
    constexpr static uint32_t TEMP_DIM_ALIGN_192 = 192;
    constexpr static uint32_t TEMP_DIM_ALIGN_128 = 128;
    constexpr static uint32_t TEMP_DIM_ALIGN_256 = 256;
    constexpr static uint32_t TEMP_DIM_ALIGN_512 = 512;
    constexpr static uint32_t TEMP_DIM_ALIGN_576 = 576;
    constexpr static uint32_t TEMP_DIM_ALIGN_640 = 640;
    constexpr static uint32_t TEMP_DIM_ALIGN_704 = 704;
    constexpr static int64_t BLOCK_BYTE_SIZE = 32;
    constexpr static int64_t QUANT_BLOCK_SIZE = 256;

    uint32_t cBlockIdx;
    // query
    uint32_t ubBaseSize;
    uint32_t qPreBlockFactor;
    uint32_t qPreBlockTotal;
    uint32_t qPreBlockTail;
    uint32_t qPostBlockTotal;
    uint32_t kPreBlockFactor;
    uint32_t kPreBlockTotal;
    uint32_t kPreBlockTail;
    uint32_t kPostBlockTotal;
    uint32_t vPreBlockFactor;
    uint32_t vPreBlockTotal;
    uint32_t vPreBlockTail;
    uint32_t vPostBlockTotal;
    uint32_t isSink = 0;
    uint64_t sinkPreBlockFactor = 0;
    uint64_t sinkPreBlockTotal = 0;
    uint64_t sinkPreBlockTail = 0;

    uint64_t initdqSize;
    uint64_t dqOffset;
    uint64_t initdkSize;
    uint64_t dkOffset;
    uint64_t initdvSize;
    uint64_t dvOffset;
    uint64_t initdsinkSize = 0;
    uint64_t dsinkOffset = 0;

    int64_t b;
    int64_t n1;
    int64_t s1;
    int64_t s2;
    int64_t d;
    int64_t bIdx = 0;
    int64_t nIdx = 0;
    int64_t sIdx = 0;
    int64_t dAlignToBlock;
    int64_t tempDimAlign;
    uint32_t layout;
    __gm__ uint8_t* actual_seq_qlen_addr;
};

FAG_PRE_FUNCTION_TEMPLATE
__aicore__ inline void FlashAttentionScoreGradS1S2BNGS1S2PreDavid<FAG_PRE_FUNCTION_PARAMS_TEMPLATE>::Init(
    __gm__ uint8_t* dq, __gm__ uint8_t* dk, __gm__ uint8_t* dv, __gm__ uint8_t* dx, __gm__ uint8_t* y,
    __gm__ uint8_t* actual_seq_qlen, __gm__ uint8_t* actual_seq_kvlen, __gm__ uint8_t* drop_mask,
    __gm__ uint8_t* deqScaleDy, __gm__ uint8_t* deqScaleO, __gm__ uint8_t* workspace, FagTilingType orgTilingData,
    TPipe* pipe_in)
{
    cBlockIdx = GetBlockIdx();

    TilingData = orgTilingData;
    pipe = pipe_in;

    // tiling_data
    qPreBlockFactor = TilingData->preTilingData.qPreBlockFactor;
    qPreBlockTotal = TilingData->preTilingData.qPreBlockTotal;
    qPreBlockTail = TilingData->preTilingData.qPreBlockTail;
    qPostBlockTotal = TilingData->postTilingData.qPostBlockTotal;
    kPreBlockFactor = TilingData->preTilingData.kPreBlockFactor;
    kPreBlockTotal = TilingData->preTilingData.kPreBlockTotal;
    kPreBlockTail = TilingData->preTilingData.kPreBlockTail;
    kPostBlockTotal = TilingData->postTilingData.kPostBlockTotal;
    vPreBlockFactor = TilingData->preTilingData.vPreBlockFactor;
    vPreBlockTotal = TilingData->preTilingData.vPreBlockTotal;
    vPreBlockTail = TilingData->preTilingData.vPreBlockTail;
    vPostBlockTotal = TilingData->postTilingData.vPostBlockTotal;

    isSink = TilingData->s1s2BNGS1S2BaseParams.sinkOptional;
    if (unlikely(isSink)) {
        sinkPreBlockFactor = TilingData->preTilingData.sinkPreBlockFactor;
        sinkPreBlockTotal = TilingData->preTilingData.sinkPreBlockTotal;
        sinkPreBlockTail = TilingData->preTilingData.sinkPreBlockTail;
        dsinkWorkSpaceGm.SetGlobalBuffer((__gm__ float*)workspace +
                                         TilingData->postTilingData.dsinkWorkSpaceOffset / sizeof(float));
        initdsinkSize = cBlockIdx == sinkPreBlockTotal - 1 ? sinkPreBlockTail : sinkPreBlockFactor;
        dsinkOffset = ((uint64_t)cBlockIdx) * sinkPreBlockFactor;
    }

    dqGm.SetGlobalBuffer((__gm__ T1*)dq);
    dkGm.SetGlobalBuffer((__gm__ T1*)dk);
    dvGm.SetGlobalBuffer((__gm__ T1*)dv);
    dxGm.SetGlobalBuffer((__gm__ T1*)dx);
    yGm.SetGlobalBuffer((__gm__ T1*)y);
    deqScaleDyGm.SetGlobalBuffer((__gm__ float*)deqScaleDy);
    deqScaleOGm.SetGlobalBuffer((__gm__ float*)deqScaleO);

    dqWorkSpaceGm.SetGlobalBuffer((__gm__ float*)workspace + TilingData->postTilingData.dqWorkSpaceOffset / sizeof(T2));
    dkWorkSpaceGm.SetGlobalBuffer((__gm__ float*)workspace + TilingData->postTilingData.dkWorkSpaceOffset / sizeof(T2));
    dvWorkSpaceGm.SetGlobalBuffer((__gm__ float*)workspace + TilingData->postTilingData.dvWorkSpaceOffset / sizeof(T2));
    sfmgWorkspaceGm.SetGlobalBuffer((__gm__ T2*)workspace +
                                    TilingData->postTilingData.sfmgWorkSpaceOffset / sizeof(T2));

    initdqSize = cBlockIdx == qPreBlockTotal - 1 ? qPreBlockTail : qPreBlockFactor;
    dqOffset = ((uint64_t)cBlockIdx) * qPreBlockFactor;
    initdkSize = cBlockIdx == kPreBlockTotal - 1 ? kPreBlockTail : kPreBlockFactor;
    dkOffset = ((uint64_t)cBlockIdx) * kPreBlockFactor;
    initdvSize = cBlockIdx == vPreBlockTotal - 1 ? vPreBlockTail : vPreBlockFactor;
    dvOffset = ((uint64_t)cBlockIdx) * vPreBlockFactor;

    b = TilingData->preSfmgTilingData.b;
    n1 = TilingData->preSfmgTilingData.n2 * TilingData->preSfmgTilingData.g;
    s1 = TilingData->preSfmgTilingData.s1;
    d = TilingData->preSfmgTilingData.value_d;
    int64_t blockNums = BLOCK_BYTE_SIZE / sizeof(T1);
    dAlignToBlock = AlignTo(d, blockNums);
    actual_seq_qlen_addr = actual_seq_qlen; // todo

    pipe->InitBuffer(input1Que[0], 1, TilingData->preSfmgTilingData.inputBufferLen);
    pipe->InitBuffer(input1Que[1], 1, TilingData->preSfmgTilingData.inputBufferLen);
    pipe->InitBuffer(input2Que[0], 1, TilingData->preSfmgTilingData.inputBufferLen);
    pipe->InitBuffer(input2Que[1], 1, TilingData->preSfmgTilingData.inputBufferLen);
    pipe->InitBuffer(out1Que, 2, TilingData->preSfmgTilingData.outputBufferLen);
}

FAG_PRE_FUNCTION_TEMPLATE
__aicore__ inline void FlashAttentionScoreGradS1S2BNGS1S2PreDavid<FAG_PRE_FUNCTION_PARAMS_TEMPLATE>::CalTempDimAlign()
{
    if constexpr (HEAD_DIM_ALIGN <= 256) {
        if (HEAD_DIM_ALIGN == TEMP_DIM_ALIGN_192 && IS_D_NO_EQUAL != 0) {
            tempDimAlign = 256;
        } else {
            tempDimAlign = HEAD_DIM_ALIGN;
        }
    } else {
        if (dAlignToBlock <= TEMP_DIM_ALIGN_384 && !IsSameType<T1, float>::value) {
            tempDimAlign = TEMP_DIM_ALIGN_384;
        } else if (dAlignToBlock <= TEMP_DIM_ALIGN_320 && IsSameType<T1, float>::value) {
            tempDimAlign = TEMP_DIM_ALIGN_320;
        } else if (dAlignToBlock > TEMP_DIM_ALIGN_320 && dAlignToBlock <= TEMP_DIM_ALIGN_384 &&
                   IsSameType<T1, float>::value) {
            tempDimAlign = TEMP_DIM_ALIGN_384;
        } else if (dAlignToBlock > TEMP_DIM_ALIGN_384 && dAlignToBlock <= TEMP_DIM_ALIGN_448 &&
                   IsSameType<T1, float>::value) {
            tempDimAlign = TEMP_DIM_ALIGN_448;
        } else if (dAlignToBlock > TEMP_DIM_ALIGN_448 && dAlignToBlock <= TEMP_DIM_ALIGN_512 &&
                   IsSameType<T1, float>::value) {
            tempDimAlign = TEMP_DIM_ALIGN_512;
        } else if (dAlignToBlock > TEMP_DIM_ALIGN_512 && dAlignToBlock <= TEMP_DIM_ALIGN_576 &&
                   IsSameType<T1, float>::value) {
            tempDimAlign = TEMP_DIM_ALIGN_576;
        } else if (dAlignToBlock > TEMP_DIM_ALIGN_576 && dAlignToBlock <= TEMP_DIM_ALIGN_640 &&
                   IsSameType<T1, float>::value) {
            tempDimAlign = TEMP_DIM_ALIGN_640;
        } else if (dAlignToBlock > TEMP_DIM_ALIGN_640 && dAlignToBlock <= TEMP_DIM_ALIGN_704 &&
                   IsSameType<T1, float>::value) {
            tempDimAlign = TEMP_DIM_ALIGN_704;
        } else {
            tempDimAlign = HEAD_DIM_ALIGN;
        }
    }
}

FAG_PRE_FUNCTION_TEMPLATE
__aicore__ inline void FlashAttentionScoreGradS1S2BNGS1S2PreDavid<FAG_PRE_FUNCTION_PARAMS_TEMPLATE>::InitIndex(
    int64_t startIdx, int64_t& curS)
{
    if constexpr (IS_TND) {
        int64_t totalLen = 0;
        for (int64_t bDimIdx = bIdx; bDimIdx < b; bDimIdx++) {
            totalLen = n1 * ((__gm__ int64_t*)actual_seq_qlen_addr)[bDimIdx] * d;
            if (totalLen > startIdx) {
                bIdx = bDimIdx;
                curS = (bIdx == 0) ? ((__gm__ int64_t*)actual_seq_qlen_addr)[bIdx] :
                                     (((__gm__ int64_t*)actual_seq_qlen_addr)[bIdx] -
                                      ((__gm__ int64_t*)actual_seq_qlen_addr)[bIdx - 1]);
                int64_t bTail = startIdx - (totalLen - n1 * curS * d);
                nIdx = bTail / (curS * d);
                int64_t nTail = bTail % (curS * d);
                sIdx = nTail / d;
                break;
            }
        }
    } else {
        bIdx = startIdx / (n1 * s1 * d);
        int64_t bTail = startIdx % (n1 * s1 * d);
        nIdx = bTail / (s1 * d);
        int64_t nTail = bTail % (s1 * d);
        sIdx = nTail / d;
    }
}

FAG_PRE_FUNCTION_TEMPLATE
__aicore__ inline void FlashAttentionScoreGradS1S2BNGS1S2PreDavid<FAG_PRE_FUNCTION_PARAMS_TEMPLATE>::DoCopyIn(
    int64_t curS, int64_t curNBurst, int64_t dstOffset)
{
    int64_t srcOffset = 0;
    int64_t transpose_stride = 0;
    if constexpr (IS_TND) {
        int64_t bOffset = bIdx == 0 ? 0 : n1 * ((__gm__ int64_t*)actual_seq_qlen_addr)[bIdx - 1] * d;
        srcOffset = bOffset + (sIdx * n1 + nIdx) * d;
        transpose_stride = (n1 * d - d) * sizeof(T1);
    } else {
        if (layout == INPUT_FROAMT_BN2GS2D) {
            srcOffset = bIdx * (n1 * s1 * d) + nIdx * (s1 * d) + sIdx * d;
            transpose_stride = 0;
        } else if (layout == INPUT_FROAMT_BS2N2GD) {
            srcOffset = bIdx * (s1 * n1 * d) + sIdx * (n1 * d) + nIdx * d;
            transpose_stride = (n1 * d - d) * sizeof(T1);
        } else if (layout == INPUT_FROAMT_S2BN2GD) {
            srcOffset = sIdx * (b * n1 * d) + bIdx * (n1 * d) + nIdx * d;
            transpose_stride = (b * n1 * d - d) * sizeof(T1);
        }
    }
    uint32_t dstBlockStride = (tempDimAlign - dAlignToBlock) * sizeof(T1) / BLOCK_BYTE_SIZE;
    DataCopyPad(input1Buf[dstOffset], dxGm[srcOffset],
                {static_cast<uint16_t>(curNBurst), static_cast<uint32_t>(d * sizeof(T1)),
                 static_cast<uint32_t>(transpose_stride), dstBlockStride, 0},
                {true, 0, static_cast<uint8_t>((dAlignToBlock - d)), 0}, true);
    DataCopyPad(input2Buf[dstOffset], yGm[srcOffset],
                {static_cast<uint16_t>(curNBurst), static_cast<uint32_t>(d * sizeof(T1)),
                 static_cast<uint32_t>(transpose_stride), dstBlockStride, 0},
                {true, 0, static_cast<uint8_t>((dAlignToBlock - d)), 0}, true);
}

FAG_PRE_FUNCTION_TEMPLATE
__aicore__ inline void FlashAttentionScoreGradS1S2BNGS1S2PreDavid<FAG_PRE_FUNCTION_PARAMS_TEMPLATE>::CopyInSfmg(
    int64_t leftNburst, int64_t& curS)
{
    int64_t dstOffset = 0;
    while (leftNburst > 0) {
        int64_t curNburst = 0;
        if (curS - sIdx < leftNburst) { // 需要借N或借B
            curNburst = curS - sIdx;
            DoCopyIn(curS, curNburst, dstOffset);
            leftNburst = leftNburst - curNburst;
            sIdx = 0;
            if (nIdx < n1 - 1) { // 需要借N
                nIdx += 1;
            } else {
                nIdx = 0;
                if (bIdx < b - 1) { // 需要借B
                    bIdx += 1;
                    if constexpr (IS_TND) {
                        curS = ((__gm__ int64_t*)actual_seq_qlen_addr)[bIdx] -
                               ((__gm__ int64_t*)actual_seq_qlen_addr)[bIdx - 1];
                    } else {
                        curS = s1;
                    }
                } else { // 没有轴可以借了，end
                    leftNburst = 0;
                }
            }
        } else { // 当前S够用
            curNburst = leftNburst;
            DoCopyIn(curS, curNburst, dstOffset);
            sIdx = sIdx + leftNburst;
            leftNburst = 0;
        }
        dstOffset = dstOffset + curNburst * tempDimAlign;
    }
}

FAG_PRE_FUNCTION_TEMPLATE
__aicore__ inline void
FlashAttentionScoreGradS1S2BNGS1S2PreDavid<FAG_PRE_FUNCTION_PARAMS_TEMPLATE>::CalculateSoftmaxGrad(
    int64_t taskId, int64_t sfmgOutputOffset, int64_t curNBurst, int64_t deqScaleIdx)
{
    LocalTensor<T1> yInTensor = input1Que[taskId & 1].DeQue<T1>();
    LocalTensor<T1> dxInTensor = input2Que[taskId & 1].DeQue<T1>();
    auto output1Buf = out1Que.AllocTensor<T2>();

    if constexpr (IsSameType<T1, float8_e5m2_t>::value || IsSameType<T1, float8_e4m3_t>::value) {
        float deqScaleDxValue = deqScaleDyGm.GetValue(deqScaleIdx);
        float deqScaleYValue = deqScaleOGm.GetValue(deqScaleIdx);
        if (HEAD_DIM_ALIGN == TEMP_DIM_ALIGN_192 && IS_D_NO_EQUAL == 1) {
            AscendC::MyAntiQuantSoftmaxGradFrontCast<T1, T2, TEMP_DIM_ALIGN_256>(
                output1Buf, dxInTensor, yInTensor, deqScaleDxValue, deqScaleYValue, static_cast<uint32_t>(curNBurst),
                static_cast<uint32_t>(dAlignToBlock));
        } else {
            AscendC::MyAntiQuantSoftmaxGradFrontCast<T1, T2, HEAD_DIM_ALIGN>(
                output1Buf, dxInTensor, yInTensor, deqScaleDxValue, deqScaleYValue, static_cast<uint32_t>(curNBurst),
                static_cast<uint32_t>(dAlignToBlock));
        }
    } else {
        if constexpr (HEAD_DIM_ALIGN <= TEMP_DIM_ALIGN_256) {
            if constexpr (HEAD_DIM_ALIGN == TEMP_DIM_ALIGN_192 && IS_D_NO_EQUAL == 1) {
                AscendC::MySoftmaxGradFrontCast<T1, T2, TEMP_DIM_ALIGN_256, HEAD_DIM_ALIGN>(
                    output1Buf, dxInTensor, yInTensor, static_cast<uint32_t>(curNBurst),
                    static_cast<uint32_t>(dAlignToBlock));
            } else {
                AscendC::MySoftmaxGradFrontCast<T1, T2, HEAD_DIM_ALIGN, HEAD_DIM_ALIGN>(
                    output1Buf, dxInTensor, yInTensor, static_cast<uint32_t>(curNBurst),
                    static_cast<uint32_t>(dAlignToBlock));
            }
        } else {
            if (dAlignToBlock <= TEMP_DIM_ALIGN_384 && !IsSameType<T1, float>::value) {
                AscendC::MySoftmaxGradFrontCast<T1, T2, TEMP_DIM_ALIGN_384, HEAD_DIM_ALIGN>(
                    output1Buf, dxInTensor, yInTensor, static_cast<uint32_t>(curNBurst),
                    static_cast<uint32_t>(dAlignToBlock));
            } else if (dAlignToBlock <= TEMP_DIM_ALIGN_320 && IsSameType<T1, float>::value) {
                AscendC::MySoftmaxGradFrontCast<T1, T2, TEMP_DIM_ALIGN_320, HEAD_DIM_ALIGN>(
                    output1Buf, dxInTensor, yInTensor, static_cast<uint32_t>(curNBurst),
                    static_cast<uint32_t>(dAlignToBlock));
            } else if (dAlignToBlock > TEMP_DIM_ALIGN_320 && dAlignToBlock <= TEMP_DIM_ALIGN_384 &&
                       IsSameType<T1, float>::value) {
                AscendC::MySoftmaxGradFrontCast<T1, T2, TEMP_DIM_ALIGN_384, HEAD_DIM_ALIGN>(
                    output1Buf, dxInTensor, yInTensor, static_cast<uint32_t>(curNBurst),
                    static_cast<uint32_t>(dAlignToBlock));
            } else if (dAlignToBlock > TEMP_DIM_ALIGN_384 && dAlignToBlock <= TEMP_DIM_ALIGN_448 &&
                       IsSameType<T1, float>::value) {
                AscendC::MySoftmaxGradFrontCast<T1, T2, TEMP_DIM_ALIGN_448, HEAD_DIM_ALIGN>(
                    output1Buf, dxInTensor, yInTensor, static_cast<uint32_t>(curNBurst),
                    static_cast<uint32_t>(dAlignToBlock));
            } else if (dAlignToBlock > TEMP_DIM_ALIGN_448 && dAlignToBlock <= TEMP_DIM_ALIGN_512 &&
                       IsSameType<T1, float>::value) {
                AscendC::MySoftmaxGradFrontCast<T1, T2, TEMP_DIM_ALIGN_512, HEAD_DIM_ALIGN>(
                    output1Buf, dxInTensor, yInTensor, static_cast<uint32_t>(curNBurst),
                    static_cast<uint32_t>(dAlignToBlock));
            } else if (dAlignToBlock > TEMP_DIM_ALIGN_512 && dAlignToBlock <= TEMP_DIM_ALIGN_576 &&
                       IsSameType<T1, float>::value) {
                AscendC::MySoftmaxGradFrontCast<T1, T2, TEMP_DIM_ALIGN_576, HEAD_DIM_ALIGN>(
                    output1Buf, dxInTensor, yInTensor, static_cast<uint32_t>(curNBurst),
                    static_cast<uint32_t>(dAlignToBlock));
            } else if (dAlignToBlock > TEMP_DIM_ALIGN_576 && dAlignToBlock <= TEMP_DIM_ALIGN_640 &&
                       IsSameType<T1, float>::value) {
                AscendC::MySoftmaxGradFrontCast<T1, T2, TEMP_DIM_ALIGN_640, HEAD_DIM_ALIGN>(
                    output1Buf, dxInTensor, yInTensor, static_cast<uint32_t>(curNBurst),
                    static_cast<uint32_t>(dAlignToBlock));
            } else if (dAlignToBlock > TEMP_DIM_ALIGN_640 && dAlignToBlock <= TEMP_DIM_ALIGN_704 &&
                       IsSameType<T1, float>::value) {
                AscendC::MySoftmaxGradFrontCast<T1, T2, TEMP_DIM_ALIGN_704, HEAD_DIM_ALIGN>(
                    output1Buf, dxInTensor, yInTensor, static_cast<uint32_t>(curNBurst),
                    static_cast<uint32_t>(dAlignToBlock));
            } else if (dAlignToBlock <= TEMP_DIM_ALIGN_512 && !IsSameType<T1, float>::value) {
                AscendC::MySoftmaxGradFrontCast<T1, T2, TEMP_DIM_ALIGN_512, HEAD_DIM_ALIGN>(
                    output1Buf, dxInTensor, yInTensor, static_cast<uint32_t>(curNBurst),
                    static_cast<uint32_t>(dAlignToBlock));
            } else if (dAlignToBlock <= TEMP_DIM_ALIGN_640 && !IsSameType<T1, float>::value) {
                AscendC::MySoftmaxGradFrontCast<T1, T2, TEMP_DIM_ALIGN_640, HEAD_DIM_ALIGN>(
                    output1Buf, dxInTensor, yInTensor, static_cast<uint32_t>(curNBurst),
                    static_cast<uint32_t>(dAlignToBlock));
            } else {
                AscendC::MySoftmaxGradFrontCast<T1, T2, HEAD_DIM_ALIGN, HEAD_DIM_ALIGN>(
                    output1Buf, dxInTensor, yInTensor, static_cast<uint32_t>(curNBurst),
                    static_cast<uint32_t>(dAlignToBlock));
            }
        }
    }
    out1Que.EnQue(output1Buf);
    out1Que.DeQue<T2>();
    DataCopyPad(sfmgWorkspaceGm[sfmgOutputOffset], output1Buf,
                {1, static_cast<uint32_t>(curNBurst * sizeof(T2)), 0, 0, 0});
    out1Que.FreeTensor(output1Buf);
}

FAG_PRE_FUNCTION_TEMPLATE
__aicore__ inline void FlashAttentionScoreGradS1S2BNGS1S2PreDavid<FAG_PRE_FUNCTION_PARAMS_TEMPLATE>::DoSoftmaxGrad()
{
    // process
    uint32_t usedCoreNums = TilingData->preSfmgTilingData.usedCoreNum;
    layout = TilingData->preSfmgTilingData.layout;
    if (cBlockIdx < usedCoreNums) {
        int64_t singleCoreLoopTimes, singleCoreLastLoopNBurstNum;
        if (cBlockIdx == usedCoreNums - 1) {
            singleCoreLoopTimes = TilingData->preSfmgTilingData.tailCoreLoopTimes; // 尾核loop次数
            singleCoreLastLoopNBurstNum =
                TilingData->preSfmgTilingData.tailCoreLastLoopNBurstNum; // 尾核最后一次处理s1大小
        } else {
            singleCoreLoopTimes = TilingData->preSfmgTilingData.normalCoreLoopTimes; // 非尾核loop次数
            singleCoreLastLoopNBurstNum =
                TilingData->preSfmgTilingData.normalCoreLastLoopNBurstNum; // 非尾核最后一次处理s1大小
        }

        int64_t sfmgOutputOffset = cBlockIdx * TilingData->preSfmgTilingData.normalCoreNBurstNums; // 核间起始地址
        int64_t nBurst = TilingData->preSfmgTilingData.singleLoopNBurstNum; // 一次普通loop处理多少个D
        int64_t curS = s1;
        int64_t taskId = 0;
        if constexpr (IS_D_NO_EQUAL != 0) {
            input1Buf = input1Que[0].AllocTensor<T1>();
            input2Buf = input2Que[0].AllocTensor<T1>();
            Duplicate<T1>(input1Buf, 0, nBurst * tempDimAlign);
            Duplicate<T1>(input2Buf, 0, nBurst * tempDimAlign);
            input1Que[0].FreeTensor(input1Buf);
            input2Que[0].FreeTensor(input2Buf);

            input1Buf = input1Que[1].AllocTensor<T1>();
            input2Buf = input2Que[1].AllocTensor<T1>();
            Duplicate<T1>(input1Buf, 0, nBurst * tempDimAlign);
            Duplicate<T1>(input2Buf, 0, nBurst * tempDimAlign);
            input1Que[1].FreeTensor(input1Buf);
            input2Que[1].FreeTensor(input2Buf);
        }
        for (int64_t i = 0; i < singleCoreLoopTimes; i++) {
            if (i == singleCoreLoopTimes - 1) {
                nBurst = singleCoreLastLoopNBurstNum;
            }

            input1Buf = input1Que[taskId & 1].AllocTensor<T1>();
            input2Buf = input2Que[taskId & 1].AllocTensor<T1>();
            // sync
            InitIndex(sfmgOutputOffset * d, curS);
            CopyInSfmg(nBurst, curS);

            input1Que[taskId & 1].EnQue(input1Buf);
            input2Que[taskId & 1].EnQue(input2Buf);
            CalculateSoftmaxGrad(taskId, sfmgOutputOffset, nBurst);
            if constexpr (IS_D_NO_EQUAL != 0) {
                if (i < singleCoreLoopTimes - 1) {
                    Duplicate<T1>(input1Buf, 0, nBurst * tempDimAlign);
                    Duplicate<T1>(input2Buf, 0, nBurst * tempDimAlign);
                }
            }
            input1Que[taskId & 1].FreeTensor(input1Buf);
            input2Que[taskId & 1].FreeTensor(input2Buf);
            sfmgOutputOffset += TilingData->preSfmgTilingData.singleLoopNBurstNum;
            taskId++;
        }
    }
}

FAG_PRE_FUNCTION_TEMPLATE
__aicore__ inline void FlashAttentionScoreGradS1S2BNGS1S2PreDavid<FAG_PRE_FUNCTION_PARAMS_TEMPLATE>::DoSoftmaxGradFp8()
{
    // process
    uint32_t usedCoreNums = TilingData->preSfmgTilingData.usedCoreNum;
    layout = TilingData->preSfmgTilingData.layout;
    if (cBlockIdx < usedCoreNums) {
        int64_t normalAxisSize = TilingData->preSfmgTilingData.normalAxisSize;
        // FP8 Per-Block 量化中保证 nBurst=128
        int64_t nBurst = TilingData->preSfmgTilingData.singleLoopNBurstNum; // 一次普通loop处理多少个D
        int64_t curS = s1;
        int64_t taskId = 0;
        int64_t dataBlockIdx = (int)cBlockIdx - (int)usedCoreNums;
        int64_t numOf128InS1 = ((int)s1 + nBurst - 1) / nBurst;
        int64_t s1Residual = (int)s1 % nBurst == 0 ? nBurst : (int)s1 % nBurst;

        if constexpr (IS_D_NO_EQUAL != 0) {
            LocalTensor<uint8_t> input1Bufb8 = input1Que[0].AllocTensor<uint8_t>();
            LocalTensor<uint8_t> input2Bufb8 = input2Que[0].AllocTensor<uint8_t>();
            Duplicate<uint8_t>(input1Bufb8, 0, nBurst * tempDimAlign);
            Duplicate<uint8_t>(input2Bufb8, 0, nBurst * tempDimAlign);
            input1Que[0].FreeTensor(input1Bufb8);
            input2Que[0].FreeTensor(input2Bufb8);

            input1Bufb8 = input1Que[1].AllocTensor<uint8_t>();
            input2Bufb8 = input2Que[1].AllocTensor<uint8_t>();
            Duplicate<uint8_t>(input1Bufb8, 0, nBurst * tempDimAlign);
            Duplicate<uint8_t>(input2Bufb8, 0, nBurst * tempDimAlign);
            input1Que[1].FreeTensor(input1Bufb8);
            input2Que[1].FreeTensor(input2Bufb8);
        }
        while (true) {
            dataBlockIdx += usedCoreNums;
            int64_t sfmgOutputOffset = dataBlockIdx * nBurst - (dataBlockIdx / numOf128InS1) * (nBurst - s1Residual);
            if (sfmgOutputOffset >= normalAxisSize) {
                break;
            }
            // Fill with 0
            LocalTensor<uint8_t> input1Buf8b = input1Que[taskId & 1].AllocTensor<uint8_t>();
            LocalTensor<uint8_t> input2Buf8b = input2Que[taskId & 1].AllocTensor<uint8_t>();
            Duplicate<uint8_t>(input1Buf8b, 0, nBurst * tempDimAlign);
            Duplicate<uint8_t>(input2Buf8b, 0, nBurst * tempDimAlign);
            SetFlag<HardEvent::V_MTE2>(static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2)));
            WaitFlag<HardEvent::V_MTE2>(static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2)));
            input1Que[taskId & 1].FreeTensor(input1Buf8b);
            input2Que[taskId & 1].FreeTensor(input2Buf8b);

            input1Buf = input1Que[taskId & 1].AllocTensor<T1>();
            input2Buf = input2Que[taskId & 1].AllocTensor<T1>();
            int64_t curNBurst = ((dataBlockIdx + 1) % numOf128InS1 == 0) ? s1Residual : nBurst;
            InitIndex(sfmgOutputOffset * d, curS);
            CopyInSfmg(curNBurst, curS);

            input1Que[taskId & 1].EnQue(input1Buf);
            input2Que[taskId & 1].EnQue(input2Buf);
            auto deqScaleIdx = dataBlockIdx * nBurst / QUANT_BLOCK_SIZE;
            CalculateSoftmaxGrad(taskId, sfmgOutputOffset, curNBurst, deqScaleIdx);
            if constexpr (IS_D_NO_EQUAL != 0) {
                Duplicate<uint8_t>(reinterpret_cast<LocalTensor<uint8_t>&>(input1Buf), 0, nBurst * tempDimAlign);
                Duplicate<uint8_t>(reinterpret_cast<LocalTensor<uint8_t>&>(input2Buf), 0, nBurst * tempDimAlign);
            }
            input1Que[taskId & 1].FreeTensor(input1Buf);
            input2Que[taskId & 1].FreeTensor(input2Buf);
            taskId++;
        }
    }
}

FAG_PRE_FUNCTION_TEMPLATE
__aicore__ inline void FlashAttentionScoreGradS1S2BNGS1S2PreDavid<FAG_PRE_FUNCTION_PARAMS_TEMPLATE>::Process()
{
    // process
    if (g_coreType == AIV && cBlockIdx < TilingData->preTilingData.maskCoreNum) {
        // clear dq dk dv workspace
        if constexpr (SPLIT_AXIS == BN2GS1S2) {
            if constexpr (IsSameType<T1, float>::value) {
                InitOutput<T1>(dqGm[dqOffset], initdqSize, 0);
                InitOutput<T1>(dkGm[dkOffset], initdkSize, 0);
                InitOutput<T1>(dvGm[dvOffset], initdvSize, 0);
            } else {
                InitOutput<float>(dqWorkSpaceGm[dqOffset], initdqSize, 0);
                InitOutput<float>(dkWorkSpaceGm[dkOffset], initdkSize, 0);
                InitOutput<float>(dvWorkSpaceGm[dvOffset], initdvSize, 0);
            }
        }
        if (unlikely(isSink && (IS_ATTEN_MASK || IS_TND))) {
            InitOutput<float>(dsinkWorkSpaceGm[dsinkOffset], initdsinkSize, 0);
        }
        CalTempDimAlign();

        if constexpr (IsSameType<T1, float8_e5m2_t>::value || IsSameType<T1, float8_e4m3_t>::value) {
            // FP8量化方案中，采用不同的数据分块方案
            DoSoftmaxGradFp8();
        } else {
            DoSoftmaxGrad();
        }
    }
}

FAG_PRE_FUNCTION_TEMPLATE
__aicore__ inline void FlashAttentionScoreGradS1S2BNGS1S2PreDavid<FAG_PRE_FUNCTION_PARAMS_TEMPLATE>::SyncALLCores()
{
    SyncAll<>();
}
#endif // _FLASH_ATTENTION_SCORE_GRAD_S1S2_BNGS1S2_PRE_KERNEL_DAVID_H_
