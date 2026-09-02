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
 * \file flash_attention_score_grad_presfmg_regbase.h
 * \brief
 */
#ifndef FLASH_ATTENTION_SCORE_GRAD_PRESFMG_KERNEL_REGBASE_H_
#define FLASH_ATTENTION_SCORE_GRAD_PRESFMG_KERNEL_REGBASE_H_
#include "math.h"
#include "kernel_basic_intf.h"
#include "vector_api/vf_anti_quant_softmax_grad_front_cast.h"
#include "vector_api/vf_softmax_grad_front_cast.h"
#include "flash_attention_score_grad_common.h"
using namespace AscendC;

#define PRE_FUNCTION_TEMPLATE \
    template <typename T1, typename T2, typename OUTDTYPE, DTemplateType dTemplateType, const int8_t IS_D_NO_EQUAL, \
              const uint8_t DETER_SPARSE_TYPE, const uint32_t IS_TND, const uint8_t SPLIT_AXIS, \
              const uint8_t IS_TND_SWIZZLE>

#define PRE_FUNCTION_ARGS_TEMPLATE \
    T1, T2, OUTDTYPE, dTemplateType, IS_D_NO_EQUAL, DETER_SPARSE_TYPE, IS_TND, SPLIT_AXIS, IS_TND_SWIZZLE

template <typename T1, typename T2, typename OUTDTYPE, DTemplateType dTemplateType = DTemplateType::Aligned128,
          const int8_t IS_D_NO_EQUAL = 0, const uint8_t DETER_SPARSE_TYPE = 0, const uint32_t IS_TND = 0,
          const uint8_t SPLIT_AXIS = 0, const uint8_t IS_TND_SWIZZLE = 0>
class FlashAttentionScoreGradPresfmgRegbase {
public:
    __aicore__ inline FlashAttentionScoreGradPresfmgRegbase(){};
    __aicore__ inline void Init(GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR dx, GM_ADDR y, GM_ADDR deqScaleDy,
                                GM_ADDR actual_seq_qlen, GM_ADDR workspace, FagTilingType ordTilingData,
                                TPipe *pipe_in);
    __aicore__ inline void Process();
    __aicore__ inline void SyncALLCores();

    __aicore__ inline void CalTempDimAlign();
    __aicore__ inline void InitIndex(uint64_t startIdx, uint64_t &curS);
    __aicore__ inline void CopyInSfmg(uint64_t leftNburst, uint64_t &curS);
    __aicore__ inline void DoCopyIn(uint64_t curS, uint64_t curNBurst, uint64_t dstOffset);
    __aicore__ inline void CalculateSoftmaxGrad(uint64_t taskId, uint64_t sfmgOutputOffset, uint64_t curNBurst,
                                                uint64_t deqScaleIdx = 0);
    __aicore__ inline void DoSoftmaxGrad();
    template <uint8_t GM_IDX>
    __aicore__ inline void CopyDqkvToGm(const uint64_t &loopIdx);

    constexpr static uint32_t HEAD_DIM_ALIGN = static_cast<uint32_t>(dTemplateType);
    constexpr static uint64_t BLOCK_BYTE_SIZE = 32;

    TPipe *pipe;
    GlobalTensor<float> dqWorkSpaceGm, dkWorkSpaceGm, dvWorkSpaceGm, deqScaleDyGm, sfmgWorkspaceGm;
    GlobalTensor<T1> dxGm;
    GlobalTensor<OUTDTYPE> dqGm, dkGm, dvGm, yGm;

    FagTilingType tilingData;

    uint32_t vBlockIdx;
    // query
    uint64_t ubBaseSize;
    uint64_t qPreBlockFactor;
    uint64_t qPreBlockTotal;
    uint64_t qPreBlockTail;
    uint64_t kPreBlockFactor;
    uint64_t kPreBlockTotal;
    uint64_t kPreBlockTail;
    uint64_t vPreBlockFactor;
    uint64_t vPreBlockTotal;
    uint64_t vPreBlockTail;

    uint64_t initdqSize;
    uint64_t dqOffset;
    uint64_t initdkSize;
    uint64_t dkOffset;
    uint64_t initdvSize;
    uint64_t dvOffset;
    uint64_t dqLoopNum;
    uint64_t dkLoopNum;
    uint64_t dvLoopNum;
    uint64_t dqBlockTail;
    uint64_t dkBlockTail;
    uint64_t dvBlockTail;
    uint64_t singleInitSize;
    event_t eventIDVToMte3;

    uint64_t b;
    uint64_t n1;
    uint64_t s1;
    uint64_t s2;
    uint64_t d;
    uint64_t bIdx = 0;
    uint64_t nIdx = 0;
    uint64_t sIdx = 0;
    uint64_t ns1d = 0;
    uint64_t s1d = 0;
    uint64_t nd = 0;
    uint64_t bnd = 0;

    uint64_t dAlignToBlock;
    uint64_t dAlignToBlockB16;
    uint64_t tempDimAlign = 0;
    uint32_t layout;
    uint32_t usedCoreNum;
    GM_ADDR actual_seq_qlen_addr;

    LocalTensor<OUTDTYPE> input1Buf;
    LocalTensor<T1> input2Buf;
    LocalTensor<T2> outputBuf;
    TBuf<> dqInitBuf;
    TBuf<> dkInitBuf;
    TBuf<> dvInitBuf;
    LocalTensor<T2> dqTensor;
    LocalTensor<T2> dkTensor;
    LocalTensor<T2> dvTensor;

    TQue<QuePosition::VECIN, 1> input1Que[2];
    TQue<QuePosition::VECIN, 1> input2Que[2];
    TQue<QuePosition::VECOUT, 1> out1Que;
};

PRE_FUNCTION_TEMPLATE
__aicore__ inline void FlashAttentionScoreGradPresfmgRegbase<PRE_FUNCTION_ARGS_TEMPLATE>::Init(
    GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR dx, GM_ADDR y, GM_ADDR deqScaleDy, GM_ADDR actual_seq_qlen,
    GM_ADDR workspace, FagTilingType orgTilingData, TPipe *pipe_in)
{
    if (g_coreType == AIV) {
        vBlockIdx = GetBlockIdx();

        tilingData = orgTilingData;
        pipe = pipe_in;

        // tiling_data
        qPreBlockFactor = tilingData->preTilingData.qPreBlockFactor;
        qPreBlockTotal = tilingData->preTilingData.qPreBlockTotal;
        qPreBlockTail = tilingData->preTilingData.qPreBlockTail;
        kPreBlockFactor = tilingData->preTilingData.kPreBlockFactor;
        kPreBlockTotal = tilingData->preTilingData.kPreBlockTotal;
        kPreBlockTail = tilingData->preTilingData.kPreBlockTail;
        vPreBlockFactor = tilingData->preTilingData.vPreBlockFactor;
        vPreBlockTotal = tilingData->preTilingData.vPreBlockTotal;
        vPreBlockTail = tilingData->preTilingData.vPreBlockTail;

        b = tilingData->s1s2BNGS1S2BaseParams.b;
        n1 = tilingData->s1s2BNGS1S2BaseParams.n2 * tilingData->s1s2BNGS1S2BaseParams.g;
        s1 = tilingData->s1s2BNGS1S2BaseParams.s1;
        d = tilingData->s1s2BNGS1S2BaseParams.d1;
        ns1d = n1 * s1 * d;
        s1d = s1 * d;
        nd = n1 * d;
        bnd = b * n1 * d;

        usedCoreNum = tilingData->preTilingData.sfmgUsedCoreNum;
        layout = tilingData->s1s2BNGS1S2BaseParams.layout;

        uint64_t blockNums = BLOCK_BYTE_SIZE / sizeof(T1);
        uint64_t blockNumsB16 = BLOCK_BYTE_SIZE / sizeof(OUTDTYPE);
        dAlignToBlock = AlignTo(d, blockNums);
        dAlignToBlockB16 = AlignTo(d, blockNumsB16);
        actual_seq_qlen_addr = actual_seq_qlen;

        dxGm.SetGlobalBuffer((__gm__ T1 *)dx);
        dqGm.SetGlobalBuffer((__gm__ OUTDTYPE *)dq);
        dkGm.SetGlobalBuffer((__gm__ OUTDTYPE *)dk);
        dvGm.SetGlobalBuffer((__gm__ OUTDTYPE *)dv);
        yGm.SetGlobalBuffer((__gm__ OUTDTYPE *)y);
        deqScaleDyGm.SetGlobalBuffer((__gm__ float *)deqScaleDy);

        dqWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace +
                                      tilingData->postTilingData.dqWorkSpaceOffset / sizeof(T2));
        dkWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace +
                                      tilingData->postTilingData.dkWorkSpaceOffset / sizeof(T2));
        dvWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace +
                                      tilingData->postTilingData.dvWorkSpaceOffset / sizeof(T2));
        sfmgWorkspaceGm.SetGlobalBuffer((__gm__ float *)workspace +
                                        tilingData->postTilingData.sfmgWorkSpaceOffset / sizeof(T2));

        initdqSize = vBlockIdx == qPreBlockTotal - 1 ? qPreBlockTail : qPreBlockFactor;
        dqOffset = ((uint64_t)vBlockIdx) * qPreBlockFactor;
        initdkSize = vBlockIdx == kPreBlockTotal - 1 ? kPreBlockTail : kPreBlockFactor;
        dkOffset = ((uint64_t)vBlockIdx) * kPreBlockFactor;
        initdvSize = vBlockIdx == vPreBlockTotal - 1 ? vPreBlockTail : vPreBlockFactor;
        dvOffset = ((uint64_t)vBlockIdx) * vPreBlockFactor;
        if (SPLIT_AXIS == BN2S2 && !(tilingData->preTilingData.sValueZeroUnderTND ||
                                     (IS_DETER_NEW(DETER_SPARSE_TYPE) && tilingData->preTilingData.hasInvalidCol))) {
            singleInitSize = PRE_INIT_UB_SIZE * NUM_THREE / sizeof(T2);
            dqLoopNum = Ceil(initdqSize, singleInitSize);
            dkLoopNum = 0;
            dvLoopNum = 0;
            dqBlockTail = initdqSize % singleInitSize == 0 ? singleInitSize : initdqSize % singleInitSize;
            dkBlockTail = 0;
            dvBlockTail = 0;
            pipe->InitBuffer(dqInitBuf, PRE_INIT_UB_SIZE * NUM_THREE);
            dqTensor = dqInitBuf.template Get<T2>();
        } else {
            singleInitSize = PRE_INIT_UB_SIZE / sizeof(T2);
            dqLoopNum = Ceil(initdqSize, singleInitSize);
            dkLoopNum = Ceil(initdkSize, singleInitSize);
            dvLoopNum = Ceil(initdvSize, singleInitSize);
            dqBlockTail = initdqSize % singleInitSize == 0 ? singleInitSize : initdqSize % singleInitSize;
            dkBlockTail = initdkSize % singleInitSize == 0 ? singleInitSize : initdkSize % singleInitSize;
            dvBlockTail = initdvSize % singleInitSize == 0 ? singleInitSize : initdvSize % singleInitSize;
            pipe->InitBuffer(dqInitBuf, PRE_INIT_UB_SIZE);
            pipe->InitBuffer(dkInitBuf, PRE_INIT_UB_SIZE);
            pipe->InitBuffer(dvInitBuf, PRE_INIT_UB_SIZE);
            dqTensor = dqInitBuf.template Get<T2>();
            dkTensor = dkInitBuf.template Get<T2>();
            dvTensor = dvInitBuf.template Get<T2>();
        }
        eventIDVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        pipe->InitBuffer(input1Que[0], 1, tilingData->preTilingData.sfmgYBufferLen);
        pipe->InitBuffer(input1Que[1], 1, tilingData->preTilingData.sfmgYBufferLen);
        pipe->InitBuffer(input2Que[0], 1, tilingData->preTilingData.sfmgDyBufferLen);
        pipe->InitBuffer(input2Que[1], 1, tilingData->preTilingData.sfmgDyBufferLen);
        pipe->InitBuffer(out1Que, 2, tilingData->preTilingData.sfmgOutputBufferLen);
    }
}

PRE_FUNCTION_TEMPLATE
__aicore__ inline void FlashAttentionScoreGradPresfmgRegbase<PRE_FUNCTION_ARGS_TEMPLATE>::Process()
{
    // process
    if (g_coreType == AIV && vBlockIdx < usedCoreNum) {
        // clear dq dk dv workspace
        if constexpr (IsSameType<T1, float>::value) {
            Duplicate<T2>(dqTensor, 0, singleInitSize);
            Duplicate<T2>(dkTensor, 0, singleInitSize);
            Duplicate<T2>(dvTensor, 0, singleInitSize);
        } else if constexpr (IsSameType<T1, half>::value || IsSameType<T1, bfloat16_t>::value) {
            Duplicate<T2>(dqTensor, 0, singleInitSize);
            if constexpr (SPLIT_AXIS == 0) {
                Duplicate<T2>(dkTensor, 0, singleInitSize);
                Duplicate<T2>(dvTensor, 0, singleInitSize);
            } else {
                if (tilingData->preTilingData.sValueZeroUnderTND ||
                    (IS_DETER_NEW(DETER_SPARSE_TYPE) && tilingData->preTilingData.hasInvalidCol)) {
                    // BN2S2针对TND中有S为0的场景 或 newDeter与无效列叠加场景，增加gm清零
                    Duplicate<T2>(dkTensor, 0, singleInitSize);
                    Duplicate<T2>(dvTensor, 0, singleInitSize);
                }
            }
        } else {
            if (tilingData->s1s2BNGS1S2SplitCoreParams.s2Outer > 1) {
                InitOutput<float>(dqWorkSpaceGm[dqOffset], initdqSize, 0);
            }
        }
        SetFlag<HardEvent::V_MTE3>(eventIDVToMte3);
        CalTempDimAlign();
        DoSoftmaxGrad();
    }
}

PRE_FUNCTION_TEMPLATE
__aicore__ inline void FlashAttentionScoreGradPresfmgRegbase<PRE_FUNCTION_ARGS_TEMPLATE>::CalTempDimAlign()
{
    tempDimAlign = HEAD_DIM_ALIGN;
}

PRE_FUNCTION_TEMPLATE
__aicore__ inline void FlashAttentionScoreGradPresfmgRegbase<PRE_FUNCTION_ARGS_TEMPLATE>::InitIndex(uint64_t startIdx,
                                                                                                    uint64_t &curS)
{
    if constexpr (IS_TND) {
        uint64_t totalLen = 0;
        for (uint64_t bDimIdx = bIdx; bDimIdx < b; bDimIdx++) {
            totalLen = n1 * ((__gm__ int64_t *)actual_seq_qlen_addr)[bDimIdx] * d;
            if (totalLen > startIdx) {
                bIdx = bDimIdx;
                curS = (bIdx == 0) ? ((__gm__ int64_t *)actual_seq_qlen_addr)[bIdx] :
                                     (((__gm__ int64_t *)actual_seq_qlen_addr)[bIdx] -
                                      ((__gm__ int64_t *)actual_seq_qlen_addr)[bIdx - 1]);
                uint64_t bTail = startIdx - (totalLen - n1 * curS * d);
                nIdx = bTail / (curS * d);
                uint64_t nTail = bTail % (curS * d);
                sIdx = nTail / d;
                break;
            }
        }
    } else {
        bIdx = startIdx / ns1d;
        uint64_t bTail = startIdx % ns1d;
        nIdx = bTail / s1d;
        uint64_t nTail = bTail % s1d;
        sIdx = nTail / d;
    }
}

PRE_FUNCTION_TEMPLATE
__aicore__ inline void FlashAttentionScoreGradPresfmgRegbase<PRE_FUNCTION_ARGS_TEMPLATE>::DoCopyIn(uint64_t curS,
                                                                                                   uint64_t curNBurst,
                                                                                                   uint64_t dstOffset)
{
    uint64_t srcOffset = 0;
    uint64_t transposeStride = 0;
    uint64_t transposeStrideB16 = 0;
    if constexpr (IS_TND) {
        uint64_t bOffset = bIdx == 0 ? 0 : n1 * ((__gm__ int64_t *)actual_seq_qlen_addr)[bIdx - 1] * d;
        srcOffset = bOffset + (sIdx * n1 + nIdx) * d;
        transposeStride = (nd - d) * sizeof(T1);
        transposeStrideB16 = (nd - d) * sizeof(OUTDTYPE);
    } else {
        if (layout == BNGSD) {
            srcOffset = bIdx * ns1d + nIdx * s1d + sIdx * d;
            transposeStride = 0;
            transposeStrideB16 = 0;
        } else if (layout == BSNGD) {
            srcOffset = bIdx * ns1d + sIdx * (nd) + nIdx * d;
            transposeStride = (nd - d) * sizeof(T1);
            transposeStrideB16 = (nd - d) * sizeof(OUTDTYPE);
        } else if (layout == SBNGD) {
            srcOffset = sIdx * bnd + bIdx * nd + nIdx * d;
            transposeStride = (bnd - d) * sizeof(T1);
            transposeStrideB16 = (bnd - d) * sizeof(OUTDTYPE);
        }
    }
    uint64_t dstBlockStrideB16 = (tempDimAlign - dAlignToBlockB16) * sizeof(OUTDTYPE) / BLOCK_BYTE_SIZE;
    uint64_t dstBlockStride = (tempDimAlign - dAlignToBlock) * sizeof(T1) / BLOCK_BYTE_SIZE;
    DataCopyPad(input1Buf[dstOffset], yGm[srcOffset],
                {static_cast<uint16_t>(curNBurst), static_cast<uint32_t>(d * sizeof(OUTDTYPE)),
                 static_cast<uint32_t>(transposeStrideB16), static_cast<uint32_t>(dstBlockStrideB16), 0},
                {true, 0, static_cast<uint8_t>((dAlignToBlockB16 - d)), 0});
    DataCopyPad(input2Buf[dstOffset].template ReinterpretCast<uint8_t>(),
                dxGm[srcOffset].template ReinterpretCast<uint8_t>(),
                {static_cast<uint16_t>(curNBurst), static_cast<uint32_t>(d * sizeof(T1)),
                 static_cast<uint32_t>(transposeStride), static_cast<uint32_t>(dstBlockStride), 0},
                {true, 0, static_cast<uint8_t>((dAlignToBlock - d)), 0});
}

PRE_FUNCTION_TEMPLATE
__aicore__ inline void FlashAttentionScoreGradPresfmgRegbase<PRE_FUNCTION_ARGS_TEMPLATE>::CopyInSfmg(
    uint64_t leftNburst, uint64_t &curS)
{
    uint64_t dstOffset = 0;
    while (leftNburst > 0) {
        uint64_t curNburst = 0;
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
                        curS = ((__gm__ int64_t *)actual_seq_qlen_addr)[bIdx] -
                               ((__gm__ int64_t *)actual_seq_qlen_addr)[bIdx - 1];
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

PRE_FUNCTION_TEMPLATE
__aicore__ inline void FlashAttentionScoreGradPresfmgRegbase<PRE_FUNCTION_ARGS_TEMPLATE>::CalculateSoftmaxGrad(
    uint64_t taskId, uint64_t sfmgOutputOffset, uint64_t curNBurst, uint64_t deqScaleIdx)
{
    LocalTensor<OUTDTYPE> yInTensor = input1Que[taskId & 1].DeQue<OUTDTYPE>();
    LocalTensor<T1> dxInTensor = input2Que[taskId & 1].DeQue<T1>();
    auto output1Buf = out1Que.AllocTensor<T2>();

    if constexpr (HEAD_DIM_ALIGN <= 256) {
        AscendC::MySoftmaxGradFrontCast<T1, T2, HEAD_DIM_ALIGN, HEAD_DIM_ALIGN>(
            output1Buf, yInTensor, dxInTensor, static_cast<uint32_t>(curNBurst),
            static_cast<uint32_t>(dAlignToBlockB16));
    } else {
        if (d <= 384) {
            AscendC::MySoftmaxGradFrontCast<T1, T2, 384, HEAD_DIM_ALIGN>(output1Buf, yInTensor, dxInTensor,
                                                                         static_cast<uint32_t>(curNBurst),
                                                                         static_cast<uint32_t>(dAlignToBlockB16));
        } else if (d <= 512) {
            AscendC::MySoftmaxGradFrontCast<T1, T2, 512, HEAD_DIM_ALIGN>(output1Buf, yInTensor, dxInTensor,
                                                                         static_cast<uint32_t>(curNBurst),
                                                                         static_cast<uint32_t>(dAlignToBlockB16));
        } else if (d <= 640) {
            AscendC::MySoftmaxGradFrontCast<T1, T2, 640, HEAD_DIM_ALIGN>(output1Buf, yInTensor, dxInTensor,
                                                                         static_cast<uint32_t>(curNBurst),
                                                                         static_cast<uint32_t>(dAlignToBlockB16));
        } else {
            AscendC::MySoftmaxGradFrontCast<T1, T2, HEAD_DIM_ALIGN, HEAD_DIM_ALIGN>(
                output1Buf, yInTensor, dxInTensor, static_cast<uint32_t>(curNBurst),
                static_cast<uint32_t>(dAlignToBlockB16));
        }
    }
    out1Que.EnQue(output1Buf);
    out1Que.DeQue<T2>();
    DataCopyPad(sfmgWorkspaceGm[sfmgOutputOffset], output1Buf,
                {1, static_cast<uint32_t>(curNBurst * sizeof(T2)), 0, 0, 0});
    out1Que.FreeTensor(output1Buf);
}

PRE_FUNCTION_TEMPLATE
template <uint8_t GM_IDX>
__aicore__ inline void FlashAttentionScoreGradPresfmgRegbase<PRE_FUNCTION_ARGS_TEMPLATE>::CopyDqkvToGm(
    const uint64_t &loopIdx)
{
    if constexpr (IsSameType<T1, float>::value) {
        if constexpr (GM_IDX == DQ_IDX) {
            if (loopIdx == dqLoopNum - 1) {
                DataCopyPad(dqGm[dqOffset + loopIdx * singleInitSize], dqTensor,
                            {1, static_cast<uint32_t>(dqBlockTail * sizeof(T2)), 0, 0, 0});
            } else {
                DataCopy(dqGm[dqOffset + loopIdx * singleInitSize], dqTensor, singleInitSize);
            }
        } else if constexpr (GM_IDX == DK_IDX) {
            if (loopIdx == dkLoopNum - 1) {
                DataCopyPad(dkGm[dkOffset + loopIdx * singleInitSize], dkTensor,
                            {1, static_cast<uint32_t>(dkBlockTail * sizeof(T2)), 0, 0, 0});
            } else {
                DataCopy(dkGm[dkOffset + loopIdx * singleInitSize], dkTensor, singleInitSize);
            }
        } else {
            if (loopIdx == dvLoopNum - 1) {
                DataCopyPad(dvGm[dvOffset + loopIdx * singleInitSize], dvTensor,
                            {1, static_cast<uint32_t>(dvBlockTail * sizeof(T2)), 0, 0, 0});
            } else {
                DataCopy(dvGm[dvOffset + loopIdx * singleInitSize], dvTensor, singleInitSize);
            }
        }
    } else {
        if constexpr (GM_IDX == DQ_IDX) {
            if (loopIdx == dqLoopNum - 1) {
                DataCopyPad(dqWorkSpaceGm[dqOffset + loopIdx * singleInitSize], dqTensor,
                            {1, static_cast<uint32_t>(dqBlockTail * sizeof(T2)), 0, 0, 0});
            } else {
                DataCopy(dqWorkSpaceGm[dqOffset + loopIdx * singleInitSize], dqTensor, singleInitSize);
            }
        } else if constexpr (GM_IDX == DK_IDX) {
            if (loopIdx == dkLoopNum - 1) {
                DataCopyPad(dkWorkSpaceGm[dkOffset + loopIdx * singleInitSize], dkTensor,
                            {1, static_cast<uint32_t>(dkBlockTail * sizeof(T2)), 0, 0, 0});
            } else {
                DataCopy(dkWorkSpaceGm[dkOffset + loopIdx * singleInitSize], dkTensor, singleInitSize);
            }
        } else {
            if (loopIdx == dvLoopNum - 1) {
                DataCopyPad(dvWorkSpaceGm[dvOffset + loopIdx * singleInitSize], dvTensor,
                            {1, static_cast<uint32_t>(dvBlockTail * sizeof(T2)), 0, 0, 0});
            } else {
                DataCopy(dvWorkSpaceGm[dvOffset + loopIdx * singleInitSize], dvTensor, singleInitSize);
            }
        }
    }
}

PRE_FUNCTION_TEMPLATE
__aicore__ inline void FlashAttentionScoreGradPresfmgRegbase<PRE_FUNCTION_ARGS_TEMPLATE>::DoSoftmaxGrad()
{
    // process
    if (vBlockIdx < usedCoreNum) {
        uint64_t singleCoreLoopTimes, singleCoreLastLoopNBurstNum;
        if (vBlockIdx == usedCoreNum - 1) {
            singleCoreLoopTimes = tilingData->preTilingData.tailCoreLoopTimes;                 // 尾核loop次数
            singleCoreLastLoopNBurstNum = tilingData->preTilingData.tailCoreLastLoopNBurstNum; // 尾核最后一次处理s1大小
        } else {
            singleCoreLoopTimes = tilingData->preTilingData.normalCoreLoopTimes; // 非尾核loop次数
            singleCoreLastLoopNBurstNum =
                tilingData->preTilingData.normalCoreLastLoopNBurstNum; // 非尾核最后一次处理s1大小
        }

        uint64_t sfmgOutputOffset = vBlockIdx * tilingData->preTilingData.normalCoreNBurstNums; // 核间起始地址
        uint64_t nBurst = tilingData->preTilingData.singleLoopNBurstNum; // 一次普通loop处理多少个D
        uint64_t curS = s1;
        uint64_t taskId = 0;
        if constexpr (IS_D_NO_EQUAL) {
            input1Buf = input1Que[0].AllocTensor<OUTDTYPE>();
            input2Buf = input2Que[0].AllocTensor<T1>();
            Duplicate<T1>(input1Buf, 0, nBurst * tempDimAlign);
            Duplicate<T1>(input2Buf, 0, nBurst * tempDimAlign);
            input1Que[0].FreeTensor(input1Buf);
            input2Que[0].FreeTensor(input2Buf);

            input1Buf = input1Que[1].AllocTensor<OUTDTYPE>();
            input2Buf = input2Que[1].AllocTensor<T1>();
            Duplicate<OUTDTYPE>(input1Buf, 0, nBurst * tempDimAlign);
            Duplicate<T1>(input2Buf, 0, nBurst * tempDimAlign);
            input1Que[1].FreeTensor(input1Buf);
            input2Que[1].FreeTensor(input2Buf);
        }
        uint64_t commonLoopTimes = Min(Min(Min(dqLoopNum, dkLoopNum), dvLoopNum), singleCoreLoopTimes);
        WaitFlag<HardEvent::V_MTE3>(eventIDVToMte3);
        for (uint64_t i = 0; i < commonLoopTimes; i++) {
            if (i == singleCoreLoopTimes - 1) {
                nBurst = singleCoreLastLoopNBurstNum;
            }

            input1Buf = input1Que[taskId & 1].AllocTensor<OUTDTYPE>();
            input2Buf = input2Que[taskId & 1].AllocTensor<T1>();
            // sync
            InitIndex(sfmgOutputOffset * d, curS);
            CopyInSfmg(nBurst, curS);
            input1Que[taskId & 1].EnQue(input1Buf);
            input2Que[taskId & 1].EnQue(input2Buf);
            CalculateSoftmaxGrad(taskId, sfmgOutputOffset, nBurst);
            CopyDqkvToGm<DQ_IDX>(i);
            CopyDqkvToGm<DK_IDX>(i);
            CopyDqkvToGm<DV_IDX>(i);

            if constexpr (IS_D_NO_EQUAL != 0) {
                if (i < singleCoreLoopTimes - 1) {
                    Duplicate<OUTDTYPE>(input1Buf, 0, nBurst * tempDimAlign);
                    Duplicate<T1>(input2Buf, 0, nBurst * tempDimAlign);
                }
            }
            input1Que[taskId & 1].FreeTensor(input1Buf);
            input2Que[taskId & 1].FreeTensor(input2Buf);
            sfmgOutputOffset += tilingData->preTilingData.singleLoopNBurstNum;
            taskId++;
        }
        for (uint64_t i = commonLoopTimes; i < singleCoreLoopTimes; i++) {
            if (i == singleCoreLoopTimes - 1) {
                nBurst = singleCoreLastLoopNBurstNum;
            }

            input1Buf = input1Que[taskId & 1].AllocTensor<OUTDTYPE>();
            input2Buf = input2Que[taskId & 1].AllocTensor<T1>();
            // sync
            InitIndex(sfmgOutputOffset * d, curS);
            CopyInSfmg(nBurst, curS);

            input1Que[taskId & 1].EnQue(input1Buf);
            input2Que[taskId & 1].EnQue(input2Buf);
            CalculateSoftmaxGrad(taskId, sfmgOutputOffset, nBurst);
            if constexpr (IS_D_NO_EQUAL != 0) {
                if (i < singleCoreLoopTimes - 1) {
                    Duplicate<OUTDTYPE>(input1Buf, 0, nBurst * tempDimAlign);
                    Duplicate<T1>(input2Buf, 0, nBurst * tempDimAlign);
                }
            }
            input1Que[taskId & 1].FreeTensor(input1Buf);
            input2Que[taskId & 1].FreeTensor(input2Buf);
            sfmgOutputOffset += tilingData->preTilingData.singleLoopNBurstNum;
            taskId++;
        }
        for (uint64_t i = commonLoopTimes; i < dqLoopNum; i++) {
            CopyDqkvToGm<DQ_IDX>(i);
        }
        for (uint64_t i = commonLoopTimes; i < dkLoopNum; i++) {
            CopyDqkvToGm<DK_IDX>(i);
        }
        for (uint64_t i = commonLoopTimes; i < dvLoopNum; i++) {
            CopyDqkvToGm<DV_IDX>(i);
        }
    }
}

PRE_FUNCTION_TEMPLATE
__aicore__ inline void FlashAttentionScoreGradPresfmgRegbase<PRE_FUNCTION_ARGS_TEMPLATE>::SyncALLCores()
{
    SyncAll<false>();
}
#endif // _FLASH_ATTENTION_SCORE_GRAD_PRE_SFMG_KERNEL_REGBASE_H_
