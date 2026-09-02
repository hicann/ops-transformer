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
 * \file vec_op.h
 * \brief
 */

#pragma once
#include "kernel_operator.h"
#include "nsa_selected_attention_grad_common_header.h"

namespace NSAG_BASIC {
struct StaticParams {
    uint32_t singleM = 0;
    uint32_t singleN = 0;
    uint32_t sftBaseM = 0;
    uint32_t sftBaseN = 0;
};

template <typename NSAGT>
class VecOp {
public:
    using TILING_CLASS = typename NSAGT::tiling_class;
    using T1 = typename NSAGT::t1;
    static constexpr uint32_t ATTEN_ENABLE = NSAGT::atten_enable;

    __aicore__ inline VecOp(){};
    __aicore__ inline void Init(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR attention_out,
                                GM_ADDR attention_out_grad, GM_ADDR softmax_max, GM_ADDR softmax_sum,
                                GM_ADDR topk_indices, GM_ADDR actual_seq_qlen, GM_ADDR actual_seq_kvlen,
                                GM_ADDR atten_mask, GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR workspace,
                                const TILING_CLASS *__restrict ordTilingData, TPipe *pipe);
    __aicore__ inline void Process(const int64_t dyGmOffset, const int64_t sumGmOffset, const uint64_t indicesGmOffset,
                                   const int64_t s1Index, const int32_t blkCntOffset, int64_t mm12Addr,
                                   int64_t mm345Addr);

protected:
    __aicore__ inline void InitParams(const TILING_CLASS *__restrict ordTilingData, GM_ADDR actual_seq_qlen,
                                      GM_ADDR actual_seq_kvlen);
    __aicore__ inline void InitGMBuffer(GM_ADDR attention_out, GM_ADDR attention_out_grad, GM_ADDR softmax_max,
                                        GM_ADDR softmax_sum, GM_ADDR topk_indices, GM_ADDR workspace);
    __aicore__ inline void InitUB(TPipe *pipe);
    __aicore__ inline void AtomicClean();
    __aicore__ inline void DumpGmZero(GlobalTensor<float> &gm, int64_t num);
    __aicore__ inline void CalRowsumAndSftCopyIn(const int64_t dyGmOffset, const int64_t sumGmOffset);
    __aicore__ inline void CalAttenMsk(const uint64_t indicesGmOffset, const int64_t s1Index, const int32_t processM,
                                       const int32_t blkCntOffset);
    __aicore__ inline void CalSoftmax(const int32_t loopIdx, const int32_t processM, const int64_t mm12Addr,
                                      const int64_t mm345Addr, const uint64_t indicesGmOffset, const int64_t s1Index,
                                      const int32_t blkCntOffset);
    __aicore__ inline void CalSoftmaxGrad(const int32_t loopIdx, const int32_t processM, const int64_t mm12Addr,
                                          const int64_t mm345Addr);
    __aicore__ inline void CalSub(LocalTensor<float> &dstTensor, LocalTensor<float> &srcTensor, int32_t baseM,
                                  int32_t baseN);

protected:
    // core info
    int64_t usedCoreNum;
    int64_t formerCoreNum;
    uint32_t cubeBlockIdx;
    uint32_t vecBlockIdx;
    StaticParams params;
    GlobalTensor<T1> attentionGm;
    GlobalTensor<T1> attentionGradGm;
    GlobalTensor<float> softmaxMaxGm;
    GlobalTensor<float> softmaxSumGm;
    GlobalTensor<int32_t> topkIndicesGm;
    // workspace
    GlobalTensor<float> mm1WorkspaceGm;
    GlobalTensor<float> mm2WorkspaceGm;
    GlobalTensor<T1> pWorkspaceGm;
    GlobalTensor<T1> dsWorkspaceGm;
    GlobalTensor<float> dqWorkspaceGm;
    GlobalTensor<float> dkWorkspaceGm;
    GlobalTensor<float> dvWorkspaceGm;

    TBuf<> vecQue;
    LocalTensor<uint8_t> helpTensor;
    LocalTensor<int32_t> topkIndicesTensor;
    LocalTensor<float> rowSumOutTensor;
    LocalTensor<float> maxTensor;
    LocalTensor<float> sumTensor;
    LocalTensor<T1> attentionT1Tensor;
    LocalTensor<float> attentionFP32Tensor;
    LocalTensor<T1> attentionGradT1Tensor;
    LocalTensor<float> attentionGradFP32Tensor;
    LocalTensor<float> pTensor;
    LocalTensor<float> dPTensor;
    LocalTensor<T1> sftOutT1Tensor;
    LocalTensor<T1> sftgOutT1Tensor;

    constexpr static const int32_t BLOCK = 32;
    constexpr static const int32_t BLOCK_T1 = BLOCK / sizeof(T1);
    constexpr static const int32_t BLOCK_FP32 = BLOCK / sizeof(float);
    constexpr static const int32_t BLOCK_INT32 = BLOCK / sizeof(int32_t);
    constexpr static const int32_t MSK_LEN = 64;
    constexpr static const uint32_t ATTEN_MASK_SCALE = 0xFF7FFFFF;

    const TILING_CLASS *__restrict tilingData;
    // Shape
    int64_t dimB;
    int64_t dimN2;
    int64_t dimS1;
    int64_t dimS2;
    int64_t dimG;
    int64_t dimD;
    int64_t dimD2;
    int64_t dimDqk; // Dqk = D
    int64_t dimDv;  // Dv = D2
    int64_t dimDAlign;
    int64_t dimD2Align;
    int64_t dimTkv{0};
    int64_t dimTq{0};
    int64_t selectedS2{0};
    float scaleValue;
    uint32_t selectedBlockCount;
    uint32_t selectedBlockSize;
    // workspace
    uint32_t mm12WorkspaceLen;
    int64_t dqWorkspaceLen;
    int64_t dkWorkspaceLen;
    int64_t dvWorkspaceLen;
    event_t vWaitMte2;
    event_t mte3WaitMte2;
    event_t mte3WaitV;
    event_t sWaitMte2;
    event_t mte2WaitMte3;
};

template <typename NSAGT>
__aicore__ inline void VecOp<NSAGT>::Init(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR attention_out,
                                          GM_ADDR attention_out_grad, GM_ADDR softmax_max, GM_ADDR softmax_sum,
                                          GM_ADDR topk_indices, GM_ADDR actual_seq_qlen, GM_ADDR actual_seq_kvlen,
                                          GM_ADDR atten_mask, GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR workspace,
                                          const TILING_CLASS *__restrict ordTilingData, TPipe *pipe)
{
    InitParams(ordTilingData, actual_seq_qlen, actual_seq_kvlen);
    InitGMBuffer(attention_out, attention_out_grad, softmax_max, softmax_sum, topk_indices, workspace);
    InitUB(pipe);
    AtomicClean();
}

template <typename NSAGT>
__aicore__ inline void VecOp<NSAGT>::InitParams(const TILING_CLASS *__restrict ordTilingData, GM_ADDR actual_seq_qlen,
                                                GM_ADDR actual_seq_kvlen)
{
    cubeBlockIdx = GetBlockIdx() / 2;
    vecBlockIdx = GetBlockIdx();
    tilingData = ordTilingData;
    usedCoreNum = tilingData->opInfo.usedCoreNum;
    formerCoreNum = tilingData->opInfo.formerCoreNum;

    dimB = tilingData->opInfo.B;
    dimN2 = tilingData->opInfo.N2;
    dimS1 = tilingData->opInfo.S1;
    dimS2 = tilingData->opInfo.S2;
    dimG = tilingData->opInfo.G;
    dimD = tilingData->opInfo.D;
    dimD2 = tilingData->opInfo.D2;
    dimDqk = tilingData->opInfo.D;
    dimDv = tilingData->opInfo.D2;
    dimDAlign = (dimD + BLOCK_T1 - 1) / BLOCK_T1 * BLOCK_T1;
    dimD2Align = (dimD2 + BLOCK_T1 - 1) / BLOCK_T1 * BLOCK_T1;

    scaleValue = tilingData->opInfo.scaleValue;
    selectedBlockCount = tilingData->opInfo.selectedBlockCount;
    selectedBlockSize = tilingData->opInfo.selectedBlockSize;
    mm12WorkspaceLen = tilingData->opInfo.mm12WorkspaceLen;
    dqWorkspaceLen = tilingData->opInfo.dqWorkspaceLen;
    dkWorkspaceLen = tilingData->opInfo.dkWorkspaceLen;
    dvWorkspaceLen = tilingData->opInfo.dvWorkspaceLen;

    params.singleM = tilingData->splitCoreParams.singleM;
    params.singleN = tilingData->splitCoreParams.singleN;
    params.sftBaseM = tilingData->splitCoreParams.sftBaseM;
    params.sftBaseN = tilingData->splitCoreParams.sftBaseN;

    selectedS2 = selectedBlockCount * selectedBlockSize;

    for (int64_t i = 0; i < dimB; i++) {
        int64_t seqS1Len = 0;
        int64_t seqS2Len = 0;
        if (unlikely(i == 0)) {
            seqS1Len = ((__gm__ int64_t *)actual_seq_qlen)[i];
            seqS2Len = ((__gm__ int64_t *)actual_seq_kvlen)[i];
        } else {
            seqS1Len = ((__gm__ int64_t *)actual_seq_qlen)[i] - ((__gm__ int64_t *)actual_seq_qlen)[i - 1];
            seqS2Len = ((__gm__ int64_t *)actual_seq_kvlen)[i] - ((__gm__ int64_t *)actual_seq_kvlen)[i - 1];
        }
        dimTq += (int64_t)seqS1Len;
        dimTkv += (int64_t)seqS2Len;
    }
}

template <typename NSAGT>
__aicore__ inline void VecOp<NSAGT>::InitGMBuffer(GM_ADDR attention_out, GM_ADDR attention_out_grad,
                                                  GM_ADDR softmax_max, GM_ADDR softmax_sum, GM_ADDR topk_indices,
                                                  GM_ADDR workspace)
{
    /*
     * 初始化输入
     */
    attentionGm.SetGlobalBuffer((__gm__ T1 *)attention_out);
    attentionGradGm.SetGlobalBuffer((__gm__ T1 *)attention_out_grad);
    softmaxMaxGm.SetGlobalBuffer((__gm__ float *)softmax_max);
    softmaxSumGm.SetGlobalBuffer((__gm__ float *)softmax_sum);
    topkIndicesGm.SetGlobalBuffer((__gm__ int32_t *)topk_indices);

    /*
     * 初始化workspace
     */
    int64_t usedWorkspaceLen = 0;
    // mm1 与 p 复用workspace
    int64_t mm1Addr = usedWorkspaceLen / sizeof(float) + cubeBlockIdx * mm12WorkspaceLen / sizeof(float);
    int64_t pAddr = usedWorkspaceLen / sizeof(T1) + cubeBlockIdx * mm12WorkspaceLen / sizeof(T1);
    usedWorkspaceLen += mm12WorkspaceLen * usedCoreNum;
    // mm2 与 ds 复用workspace
    int64_t mm2Addr = usedWorkspaceLen / sizeof(float) + cubeBlockIdx * mm12WorkspaceLen / sizeof(float);
    int64_t dsAddr = usedWorkspaceLen / sizeof(T1) + cubeBlockIdx * mm12WorkspaceLen / sizeof(T1);
    usedWorkspaceLen += mm12WorkspaceLen * usedCoreNum;
    // post
    int64_t dqAddr = usedWorkspaceLen / sizeof(float);
    int64_t dkAddr = dqAddr + dqWorkspaceLen / sizeof(float);
    int64_t dvAddr = dkAddr + dkWorkspaceLen / sizeof(float);
    usedWorkspaceLen += dqWorkspaceLen + dkWorkspaceLen + dvWorkspaceLen;

    mm1WorkspaceGm.SetGlobalBuffer((__gm__ float *)workspace + mm1Addr);
    mm2WorkspaceGm.SetGlobalBuffer((__gm__ float *)workspace + mm2Addr);
    pWorkspaceGm.SetGlobalBuffer((__gm__ T1 *)workspace + pAddr);
    dsWorkspaceGm.SetGlobalBuffer((__gm__ T1 *)workspace + dsAddr);
    dqWorkspaceGm.SetGlobalBuffer((__gm__ float *)workspace + dqAddr);
    dkWorkspaceGm.SetGlobalBuffer((__gm__ float *)workspace + dkAddr);
    dvWorkspaceGm.SetGlobalBuffer((__gm__ float *)workspace + dvAddr);
}

template <typename NSAGT>
__aicore__ inline void VecOp<NSAGT>::InitUB(TPipe *pipe)
{
    uint32_t dataSize = params.singleM * dimDv;
    uint32_t ubOffset = 0;
    uint32_t rowsumUbOffset = 0;
    uint32_t totalUbSpace = 191 * 1024;

    pipe->InitBuffer(vecQue, totalUbSpace);

    // topk
    int32_t topkNumber = AlignTo<int32_t>(selectedBlockCount, BLOCK);
    topkIndicesTensor = vecQue.GetWithOffset<int32_t>(topkNumber, ubOffset);
    ubOffset += topkNumber * sizeof(int32_t);

    // rowsum out
    rowSumOutTensor = vecQue.GetWithOffset<float>(dimG * BLOCK_FP32, ubOffset);
    ubOffset += dimG * BLOCK_FP32 * sizeof(float);
    maxTensor = vecQue.GetWithOffset<float>(dimG * BLOCK_FP32, ubOffset);
    ubOffset += dimG * BLOCK_FP32 * sizeof(float);
    sumTensor = vecQue.GetWithOffset<float>(dimG * BLOCK_FP32, ubOffset);
    ubOffset += dimG * BLOCK_FP32 * sizeof(float);
    rowsumUbOffset = ubOffset;

    // rowsum cal
    attentionGradT1Tensor = vecQue.GetWithOffset<T1>(dimG * dimDv, ubOffset);
    ubOffset += dimG * dimDv * sizeof(T1);
    attentionGradFP32Tensor = vecQue.GetWithOffset<float>(dimG * dimDv, ubOffset);
    ubOffset += dimG * dimDv * sizeof(float);
    attentionT1Tensor = vecQue.GetWithOffset<T1>(dimG * dimDv, ubOffset);
    ubOffset += dimG * dimDv * sizeof(T1);
    attentionFP32Tensor = vecQue.GetWithOffset<float>(dimG * dimDv, ubOffset);
    ubOffset += dimG * dimDv * sizeof(float);

    // sftGard
    int32_t sftDataSize = params.sftBaseM * params.sftBaseN;
    pTensor = vecQue.GetWithOffset<float>(sftDataSize, ubOffset);
    ubOffset += sftDataSize * sizeof(float);
    dPTensor = vecQue.GetWithOffset<float>(sftDataSize, ubOffset);
    ubOffset += sftDataSize * sizeof(float);

    sftOutT1Tensor = vecQue.GetWithOffset<T1>(sftDataSize, ubOffset);
    ubOffset += sftDataSize * sizeof(T1);
    sftgOutT1Tensor = vecQue.GetWithOffset<T1>(sftDataSize, ubOffset);
    ubOffset += sftDataSize * sizeof(T1);

    helpTensor = vecQue.GetWithOffset<uint8_t>((totalUbSpace - rowsumUbOffset) / sizeof(uint8_t), rowsumUbOffset);

    uint32_t attentionShape[2] = {params.singleM, static_cast<uint32_t>(dimDv)};
    uint32_t softmaxGradShape[2] = {params.singleM, BLOCK_FP32};
    attentionGradFP32Tensor.SetShapeInfo(ShapeInfo(2, attentionShape, DataFormat::ND));
    attentionFP32Tensor.SetShapeInfo(ShapeInfo(2, attentionShape, DataFormat::ND));
    rowSumOutTensor.SetShapeInfo(ShapeInfo(2, softmaxGradShape, DataFormat::ND));
    rowSumOutTensor.SetSize(params.singleM * BLOCK_FP32);
    attentionGradT1Tensor.SetSize(dataSize);
    attentionGradFP32Tensor.SetSize(dataSize);
    attentionT1Tensor.SetSize(dataSize);
    attentionFP32Tensor.SetSize(dataSize);
    // params
    sumTensor.SetSize(params.singleM * 8);
    maxTensor.SetSize(params.singleM * 8);

    // set shape
    uint32_t softmaxShape[2] = {static_cast<uint32_t>(params.singleM), 8};
    sumTensor.SetShapeInfo(ShapeInfo(2, softmaxShape, DataFormat::ND));
    maxTensor.SetShapeInfo(ShapeInfo(2, softmaxShape, DataFormat::ND));

    uint32_t dstSoftShape[2] = {static_cast<uint32_t>(params.sftBaseM), static_cast<uint32_t>(params.sftBaseN)};
    pTensor.SetShapeInfo(ShapeInfo(2, dstSoftShape, DataFormat::ND));
    mte2WaitMte3 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>());
    mte3WaitMte2 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE3>());
    sWaitMte2 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_S>());
    vWaitMte2 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
    mte3WaitV = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());
}

template <typename NSAGT>
__aicore__ inline void VecOp<NSAGT>::AtomicClean()
{
    int64_t dqSize, dkSize, dvSize;
    dqSize = dimTq * dimN2 * dimG * dimDAlign;
    dkSize = dimTkv * dimN2 * dimDAlign;
    dvSize = dimTkv * dimN2 * dimD2Align;
    dqSize = (dqSize + BLOCK_INT32 - 1) / BLOCK_INT32 * BLOCK_INT32;
    dkSize = (dkSize + BLOCK_INT32 - 1) / BLOCK_INT32 * BLOCK_INT32;
    dvSize = (dvSize + BLOCK_INT32 - 1) / BLOCK_INT32 * BLOCK_INT32;
    DumpGmZero(dqWorkspaceGm, dqSize);
    DumpGmZero(dkWorkspaceGm, dkSize);
    DumpGmZero(dvWorkspaceGm, dvSize); // FP32 FUNC:T1 FP16
}

template <typename NSAGT>
__aicore__ inline void VecOp<NSAGT>::DumpGmZero(GlobalTensor<float> &gm, int64_t num)
{
    int64_t perSize = (num + tilingData->opInfo.castUsedCoreNum - 1) / tilingData->opInfo.castUsedCoreNum;
    int64_t coreNum = (num + perSize - 1) / perSize;
    int64_t tailSize = num - perSize * (coreNum - 1);
    int64_t initSize = perSize;

    if (vecBlockIdx == coreNum - 1) {
        initSize = tailSize;
    }

    if (vecBlockIdx < coreNum) {
        InitOutput<float>(gm[vecBlockIdx * perSize], initSize, 0);
    }
}

template <typename NSAGT>
__aicore__ inline void VecOp<NSAGT>::CalSub(LocalTensor<float> &dstTensor, LocalTensor<float> &srcTensor, int32_t baseM,
                                            int32_t baseN)
{
    constexpr static int32_t subMsk = 64;
    int8_t s2Repeat = baseN / subMsk;
    int8_t s2RepeatTail = baseN % subMsk;

    for (int32_t i = 0; i < baseM; i++) {
        Sub(dstTensor[i * baseN], dstTensor[i * baseN], srcTensor[i * BLOCK_FP32], subMsk, s2Repeat,
            {1, 1, 0, 8, 8, 0});
    }

    if (s2RepeatTail) {
        for (int32_t i = 0; i < baseM; i++) {
            Sub(dstTensor[s2Repeat * subMsk + i * baseN], dstTensor[s2Repeat * subMsk + i * baseN],
                srcTensor[i * BLOCK_FP32], s2RepeatTail, 1, {1, 1, 0, 8, 8, 0});
        }
    }
}

template <typename NSAGT>
__aicore__ inline void VecOp<NSAGT>::CalRowsumAndSftCopyIn(const int64_t dyGmOffset, const int64_t sumGmOffset)
{
    uint32_t dataSize = params.singleM * dimDv;
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(dimG * dimDv * sizeof(T1)), 0, 0, 0};
    DataCopyExtParams copyParams2{1, static_cast<uint32_t>(dimG * BLOCK), 0, 0, 0};
    DataCopyPadExtParams<T1> padParams{false, 0, 0, 0};
    DataCopyPadExtParams<float> padParams2{false, 0, 0, 0};

    DataCopyPad(attentionGradT1Tensor, attentionGradGm[dyGmOffset], copyParams, padParams);
    AscendC::SetFlag<HardEvent::MTE2_V>(static_cast<int32_t>(vWaitMte2));
    DataCopyPad(attentionT1Tensor, attentionGm[dyGmOffset], copyParams, padParams);
    AscendC::SetFlag<HardEvent::MTE2_V>(static_cast<int32_t>(vWaitMte2));
    DataCopyPad(maxTensor, softmaxMaxGm[sumGmOffset], copyParams2, padParams2);
    DataCopyPad(sumTensor, softmaxSumGm[sumGmOffset], copyParams2, padParams2);
    AscendC::SetFlag<HardEvent::MTE2_V>(static_cast<int32_t>(vWaitMte2));
    AscendC::WaitFlag<HardEvent::MTE2_V>(static_cast<int32_t>(vWaitMte2));
    Cast(attentionGradFP32Tensor, attentionGradT1Tensor, RoundMode::CAST_NONE, dataSize);
    AscendC::WaitFlag<HardEvent::MTE2_V>(static_cast<int32_t>(vWaitMte2));
    Cast(attentionFP32Tensor, attentionT1Tensor, RoundMode::CAST_NONE, dataSize);
    AscendC::PipeBarrier<PIPE_V>();
    SoftmaxGradFront<float, false>(rowSumOutTensor, attentionGradFP32Tensor, attentionFP32Tensor, helpTensor,
                                   tilingData->softmaxGradTilingData);
}

template <typename NSAGT>
__aicore__ inline void VecOp<NSAGT>::CalAttenMsk(const uint64_t indicesGmOffset, const int64_t s1Index,
                                                 const int32_t processM, const int32_t blkCntOffset)
{ /*
   * 计算attenmask时对应blockSize需要保留的长度。
   * 例如：对于blockSize=64的块，需要保留0-9块，则attenMskRsvLen=10；
   */
    int32_t attenMskRsvLen = 0;
    int32_t attenMskStartIdx = -1; // 计算attenmask时attenmask开始计算的下标
    float scalar = *((float *)&ATTEN_MASK_SCALE);
    uint64_t mask[1];
    LocalTensor<float> tmpTensor;

    for (int32_t i = blkCntOffset; i < blkCntOffset + 8; i++) {
        int32_t topkIdx = topkIndicesGm[indicesGmOffset].GetValue(i);
        // 处于对角线上的block
        if (topkIdx * selectedBlockSize <= s1Index && (topkIdx + 1) * selectedBlockSize > s1Index) {
            attenMskRsvLen = s1Index + 1 - (topkIdx * selectedBlockSize);
            attenMskStartIdx = i * selectedBlockSize + attenMskRsvLen - blkCntOffset * selectedBlockSize;
        }
    }

    if (attenMskStartIdx == -1) {
        if (s1Index + 1 <= params.sftBaseN) {
            for (int32_t i = 0; i < processM; i++) {
                int32_t pOffset = i * params.sftBaseN;
                Duplicate(pTensor[pOffset], scalar, params.sftBaseN);
            }
        }
        return;
    }

    for (int32_t i = 0; i < processM; i++) {
        int32_t pOffset = i * params.sftBaseN;
        int32_t selectIdx = (attenMskStartIdx - 1) / selectedBlockSize;

        if (attenMskRsvLen != MSK_LEN) {
            tmpTensor = pTensor[selectIdx * selectedBlockSize + pOffset];
            mask[0] = UINT64_MAX - ((1ULL << attenMskRsvLen) - 1);
            Duplicate(tmpTensor, scalar, mask, 1, 0, 0);
        }

        if (s1Index + 1 < selectedS2) {
            uint32_t mskLen = params.sftBaseN - (selectIdx + 1) * selectedBlockSize;
            if (mskLen != 0) {
                tmpTensor = pTensor[(selectIdx + 1) * selectedBlockSize + pOffset];
                Duplicate(tmpTensor, scalar, mskLen);
            }
        }
    }
}

template <typename NSAGT>
__aicore__ inline void VecOp<NSAGT>::CalSoftmax(const int32_t loopIdx, const int32_t processM, const int64_t mm12Addr,
                                                const int64_t mm345Addr, const uint64_t indicesGmOffset,
                                                const int64_t s1Index, const int32_t blkCntOffset)
{
    int64_t dataSize = static_cast<int64_t>(processM) * static_cast<int64_t>(params.sftBaseN);
    int64_t maxOffset = loopIdx * params.sftBaseM * BLOCK_FP32;
    auto tmpMaxTensor = maxTensor[maxOffset];
    auto tmpSumTensor = sumTensor[maxOffset];

    DataCopyPad(pTensor, mm1WorkspaceGm[mm12Addr], {1, static_cast<uint32_t>(dataSize * sizeof(float)), 0, 0, 0},
                {false, 0, 0, 0});
    AscendC::SetFlag<HardEvent::MTE2_V>(static_cast<int32_t>(vWaitMte2));
    AscendC::WaitFlag<HardEvent::MTE2_V>(static_cast<int32_t>(vWaitMte2));

    Muls(pTensor, pTensor, scaleValue, dataSize);
    AscendC::PipeBarrier<PIPE_V>();

    if constexpr (ATTEN_ENABLE) {
        CalAttenMsk(indicesGmOffset, s1Index, processM, blkCntOffset);
        AscendC::PipeBarrier<PIPE_V>();
    }

    SimpleSoftMax<float, true, false>(pTensor, tmpSumTensor, tmpMaxTensor, pTensor, helpTensor,
                                      tilingData->softmaxTilingData);
    AscendC::PipeBarrier<PIPE_V>();
    Cast(sftOutT1Tensor, pTensor, RoundMode::CAST_ROUND, dataSize);
    AscendC::SetFlag<HardEvent::V_MTE3>(static_cast<int32_t>(mte3WaitV));
    AscendC::WaitFlag<HardEvent::V_MTE3>(static_cast<int32_t>(mte3WaitV));

    DataCopyPad(pWorkspaceGm[mm345Addr], sftOutT1Tensor, {1, static_cast<uint32_t>(dataSize * sizeof(T1)), 0, 0, 0});
    AscendC::SetFlag<HardEvent::MTE3_MTE2>(static_cast<int32_t>(mte2WaitMte3));
    AscendC::WaitFlag<HardEvent::MTE3_MTE2>(static_cast<int32_t>(mte2WaitMte3));
}

template <typename NSAGT>
__aicore__ inline void VecOp<NSAGT>::CalSoftmaxGrad(const int32_t loopIdx, const int32_t processM,
                                                    const int64_t mm12Addr, const int64_t mm345Addr)
{
    int64_t dataSize = processM * params.sftBaseN;
    int64_t rowsumOffset = loopIdx * params.sftBaseM * BLOCK_FP32;
    auto tmpRowSumOutTensor = rowSumOutTensor[rowsumOffset];

    DataCopyPad(dPTensor, mm2WorkspaceGm[mm12Addr], {1, static_cast<uint32_t>(dataSize * sizeof(float)), 0, 0, 0},
                {false, 0, 0, 0});
    AscendC::SetFlag<HardEvent::MTE2_V>(static_cast<int32_t>(vWaitMte2));
    AscendC::WaitFlag<HardEvent::MTE2_V>(static_cast<int32_t>(vWaitMte2));
    CalSub(dPTensor, tmpRowSumOutTensor, processM, params.sftBaseN);
    AscendC::PipeBarrier<PIPE_V>();
    Mul(pTensor, pTensor, dPTensor, dataSize);
    AscendC::PipeBarrier<PIPE_V>();
    Cast(sftgOutT1Tensor, pTensor, RoundMode::CAST_ROUND, dataSize);
    AscendC::SetFlag<HardEvent::V_MTE3>(static_cast<int32_t>(mte3WaitV));
    AscendC::WaitFlag<HardEvent::V_MTE3>(static_cast<int32_t>(mte3WaitV));

    DataCopyPad(dsWorkspaceGm[mm345Addr], sftgOutT1Tensor, {1, static_cast<uint32_t>(dataSize * sizeof(T1)), 0, 0, 0});
    AscendC::SetFlag<HardEvent::MTE3_MTE2>(static_cast<int32_t>(mte2WaitMte3));
    AscendC::WaitFlag<HardEvent::MTE3_MTE2>(static_cast<int32_t>(mte2WaitMte3));
}

template <typename NSAGT>
__aicore__ inline void VecOp<NSAGT>::Process(const int64_t dyGmOffset, const int64_t sumGmOffset,
                                             const uint64_t indicesGmOffset, const int64_t s1Index,
                                             const int32_t blkCntOffset, int64_t mm12Addr, int64_t mm345Addr)
{
    CalRowsumAndSftCopyIn(dyGmOffset, sumGmOffset);

    int32_t loop = (params.singleM + params.sftBaseM - 1) / params.sftBaseM;
    int32_t processM = params.sftBaseM;
    int32_t tailM = params.singleM % params.sftBaseM;
    int64_t dataSize = processM * params.sftBaseN;

    for (int32_t i = 0; i < loop; i++) {
        if (i == 0) {
            AscendC::WaitFlag<HardEvent::MTE2_V>(
                static_cast<int32_t>(vWaitMte2)); // wait softmax_max and softmax_sum MTE2
        }
        if (i == loop - 1 && tailM != 0) {
            processM = tailM;
        }
        CalSoftmax(i, processM, mm12Addr, mm345Addr, indicesGmOffset, s1Index, blkCntOffset);
        CalSoftmaxGrad(i, processM, mm12Addr, mm345Addr);

        mm12Addr += dataSize;
        mm345Addr += dataSize;
    }
}

} // namespace NSAG_BASIC
