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
#include "sparse_flash_mla_grad_common_header.h"

namespace SMLAG_BASIC {
struct StaticParams {
    uint32_t singleM = 0;
    uint32_t singleN = 0;
    uint32_t sftBaseM = 0;
    uint32_t sftBaseN = 0;
    int64_t maxGatherSize = 0;
};

template <typename SMLAGT>
class VecOp {
public:
    using TILING_CLASS = typename SMLAGT::tiling_class;
    using T1 = typename SMLAGT::t1;
    static constexpr bool IS_BSND = SMLAGT::is_bsnd;
    static constexpr uint32_t MODE = SMLAGT::mode;

    __aicore__ inline VecOp(){};
    __aicore__ inline void Init(GM_ADDR ori_kv, GM_ADDR cmp_kv, GM_ADDR attention_out, GM_ADDR attention_out_grad,
                                GM_ADDR lse, GM_ADDR topk_indices, GM_ADDR sinks, GM_ADDR dsinks, GM_ADDR cu_seqlens_q,
                                GM_ADDR cu_seqlens_ori_kv, GM_ADDR cu_seqlens_cmp_kv, GM_ADDR cmp_softmax_l1_norm,
                                GM_ADDR workspace, const TILING_CLASS *__restrict ordTilingData, TPipe *pipe);
    __aicore__ inline void Process(const RunInfo &runInfo);
    __aicore__ inline void GatherKV(const int64_t n2Index, uint64_t currentS1Offset, const RunInfo &runInfo);
    __aicore__ inline void ScatterAdd(const RunInfo &runInfo);
    __aicore__ inline void CopyOutDsinks();

protected:
    __aicore__ inline void InitParams(const TILING_CLASS *__restrict ordTilingData);
    __aicore__ inline void InitGMBuffer(GM_ADDR ori_kv, GM_ADDR cmp_kv, GM_ADDR attention_out,
                                        GM_ADDR attention_out_grad, GM_ADDR lse, GM_ADDR topk_indices, GM_ADDR sinks,
                                        GM_ADDR dsinks, GM_ADDR cmp_softmax_l1_norm, GM_ADDR workspace);
    __aicore__ inline void InitUB(TPipe *pipe);
    __aicore__ inline void AtomicClean();
    __aicore__ inline void DumpGmZero(GlobalTensor<float> &gm, int64_t num);
    __aicore__ inline void CalRowsumAndSftCopyIn(const int64_t dyGmOffset, const int64_t lseGmOffset,
                                                 const int32_t processM);
    __aicore__ inline void CalAttenMsk(const int32_t processM, const RunInfo &runInfo, const float maskVal);
    __aicore__ inline void CalSoftmax(const int32_t processM, const int64_t mm12Addr, const int64_t mm345Addr,
                                      const RunInfo &runInfo, const int32_t mOffset, const int64_t s1Index);
    __aicore__ inline void CalSoftmaxGrad(const int32_t processM, const int64_t mm12Addr, const int64_t mm345Addr,
                                          const RunInfo &runInfo);
    __aicore__ inline void CalSub(LocalTensor<float> &dstTensor, LocalTensor<float> &srcTensor, int32_t baseM,
                                  int32_t baseN);
    __aicore__ inline void CalDsinks(const int32_t processM, const RunInfo &runInfo, const int32_t mOffset);
    __aicore__ inline void ScatterCopyOutPairwise(const LocalTensor<float> &srcTensor, GlobalTensor<float> &dstGm,
                                                  GlobalTensor<int32_t> &indicesGm, int64_t validRows,
                                                  int64_t &curS2Index);

protected:
    // core info
    int64_t usedCoreNum;
    int64_t formerCoreNum;
    uint32_t cubeBlockIdx;
    uint32_t vecBlockIdx;
    uint32_t subBlockIdx;
    StaticParams params;
    GlobalTensor<T1> attentionGm;
    GlobalTensor<T1> attentionGradGm;
    GlobalTensor<float> lseGm;
    GlobalTensor<int32_t> topkIndicesGm;
    GlobalTensor<T1> oriKvGm;
    GlobalTensor<T1> cmpKvGm;
    GlobalTensor<float> sinksGm;
    GlobalTensor<float> dSinksGm;
    GlobalTensor<float> cmpSoftmaxL1Gm;

    // workspace
    GlobalTensor<T1> selectedKWorkspaceGm;
    GlobalTensor<float> mm1WorkspaceGm;
    GlobalTensor<float> mm2WorkspaceGm;
    GlobalTensor<T1> pWorkspaceGm;
    GlobalTensor<T1> dsWorkspaceGm;
    GlobalTensor<float> dqWorkspaceGm;
    GlobalTensor<float> dkWorkspaceGm;
    GlobalTensor<float> mm4ResWorkspaceGm; // 24 * 2 * K * Dk
    GlobalTensor<float> mm5ResWorkspaceGm; // 24 * 2 * K * Dv
    GlobalTensor<float> additionalWorkspaceGm;

    TBuf<> vecQue;
    LocalTensor<uint8_t> helpTensor;
    LocalTensor<int32_t> topkIndicesTensor;
    LocalTensor<float> rowSumOutTensor;
    LocalTensor<float> lseTensor;
    LocalTensor<float> lseTmp;
    LocalTensor<T1> attentionT1Tensor;
    LocalTensor<float> attentionFP32Tensor;
    LocalTensor<T1> attentionGradT1Tensor;
    LocalTensor<float> attentionGradFP32Tensor;
    LocalTensor<float> pTensor;
    LocalTensor<float> dPTensor;
    LocalTensor<float> dSinkSumTensor;
    LocalTensor<float> sinkTmp;
    LocalTensor<float> sinkTensor;
    LocalTensor<float> sinkTmpBuf;
    LocalTensor<float> cmpL1NormTensor;
    LocalTensor<float> dSinkTensor;
    LocalTensor<T1> sftOutT1Tensor;
    LocalTensor<T1> sftgOutT1Tensor;
    LocalTensor<T1> gatherTensorPing;
    LocalTensor<T1> gatherTensorPong;
    LocalTensor<float> scatterAddTensorK;
    LocalTensor<float> scatterAddTensorV;

    // 地址相关
    int64_t selectedKWspOffset{0};
    int64_t selectedVWspOffset{0};

    constexpr static const int32_t BLOCK = 32;
    constexpr static const int32_t BLOCK_T1 = BLOCK / sizeof(T1);
    constexpr static const int32_t BLOCK_FP32 = BLOCK / sizeof(float);
    constexpr static const int32_t BLOCK_INT32 = BLOCK / sizeof(int32_t);
    constexpr static const int32_t MSK_LEN = 64;
    constexpr static const uint32_t ATTEN_MASK_SCALE = 0xFF7FFFFF;
    constexpr static const int64_t MAX_ADD_NUM = 64;

    const TILING_CLASS *__restrict tilingData;
    // Shape
    int64_t dimB;
    int64_t dimN2;
    int64_t dimS1;
    int64_t dimS2;
    int64_t dimS3;
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
    int64_t curS1;
    int64_t curS2;
    int64_t lastBlock;
    int64_t s1BasicSize;
    int64_t actualSelS2;
    int64_t actualSelS2Align;
    int64_t curActualLeft;
    int64_t curActualRight;
    float scaleValue;
    uint32_t selectedBlockCount;
    uint32_t selectedBlockSize;
    uint32_t selectedCountOffset;
    uint32_t actualSelectedCount;
    int32_t maxSelCnt;
    int64_t dimRope;
    int64_t cmpRatio;
    int64_t dOriKvSize;
    int64_t oriWinLeft;
    int64_t oriWinRight;
    int64_t oriWinLen;
    // workspace
    uint32_t mm12WorkspaceLen;
    int64_t dqWorkspaceLen;
    int64_t dkWorkspaceLen;
    int64_t dvWorkspaceLen;
    int64_t additionalWorkspaceLen;
    int64_t selectedKWorkspaceLen;
    int64_t selectedVWorkspaceLen;
    int64_t concatQWorkspaceLen;
    event_t vWaitMte2;
    event_t vWaitMte2Pong;
    event_t vWaitMte3;
    event_t mte3WaitMte2;
    event_t mte3WaitV;
    event_t sWaitMte2;
    event_t mte2WaitMte3;
    event_t mte2WaitMte3Pong;
    event_t mte3WaitMte2Pong;
    event_t mte2WaitV;

    DataCopyPadExtParams<T1> padParams;
    DataCopyExtParams intriParamsKey;
    DataCopyExtParams outParamK;
    uint32_t mergePingPong{0};
    int32_t pingPongIdx = 0;
};

template <typename SMLAGT>
__aicore__ inline void VecOp<SMLAGT>::Init(GM_ADDR ori_kv, GM_ADDR cmp_kv, GM_ADDR attention_out,
                                           GM_ADDR attention_out_grad, GM_ADDR lse, GM_ADDR topk_indices, GM_ADDR sinks,
                                           GM_ADDR dsinks, GM_ADDR cu_seqlens_q, GM_ADDR cu_seqlens_ori_kv,
                                           GM_ADDR cu_seqlens_cmp_kv, GM_ADDR cmp_softmax_l1_norm, GM_ADDR workspace,
                                           const TILING_CLASS *__restrict ordTilingData, TPipe *pipe)
{
    InitParams(ordTilingData);
    InitGMBuffer(ori_kv, cmp_kv, attention_out, attention_out_grad, lse, topk_indices, sinks, dsinks,
                 cmp_softmax_l1_norm, workspace);
    InitUB(pipe);
    AtomicClean();
}

template <typename SMLAGT>
__aicore__ inline void VecOp<SMLAGT>::InitParams(const TILING_CLASS *__restrict ordTilingData)
{
    cubeBlockIdx = GetBlockIdx() / 2;
    vecBlockIdx = GetBlockIdx();
    subBlockIdx = GetSubBlockIdx();
    tilingData = ordTilingData;
    usedCoreNum = tilingData->opInfo.usedCoreNum;
    formerCoreNum = tilingData->opInfo.formerCoreNum;

    dimB = tilingData->opInfo.B;
    dimN2 = tilingData->opInfo.N2;
    dimS1 = tilingData->opInfo.S1;
    dimS2 = tilingData->opInfo.S2;
    dimS3 = tilingData->opInfo.S3;
    dimG = tilingData->opInfo.G;
    dimD = tilingData->opInfo.D;
    dimD2 = dimD;
    dimDqk = tilingData->opInfo.D;
    dimDv = dimD;
    dimRope = 0;
    dimDAlign = (dimD + BLOCK_T1 - 1) / BLOCK_T1 * BLOCK_T1;
    dimD2Align = (dimD2 + BLOCK_T1 - 1) / BLOCK_T1 * BLOCK_T1;
    s1BasicSize = tilingData->splitCoreParams.s1BasicSize;

    cmpRatio = tilingData->opInfo.cmpRatio;
    scaleValue = tilingData->opInfo.scaleValue;
    selectedBlockCount = tilingData->opInfo.selectedBlockCount;
    selectedBlockSize = tilingData->opInfo.selectedBlockSize;
    mm12WorkspaceLen = tilingData->opInfo.mm12WorkspaceLen;
    dqWorkspaceLen = tilingData->opInfo.dqWorkspaceLen;
    dkWorkspaceLen = tilingData->opInfo.dkWorkspaceLen;
    additionalWorkspaceLen = tilingData->opInfo.additionalWorkspaceLen;
    selectedKWorkspaceLen = tilingData->opInfo.selectedKWorkspaceLen;

    params.singleM = tilingData->splitCoreParams.singleM;
    params.singleN = tilingData->splitCoreParams.singleN;
    params.sftBaseM = tilingData->splitCoreParams.sftBaseM;
    params.sftBaseN = tilingData->splitCoreParams.sftBaseN;
    params.maxGatherSize = tilingData->splitCoreParams.maxGatherSize;

    selectedS2 = selectedBlockCount * selectedBlockSize;
    selectedCountOffset = tilingData->splitCoreParams.singleN;
    if (tilingData->opInfo.selectedBlockSize * tilingData->opInfo.selectedBlockCount <=
        tilingData->splitCoreParams.singleN) {
        selectedCountOffset = tilingData->opInfo.selectedBlockCount;
    }

    oriWinLeft = tilingData->opInfo.oriWinLeft;
    oriWinRight = tilingData->opInfo.oriWinRight;
    oriWinLen = oriWinLeft + oriWinRight + 1;

    if constexpr (IS_BSND == false) {
        dimTq = tilingData->opInfo.S1;
        dimTkv = (tilingData->opInfo.S2 + tilingData->opInfo.S3);
        dOriKvSize = tilingData->opInfo.S2 * dimN2 * dimDqk;
    } else {
        dimTq = dimB * dimS1;
        dimTkv = dimB * (dimS2 + dimS3);
        dOriKvSize = dimB * dimS2 * dimN2 * dimDqk;
    }

    intriParamsKey.blockLen = dimDqk * sizeof(T1);
    intriParamsKey.dstStride = 0;
    intriParamsKey.blockCount = 2;

    outParamK.blockCount = params.maxGatherSize;
    outParamK.blockLen = dimDqk * sizeof(T1);
    outParamK.srcStride = 0;
    outParamK.dstStride = dimRope * sizeof(T1);
}

template <typename SMLAGT>
__aicore__ inline void VecOp<SMLAGT>::InitGMBuffer(GM_ADDR ori_kv, GM_ADDR cmp_kv, GM_ADDR attention_out,
                                                   GM_ADDR attention_out_grad, GM_ADDR lse, GM_ADDR topk_indices,
                                                   GM_ADDR sinks, GM_ADDR dsinks, GM_ADDR cmp_softmax_l1_norm,
                                                   GM_ADDR workspace)
{
    /*
     * 初始化输入
     */
    attentionGm.SetGlobalBuffer((__gm__ T1 *)attention_out);
    attentionGradGm.SetGlobalBuffer((__gm__ T1 *)attention_out_grad);
    lseGm.SetGlobalBuffer((__gm__ float *)lse);
    topkIndicesGm.SetGlobalBuffer((__gm__ int32_t *)topk_indices);
    oriKvGm.SetGlobalBuffer((__gm__ T1 *)ori_kv);
    cmpKvGm.SetGlobalBuffer((__gm__ T1 *)cmp_kv);
    dSinksGm.SetGlobalBuffer((__gm__ float *)dsinks);
    sinksGm.SetGlobalBuffer((__gm__ float *)sinks);
    cmpSoftmaxL1Gm.SetGlobalBuffer((__gm__ float *)cmp_softmax_l1_norm);

    /*
     * 初始化workspace
     */
    int64_t usedWorkspaceLen = 0;
    // select
    int64_t selectedKAddr = usedWorkspaceLen / sizeof(T1) + cubeBlockIdx * selectedKWorkspaceLen / sizeof(T1);
    usedWorkspaceLen += selectedKWorkspaceLen * usedCoreNum;
    // mm1 与 p 复用workspace
    int64_t mm1Addr = usedWorkspaceLen / sizeof(float) + cubeBlockIdx * mm12WorkspaceLen / sizeof(float);
    usedWorkspaceLen += mm12WorkspaceLen * usedCoreNum;
    int64_t pAddr = usedWorkspaceLen / sizeof(T1) + cubeBlockIdx * mm12WorkspaceLen / sizeof(T1);
    usedWorkspaceLen += mm12WorkspaceLen * usedCoreNum;
    // mm2 与 ds 复用workspace
    int64_t mm2Addr = usedWorkspaceLen / sizeof(float) + cubeBlockIdx * mm12WorkspaceLen / sizeof(float);
    usedWorkspaceLen += mm12WorkspaceLen * usedCoreNum;
    int64_t dsAddr = usedWorkspaceLen / sizeof(T1) + cubeBlockIdx * mm12WorkspaceLen / sizeof(T1);
    usedWorkspaceLen += mm12WorkspaceLen * usedCoreNum;
    // post
    int64_t dqAddr = usedWorkspaceLen / sizeof(float);
    int64_t dkAddr = dqAddr + dqWorkspaceLen / sizeof(float);
    usedWorkspaceLen += dqWorkspaceLen + dkWorkspaceLen;
    int64_t additionalAddr = usedWorkspaceLen / sizeof(float);

    // scatter add
    int64_t mm4ResAddr = (usedWorkspaceLen + additionalWorkspaceLen) / sizeof(float);
    int64_t mm5ResAddr = mm4ResAddr + MAX_CORE_NUM * selectedBlockCount * dimDAlign * 2;
    usedWorkspaceLen +=
        additionalWorkspaceLen + MAX_CORE_NUM * selectedBlockCount * (dimDAlign + dimDAlign) * 2 * sizeof(float);

    mm1WorkspaceGm.SetGlobalBuffer((__gm__ float *)workspace + mm1Addr);
    mm2WorkspaceGm.SetGlobalBuffer((__gm__ float *)workspace + mm2Addr);
    pWorkspaceGm.SetGlobalBuffer((__gm__ T1 *)workspace + pAddr);
    dsWorkspaceGm.SetGlobalBuffer((__gm__ T1 *)workspace + dsAddr);
    dqWorkspaceGm.SetGlobalBuffer((__gm__ float *)workspace + dqAddr);
    dkWorkspaceGm.SetGlobalBuffer((__gm__ float *)workspace + dkAddr);
    selectedKWorkspaceGm.SetGlobalBuffer((__gm__ T1 *)workspace + selectedKAddr);

    mm4ResWorkspaceGm.SetGlobalBuffer((__gm__ float *)workspace + mm4ResAddr);
    mm5ResWorkspaceGm.SetGlobalBuffer((__gm__ float *)workspace + mm5ResAddr);
    additionalWorkspaceGm.SetGlobalBuffer((__gm__ float *)workspace + additionalAddr);
}

template <typename SMLAGT>
__aicore__ inline void VecOp<SMLAGT>::InitUB(TPipe *pipe)
{
    uint32_t dataSize = params.sftBaseM * dimDv;
    uint32_t ubOffset = 0;
    uint32_t rowsumUbOffset = 0;
    uint32_t totalUbSpace = 191 * 1024;

    pipe->InitBuffer(vecQue, totalUbSpace);

    int32_t sftBaseNAlign = AlignUp(params.sftBaseN, BLOCK_FP32);
    cmpL1NormTensor = vecQue.GetWithOffset<float>(sftBaseNAlign, ubOffset);
    ubOffset += sftBaseNAlign * sizeof(float);

    dSinkTensor = vecQue.GetWithOffset<float>(dimG * BLOCK_FP32, ubOffset);
    ubOffset += dimG * BLOCK_FP32 * sizeof(float);
    uint32_t dSinksUbOffset = ubOffset;

    // rowsum out
    rowSumOutTensor = vecQue.GetWithOffset<float>(params.sftBaseM * BLOCK_FP32, ubOffset);
    ubOffset += params.sftBaseM * BLOCK_FP32 * sizeof(float);
    lseTensor = vecQue.GetWithOffset<float>(params.sftBaseM * BLOCK_FP32, ubOffset);
    ubOffset += params.sftBaseM * BLOCK_FP32 * sizeof(float);

    int32_t maxTmpSize = AlignUp(params.sftBaseM, BLOCK_FP32);
    lseTmp = vecQue.GetWithOffset<float>(maxTmpSize, ubOffset);
    ubOffset += maxTmpSize * sizeof(float);
    rowsumUbOffset = ubOffset;

    helpTensor =
        vecQue.GetWithOffset<uint8_t>((params.sftBaseM * dimDv * sizeof(T1)) / sizeof(uint8_t), rowsumUbOffset);

    // rowsum cal
    attentionGradT1Tensor = vecQue.GetWithOffset<T1>(params.sftBaseM * dimDv, ubOffset);
    ubOffset += params.sftBaseM * dimDv * sizeof(T1); //
    attentionGradFP32Tensor = vecQue.GetWithOffset<float>(params.sftBaseM * dimDv, ubOffset);
    ubOffset += params.sftBaseM * dimDv * sizeof(float);
    attentionT1Tensor = vecQue.GetWithOffset<T1>(params.sftBaseM * dimDv, ubOffset);
    ubOffset += params.sftBaseM * dimDv * sizeof(T1);
    attentionFP32Tensor = vecQue.GetWithOffset<float>(params.sftBaseM * dimDv, ubOffset);
    ubOffset += params.sftBaseM * dimDv * sizeof(float);

    // sftGard
    int32_t sftDataSize = params.sftBaseM * params.sftBaseN;
    pTensor = vecQue.GetWithOffset<float>(sftDataSize, ubOffset);
    sinkTmpBuf = vecQue.GetWithOffset<float>(sftDataSize, ubOffset);
    ubOffset += sftDataSize * sizeof(float);
    dPTensor = vecQue.GetWithOffset<float>(sftDataSize, ubOffset);
    sftgOutT1Tensor = vecQue.GetWithOffset<T1>(sftDataSize, ubOffset);
    ubOffset += sftDataSize * sizeof(float);

    sftOutT1Tensor = vecQue.GetWithOffset<T1>(sftDataSize, ubOffset);
    ubOffset += sftDataSize * sizeof(T1);

    sinkTmp = vecQue.GetWithOffset<float>(maxTmpSize, ubOffset);
    ubOffset += maxTmpSize * sizeof(float);
    sinkTensor = vecQue.GetWithOffset<float>(params.sftBaseM * BLOCK_FP32, ubOffset);
    ubOffset += params.sftBaseM * BLOCK_FP32 * sizeof(float);
    dSinkSumTensor = vecQue.GetWithOffset<float>(sftDataSize, ubOffset);
    ubOffset += sftDataSize * sizeof(float);

    scatterAddTensorK = vecQue.GetWithOffset<float>(16 * dimDAlign * 2, rowsumUbOffset);
    rowsumUbOffset += 16 * dimDAlign * 2 * sizeof(float);
    scatterAddTensorV = vecQue.GetWithOffset<float>(16 * dimD2Align * 2, rowsumUbOffset);
    rowsumUbOffset += 16 * dimD2Align * 2 * sizeof(float);

    maxSelCnt = 2;
    gatherTensorPing = vecQue.GetWithOffset<T1>(params.maxGatherSize * dimDqk, dSinksUbOffset);
    dSinksUbOffset += params.maxGatherSize * dimDqk * sizeof(T1);
    gatherTensorPong = vecQue.GetWithOffset<T1>(params.maxGatherSize * dimDqk, dSinksUbOffset);
    dSinksUbOffset += params.maxGatherSize * dimDqk * sizeof(T1);

    uint32_t attentionShape[2] = {params.sftBaseM, static_cast<uint32_t>(dimDv)};
    uint32_t softmaxGradShape[2] = {params.sftBaseM, BLOCK_FP32};
    attentionGradFP32Tensor.SetShapeInfo(ShapeInfo(2, attentionShape, DataFormat::ND));
    attentionFP32Tensor.SetShapeInfo(ShapeInfo(2, attentionShape, DataFormat::ND));
    rowSumOutTensor.SetShapeInfo(ShapeInfo(2, softmaxGradShape, DataFormat::ND));
    rowSumOutTensor.SetSize(params.sftBaseM * BLOCK_FP32);
    attentionGradT1Tensor.SetSize(dataSize);
    attentionGradFP32Tensor.SetSize(dataSize);
    attentionT1Tensor.SetSize(dataSize);
    attentionFP32Tensor.SetSize(dataSize);
    // params
    lseTensor.SetSize(params.sftBaseM * 8);
    sinkTensor.SetSize(params.sftBaseM * 8);

    // set shape
    uint32_t softmaxShape[2] = {static_cast<uint32_t>(params.sftBaseM), 8};
    lseTensor.SetShapeInfo(ShapeInfo(2, softmaxShape, DataFormat::ND));
    sinkTensor.SetShapeInfo(ShapeInfo(2, softmaxShape, DataFormat::ND));

    uint32_t dstSoftShape[2] = {static_cast<uint32_t>(params.sftBaseM), static_cast<uint32_t>(params.sftBaseN)};
    pTensor.SetShapeInfo(ShapeInfo(2, dstSoftShape, DataFormat::ND));
    mte2WaitMte3 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>());
    mte3WaitMte2 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE3>());
    mte2WaitMte3Pong = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>());
    mte3WaitMte2Pong = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE3>());
    sWaitMte2 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_S>());
    vWaitMte2 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
    vWaitMte3 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>());
    mte3WaitV = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());
    vWaitMte2Pong = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
    mte2WaitV = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());

    Duplicate(dSinkTensor, (float)0.0, dimG * BLOCK_FP32);
}

template <typename SMLAGT>
__aicore__ inline void VecOp<SMLAGT>::AtomicClean()
{
    DumpGmZero(dqWorkspaceGm, dqWorkspaceLen / sizeof(float));
    if constexpr (MODE != SMLAG_SCFA_MODE) {
        DumpGmZero(dkWorkspaceGm, (dkWorkspaceLen * 2) / sizeof(float));
    } else {
        DumpGmZero(dkWorkspaceGm, dkWorkspaceLen / sizeof(float));
    }
    DumpGmZero(dSinksGm, dimN2 * dimG);
    if constexpr (MODE == SMLAG_SCFA_MODE) {
        DumpGmZero(additionalWorkspaceGm, additionalWorkspaceLen / sizeof(float));
        DumpGmZero(cmpSoftmaxL1Gm, dimTq * dimN2 * selectedBlockCount);
    }
}

template <typename SMLAGT>
__aicore__ inline void VecOp<SMLAGT>::DumpGmZero(GlobalTensor<float> &gm, int64_t num)
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

template <typename SMLAGT>
__aicore__ inline void VecOp<SMLAGT>::CalSub(LocalTensor<float> &dstTensor, LocalTensor<float> &srcTensor,
                                             int32_t baseM, int32_t baseN)
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

template <typename SMLAGT>
__aicore__ inline void VecOp<SMLAGT>::CalRowsumAndSftCopyIn(const int64_t dyGmOffset, const int64_t lseGmOffset,
                                                            const int32_t processM)
{
    uint32_t dataSize = processM * dimDv;
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(processM * dimDv * sizeof(T1)), 0, 0, 0};
    DataCopyExtParams copyParams2{1, static_cast<uint32_t>(processM * sizeof(float)), 0, 0, 0};
    DataCopyPadExtParams<T1> padParams{false, 0, 0, 0};
    DataCopyPadExtParams<float> padParams2{false, 0, 0, 0};

    DataCopyPad(attentionGradT1Tensor, attentionGradGm[dyGmOffset], copyParams, padParams);
    DataCopyPad(attentionT1Tensor, attentionGm[dyGmOffset], copyParams, padParams);
    SET_FLAG(MTE2, V, vWaitMte2);
    WAIT_FLAG(MTE2, V, vWaitMte2);

    Cast(attentionGradFP32Tensor, attentionGradT1Tensor, RoundMode::CAST_NONE, dataSize);
    Cast(attentionFP32Tensor, attentionT1Tensor, RoundMode::CAST_NONE, dataSize);
    PIPE_BARRIER(PIPE_V);
    SoftmaxGradFront<float, false>(rowSumOutTensor, attentionGradFP32Tensor, attentionFP32Tensor, helpTensor,
                                   tilingData->softmaxGradTilingData);

    // [N2, T1, G] or [B, N2, S1, G] --> [sftBaseM, 1] --> [sftBaseM, 8]
    DataCopyPad(lseTmp, lseGm[lseGmOffset], copyParams2, padParams2);
    SET_FLAG(MTE2, V, vWaitMte2);

    WAIT_FLAG(MTE2, V, vWaitMte2);
    Brcb(lseTensor, lseTmp, CeilDiv(params.sftBaseM, BLOCK_FP32), {1, 8});
    PIPE_BARRIER(PIPE_V);
}

template <typename SMLAGT>
__aicore__ inline void VecOp<SMLAGT>::CalAttenMsk(const int32_t processM, const RunInfo &runInfo, const float maskVal)
{
    if (curActualLeft > 0) {
        // mask: [0, curActualLeft)
        int32_t leftLoop = CeilDiv(curActualLeft, 64);
        for (int32_t loop = 0; loop < leftLoop; loop++) {
            uint64_t mask = (loop == leftLoop - 1) ? curActualLeft - (loop * 64) : 64;
            Duplicate(pTensor[loop * 64], maskVal, mask, processM, 1, (actualSelS2Align * sizeof(float)) / BLOCK);
        }
    }

    if (curActualRight < actualSelS2Align) {
        // mask: [curActualRight, actualSelS2Align)
        int32_t rightStartOffset = curActualRight / 8 * 8; // UB上起始位置需向下32B对齐
        int32_t attenMskRsv = curActualRight - rightStartOffset;
        int32_t attenMskEnd = Min(actualSelS2Align - rightStartOffset, 64);
        int32_t rightMaskLen = actualSelS2Align - rightStartOffset;
        int32_t rightLoop = CeilDiv(rightMaskLen, 64);
        int32_t remain = rightMaskLen % 64 ? rightMaskLen % 64 : 64;
        for (int32_t loop = 0; loop < rightLoop; loop++) {
            if (loop == 0) {
                uint64_t maskEnd = (attenMskEnd == 64) ? UINT64_MAX : (1ULL << attenMskEnd) - 1;
                uint64_t maskRsv = (attenMskRsv == 64) ? UINT64_MAX : (1ULL << attenMskRsv) - 1;
                uint64_t mask[1] = {maskEnd & ~maskRsv};
                Duplicate(pTensor[loop * 64 + rightStartOffset], maskVal, mask, processM, 1,
                          (actualSelS2Align * sizeof(float)) / BLOCK);
            } else {
                uint64_t maskLen = (loop == rightLoop - 1) ? remain : 64;
                Duplicate(pTensor[loop * 64 + rightStartOffset], maskVal, maskLen, processM, 1,
                          (actualSelS2Align * sizeof(float)) / BLOCK);
            }
        }
    }
}

template <typename SMLAGT>
__aicore__ inline void VecOp<SMLAGT>::CalSoftmax(const int32_t processM, const int64_t mm12Addr,
                                                 const int64_t mm345Addr, const RunInfo &runInfo, const int32_t mOffset,
                                                 const int64_t s1Index)
{
    WAIT_FLAG(MTE3, MTE2, mte2WaitMte3);
    if constexpr (MODE != SMLAG_SCFA_MODE) {
        if (MODE == SMLAG_SWA_MODE || runInfo.isOri) {
            int64_t curOriWinStart = Max(s1Index - oriWinLeft + runInfo.oriWinDiagOffset, 0);
            int64_t curOriWinEnd = Min(s1Index + oriWinRight + 1 + runInfo.oriWinDiagOffset, runInfo.curS2);
            int64_t blkStart = runInfo.oriWinStart + runInfo.blkCntOffset;
            curActualRight =
                Min(runInfo.oriWinStart + runInfo.blkCntOffset + runInfo.actualSelCntOffset, curOriWinEnd) - blkStart;
            curActualLeft = Max(runInfo.oriWinStart + runInfo.blkCntOffset, curOriWinStart) - blkStart;
            curActualRight = Max(curActualRight, curActualLeft);
        } else {
            curActualLeft = 0;
            curActualRight = Max((runInfo.cmpDiagOffset + s1Index + 1) / cmpRatio - runInfo.blkCntOffset, 0);
            curActualRight = Min(curActualRight, actualSelS2);
        }
    }
    actualSelS2Align = AlignUp(actualSelS2, 8);
    int64_t dataSize = processM * actualSelS2Align;

    WAIT_FLAG(V, MTE2, mte2WaitV); // wait sinkTmp
    DataCopyPad(pTensor, mm1WorkspaceGm[mm12Addr],
                {static_cast<uint16_t>(processM), static_cast<uint32_t>(actualSelS2 * sizeof(float)),
                 static_cast<uint32_t>((params.sftBaseN - actualSelS2) * sizeof(float)), 0, 0},
                {false, 0, 0, 0});
    DataCopyPad(sinkTmp, sinksGm[mOffset], {1, static_cast<uint32_t>(processM * sizeof(float)), 0, 0, 0},
                {false, 0, 0, 0});
    SET_FLAG(MTE2, V, vWaitMte2);
    WAIT_FLAG(MTE2, V, vWaitMte2);

    Brcb(sinkTensor, sinkTmp, CeilDiv(params.sftBaseM, BLOCK_FP32), {1, 8});
    PIPE_BARRIER(PIPE_V);

    Muls(pTensor, pTensor, scaleValue, dataSize);
    PIPE_BARRIER(PIPE_V);

    if constexpr (MODE != SMLAG_SCFA_MODE) {
        CalAttenMsk(processM, runInfo, *((float *)&ATTEN_MASK_SCALE));
        PIPE_BARRIER(PIPE_V);
    }

    // pTensor: [processM, actualSelS2Align]
    // lseTensor: [sftBaseM, 8]
    bool isSmallS2 = CeilDiv(actualSelS2Align, 8) < 256;
    uint8_t repeatTime = isSmallS2 ? processM : 1;
    uint8_t repeatStrideFp32 = isSmallS2 ? static_cast<uint8_t>(CeilDiv(actualSelS2Align, 8)) : 0;
    uint8_t repeatStrideT1 = isSmallS2 ? static_cast<uint8_t>(CeilDiv(actualSelS2Align, 16)) : 0;
    auto castS2Loop = CeilDiv(actualSelS2Align, 64);
    for (int outerLoop = 0; outerLoop < processM; outerLoop += repeatTime) {
        auto offset = outerLoop * actualSelS2Align;
        for (int i = 0; i < castS2Loop; ++i) {
            uint64_t mask = i == (castS2Loop - 1) ? actualSelS2Align - (i * 64) : 64;
            Sub(pTensor[offset + i * 64], pTensor[offset + i * 64], lseTensor[outerLoop * 8], mask, repeatTime,
                {1, 1, 0, repeatStrideFp32, repeatStrideFp32, 1});
            PIPE_BARRIER(PIPE_V);

            Exp(pTensor[offset + i * 64], pTensor[offset + i * 64], mask, repeatTime,
                {1, 1, repeatStrideFp32, repeatStrideFp32});
            PIPE_BARRIER(PIPE_V);

            Cast(sftOutT1Tensor[offset + i * 64], pTensor[offset + i * 64], RoundMode::CAST_ROUND, mask, repeatTime,
                 {1, 1, repeatStrideT1, repeatStrideFp32});
        }
    }
    SET_FLAG(V, MTE3, mte3WaitV);
    WAIT_FLAG(V, MTE3, mte3WaitV);

    if constexpr (MODE == SMLAG_SCFA_MODE) {
        if (!runInfo.isOri) {
            // ReduceG
            // 后续可优化成二分累加
            for (int32_t i = 0; i < processM; ++i) {
                Add(cmpL1NormTensor, cmpL1NormTensor, pTensor[i * actualSelS2Align], actualSelS2);
                PIPE_BARRIER(PIPE_V);
            }
        }
    }

    Sub(sinkTensor, sinkTensor, lseTensor, processM * 8);
    PIPE_BARRIER(PIPE_V);
    Exp(sinkTensor, sinkTensor, processM * 8);
    PIPE_BARRIER(PIPE_V);

    DataCopyPad(pWorkspaceGm[mm345Addr], sftOutT1Tensor,
                {static_cast<uint16_t>(processM), static_cast<uint32_t>(actualSelS2 * sizeof(T1)), 0,
                 static_cast<uint32_t>((params.sftBaseN - actualSelS2) * sizeof(T1)), 0});
    SET_FLAG(MTE3, MTE2, mte2WaitMte3);
    WAIT_FLAG(MTE3, MTE2, mte2WaitMte3);
}

template <typename SMLAGT>
__aicore__ inline void VecOp<SMLAGT>::CalSoftmaxGrad(const int32_t processM, const int64_t mm12Addr,
                                                     const int64_t mm345Addr, const RunInfo &runInfo)
{
    int64_t dataSize = processM * actualSelS2Align;
    DataCopyPad(dPTensor, mm2WorkspaceGm[mm12Addr],
                {static_cast<uint16_t>(processM), static_cast<uint32_t>(actualSelS2 * sizeof(float)),
                 static_cast<uint32_t>((params.sftBaseN - actualSelS2) * sizeof(float)), 0, 0},
                {false, 0, 0, 0});
    SET_FLAG(MTE2, V, vWaitMte2);
    WAIT_FLAG(MTE2, V, vWaitMte2);

    Mul(dSinkSumTensor, dPTensor, pTensor, dataSize);
    PIPE_BARRIER(PIPE_V);

    CalSub(dPTensor, rowSumOutTensor, processM, actualSelS2Align);
    PIPE_BARRIER(PIPE_V);

    Mul(pTensor, pTensor, dPTensor, dataSize);
    PIPE_BARRIER(PIPE_V);

    if constexpr (MODE != SMLAG_SCFA_MODE) {
        CalAttenMsk(processM, runInfo, 0.0);
        PIPE_BARRIER(PIPE_V);
    }

    bool isSmallS2 = CeilDiv(actualSelS2Align, 8) < 256;
    uint8_t repeatTime = isSmallS2 ? params.sftBaseM : 1;
    uint8_t repeatStrideFp32 = isSmallS2 ? static_cast<uint8_t>(CeilDiv(actualSelS2Align, 8)) : 0;
    uint8_t repeatStrideT1 = isSmallS2 ? static_cast<uint8_t>(CeilDiv(actualSelS2Align, 16)) : 0;
    auto castS2Loop = CeilDiv(actualSelS2Align, 64);
    for (int outerLoop = 0; outerLoop < params.sftBaseM; outerLoop += repeatTime) {
        auto offset = outerLoop * actualSelS2Align;
        for (int i = 0; i < castS2Loop; ++i) {
            uint64_t mask = i == (castS2Loop - 1) ? actualSelS2Align - (i * 64) : 64;
            Cast(sftgOutT1Tensor[offset + i * 64], pTensor[offset + i * 64], RoundMode::CAST_ROUND, mask, repeatTime,
                 {1, 1, repeatStrideT1, repeatStrideFp32});
        }
    }
    SET_FLAG(V, MTE3, mte3WaitV);
    WAIT_FLAG(V, MTE3, mte3WaitV);

    DataCopyPad(dsWorkspaceGm[mm345Addr], sftgOutT1Tensor,
                {static_cast<uint16_t>(processM), static_cast<uint32_t>(actualSelS2 * sizeof(T1)), 0,
                 static_cast<uint32_t>((params.sftBaseN - actualSelS2) * sizeof(T1)), 0});
    SET_FLAG(MTE3, MTE2, mte2WaitMte3);
    WAIT_FLAG(MTE3, MTE2, mte2WaitMte3);
}

template <typename SMLAGT>
__aicore__ inline void VecOp<SMLAGT>::CalDsinks(const int32_t processM, const RunInfo &runInfo, const int32_t mOffset)
{
    if (MODE != SMLAG_SCFA_MODE && curActualRight - curActualLeft == 0) {
        return;
    }
    uint64_t tailS2 = actualSelS2Align % 64 ? actualSelS2Align % 64 : 64;
    uint64_t totalLoop = CeilDiv(actualSelS2Align, 64);
    for (int32_t i = 0; i < processM; ++i) {
        for (int32_t loop = 0; loop < totalLoop; ++loop) {
            uint64_t mask = loop == totalLoop - 1 ? tailS2 : 64;
            Mul(dSinkSumTensor[i * actualSelS2Align + loop * 64], dSinkSumTensor[i * actualSelS2Align + loop * 64],
                sinkTensor[i * BLOCK_FP32], mask, 1, {1, 1, 0, 0, 0, 0});
        }
        PIPE_BARRIER(PIPE_V);

        if constexpr (MODE != SMLAG_SCFA_MODE) {
            ReduceSum(sinkTensor[i * BLOCK_FP32], dSinkSumTensor[i * actualSelS2Align], sinkTmpBuf, curActualRight);
        } else {
            ReduceSum(sinkTensor[i * BLOCK_FP32], dSinkSumTensor[i * actualSelS2Align], sinkTmpBuf, actualSelS2);
        }
        PIPE_BARRIER(PIPE_V);
    }

    Muls(sinkTensor, sinkTensor, -1.0f, processM * BLOCK_FP32);
    PIPE_BARRIER(PIPE_V);
    Add(dSinkTensor[mOffset * BLOCK_FP32], dSinkTensor[mOffset * BLOCK_FP32], sinkTensor, processM * BLOCK_FP32);

    SET_FLAG(V, MTE3, mte3WaitV);
    WAIT_FLAG(V, MTE3, mte3WaitV);
}

template <typename SMLAGT>
__aicore__ inline void VecOp<SMLAGT>::CopyOutDsinks()
{
    SetAtomicAdd<float>();
    DataCopyPad(dSinksGm, dSinkTensor, {static_cast<uint16_t>(dimG), sizeof(float), 0, 0});
    SetAtomicNone();
}

template <typename SMLAGT>
__aicore__ inline void VecOp<SMLAGT>::GatherKV(const int64_t n2Index, uint64_t currentS1Offset, const RunInfo &runInfo)
{
    uint64_t kSelectedWsAddr = runInfo.selectedKGmOffset;
    uint64_t s1Offset = IS_BSND ? currentS1Offset : currentS1Offset + runInfo.s1Index;
    uint64_t gmOffset = s1Offset * (dimN2 * selectedBlockCount) + n2Index * selectedBlockCount + runInfo.blkCntOffset;

    uint64_t totalD = dimDqk + dimRope;
    event_t mte2WaitMte3EventId;
    event_t mte3WaitMte2EventId;

    SET_FLAG(MTE3, MTE2, mte2WaitMte3Pong);
    SET_FLAG(MTE3, MTE2, mte2WaitMte3);

    // ------------- MergeKv --------------
    uint32_t s2Pair = CeilDiv(runInfo.actualSelCntOffset, 2) * 2;
    uint32_t firstVecEnd = (s2Pair / 2);
    uint32_t curBlk = subBlockIdx == 0 ? 0 : firstVecEnd;
    uint32_t curActualSelCntEnd = subBlockIdx == 0 ? firstVecEnd : runInfo.actualSelCntOffset;
    uint32_t curActualSelCntOffset = curActualSelCntEnd - curBlk;
    uint64_t outWsOffset = subBlockIdx == 0 ? 0 : firstVecEnd * totalD;
    uint64_t orgOutWsOffset = outWsOffset;
    uint32_t i;

    bool isLast = runInfo.isLastBasicBlock && subBlockIdx == 1;

    int64_t curGatherCount = 0;
    for (i = curBlk; i < curBlk + curActualSelCntOffset / maxSelCnt * maxSelCnt; i += maxSelCnt) {
        int64_t keyOffset1 = topkIndicesGm.GetValue(gmOffset + i);
        int64_t keyOffset2 = topkIndicesGm.GetValue(gmOffset + i + 1);

        uint32_t s2OrgStride = keyOffset2 - keyOffset1 - selectedBlockSize;
        intriParamsKey.blockCount = 2;
        intriParamsKey.blockLen = dimDqk * sizeof(T1);

        if (curGatherCount == 0) {
            mte2WaitMte3EventId = mergePingPong ? mte2WaitMte3Pong : mte2WaitMte3;
            mte3WaitMte2EventId = mergePingPong ? mte3WaitMte2Pong : mte3WaitMte2;
            WAIT_FLAG(MTE3, MTE2, mte2WaitMte3EventId);
        }

        // CopyIn
        intriParamsKey.srcStride = s2OrgStride * dimN2 * dimDqk * sizeof(T1);
        LocalTensor<T1> &gatherTensor = mergePingPong ? gatherTensorPing : gatherTensorPong;

        bool isActualLast = isLast && i >= runInfo.actualSelCntOffset - 2;
        int64_t curOffset = curGatherCount * dimDqk;
        if (keyOffset2 <= keyOffset1 || isActualLast) {
            intriParamsKey.blockCount = 1;
            DataCopyPad(gatherTensor[curOffset], cmpKvGm[runInfo.cmpKeyGmOffset + keyOffset1 * dimN2 * dimDqk],
                        intriParamsKey, padParams);
            intriParamsKey.blockLen = dimDqk * sizeof(T1);
            DataCopyPad(gatherTensor[curOffset + dimDqk], cmpKvGm[runInfo.cmpKeyGmOffset + keyOffset2 * dimN2 * dimDqk],
                        intriParamsKey, padParams);
        } else {
            DataCopyPad(gatherTensor[curOffset], cmpKvGm[runInfo.cmpKeyGmOffset + keyOffset1 * dimN2 * dimDqk],
                        intriParamsKey, padParams);
        }

        curGatherCount += maxSelCnt;

        if (curGatherCount == params.maxGatherSize) {
            SET_FLAG(MTE2, MTE3, mte3WaitMte2EventId);
            WAIT_FLAG(MTE2, MTE3, mte3WaitMte2EventId);

            outParamK.blockCount = curGatherCount;

            DataCopyPad(selectedKWorkspaceGm[kSelectedWsAddr + outWsOffset], gatherTensor, outParamK);

            SET_FLAG(MTE3, MTE2, mte2WaitMte3EventId);

            outWsOffset += curGatherCount * totalD;
            mergePingPong = 1 - mergePingPong;
            curGatherCount = 0;
        }
    }
    if (i < curActualSelCntEnd) {
        int64_t keyOffset1 = topkIndicesGm.GetValue(gmOffset + i);

        if (curGatherCount == 0) {
            mte2WaitMte3EventId = mergePingPong ? mte2WaitMte3Pong : mte2WaitMte3;
            mte3WaitMte2EventId = mergePingPong ? mte3WaitMte2Pong : mte3WaitMte2;
            WAIT_FLAG(MTE3, MTE2, mte2WaitMte3EventId);
        }

        // CopyIn
        LocalTensor<T1> &gatherTensor = mergePingPong ? gatherTensorPing : gatherTensorPong;

        intriParamsKey.blockCount = 1;
        if (i == runInfo.actualSelCntOffset - 1 && isLast) {
            intriParamsKey.blockLen = runInfo.lastBlockSize * dimDqk * sizeof(T1);
        }
        DataCopyPad(gatherTensor[curGatherCount * dimDqk],
                    cmpKvGm[runInfo.cmpKeyGmOffset + keyOffset1 * dimN2 * dimDqk], intriParamsKey, padParams);

        curGatherCount += 1;
    }
    if (curGatherCount != 0) {
        LocalTensor<T1> &gatherTensor = mergePingPong ? gatherTensorPing : gatherTensorPong;
        SET_FLAG(MTE2, MTE3, mte3WaitMte2EventId);
        WAIT_FLAG(MTE2, MTE3, mte3WaitMte2EventId);

        outParamK.blockCount = curGatherCount;
        DataCopyPad(selectedKWorkspaceGm[kSelectedWsAddr + outWsOffset], gatherTensor, outParamK);

        SET_FLAG(MTE3, MTE2, mte2WaitMte3EventId);

        mergePingPong = 1 - mergePingPong;
    }
    WAIT_FLAG(MTE3, MTE2, mte2WaitMte3Pong);
    WAIT_FLAG(MTE3, MTE2, mte2WaitMte3);
}

template <typename SMLAGT>
__aicore__ inline void VecOp<SMLAGT>::Process(const RunInfo &runInfo)
{
    int32_t totalLines = dimG * runInfo.curS1Basic;
    int32_t firstCoreLines = CeilDiv(totalLines, 2);

    int32_t curCoreBegin = subBlockIdx == 0 ? 0 : firstCoreLines;
    int32_t curCoreEnd = subBlockIdx == 0 ? firstCoreLines : totalLines;

    int32_t processM = params.sftBaseM;
    int64_t dataSize = processM * params.sftBaseN;

    int64_t dyGmOffset = runInfo.dyGmOffset + curCoreBegin * dimDv;
    int64_t lseGmOffset = runInfo.lseGmOffset + curCoreBegin;
    int64_t mm12Addr = runInfo.mm12GmOffset + curCoreBegin * params.sftBaseN;
    int64_t mm345Addr = runInfo.mm345GmOffset + curCoreBegin * params.sftBaseN;

    int32_t mOffset = 0;
    int64_t s1Index = runInfo.s1Index;

    actualSelS2 = runInfo.actualSelCntOffset;
    if constexpr (MODE == SMLAG_SCFA_MODE) {
        if (!runInfo.isOri) {
            WAIT_FLAG(MTE3, V, runInfo.vWaitMte3Proc);
            Duplicate(cmpL1NormTensor, (float)0.0, actualSelS2);
            PIPE_BARRIER(PIPE_V);
        }
    }
    SET_FLAG(MTE3, MTE2, mte2WaitMte3);
    SET_FLAG(V, MTE2, mte2WaitV);
    for (int32_t curLine = curCoreBegin; curLine < curCoreEnd;) {
        s1Index = curLine / dimG + runInfo.s1Index;
        mOffset = curLine % dimG;
        processM = mOffset + params.sftBaseM > dimG ? dimG - mOffset : params.sftBaseM;
        processM = curLine + processM > curCoreEnd ? curCoreEnd - curLine : processM;

        dataSize = processM * params.sftBaseN;
        CalRowsumAndSftCopyIn(dyGmOffset, lseGmOffset, processM);
        dyGmOffset += (processM * dimDv);
        lseGmOffset += processM;

        CalSoftmax(processM, mm12Addr, mm345Addr, runInfo, mOffset, s1Index);
        CalSoftmaxGrad(processM, mm12Addr, mm345Addr, runInfo);
        CalDsinks(processM, runInfo, mOffset);
        SET_FLAG(V, MTE2, mte2WaitV);
        SET_FLAG(MTE3, MTE2, mte2WaitMte3);

        mm12Addr += dataSize;
        mm345Addr += dataSize;
        curLine += processM;
    }
    if constexpr (MODE == SMLAG_SCFA_MODE) {
        if (!runInfo.isOri) {
            Muls(cmpL1NormTensor, cmpL1NormTensor, 1.0f / (float)dimG, actualSelS2);
            SET_FLAG(V, MTE3, mte3WaitV);
            WAIT_FLAG(V, MTE3, mte3WaitV);
            SetAtomicAdd<float>();
            DataCopyPad(cmpSoftmaxL1Gm[runInfo.indicesGmOffset + runInfo.blkCntOffset], cmpL1NormTensor,
                        {1, (uint32_t)(actualSelS2 * sizeof(float)), 0, 0, 0});
            SetAtomicNone();
            SET_FLAG(MTE3, V, runInfo.vWaitMte3Proc);
        }
    }

    WAIT_FLAG(MTE3, MTE2, mte2WaitMte3);
    WAIT_FLAG(V, MTE2, mte2WaitV);
}

template <typename SMLAGT>
__aicore__ inline void VecOp<SMLAGT>::ScatterCopyOutPairwise(const LocalTensor<float> &srcTensor,
                                                             GlobalTensor<float> &dstGm,
                                                             GlobalTensor<int32_t> &indicesGm, int64_t validRows,
                                                             int64_t &curS2Index)
{
    int64_t row = 0;
    while (row < validRows) {
        int64_t idx0 = indicesGm.GetValue(curS2Index);
        bool canPair = row + 1 < validRows;
        int64_t idx1 = -1;
        uint64_t gapBytes = 0;
        if (canPair) {
            idx1 = indicesGm.GetValue(curS2Index + 1);
            if (idx0 >= 0 && idx1 > idx0) {
                gapBytes = static_cast<uint64_t>(idx1 - idx0 - 1) * dimDAlign * sizeof(float);
                canPair = gapBytes <= UINT32_MAX;
            } else {
                canPair = false;
            }
        }

        if (canPair) {
            DataCopyExtParams pairParams{2, static_cast<uint32_t>(dimDAlign * sizeof(float)), 0,
                                         static_cast<uint32_t>(gapBytes), 0};
            DataCopyPad(dstGm[idx0 * dimDAlign], srcTensor[row * dimDAlign], pairParams);
            row += 2;
            curS2Index += 2;
        } else {
            if (idx0 >= 0) {
                DataCopy(dstGm[idx0 * dimDAlign], srcTensor[row * dimDAlign], dimDAlign);
            }
            ++row;
            ++curS2Index;
        }
    }
}

template <typename SMLAGT>
__aicore__ inline void VecOp<SMLAGT>::ScatterAdd(const RunInfo &runInfo)
{
    if (runInfo.isOri) {
        return;
    }
    LocalTensor<float> dkInUb;
    LocalTensor<float> dvInUb;
    int64_t UB_ROW_SIZE = 16;

    GlobalTensor<float> dCmpKvOutGm = dkWorkspaceGm[runInfo.cmpKeyGmOffset + dOriKvSize];
    int64_t s2RealSize = runInfo.actualSelectedBlockCount;
    int64_t firstCoreKSize = s2RealSize / 2;
    int64_t currentCoreKSize = subBlockIdx == 0 ? firstCoreKSize : s2RealSize - firstCoreKSize;
    int64_t curS2Index = subBlockIdx == 0 ? 0 : firstCoreKSize;

    if (currentCoreKSize == 0) {
        return;
    }
    SetAtomicAdd<float>();

    int64_t maxLoops = CeilDiv(currentCoreKSize, UB_ROW_SIZE);
    int64_t tailRows = currentCoreKSize - (maxLoops - 1) * UB_ROW_SIZE;

    int64_t currentDkSrcOffset = runInfo.scatterTaskId * MAX_CORE_NUM * selectedBlockCount * dimDAlign +
                                 cubeBlockIdx * selectedBlockCount * dimDAlign +
                                 subBlockIdx * firstCoreKSize * dimDAlign;
    int64_t currentDvSrcOffset = runInfo.scatterTaskId * MAX_CORE_NUM * selectedBlockCount * dimD2Align +
                                 cubeBlockIdx * selectedBlockCount * dimD2Align +
                                 subBlockIdx * firstCoreKSize * dimD2Align;
    GlobalTensor<float> dkSrcGm = mm4ResWorkspaceGm[currentDkSrcOffset];
    GlobalTensor<float> dvSrcGm = mm5ResWorkspaceGm[currentDvSrcOffset];
    GlobalTensor<int32_t> indicesGm = topkIndicesGm[runInfo.indicesGmOffset];

    SetFlag<AscendC::HardEvent::MTE3_MTE2>(mte2WaitMte3);
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(mte2WaitMte3Pong);
    for (int64_t loop = 0; loop < maxLoops - 1; loop++) {
        event_t backEvent = pingPongIdx == 0 ? mte2WaitMte3 : mte2WaitMte3Pong;
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(backEvent);
        dkInUb = scatterAddTensorK[pingPongIdx * (UB_ROW_SIZE * dimDAlign)];
        dvInUb = scatterAddTensorV[pingPongIdx * (UB_ROW_SIZE * dimD2Align)];
        DataCopy(dkInUb, dkSrcGm[loop * UB_ROW_SIZE * dimDAlign], UB_ROW_SIZE * dimDAlign);
        event_t event = pingPongIdx == 0 ? vWaitMte2 : vWaitMte2Pong;
        SetFlag<AscendC::HardEvent::MTE2_V>(event);
        WaitFlag<AscendC::HardEvent::MTE2_V>(event);
        Muls(dkInUb, dkInUb, (float)tilingData->opInfo.scaleValue, UB_ROW_SIZE * dimDAlign);
        DataCopy(dvInUb, dvSrcGm[loop * UB_ROW_SIZE * dimD2Align], UB_ROW_SIZE * dimD2Align);
        SetFlag<AscendC::HardEvent::MTE2_V>(event);
        WaitFlag<AscendC::HardEvent::MTE2_V>(event);
        PipeBarrier<PIPE_V>();
        Add(dkInUb, dkInUb, dvInUb, UB_ROW_SIZE * dimDAlign);
        SetFlag<AscendC::HardEvent::V_MTE3>(mte3WaitV);
        WaitFlag<AscendC::HardEvent::V_MTE3>(mte3WaitV);
        ScatterCopyOutPairwise(dkInUb, dCmpKvOutGm, indicesGm, UB_ROW_SIZE, curS2Index);
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(backEvent);
        pingPongIdx = 1 - pingPongIdx;
    }
    event_t backEvent = pingPongIdx == 0 ? mte2WaitMte3 : mte2WaitMte3Pong;
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(backEvent);
    dkInUb = scatterAddTensorK[pingPongIdx * (UB_ROW_SIZE * dimDAlign)];
    dvInUb = scatterAddTensorV[pingPongIdx * (UB_ROW_SIZE * dimD2Align)];
    DataCopy(dkInUb, dkSrcGm[(maxLoops - 1) * UB_ROW_SIZE * dimDAlign], tailRows * dimDAlign);
    event_t event = pingPongIdx == 0 ? vWaitMte2 : vWaitMte2Pong;
    SetFlag<AscendC::HardEvent::MTE2_V>(event);
    WaitFlag<AscendC::HardEvent::MTE2_V>(event);
    Muls(dkInUb, dkInUb, (float)tilingData->opInfo.scaleValue, tailRows * dimDAlign);
    DataCopy(dvInUb, dvSrcGm[(maxLoops - 1) * UB_ROW_SIZE * dimD2Align], tailRows * dimD2Align);
    SetFlag<AscendC::HardEvent::MTE2_V>(event);
    WaitFlag<AscendC::HardEvent::MTE2_V>(event);
    PipeBarrier<PIPE_V>();
    Add(dkInUb, dkInUb, dvInUb, tailRows * dimDAlign);
    SetFlag<AscendC::HardEvent::V_MTE3>(mte3WaitV);
    WaitFlag<AscendC::HardEvent::V_MTE3>(mte3WaitV);
    ScatterCopyOutPairwise(dkInUb, dCmpKvOutGm, indicesGm, tailRows, curS2Index);
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(backEvent);
    pingPongIdx = 1 - pingPongIdx;
    SetAtomicNone();

    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte2WaitMte3);
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte2WaitMte3Pong);

    SetFlag<AscendC::HardEvent::MTE3_V>(vWaitMte3);
    WaitFlag<AscendC::HardEvent::MTE3_V>(vWaitMte3);
}

} // namespace SMLAG_BASIC
