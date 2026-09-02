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
 * \file nsa_selected_attention_grad.h
 * \brief
 */

#pragma once
#include "lib/matmul_intf.h"
#include "kernel_operator.h"
#include "../basic_modules/nsa_selected_attention_grad_common_header.h"
using namespace AscendC;
using matmul::Matmul;
using matmul::MatmulType;

struct StaticParams {
    uint32_t singleM = 0;
    uint32_t singleN = 0;
    uint32_t sftBaseM = 0;
    uint32_t sftBaseN = 0;
};
static constexpr MatmulConfig MM_CFG = GetNormalConfig(false);

template <typename NSAGT>
class SelectedAttentionGrad {
    using TILING_CLASS = typename NSAGT::tiling_class;
    using T1 = typename NSAGT::t1;
    static constexpr uint32_t ATTEN_ENABLE = NSAGT::atten_enable;

public:
    __aicore__ inline SelectedAttentionGrad(){};
    __aicore__ inline void Init(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR attention_out,
                                GM_ADDR attention_out_grad, GM_ADDR softmax_max, GM_ADDR softmax_sum,
                                GM_ADDR topk_indices, GM_ADDR actual_seq_qlen, GM_ADDR actual_seq_kvlen,
                                GM_ADDR atten_mask, GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR workspace,
                                const TILING_CLASS *__restrict ordTilingData, TPipe *pipe);
    __aicore__ inline void Process();
    __aicore__ inline void DeterministicProcess();

    using aType = MatmulType<TPosition::GM, CubeFormat::ND, T1>;
    using aTypeMM34 = MatmulType<TPosition::GM, CubeFormat::ND, T1>;
    using bType = MatmulType<TPosition::GM, CubeFormat::ND, T1>;
    using aTypeTranspose = MatmulType<TPosition::GM, CubeFormat::ND, T1, true>;
    using bTypeTranspose = MatmulType<TPosition::GM, CubeFormat::ND, T1, true>;
    using cType = MatmulType<TPosition::GM, CubeFormat::ND, float>;
    using cTypeMMAlign = MatmulType<TPosition::GM, CubeFormat::ND_ALIGN, float>;
    using biasType = MatmulType<TPosition::GM, CubeFormat::ND, float>;

    using modeTypemm12 = Matmul<aType, bTypeTranspose, cTypeMMAlign, biasType, MM_CFG>;
    using modeTypeDq = Matmul<aTypeMM34, bType, cType, biasType, MM_CFG>;
    using modeTypeDkv = Matmul<aTypeTranspose, bType, cType, biasType, MM_CFG>;

    modeTypemm12 mm1;
    modeTypemm12 mm2;
    modeTypeDq mm3;
    modeTypeDkv mm4;
    modeTypeDkv mm5;

protected:
    // init
    __aicore__ inline void InitGMBuffer(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR attention_out,
                                        GM_ADDR attention_out_grad, GM_ADDR softmax_max, GM_ADDR softmax_sum,
                                        GM_ADDR topk_indices, GM_ADDR dq, GM_ADDR dk, GM_ADDR dv);
    __aicore__ inline void InitParams(const TILING_CLASS *__restrict ordTilingData, GM_ADDR actual_seq_qlen,
                                      GM_ADDR actual_seq_kvlen);
    __aicore__ inline void InitWorkspace(GM_ADDR workspace);
    __aicore__ inline void InitAddrArray(int64_t *scatterDkAddrArray, int64_t *scatterDvAddrArray,
                                         int64_t *mmPingpongIdxArray);
    __aicore__ inline void AtomicClean();
    __aicore__ inline void DumpGmZero(GlobalTensor<float> &gm, int64_t num);
    __aicore__ inline void InitUB(TPipe *pipe);
    // process
    __aicore__ inline void GetTndSeqLen(const int64_t t1Idx);
    __aicore__ inline void Getbindex(const int64_t ProcessID, const int64_t coreNum, int64_t *bIndexArray);
    __aicore__ inline void CVProcess();
    __aicore__ inline void CVProcessWithOutScatter();
    __aicore__ inline void SelectAndGather();
    __aicore__ inline void ScatterProcess(GlobalTensor<float> &inGmTensor, GlobalTensor<float> &outGmTensor,
                                          int64_t scatterGmOffset, int64_t headDim);
    __aicore__ inline void Scatter();
    __aicore__ inline void recordSecondToLastInfo();
    __aicore__ inline void Gather(GlobalTensor<T1> &inGmTensor, GlobalTensor<T1> &outGmTensor, uint64_t workSpaceOffset,
                                  int64_t headDim);
    __aicore__ inline void SendMatmul12(const int64_t mm2_a_in_addr, const int64_t mm1_a_in_addr,
                                        const int64_t out_addr);
    __aicore__ inline void SendMatmulDV(const int64_t a_in_addr, const int64_t b_in_addr, const int64_t out_addr,
                                        const bool is_sync);
    __aicore__ inline void SendMatmulDQ(const int64_t a_in_addr, const int64_t b_in_addr, const int64_t out_addr,
                                        const bool is_sync);
    __aicore__ inline void SendMatmulDK(const int64_t a_in_addr, const int64_t b_in_addr, const int64_t out_addr,
                                        const bool is_sync);
    __aicore__ inline void VectorProcess(int64_t mm12Addr, int64_t mm345Addr);
    __aicore__ inline void CalRowsumAndSftCopyIn();
    __aicore__ inline void CalAttenMsk(const int32_t processM);
    __aicore__ inline void CalSoftmax(const int32_t loopIdx, const int32_t processM, const int64_t mm12Addr,
                                      const int64_t mm345Addr);
    __aicore__ inline void CalSoftmaxGrad(const int32_t loopIdx, const int32_t processM, const int64_t mm12Addr,
                                          const int64_t mm345Addr);
    __aicore__ inline void CalSub(LocalTensor<float> &dstTensor, LocalTensor<float> &srcTensor, int32_t baseM,
                                  int32_t baseN);
    __aicore__ inline void SetLastComputeInfo(int64_t mm12GmAddr, int64_t scatterDkGmAddr, int64_t scatterDvGmAddr);
    __aicore__ inline void SetAddrInfo();
    __aicore__ inline void ScatterUseOneCore(int64_t *scatterDkAddrArray, int64_t *scatterDvAddrArray,
                                             int64_t *mmPingpongIdxArray, int64_t *bIndexArray, int64_t processNum);

protected:
    StaticParams params;
    GlobalTensor<T1> queryGm;
    GlobalTensor<T1> keyGm;
    GlobalTensor<T1> valueGm;
    GlobalTensor<T1> attentionGm;
    GlobalTensor<T1> attentionGradGm;
    GlobalTensor<float> softmaxMaxGm;
    GlobalTensor<float> softmaxSumGm;
    GlobalTensor<int32_t> topkIndicesGm;
    GlobalTensor<T1> dqGm;
    GlobalTensor<T1> dkGm;
    GlobalTensor<T1> dvGm;
    // workspace
    GlobalTensor<T1> selectedKWorkspaceGm;
    GlobalTensor<T1> selectedVWorkspaceGm;
    GlobalTensor<float> scatterDkWorkspaceGm;
    GlobalTensor<float> scatterDvWorkspaceGm;
    GlobalTensor<float> mm1WorkspaceGm;
    GlobalTensor<float> mm2WorkspaceGm;
    GlobalTensor<T1> dsInputWorkspaceGm;
    GlobalTensor<T1> dvInputWorkspaceGm;
    GlobalTensor<float> dqWorkspaceGm;
    GlobalTensor<float> dkWorkspaceGm;
    GlobalTensor<float> dvWorkspaceGm;

    TBuf<> vecQue;
    LocalTensor<uint8_t> helpTensor;
    LocalTensor<int32_t> topkIndicesTensor;
    LocalTensor<T1> gatherTensor;
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
    LocalTensor<float> scatterTensor;
    GM_ADDR actual_seq_qlen_addr;
    GM_ADDR actual_seq_kvlen_addr;
    GM_ADDR workspace_;

    constexpr static const int32_t BLOCK = 32;
    constexpr static const int32_t BLOCK_T1 = BLOCK / sizeof(T1);
    constexpr static const int32_t BLOCK_FP32 = BLOCK / sizeof(float);
    constexpr static const int32_t BLOCK_INT32 = BLOCK / sizeof(int32_t);
    constexpr static const int32_t MSK_LEN = 16;
    constexpr static const uint32_t ATTEN_MASK_SCALE = 0xFF7FFFFF;
    // core
    int64_t usedCoreNum;
    int64_t formerCoreNum;
    uint32_t blockIdx;
    const TILING_CLASS *__restrict tilingData;
    // Shape
    int64_t dimB;
    int64_t dimN2;
    int64_t dimS1;
    int64_t dimS2;
    int64_t currentS1{0};
    int64_t currentS2{0};
    int64_t currentS1Offset{0};
    int64_t currentS2Offset{0};
    int64_t lastS1Offset;
    int64_t lastS2Offset;
    int64_t secondLastS1Offset;
    int64_t secondLastS2Offset;
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
    uint32_t selectedKWorkspaceLen;
    uint32_t selectedVWorkspaceLen;
    uint32_t scatterDkWorkspaceLen;
    uint32_t scatterDvWorkspaceLen;
    uint32_t mm1WorkspaceLen;
    uint32_t mm2WorkspaceLen;
    uint32_t mm3InputWorkspaceLen;
    uint32_t mm4InputWorkspaceLen;
    int64_t dqWorkspaceLen;
    int64_t dkWorkspaceLen;
    int64_t dvWorkspaceLen;
    // Index
    uint32_t processBS1ByCore;
    int64_t formerCoreProcessNNum{0};
    int64_t scatterCoreNum{0};

    int64_t bIndex{0};
    int64_t s1Index{0};
    int64_t n2Index{0};
    int64_t lastBIndex{0};
    int64_t lastS1Index{0};
    int64_t lastN2Index{0};
    int64_t secondLastS1Index{0};
    int64_t secondLastN2Index{0};
    uint64_t currentLoop{0};
    /*
     * 计算attenmask时attenmask开始计算的下标
     */
    int32_t attenMskStartIdx{0};
    /*
     * 计算attenmask时对应blockSize需要保留的长度。
     * 例如：对于blockSize=64的块，需要保留0-9块，则attenMskRsvLen=10；
     */
    int32_t attenMskRsvLen{0};
    int32_t lastAttenMskStartIdx{0};
    int32_t lastAttenMskRsvLen{0};
    // 地址相关
    int64_t s1QGmAddr{0};
    int64_t s1AGGmAddr{0};
    int64_t s2KGmAddr{0};
    int64_t s2VGmAddr{0};
    int64_t selectedKWspOffset{0};
    int64_t selectedVWspOffset{0};
    int64_t scatterdkWspOffset{0};
    int64_t scatterdvWspOffset{0};
    int64_t mm12InputWspOffset;
    int64_t mm3PangInputWspOffset{0};
    int64_t mm4PangInputWspOffset{0};
    int64_t lastMM12WorkspaceAddr{0};
    int64_t lastMM345InputAddr{0};
    int64_t lastS1QGmAddr{0};
    int64_t lastS1AGGmAddr{0};
    int64_t selectedKGmAddr{0};
    int64_t selectedVGmAddr{0};
    int64_t lastSelectedGmAddr{0};
    int64_t lastSelectedKGmAddr{0};
    int64_t lastScatterDkGmAddr{0};
    int64_t lastScatterDvGmAddr{0};
    int64_t secondLastScatterDkGmAddr{0};
    int64_t secondLastScatterDvGmAddr{0};
    bool isLastBN2S1{false};
    uint32_t mmPingpongIdx{0};
    uint32_t selectdKPPPidx{0};

    event_t vWaitMte2;
    event_t mte3WaitMte2;
    event_t mte3WaitV;
    event_t sWaitMte2;
    event_t mte2WaitMte3;
};

template <typename NSAGT>
__aicore__ inline void SelectedAttentionGrad<NSAGT>::Init(
    GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR attention_out, GM_ADDR attention_out_grad, GM_ADDR softmax_max,
    GM_ADDR softmax_sum, GM_ADDR topk_indices, GM_ADDR actual_seq_qlen, GM_ADDR actual_seq_kvlen, GM_ADDR atten_mask,
    GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR workspace, const TILING_CLASS *__restrict ordTilingData, TPipe *pipe)
{
    InitGMBuffer(query, key, value, attention_out, attention_out_grad, softmax_max, softmax_sum, topk_indices, dq, dk,
                 dv);
    InitParams(ordTilingData, actual_seq_qlen, actual_seq_kvlen);
    InitWorkspace(workspace);
    AtomicClean();
    InitUB(pipe);
}

template <typename NSAGT>
__aicore__ inline void SelectedAttentionGrad<NSAGT>::InitGMBuffer(GM_ADDR query, GM_ADDR key, GM_ADDR value,
                                                                  GM_ADDR attention_out, GM_ADDR attention_out_grad,
                                                                  GM_ADDR softmax_max, GM_ADDR softmax_sum,
                                                                  GM_ADDR topk_indices, GM_ADDR dq, GM_ADDR dk,
                                                                  GM_ADDR dv)
{
    // 必选输入初始化
    queryGm.SetGlobalBuffer((__gm__ T1 *)query);
    keyGm.SetGlobalBuffer((__gm__ T1 *)key);
    valueGm.SetGlobalBuffer((__gm__ T1 *)value);
    attentionGm.SetGlobalBuffer((__gm__ T1 *)attention_out);
    attentionGradGm.SetGlobalBuffer((__gm__ T1 *)attention_out_grad);
    softmaxMaxGm.SetGlobalBuffer((__gm__ float *)softmax_max);
    softmaxSumGm.SetGlobalBuffer((__gm__ float *)softmax_sum);
    topkIndicesGm.SetGlobalBuffer((__gm__ int32_t *)topk_indices);

    // 输出初始化
    dqGm.SetGlobalBuffer((__gm__ T1 *)dq);
    dkGm.SetGlobalBuffer((__gm__ T1 *)dk);
    dvGm.SetGlobalBuffer((__gm__ T1 *)dv);
}

template <typename NSAGT>
__aicore__ inline void SelectedAttentionGrad<NSAGT>::InitParams(const TILING_CLASS *__restrict ordTilingData,
                                                                GM_ADDR actual_seq_qlen, GM_ADDR actual_seq_kvlen)
{
    // op_info
    blockIdx = GetBlockIdx();
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

    actual_seq_qlen_addr = actual_seq_qlen;
    actual_seq_kvlen_addr = actual_seq_kvlen;

    scaleValue = tilingData->opInfo.scaleValue;
    selectedBlockCount = tilingData->opInfo.selectedBlockCount;
    selectedBlockSize = tilingData->opInfo.selectedBlockSize;
    selectedKWorkspaceLen = tilingData->opInfo.selectedKWorkspaceLen;
    selectedVWorkspaceLen = tilingData->opInfo.selectedVWorkspaceLen;
    scatterDkWorkspaceLen = tilingData->opInfo.scatterDkWorkspaceLen;
    scatterDvWorkspaceLen = tilingData->opInfo.scatterDvWorkspaceLen;
    mm1WorkspaceLen = tilingData->opInfo.mm1WorkspaceLen;
    mm2WorkspaceLen = tilingData->opInfo.mm2WorkspaceLen;
    mm3InputWorkspaceLen = tilingData->opInfo.mm3InputWorkspaceLen;
    mm4InputWorkspaceLen = tilingData->opInfo.mm4InputWorkspaceLen;
    dqWorkspaceLen = tilingData->opInfo.dqWorkspaceLen;
    dkWorkspaceLen = tilingData->opInfo.dkWorkspaceLen;
    dvWorkspaceLen = tilingData->opInfo.dvWorkspaceLen;

    selectedKWspOffset = selectedKWorkspaceLen / sizeof(T1) / 3;
    selectedVWspOffset = selectedVWorkspaceLen / sizeof(T1) / 2;
    scatterdkWspOffset = scatterDkWorkspaceLen / sizeof(float) / 2;
    scatterdvWspOffset = scatterDvWorkspaceLen / sizeof(float) / 2;
    mm12InputWspOffset = mm1WorkspaceLen / sizeof(float) / 2;
    mm3PangInputWspOffset = mm3InputWorkspaceLen / sizeof(T1) / 2;
    mm4PangInputWspOffset = mm4InputWorkspaceLen / sizeof(T1) / 2;

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

    if (blockIdx < formerCoreNum) {
        processBS1ByCore = tilingData->opInfo.formerCoreProcessNNum;
    } else {
        processBS1ByCore = tilingData->opInfo.remainCoreProcessNNum;
    }
    formerCoreProcessNNum = tilingData->opInfo.formerCoreProcessNNum;
}

template <typename NSAGT>
__aicore__ inline void SelectedAttentionGrad<NSAGT>::GetTndSeqLen(const int64_t t1Idx)
{
    int64_t curT1 = ((__gm__ int64_t *)actual_seq_qlen_addr)[bIndex];
    while (t1Idx >= curT1) {
        curT1 = ((__gm__ int64_t *)actual_seq_qlen_addr)[++bIndex];
    }

    if (unlikely(bIndex == 0)) {
        currentS1 = ((__gm__ int64_t *)actual_seq_qlen_addr)[0];
        currentS2 = ((__gm__ int64_t *)actual_seq_kvlen_addr)[0];
        currentS1Offset = 0;
        currentS2Offset = 0;
    } else {
        currentS1 =
            ((__gm__ int64_t *)actual_seq_qlen_addr)[bIndex] - ((__gm__ int64_t *)actual_seq_qlen_addr)[bIndex - 1];
        currentS2 =
            ((__gm__ int64_t *)actual_seq_kvlen_addr)[bIndex] - ((__gm__ int64_t *)actual_seq_kvlen_addr)[bIndex - 1];
        currentS1Offset = ((__gm__ int64_t *)actual_seq_qlen_addr)[bIndex - 1];
        currentS2Offset = ((__gm__ int64_t *)actual_seq_kvlen_addr)[bIndex - 1];
    }

    s1Index = t1Idx - currentS1Offset;
}

template <typename NSAGT>
__aicore__ inline void SelectedAttentionGrad<NSAGT>::Getbindex(const int64_t ProcessID, const int64_t coreNum,
                                                               int64_t *bIndexArray)
{
    for (int64_t i = 0; i < coreNum; i++) {
        int64_t curT1 = ((__gm__ int64_t *)actual_seq_qlen_addr)[bIndex];
        while (ProcessID * usedCoreNum + i >= curT1) {
            curT1 = ((__gm__ int64_t *)actual_seq_qlen_addr)[++bIndex];
        }
        bIndexArray[i] = bIndex;
    }
}

template <typename NSAGT>
__aicore__ inline void SelectedAttentionGrad<NSAGT>::InitUB(TPipe *pipe)
{
    uint32_t dataSize = params.singleM * dimDv;
    uint32_t ubOffset = 0;
    uint32_t rowsumUbOffset = 0;
    uint32_t totalUbSpace = 191 * 1024;

    pipe->InitBuffer(vecQue, totalUbSpace);

    // topk
    int32_t topkNumber = NSAG_BASIC::AlignTo<int32_t>(selectedBlockCount, BLOCK);
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
    gatherTensor = vecQue.GetWithOffset<T1>((totalUbSpace - rowsumUbOffset) / sizeof(T1), rowsumUbOffset);
    scatterTensor = vecQue.GetWithOffset<float>((totalUbSpace - rowsumUbOffset) / sizeof(float), rowsumUbOffset);

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
__aicore__ inline void SelectedAttentionGrad<NSAGT>::DumpGmZero(GlobalTensor<float> &gm, int64_t num)
{
    // dump 0 to gm by blockIdx
    int64_t perSize = (num + tilingData->opInfo.castUsedCoreNum - 1) / tilingData->opInfo.castUsedCoreNum;
    int64_t coreNum = (num + perSize - 1) / perSize;
    int64_t tailSize = num - perSize * (coreNum - 1);
    int64_t initSize = perSize;

    if (blockIdx == coreNum - 1) {
        initSize = tailSize;
    }

    if (blockIdx < coreNum) {
        InitOutput<float>(gm[blockIdx * perSize], initSize, 0);
    }
}

template <typename NSAGT>
__aicore__ inline void SelectedAttentionGrad<NSAGT>::AtomicClean()
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
    InitOutput<T1>(selectedKWorkspaceGm, selectedKWorkspaceLen / sizeof(T1), 0);
    InitOutput<T1>(selectedVWorkspaceGm, selectedVWorkspaceLen / sizeof(T1), 0);
}
template <typename NSAGT>
__aicore__ inline void SelectedAttentionGrad<NSAGT>::InitAddrArray(int64_t *scatterDkAddrArray,
                                                                   int64_t *scatterDvAddrArray,
                                                                   int64_t *mmPingpongIdxArray)
{
    int64_t usedWorkspaceLen{0};
    // select
    int64_t selectedKAddr = usedWorkspaceLen / sizeof(T1) + blockIdx * selectedKWorkspaceLen / sizeof(T1);
    usedWorkspaceLen += selectedKWorkspaceLen * usedCoreNum;

    int64_t selectedVAddr = usedWorkspaceLen / sizeof(T1) + blockIdx * selectedVWorkspaceLen / sizeof(T1);
    usedWorkspaceLen += selectedVWorkspaceLen * usedCoreNum;
    usedWorkspaceLen += mm1WorkspaceLen * usedCoreNum;
    usedWorkspaceLen += mm2WorkspaceLen * usedCoreNum;
    usedWorkspaceLen += mm3InputWorkspaceLen * usedCoreNum;
    usedWorkspaceLen += mm4InputWorkspaceLen * usedCoreNum;
    // scatter
    int64_t scatterDkAddrLen = usedWorkspaceLen;

    int64_t scatterDkAddr = usedWorkspaceLen / sizeof(float) + blockIdx * scatterDkWorkspaceLen / sizeof(float);

    usedWorkspaceLen += scatterDkWorkspaceLen * usedCoreNum;
    int64_t scatterDvAddrLen = usedWorkspaceLen;

    int64_t scatterDvAddr = usedWorkspaceLen / sizeof(float) + blockIdx * scatterDvWorkspaceLen / sizeof(float);
    usedWorkspaceLen += scatterDvWorkspaceLen * usedCoreNum;
    for (int64_t i = 0; i < usedCoreNum; i++) {
        scatterDkAddrArray[i] = scatterDkAddrLen / sizeof(float) + i * scatterDkWorkspaceLen / sizeof(float);
        scatterDvAddrArray[i] = scatterDvAddrLen / sizeof(float) + i * scatterDvWorkspaceLen / sizeof(float);
        mmPingpongIdxArray[i] = 0;
    }
    // post
    int64_t dqAddr = usedWorkspaceLen / sizeof(float);
    int64_t dkAddr = dqAddr + dqWorkspaceLen / sizeof(float);
    int64_t dvAddr = dkAddr + dkWorkspaceLen / sizeof(float);
    usedWorkspaceLen += dqWorkspaceLen + dkWorkspaceLen + dvWorkspaceLen;
}

template <typename NSAGT>
__aicore__ inline void SelectedAttentionGrad<NSAGT>::InitWorkspace(GM_ADDR workspace)
{
    workspace_ = workspace;
    int64_t usedWorkspaceLen{0};
    // select
    int64_t selectedKAddr = usedWorkspaceLen / sizeof(T1) + blockIdx * selectedKWorkspaceLen / sizeof(T1);
    usedWorkspaceLen += selectedKWorkspaceLen * usedCoreNum;
    int64_t selectedVAddr = usedWorkspaceLen / sizeof(T1) + blockIdx * selectedVWorkspaceLen / sizeof(T1);
    usedWorkspaceLen += selectedVWorkspaceLen * usedCoreNum;
    // mm12
    int64_t mm1Addr = usedWorkspaceLen / sizeof(float) + blockIdx * mm1WorkspaceLen / sizeof(float);
    usedWorkspaceLen += mm1WorkspaceLen * usedCoreNum;
    int64_t mm2Addr = usedWorkspaceLen / sizeof(float) + blockIdx * mm2WorkspaceLen / sizeof(float);
    usedWorkspaceLen += mm2WorkspaceLen * usedCoreNum;
    // mm345
    int64_t mm3Addr = usedWorkspaceLen / sizeof(T1) + blockIdx * mm3InputWorkspaceLen / sizeof(T1);
    usedWorkspaceLen += mm3InputWorkspaceLen * usedCoreNum;
    int64_t mm4Addr = usedWorkspaceLen / sizeof(T1) + blockIdx * mm4InputWorkspaceLen / sizeof(T1);
    usedWorkspaceLen += mm4InputWorkspaceLen * usedCoreNum;
    // scatter
    int64_t scatterDkAddr = usedWorkspaceLen / sizeof(float) + blockIdx * scatterDkWorkspaceLen / sizeof(float);
    usedWorkspaceLen += scatterDkWorkspaceLen * usedCoreNum;
    int64_t scatterDvAddr = usedWorkspaceLen / sizeof(float) + blockIdx * scatterDvWorkspaceLen / sizeof(float);
    usedWorkspaceLen += scatterDvWorkspaceLen * usedCoreNum;
    // post
    int64_t dqAddr = usedWorkspaceLen / sizeof(float);
    int64_t dkAddr = dqAddr + dqWorkspaceLen / sizeof(float);
    int64_t dvAddr = dkAddr + dkWorkspaceLen / sizeof(float);
    usedWorkspaceLen += dqWorkspaceLen + dkWorkspaceLen + dvWorkspaceLen;

    mm1WorkspaceGm.SetGlobalBuffer((__gm__ float *)workspace + mm1Addr);
    mm2WorkspaceGm.SetGlobalBuffer((__gm__ float *)workspace + mm2Addr);
    dsInputWorkspaceGm.SetGlobalBuffer((__gm__ T1 *)workspace + mm3Addr);
    dvInputWorkspaceGm.SetGlobalBuffer((__gm__ T1 *)workspace + mm4Addr);
    dqWorkspaceGm.SetGlobalBuffer((__gm__ float *)workspace + dqAddr);
    dkWorkspaceGm.SetGlobalBuffer((__gm__ float *)workspace + dkAddr);
    dvWorkspaceGm.SetGlobalBuffer((__gm__ float *)workspace + dvAddr);
    selectedKWorkspaceGm.SetGlobalBuffer((__gm__ T1 *)workspace + selectedKAddr);
    selectedVWorkspaceGm.SetGlobalBuffer((__gm__ T1 *)workspace + selectedVAddr);
    scatterDkWorkspaceGm.SetGlobalBuffer((__gm__ float *)workspace + scatterDkAddr);
    scatterDvWorkspaceGm.SetGlobalBuffer((__gm__ float *)workspace + scatterDvAddr);
}

template <typename NSAGT>
__aicore__ inline void SelectedAttentionGrad<NSAGT>::SendMatmul12(const int64_t mm2_a_in_addr,
                                                                  const int64_t mm1_a_in_addr, const int64_t out_addr)
{
    // mm2
    mm2.SetTensorA(queryGm[mm2_a_in_addr]);
    mm2.SetTensorB(selectedKWorkspaceGm[selectedKGmAddr], true);
    mm2.template IterateAll<false>(mm2WorkspaceGm[out_addr], false, false, true);
    mm2.End();

    // mm1
    mm1.SetTensorA(attentionGradGm[mm1_a_in_addr]);
    mm1.SetTensorB(selectedVWorkspaceGm[selectedVGmAddr], true);
    mm1.template IterateAll<false>(mm1WorkspaceGm[out_addr], false, false, true);
    mm1.End();
}

template <typename NSAGT>
__aicore__ inline void SelectedAttentionGrad<NSAGT>::SendMatmulDQ(const int64_t a_in_addr, const int64_t b_in_addr,
                                                                  const int64_t out_addr, const bool is_sync)
{
    mm3.SetTensorA(dsInputWorkspaceGm[a_in_addr]);
    mm3.SetTensorB(selectedKWorkspaceGm[b_in_addr]);

    if (is_sync) {
        mm3.template IterateAll<true>(dqWorkspaceGm[out_addr]);
    } else {
        mm3.template IterateAll<false>(dqWorkspaceGm[out_addr]);
    }

    mm3.End();
}

template <typename NSAGT>
__aicore__ inline void SelectedAttentionGrad<NSAGT>::SendMatmulDV(const int64_t a_in_addr, const int64_t b_in_addr,
                                                                  const int64_t out_addr, const bool is_sync)
{
    mm4.SetTensorA(dvInputWorkspaceGm[a_in_addr], true);
    mm4.SetTensorB(attentionGradGm[b_in_addr]);

    if (is_sync) {
        mm4.template IterateAll<true>(scatterDvWorkspaceGm[out_addr]);
    } else {
        mm4.template IterateAll<false>(scatterDvWorkspaceGm[out_addr], 0, false, true);
    }
    mm4.End();
}

template <typename NSAGT>
__aicore__ inline void SelectedAttentionGrad<NSAGT>::SendMatmulDK(const int64_t a_in_addr, const int64_t b_in_addr,
                                                                  const int64_t out_addr, const bool is_sync)
{
    mm5.SetTensorA(dsInputWorkspaceGm[a_in_addr], true);
    mm5.SetTensorB(queryGm[b_in_addr]);

    if (is_sync) {
        mm5.template IterateAll<true>(scatterDkWorkspaceGm[out_addr]);
    } else {
        mm5.template IterateAll<false>(scatterDkWorkspaceGm[out_addr], 0, false, true);
    }
    mm5.End();
}

template <typename NSAGT>
__aicore__ inline void SelectedAttentionGrad<NSAGT>::CalSub(LocalTensor<float> &dstTensor,
                                                            LocalTensor<float> &srcTensor, int32_t baseM, int32_t baseN)
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
__aicore__ inline void SelectedAttentionGrad<NSAGT>::CalRowsumAndSftCopyIn()
{
    uint32_t dataSize = params.singleM * dimDv;
    uint64_t sumMaxAddr = lastS1Offset * dimN2 * dimG * BLOCK_FP32 + lastS1Index * dimN2 * dimG * BLOCK_FP32 +
                          lastN2Index * dimG * BLOCK_FP32;
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(dimG * dimDv * sizeof(T1)), 0, 0, 0};
    DataCopyExtParams copyParams2{1, static_cast<uint32_t>(dimG * BLOCK), 0, 0, 0};
    DataCopyPadExtParams<T1> padParams{false, 0, 0, 0};
    DataCopyPadExtParams<float> padParams2{false, 0, 0, 0};

    DataCopyPad(attentionGradT1Tensor, attentionGradGm[lastS1AGGmAddr], copyParams, padParams);
    SetFlag<HardEvent::MTE2_V>(static_cast<int32_t>(vWaitMte2));
    DataCopyPad(attentionT1Tensor, attentionGm[lastS1AGGmAddr], copyParams, padParams);
    SetFlag<HardEvent::MTE2_V>(static_cast<int32_t>(vWaitMte2));
    DataCopyPad(maxTensor, softmaxMaxGm[sumMaxAddr], copyParams2, padParams2);
    DataCopyPad(sumTensor, softmaxSumGm[sumMaxAddr], copyParams2, padParams2);
    SetFlag<HardEvent::MTE2_V>(static_cast<int32_t>(vWaitMte2));
    WaitFlag<HardEvent::MTE2_V>(static_cast<int32_t>(vWaitMte2));
    Cast(attentionGradFP32Tensor, attentionGradT1Tensor, RoundMode::CAST_NONE, dataSize);
    WaitFlag<HardEvent::MTE2_V>(static_cast<int32_t>(vWaitMte2));
    Cast(attentionFP32Tensor, attentionT1Tensor, RoundMode::CAST_NONE, dataSize);
    PipeBarrier<PIPE_V>();
    SoftmaxGradFront<float, false>(rowSumOutTensor, attentionGradFP32Tensor, attentionFP32Tensor, helpTensor,
                                   tilingData->softmaxGradTilingData);
}

template <typename NSAGT>
__aicore__ inline void SelectedAttentionGrad<NSAGT>::CalAttenMsk(const int32_t processM)
{
    if (lastAttenMskStartIdx == -1) {
        return;
    }
    LocalTensor<float> tmpTensor;
    uint64_t mask[1];
    float scalar = *((float *)&ATTEN_MASK_SCALE);

    for (int32_t i = 0; i < processM; i++) {
        int32_t pOffset = i * params.sftBaseN;
        int32_t selectIdx = (lastAttenMskStartIdx - 1) / selectedBlockSize;
        /*
         * 1.处理对角线上的块。目前仅能处理blockSize=64。
         * 2.repeatime=1，一次能处理64个FP32数据。
         * 3.当lastAttenMskRsvLen=0时，mask[0]=UINT64_MAX，表示所有数据需要被掩掉。
         * 4.当lastAttenMskRsvLen=64时，表示不存在对角线的块。
         */
        if (lastAttenMskRsvLen != selectedBlockSize) {
            tmpTensor = pTensor[selectIdx * selectedBlockSize + pOffset];
            int32_t mskLoop = selectedBlockSize / MSK_LEN;
            int32_t mskLen = lastAttenMskRsvLen;
            for (int32_t i = 0; i < mskLoop; i++) {
                if (mskLen > 0 && mskLen < MSK_LEN) {
                    mask[0] = UINT16_MAX - ((1ULL << mskLen) - 1);
                    Duplicate(tmpTensor, scalar, mask, 1, 0, 0);
                } else if (mskLen <= 0) {
                    mask[0] = UINT16_MAX;
                    Duplicate(tmpTensor, scalar, mask, 1, 0, 0);
                }
                mskLen -= MSK_LEN;
                tmpTensor = pTensor[selectIdx * selectedBlockSize + pOffset + (i + 1) * MSK_LEN];
            }
        }

        /*
         * 1.matmul一次计算selectedS2
         * 2.如果需要掩掉的块不仅仅在对角线上，则后面所有的块都需要被掩掉
         */
        if (lastS1Index + 1 < selectedS2) {
            uint32_t mskLen = (selectedBlockCount - (selectIdx + 1)) * selectedBlockSize;
            if (mskLen != 0) {
                tmpTensor = pTensor[(selectIdx + 1) * selectedBlockSize + pOffset];
                Duplicate(tmpTensor, scalar, (selectedBlockCount - (selectIdx + 1)) * selectedBlockSize);
            }
        }
    }
}

template <typename NSAGT>
__aicore__ inline void SelectedAttentionGrad<NSAGT>::CalSoftmax(const int32_t loopIdx, const int32_t processM,
                                                                const int64_t mm12Addr, const int64_t mm345Addr)
{
    int64_t dataSize = static_cast<int64_t>(processM) * static_cast<int64_t>(params.sftBaseN);
    int64_t maxOffset = loopIdx * params.sftBaseM * BLOCK_FP32;
    auto tmpMaxTensor = maxTensor[maxOffset];
    auto tmpSumTensor = sumTensor[maxOffset];

    DataCopyPad(pTensor, mm2WorkspaceGm[mm12Addr], {1, static_cast<uint32_t>(dataSize * sizeof(float)), 0, 0, 0},
                {false, 0, 0, 0});
    SetFlag<HardEvent::MTE2_V>(static_cast<int32_t>(vWaitMte2));
    WaitFlag<HardEvent::MTE2_V>(static_cast<int32_t>(vWaitMte2));

    Muls(pTensor, pTensor, scaleValue, dataSize);
    PipeBarrier<PIPE_V>();

    if constexpr (ATTEN_ENABLE) {
        CalAttenMsk(processM);
        PipeBarrier<PIPE_V>();
    }

    SimpleSoftMax<float, true, false>(pTensor, tmpSumTensor, tmpMaxTensor, pTensor, helpTensor,
                                      tilingData->softmaxTilingData);
    PipeBarrier<PIPE_V>();
    Cast(sftOutT1Tensor, pTensor, RoundMode::CAST_ROUND, dataSize);
    SetFlag<HardEvent::V_MTE3>(static_cast<int32_t>(mte3WaitV));
    WaitFlag<HardEvent::V_MTE3>(static_cast<int32_t>(mte3WaitV));

    DataCopyPad(dvInputWorkspaceGm[mm345Addr], sftOutT1Tensor,
                {1, static_cast<uint32_t>(dataSize * sizeof(T1)), 0, 0, 0});
    SetFlag<HardEvent::MTE3_MTE2>(static_cast<int32_t>(mte2WaitMte3));
    WaitFlag<HardEvent::MTE3_MTE2>(static_cast<int32_t>(mte2WaitMte3));
}

template <typename NSAGT>
__aicore__ inline void SelectedAttentionGrad<NSAGT>::CalSoftmaxGrad(const int32_t loopIdx, const int32_t processM,
                                                                    const int64_t mm12Addr, const int64_t mm345Addr)
{
    int64_t dataSize = static_cast<int64_t>(processM) * static_cast<int64_t>(params.sftBaseN);
    int64_t rowsumOffset = loopIdx * params.sftBaseM * BLOCK_FP32;
    auto tmpRowSumOutTensor = rowSumOutTensor[rowsumOffset];

    DataCopyPad(dPTensor, mm1WorkspaceGm[mm12Addr], {1, static_cast<uint32_t>(dataSize * sizeof(float)), 0, 0, 0},
                {false, 0, 0, 0});
    SetFlag<HardEvent::MTE2_V>(static_cast<int32_t>(vWaitMte2));
    WaitFlag<HardEvent::MTE2_V>(static_cast<int32_t>(vWaitMte2));
    CalSub(dPTensor, tmpRowSumOutTensor, processM, params.sftBaseN);
    PipeBarrier<PIPE_V>();
    Mul(pTensor, pTensor, dPTensor, dataSize);
    PipeBarrier<PIPE_V>();
    Cast(sftgOutT1Tensor, pTensor, RoundMode::CAST_ROUND, dataSize);
    SetFlag<HardEvent::V_MTE3>(static_cast<int32_t>(mte3WaitV));
    WaitFlag<HardEvent::V_MTE3>(static_cast<int32_t>(mte3WaitV));

    DataCopyPad(dsInputWorkspaceGm[mm345Addr], sftgOutT1Tensor,
                {1, static_cast<uint32_t>(dataSize * sizeof(T1)), 0, 0, 0});
    SetFlag<HardEvent::MTE3_MTE2>(static_cast<int32_t>(mte2WaitMte3));
    WaitFlag<HardEvent::MTE3_MTE2>(static_cast<int32_t>(mte2WaitMte3));
}

template <typename NSAGT>
__aicore__ inline void SelectedAttentionGrad<NSAGT>::VectorProcess(int64_t mm12Addr, int64_t mm345Addr)
{
    CalRowsumAndSftCopyIn();

    int32_t loop = (params.singleM + params.sftBaseM - 1) / params.sftBaseM;
    int32_t processM = params.sftBaseM;
    int32_t tailM = params.singleM % params.sftBaseM;
    int64_t dataSize = static_cast<int64_t>(processM) * static_cast<int64_t>(params.sftBaseN);

    for (int32_t i = 0; i < loop; i++) {
        if (i == 0) {
            mm2.WaitIterateAll();
            WaitFlag<HardEvent::MTE2_V>(static_cast<int32_t>(vWaitMte2)); // wait softmax_max and softmax_sum MTE2
        }
        if (i == loop - 1 && tailM != 0) {
            processM = tailM;
        }
        CalSoftmax(i, processM, mm12Addr, mm345Addr);

        if (i == 0) {
            mm1.WaitIterateAll();
        }
        CalSoftmaxGrad(i, processM, mm12Addr, mm345Addr);

        mm12Addr += dataSize;
        mm345Addr += dataSize;
    }
}

template <typename NSAGT>
__aicore__ inline void SelectedAttentionGrad<NSAGT>::ScatterProcess(GlobalTensor<float> &inGmTensor,
                                                                    GlobalTensor<float> &outGmTensor,
                                                                    int64_t scatterGmOffset, int64_t headDim)
{
    DataCopyParams intriParams;
    intriParams.blockCount = selectedBlockSize;
    intriParams.blockLen = headDim * sizeof(float);
    intriParams.srcStride = 0;
    intriParams.dstStride = (dimN2 - 1) * headDim * sizeof(float);
    for (int32_t i = 0; i < selectedBlockCount; i++) {
        int32_t topkIdx = topkIndicesTensor.GetValue(i);

        int32_t s2Idx = topkIdx * selectedBlockSize;
        int64_t outOffset =
            secondLastS2Offset * (dimN2 * headDim) + s2Idx * (dimN2 * headDim) + secondLastN2Index * headDim;
        int32_t scatterOffset = scatterGmOffset + i * selectedBlockSize * headDim;

        DataCopy(scatterTensor, inGmTensor[scatterOffset], selectedBlockSize * headDim);

        SetFlag<HardEvent::MTE2_MTE3>(static_cast<int32_t>(mte3WaitMte2));
        WaitFlag<HardEvent::MTE2_MTE3>(static_cast<int32_t>(mte3WaitMte2));

        SetAtomicAdd<float>();
        DataCopyPad(outGmTensor[outOffset], scatterTensor, intriParams);
        SetAtomicNone();

        SetFlag<HardEvent::MTE3_MTE2>(static_cast<int32_t>(mte2WaitMte3));
        WaitFlag<HardEvent::MTE3_MTE2>(static_cast<int32_t>(mte2WaitMte3));
    }
}

template <typename NSAGT>
__aicore__ inline void SelectedAttentionGrad<NSAGT>::Scatter()
{
    PipeBarrier<PIPE_ALL>();
    // copy tokIndices in
    uint64_t gmOffset = secondLastS1Offset * (dimN2 * selectedBlockCount) +
                        secondLastS1Index * (dimN2 * selectedBlockCount) + secondLastN2Index * selectedBlockCount;
    DataCopyParams intriParams;
    intriParams.blockCount = 1;
    intriParams.blockLen = selectedBlockCount * sizeof(int32_t);
    intriParams.srcStride = 0;
    intriParams.dstStride = 0;
    DataCopyPad(topkIndicesTensor, topkIndicesGm[gmOffset], intriParams, {false, 0, 0, 0});
    SetFlag<HardEvent::MTE2_S>(static_cast<int32_t>(sWaitMte2));
    WaitFlag<HardEvent::MTE2_S>(static_cast<int32_t>(sWaitMte2));

    ScatterProcess(scatterDkWorkspaceGm, dkWorkspaceGm, secondLastScatterDkGmAddr, dimDqk);
    ScatterProcess(scatterDvWorkspaceGm, dvWorkspaceGm, secondLastScatterDvGmAddr, dimDv);
    PipeBarrier<PIPE_ALL>();
}

template <typename NSAGT>
__aicore__ inline void SelectedAttentionGrad<NSAGT>::Gather(GlobalTensor<T1> &inGmTensor, GlobalTensor<T1> &outGmTensor,
                                                            uint64_t workSpaceOffset, int64_t headDim)
{
    DataCopyParams intriParams;
    int32_t totalProcessLen = selectedS2;
    int32_t loop = selectedBlockCount;
    int32_t processLen = selectedBlockSize;
    int32_t tailLen = totalProcessLen % processLen;
    int32_t startS2Idx = 0;
    int32_t dataSize = processLen * headDim;
    uint64_t inGmOffset;

    if constexpr (ATTEN_ENABLE) {
        attenMskStartIdx = -1;
        if (s1Index + 1 < selectedS2) {
            totalProcessLen = s1Index + 1;
            loop = (totalProcessLen + selectedBlockSize - 1) / selectedBlockSize;
            tailLen = totalProcessLen % processLen;
        }
    }

    for (uint32_t i = 0; i < loop; i++) {
        if (i == loop - 1 && tailLen != 0) {
            processLen = tailLen;
            dataSize = processLen * headDim;
        }
        int32_t topkIdx = topkIndicesTensor.GetValue(i);

        if constexpr (ATTEN_ENABLE) {
            // 处于对角线上的block
            if (topkIdx * selectedBlockSize <= s1Index && (topkIdx + 1) * selectedBlockSize > s1Index) {
                attenMskRsvLen = s1Index + 1 - (topkIdx * selectedBlockSize);
                attenMskStartIdx = i * selectedBlockSize + attenMskRsvLen;
            }

            if (s1Index + 1 < selectedS2) {
                startS2Idx = i * selectedBlockSize;
            } else {
                startS2Idx = topkIdx * selectedBlockSize;
            }
        } else {
            startS2Idx = topkIdx * selectedBlockSize;
        }

        inGmOffset = currentS2Offset * (dimN2 * headDim) + startS2Idx * (dimN2 * headDim) + n2Index * headDim;
        intriParams.blockCount = processLen;
        intriParams.blockLen = headDim * sizeof(T1);
        intriParams.srcStride = (dimN2 - 1) * headDim * sizeof(T1);
        intriParams.dstStride = 0;
        DataCopyPad(gatherTensor, inGmTensor[inGmOffset], intriParams, {false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_MTE3>(static_cast<int32_t>(mte3WaitMte2));
        WaitFlag<HardEvent::MTE2_MTE3>(static_cast<int32_t>(mte3WaitMte2));

        intriParams.blockCount = 1;
        intriParams.blockLen = dataSize * sizeof(T1);
        intriParams.srcStride = 0;
        intriParams.dstStride = 0;
        DataCopyPad(outGmTensor[workSpaceOffset], gatherTensor, intriParams);
        SetFlag<HardEvent::MTE3_MTE2>(static_cast<int32_t>(mte2WaitMte3));
        WaitFlag<HardEvent::MTE3_MTE2>(static_cast<int32_t>(mte2WaitMte3));

        workSpaceOffset += dataSize;
    }
}

template <typename NSAGT>
__aicore__ inline void SelectedAttentionGrad<NSAGT>::SelectAndGather()
{
    // copy tokIndices in
    DataCopyParams intriParams;
    intriParams.blockCount = 1;
    intriParams.blockLen = selectedBlockCount * sizeof(int32_t);
    intriParams.srcStride = 0;
    intriParams.dstStride = 0;
    uint64_t gmOffset = currentS1Offset * (dimN2 * selectedBlockCount) + s1Index * (dimN2 * selectedBlockCount) +
                        n2Index * selectedBlockCount;
    DataCopyPad(topkIndicesTensor, topkIndicesGm[gmOffset], intriParams, {false, 0, 0, 0});
    SetFlag<HardEvent::MTE2_S>(static_cast<int32_t>(sWaitMte2));
    WaitFlag<HardEvent::MTE2_S>(static_cast<int32_t>(sWaitMte2));

    Gather(keyGm, selectedKWorkspaceGm, selectedKGmAddr, dimDqk);
    Gather(valueGm, selectedVWorkspaceGm, selectedVGmAddr, dimDv);
    PipeBarrier<PIPE_ALL>();
}

template <typename NSAGT>
__aicore__ inline void SelectedAttentionGrad<NSAGT>::recordSecondToLastInfo()
{
    secondLastScatterDkGmAddr = lastScatterDkGmAddr;
    secondLastScatterDvGmAddr = lastScatterDvGmAddr;
    secondLastS1Offset = lastS1Offset;
    secondLastS2Offset = lastS2Offset;
    secondLastS1Index = lastS1Index;
    secondLastN2Index = lastN2Index;
}

template <typename NSAGT>
__aicore__ inline void SelectedAttentionGrad<NSAGT>::SetLastComputeInfo(int64_t mm12GmAddr, int64_t scatterDkGmAddr,
                                                                        int64_t scatterDvGmAddr)
{
    lastScatterDkGmAddr = scatterDkGmAddr;
    lastScatterDvGmAddr = scatterDvGmAddr;
    lastMM12WorkspaceAddr = mm12GmAddr;
    lastMM345InputAddr = mmPingpongIdx * mm4PangInputWspOffset;
    lastS1QGmAddr = s1QGmAddr;
    lastS1AGGmAddr = s1AGGmAddr;
    lastSelectedKGmAddr = selectedKGmAddr;
    mmPingpongIdx = 1 - mmPingpongIdx;
    selectdKPPPidx = (selectdKPPPidx + 1) % 3;
    currentLoop++; // 记录生效的绝对次数
    lastBIndex = bIndex;
    lastS1Offset = currentS1Offset;
    lastS2Offset = currentS2Offset;
    lastS1Index = s1Index;
    lastN2Index = n2Index;
    lastAttenMskRsvLen = attenMskRsvLen;
    lastAttenMskStartIdx = attenMskStartIdx;
}

template <typename NSAGT>
__aicore__ inline void SelectedAttentionGrad<NSAGT>::SetAddrInfo()
{
    // Query: B S1 N2 G D
    s1QGmAddr = currentS1Offset * (dimN2 * dimG * dimD) + s1Index * (dimN2 * dimG * dimD) + n2Index * (dimG * dimD);
    // attentionGrad: B S1 N2 G D2
    s1AGGmAddr = currentS1Offset * (dimN2 * dimG * dimD2) + s1Index * (dimN2 * dimG * dimD2) + n2Index * (dimG * dimD2);
    // Key: B S2 N2 1 D
    s2KGmAddr = currentS2Offset * (dimN2 * dimD) + n2Index * dimD;
    // Value: B S2 N2 1 D2
    s2VGmAddr = currentS2Offset * (dimN2 * dimD2) + n2Index * dimD2;
    selectedKGmAddr = selectdKPPPidx * selectedKWspOffset;
    selectedVGmAddr = mmPingpongIdx * selectedVWspOffset;
}

template <typename NSAGT>
__aicore__ inline void SelectedAttentionGrad<NSAGT>::CVProcess()
{
    SetAddrInfo();
    int64_t mm12GmAddr = mmPingpongIdx * mm12InputWspOffset;
    int64_t scatterDkGmAddr = mmPingpongIdx * scatterdkWspOffset;
    int64_t scatterDvGmAddr = mmPingpongIdx * scatterdvWspOffset;

    SelectAndGather();
    SendMatmul12(s1QGmAddr, s1AGGmAddr, mm12GmAddr);

    if (likely(currentLoop > 0)) {
        VectorProcess(lastMM12WorkspaceAddr, lastMM345InputAddr);
        SendMatmulDV(lastMM345InputAddr, lastS1AGGmAddr, lastScatterDvGmAddr, false);
        SendMatmulDQ(lastMM345InputAddr, lastSelectedKGmAddr, lastS1QGmAddr, false);
        SendMatmulDK(lastMM345InputAddr, lastS1QGmAddr, lastScatterDkGmAddr, false);
        if (currentLoop > 1) {
            mm4.WaitIterateAll();
            mm5.WaitIterateAll();
            Scatter();
        }
        recordSecondToLastInfo();
    }
    SetLastComputeInfo(mm12GmAddr, scatterDkGmAddr, scatterDvGmAddr);

    // 当前核的最后一次
    bool is_last = isLastBN2S1;
    if (unlikely(is_last)) {
        if (secondLastScatterDkGmAddr != lastScatterDkGmAddr) {
            mm4.WaitIterateAll();
            mm5.WaitIterateAll();
            Scatter();
        }
        recordSecondToLastInfo();
        VectorProcess(lastMM12WorkspaceAddr, lastMM345InputAddr);
        SendMatmulDV(lastMM345InputAddr, lastS1AGGmAddr, lastScatterDvGmAddr, true);
        SendMatmulDQ(lastMM345InputAddr, lastSelectedKGmAddr, lastS1QGmAddr, true);
        SendMatmulDK(lastMM345InputAddr, lastS1QGmAddr, lastScatterDkGmAddr, true);
        Scatter();
    }
}

template <typename NSAGT>
__aicore__ inline void SelectedAttentionGrad<NSAGT>::CVProcessWithOutScatter()
{
    SetAddrInfo();

    int64_t mm12GmAddr = mmPingpongIdx * mm12InputWspOffset;
    int64_t scatterDkGmAddr = mmPingpongIdx * scatterdkWspOffset;
    int64_t scatterDvGmAddr = mmPingpongIdx * scatterdvWspOffset;

    SelectAndGather();
    SendMatmul12(s1QGmAddr, s1AGGmAddr, mm12GmAddr);
    SetLastComputeInfo(mm12GmAddr, scatterDkGmAddr, scatterDvGmAddr);

    VectorProcess(lastMM12WorkspaceAddr, lastMM345InputAddr);
    SendMatmulDV(lastMM345InputAddr, lastS1AGGmAddr, lastScatterDvGmAddr, false);
    SendMatmulDQ(lastMM345InputAddr, lastSelectedKGmAddr, lastS1QGmAddr, false);
    SendMatmulDK(lastMM345InputAddr, lastS1QGmAddr, lastScatterDkGmAddr, false);
    mm4.WaitIterateAll();
    mm5.WaitIterateAll();
    SyncAll();
}

template <typename NSAGT>
__aicore__ inline void SelectedAttentionGrad<NSAGT>::Process()
{
    if (blockIdx >= usedCoreNum) {
        return;
    }
    PipeBarrier<PIPE_ALL>(); // wait init GM

    for (int64_t i = 0; i < processBS1ByCore; i++) {
        GetTndSeqLen(blockIdx + usedCoreNum * i);
        for (int64_t n2Idx = 0; n2Idx < dimN2; n2Idx++) {
            n2Index = n2Idx;
            isLastBN2S1 = (i == processBS1ByCore - 1 && n2Idx == dimN2 - 1) ? true : false;
            CVProcess();
        }
    }

    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_S>(sWaitMte2);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_MTE3>(mte3WaitMte2);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_MTE3>(mte2WaitMte3);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(vWaitMte2);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(mte3WaitV);

    SyncAll();
}

template <typename NSAGT>
__aicore__ inline void SelectedAttentionGrad<NSAGT>::ScatterUseOneCore(int64_t *scatterDkAddrArray,
                                                                       int64_t *scatterDvAddrArray,
                                                                       int64_t *mmPingpongIdxArray,
                                                                       int64_t *bIndexArray, int64_t processNum)
{
    for (int64_t n2Idx = 0; n2Idx < dimN2; n2Idx++) {
        SyncAll();
        for (int64_t nowblockIdx = 0; nowblockIdx < scatterCoreNum; nowblockIdx++) {
            if (unlikely(bIndexArray[nowblockIdx] == 0)) {
                currentS1 = ((__gm__ int64_t *)actual_seq_qlen_addr)[0];
                currentS2 = ((__gm__ int64_t *)actual_seq_kvlen_addr)[0];
                currentS1Offset = 0;
                currentS2Offset = 0;
            } else {
                currentS1 = ((__gm__ int64_t *)actual_seq_qlen_addr)[bIndexArray[nowblockIdx]] -
                            ((__gm__ int64_t *)actual_seq_qlen_addr)[bIndexArray[nowblockIdx] - 1];
                currentS2 = ((__gm__ int64_t *)actual_seq_kvlen_addr)[bIndexArray[nowblockIdx]] -
                            ((__gm__ int64_t *)actual_seq_kvlen_addr)[bIndexArray[nowblockIdx] - 1];
                currentS1Offset = ((__gm__ int64_t *)actual_seq_qlen_addr)[bIndexArray[nowblockIdx] - 1];
                currentS2Offset = ((__gm__ int64_t *)actual_seq_kvlen_addr)[bIndexArray[nowblockIdx] - 1];
            }
            s1Index = nowblockIdx + usedCoreNum * processNum - currentS1Offset;
            scatterDkWorkspaceGm.SetGlobalBuffer((__gm__ float *)workspace_ + scatterDkAddrArray[nowblockIdx]);
            scatterDvWorkspaceGm.SetGlobalBuffer((__gm__ float *)workspace_ + scatterDvAddrArray[nowblockIdx]);
            n2Index = n2Idx;
            secondLastScatterDkGmAddr = mmPingpongIdxArray[nowblockIdx] * scatterdkWspOffset;
            secondLastScatterDvGmAddr = mmPingpongIdxArray[nowblockIdx] * scatterdvWspOffset;
            secondLastS1Offset = currentS1Offset;
            secondLastS2Offset = currentS2Offset;
            secondLastS1Index = s1Index;
            secondLastN2Index = n2Index;
            Scatter();
            mmPingpongIdxArray[nowblockIdx] = 1 - mmPingpongIdxArray[nowblockIdx];
        }
    }
}

template <typename NSAGT>
__aicore__ inline void SelectedAttentionGrad<NSAGT>::DeterministicProcess()
{
    if (blockIdx < usedCoreNum) {
        PipeBarrier<PIPE_ALL>(); // wait init GM
        for (int64_t i = 0; i < processBS1ByCore; i++) {
            GetTndSeqLen(blockIdx + usedCoreNum * i);
            for (int64_t n2Idx = 0; n2Idx < dimN2; n2Idx++) {
                n2Index = n2Idx;
                isLastBN2S1 = (i == processBS1ByCore - 1 && n2Idx == dimN2 - 1) ? true : false;
                CVProcessWithOutScatter();
            }
        }
        if (formerCoreProcessNNum > processBS1ByCore) {
            for (int64_t n2Idx = 0; n2Idx < dimN2; n2Idx++) {
                SyncAll();
            }
        }
    } else {
        int64_t scatterDkAddrArray[96];
        int64_t scatterDvAddrArray[96];
        int64_t bIndexArray[96];
        int64_t mmPingpongIdxArray[96];
        int64_t scatterDkGmAddr;
        int64_t scatterDvGmAddr;
        InitAddrArray(scatterDkAddrArray, scatterDvAddrArray, mmPingpongIdxArray);
        PipeBarrier<PIPE_ALL>(); // wait init GM
        scatterCoreNum = usedCoreNum;
        for (int64_t i = 0; i < formerCoreProcessNNum - 1; i++) {
            Getbindex(i, usedCoreNum, bIndexArray);
            ScatterUseOneCore(scatterDkAddrArray, scatterDvAddrArray, mmPingpongIdxArray, bIndexArray, i);
        }

        Getbindex(formerCoreProcessNNum - 1, formerCoreNum, bIndexArray);
        scatterCoreNum = formerCoreNum;
        ScatterUseOneCore(scatterDkAddrArray, scatterDvAddrArray, mmPingpongIdxArray, bIndexArray,
                          formerCoreProcessNNum - 1);
    }

    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_S>(sWaitMte2);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_MTE3>(mte3WaitMte2);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_MTE3>(mte2WaitMte3);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(vWaitMte2);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(mte3WaitV);

    SyncAll();
}
