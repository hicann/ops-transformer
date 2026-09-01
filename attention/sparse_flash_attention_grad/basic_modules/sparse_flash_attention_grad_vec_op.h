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
#include "sparse_flash_attention_grad_common_header.h"

namespace SFAG_BASIC {
struct StaticParams {
    uint32_t singleM = 0;
    uint32_t singleN = 0;
    uint32_t sftBaseM = 0;
    uint32_t sftBaseN = 0;
};

constexpr static int32_t repeatMaxBytes = 256;
constexpr static int32_t repeatMaxTimes = 255;
constexpr static int32_t repeatMaxSize = repeatMaxBytes / 4; // 4 means sizeof(T)

template <typename SFAGT>
class VecOp {
public:
    using TILING_CLASS = typename SFAGT::tiling_class;
    using T1 = typename SFAGT::t1;
    static constexpr uint32_t ATTEN_ENABLE = SFAGT::atten_enable;
    static constexpr bool HAS_ROPE = SFAGT::has_rope;
    static constexpr bool IS_BSND = SFAGT::is_bsnd;
    static constexpr bool IS_DETERMINISTIC = SFAGT::is_deterministic;
    static constexpr bool KV_MERGE = SFAGT::kv_merge;

    __aicore__ inline VecOp(){};
    __aicore__ inline void Init(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR attention_out,
                                GM_ADDR attention_out_grad, GM_ADDR softmax_max, GM_ADDR softmax_sum,
                                GM_ADDR topk_indices, GM_ADDR actual_seq_qlen, GM_ADDR actual_seq_kvlen,
                                GM_ADDR key_rope, GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR workspace,
                                const TILING_CLASS *__restrict ordTilingData, TPipe *pipe);
    __aicore__ inline void Process(const RunInfo &runInfo);
    __aicore__ inline void GatherKV(const int64_t n2Index, uint64_t currentS1Offset, const RunInfo &runInfo);
    __aicore__ inline void ScatterAddUnDeter(const RunInfo &runInfo);
    __aicore__ inline void ScatterAddDeter(const RunInfo &runInfo);
    __aicore__ inline void ScatterAdd(const RunInfo &runInfo);
    __aicore__ inline void CleanDqRow(const RunInfo &runInfo);

protected:
    __aicore__ inline void InitParams(const TILING_CLASS *__restrict ordTilingData, GM_ADDR actual_seq_qlen,
                                      GM_ADDR actual_seq_kvlen);
    __aicore__ inline void InitGMBuffer(GM_ADDR key, GM_ADDR value, GM_ADDR dv, GM_ADDR attention_out,
                                        GM_ADDR attention_out_grad, GM_ADDR softmax_max, GM_ADDR softmax_sum,
                                        GM_ADDR topk_indices, GM_ADDR key_rope, GM_ADDR workspace);
    __aicore__ inline void InitUB(TPipe *pipe);
    __aicore__ inline void AtomicClean();
    __aicore__ inline void GatherKVNonOptimized(const int64_t n2Index, uint64_t currentS1Offset,
                                                const RunInfo &runInfo);
    __aicore__ inline void GatherKVOptimized(const int64_t n2Index, uint64_t currentS1Offset, const RunInfo &runInfo);
    template <typename GM_TYPE>
    __aicore__ inline void DumpGmZero(GlobalTensor<GM_TYPE> &gm, int64_t num);
    __aicore__ inline void CalRowsumAndSftCopyIn(const int32_t loopIdx, const int64_t dyGmOffset,
                                                 const int64_t sumGmOffset, const int32_t processM,
                                                 const bool calcSoftmaxGradFront);
    __aicore__ inline void CalAttenMsk(const int32_t processM, const int32_t actualSelS2Align, const RunInfo &runInfo);
    __aicore__ inline void CalSoftmax(const int32_t loopIdx, const int32_t processM, const int64_t mm12Addr,
                                      const int64_t mm345Addr, const RunInfo &runInfo);
    __aicore__ inline void CalSoftmaxGrad(const int32_t loopIdx, const int32_t processM, const int64_t mm12Addr,
                                          const int64_t mm345Addr, const RunInfo &runInfo);
    __aicore__ inline void CalSub(LocalTensor<float> &dstTensor, LocalTensor<float> &srcTensor, int32_t baseM,
                                  int32_t baseN);
    __aicore__ inline void NzToNd(Nz2NdInfo &nz2NdInfo, const GlobalTensor<float> &bmmResGm, LocalTensor<float> &tempUb,
                                  LocalTensor<float> &bmmResUb, event_t eventIdMte2ToV, event_t backEventTmp,
                                  event_t backEvent, bool isBackEvent = false, int64_t dstLastAxis = 0);

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
    GlobalTensor<float> softmaxMaxGm;
    GlobalTensor<float> softmaxSumGm;
    GlobalTensor<int32_t> topkIndicesGm;
    GlobalTensor<T1> keyGm;
    GlobalTensor<T1> valueGm;
    GlobalTensor<T1> keyRopeGm;

    // workspace
    GlobalTensor<T1> selectedKWorkspaceGm;
    GlobalTensor<float> mm1WorkspaceGm;
    GlobalTensor<float> mm2WorkspaceGm;
    GlobalTensor<T1> pWorkspaceGm;
    GlobalTensor<T1> dsWorkspaceGm;
    GlobalTensor<float> dqWorkspaceGm;
    GlobalTensor<float> dkWorkspaceGm;
    GlobalTensor<float> dvWorkspaceGm;
    GlobalTensor<float> mm4ResWorkspaceGm; // 24 * SCATTER_BUFFER_NUM * K * Dk
    GlobalTensor<float> mm5ResWorkspaceGm; // 24 * SCATTER_BUFFER_NUM * K * Dv

    TBuf<> vecQue;
    LocalTensor<uint8_t> helpTensor;
    LocalTensor<float> rowSumOutTensor;
    LocalTensor<float> maxTensor;
    LocalTensor<float> sumTensor;
    LocalTensor<float> maxTmp;
    LocalTensor<float> sumTmp;
    LocalTensor<T1> attentionT1Tensor;
    LocalTensor<float> attentionFP32Tensor;
    LocalTensor<T1> attentionGradT1Tensor;
    LocalTensor<float> attentionGradFP32Tensor;
    LocalTensor<float> pTensor;
    LocalTensor<float> dPTensor;
    LocalTensor<T1> sftOutT1Tensor;
    LocalTensor<T1> sftgOutT1Tensor;
    LocalTensor<T1> gatherTensorPing;
    LocalTensor<T1> gatherTensorPong;
    LocalTensor<T1> gatherRopePing;
    LocalTensor<T1> gatherRopePong;
    LocalTensor<float> scatterAddTensorK;
    LocalTensor<float> scatterAddTensorV;
    LocalTensor<float> tempTensorK;
    LocalTensor<float> tempTensorV;

    // 地址相关
    int64_t selectedKWspOffset{0};
    int64_t selectedVWspOffset{0};
    bool enableOptimizedScatter{false};

    constexpr static const int32_t BLOCK = 32;
    constexpr static const int32_t BLOCK_T1 = BLOCK / sizeof(T1);
    constexpr static const int32_t BLOCK_FP32 = BLOCK / sizeof(float);
    constexpr static const int32_t BLOCK_INT32 = BLOCK / sizeof(int32_t);
    constexpr static const int32_t MSK_LEN = 64;
    constexpr static const uint32_t ATTEN_MASK_SCALE = 0xFF7FFFFF;
    constexpr static const uint32_t MAX_GATHER_SIZE = 32;
    constexpr static const uint32_t UB_ROW_SIZE = 4;
    constexpr static const uint32_t NON_OPT_MAX_GATHER_SIZE = 64;
    constexpr static const uint32_t NON_OPT_UB_ROW_SIZE = 16;

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
    int64_t curS1;
    int64_t curS2;
    int64_t lastBlock;
    int64_t selectedBlockSizeDqk;
    int64_t selectedBlockSizeDrope;
    int64_t selectedBlockSizeDimDAlign;
    int64_t selectedBlockSizeDimD2Align;
    int64_t ubRowSizeDAlign;
    int64_t ubRowSizeD2Align;
    float scaleValue;
    uint32_t selectedBlockCount;
    uint32_t selectedBlockSize;
    uint32_t selectedCountOffset;
    uint32_t actualSelectedCount;
    int32_t maxSelCnt;
    int64_t dimRope;
    // workspace
    uint64_t mm12WorkspaceLen;
    int64_t dqWorkspaceLen;
    int64_t dkWorkspaceLen;
    int64_t dvWorkspaceLen;
    int64_t selectedKWorkspaceLen;
    int64_t concatQWorkspaceLen;
    event_t vWaitMte2;
    event_t vWaitMte2Pong;
    event_t vWaitMte3;
    event_t mte3WaitMte2;
    event_t mte3WaitV;
    event_t sWaitMte2;
    event_t mte3WaitMte2Pong;
    event_t mte2WaitMte3Proc;
    event_t mte2WaitMte3;
    event_t mte2WaitMte3Pong;

    DataCopyPadExtParams<T1> padParams;
    DataCopyExtParams intriParamsKey;
    DataCopyExtParams intriParamsRope;
    DataCopyExtParams outParamK;
    DataCopyExtParams outParamRope;
    DataCopyExtParams scatterParams;
    uint32_t mergePingPong{0};
    int32_t pingPongIdx = 0;
};

template <typename SFAGT>
__aicore__ inline void VecOp<SFAGT>::Init(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR attention_out,
                                          GM_ADDR attention_out_grad, GM_ADDR softmax_max, GM_ADDR softmax_sum,
                                          GM_ADDR topk_indices, GM_ADDR actual_seq_qlen, GM_ADDR actual_seq_kvlen,
                                          GM_ADDR key_rope, GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR workspace,
                                          const TILING_CLASS *__restrict ordTilingData, TPipe *pipe)
{
    InitParams(ordTilingData, actual_seq_qlen, actual_seq_kvlen);
    InitGMBuffer(key, value, dv, attention_out, attention_out_grad, softmax_max, softmax_sum, topk_indices, key_rope,
                 workspace);
    InitUB(pipe);
    AtomicClean();
}

template <typename SFAGT>
__aicore__ inline void VecOp<SFAGT>::InitParams(const TILING_CLASS *__restrict ordTilingData, GM_ADDR actual_seq_qlen,
                                                GM_ADDR actual_seq_kvlen)
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
    dimG = tilingData->opInfo.G;
    dimD = tilingData->opInfo.D;
    dimD2 = tilingData->opInfo.D2;
    dimDqk = tilingData->opInfo.D;
    dimDv = tilingData->opInfo.D2;
    dimRope = tilingData->opInfo.ropeD;
    dimDAlign = (dimD + dimRope + BLOCK_T1 - 1) / BLOCK_T1 * BLOCK_T1;
    dimD2Align = (dimD2 + BLOCK_T1 - 1) / BLOCK_T1 * BLOCK_T1;

    scaleValue = tilingData->opInfo.scaleValue;
    selectedBlockCount = tilingData->opInfo.selectedBlockCount;
    selectedBlockSize = tilingData->opInfo.selectedBlockSize;
    mm12WorkspaceLen = tilingData->opInfo.mm12WorkspaceLen;
    dqWorkspaceLen = tilingData->opInfo.dqWorkspaceLen;
    dkWorkspaceLen = tilingData->opInfo.dkWorkspaceLen;
    dvWorkspaceLen = tilingData->opInfo.dvWorkspaceLen;
    selectedKWorkspaceLen = tilingData->opInfo.selectedKWorkspaceLen;
    selectedBlockSizeDqk = selectedBlockSize * dimDqk;
    selectedBlockSizeDrope = selectedBlockSize * dimRope;
    selectedBlockSizeDimDAlign = selectedBlockSize * dimDAlign;
    selectedBlockSizeDimD2Align = selectedBlockSize * dimD2Align;
    enableOptimizedScatter = tilingData->opInfo.enableOptimizedScatter;
    ubRowSizeDAlign = UB_ROW_SIZE * dimDAlign;
    ubRowSizeD2Align = UB_ROW_SIZE * dimD2Align;

    params.singleM = tilingData->splitCoreParams.singleM;
    params.singleN = tilingData->splitCoreParams.singleN;
    params.sftBaseM = tilingData->splitCoreParams.sftBaseM;
    params.sftBaseN = tilingData->splitCoreParams.sftBaseN;

    selectedS2 = selectedBlockCount * selectedBlockSize;
    selectedCountOffset = PER_LOOP_BLOCK_SIZE / selectedBlockSize;
    if (tilingData->opInfo.selectedBlockSize * tilingData->opInfo.selectedBlockCount <= PER_LOOP_BLOCK_SIZE) {
        selectedCountOffset = tilingData->opInfo.selectedBlockCount;
    }

    if constexpr (IS_BSND == false) {
        dimTq = dimS1;
        dimTkv = dimS2;
    } else {
        dimTq = dimB * dimS1;
        dimTkv = dimB * dimS2;
    }

    intriParamsKey.blockLen = selectedBlockSize * dimDqk * sizeof(T1);
    intriParamsKey.dstStride = 0;
    intriParamsKey.blockCount = 2;

    intriParamsRope.blockLen = selectedBlockSize * dimRope * sizeof(T1);
    intriParamsRope.dstStride = 0;
    intriParamsRope.blockCount = 2;

    outParamK.blockCount = 2 * selectedBlockSize;
    outParamK.blockLen = dimDqk * sizeof(T1);
    outParamK.srcStride = 0;
    outParamK.dstStride = 0;

    outParamRope.blockCount = 2 * selectedBlockSize;
    outParamRope.blockLen = dimRope * sizeof(T1);
    outParamRope.srcStride = 0;
    outParamRope.dstStride = 0;

    scatterParams.blockCount = 2;
    scatterParams.srcStride = 0;
}

template <typename SFAGT>
__aicore__ inline void VecOp<SFAGT>::InitGMBuffer(GM_ADDR key, GM_ADDR value, GM_ADDR dv, GM_ADDR attention_out,
                                                  GM_ADDR attention_out_grad, GM_ADDR softmax_max, GM_ADDR softmax_sum,
                                                  GM_ADDR topk_indices, GM_ADDR key_rope, GM_ADDR workspace)
{
    (void)dv;
    /*
     * 初始化输入
     */
    attentionGm.SetGlobalBuffer((__gm__ T1 *)attention_out);
    attentionGradGm.SetGlobalBuffer((__gm__ T1 *)attention_out_grad);
    softmaxMaxGm.SetGlobalBuffer((__gm__ float *)softmax_max);
    softmaxSumGm.SetGlobalBuffer((__gm__ float *)softmax_sum);
    topkIndicesGm.SetGlobalBuffer((__gm__ int32_t *)topk_indices);
    keyGm.SetGlobalBuffer((__gm__ T1 *)key);
    valueGm.SetGlobalBuffer((__gm__ T1 *)value);
    keyRopeGm.SetGlobalBuffer((__gm__ T1 *)key_rope);

    /*
     * 初始化workspace
     */
    int64_t usedWorkspaceLen = 0;
    // select
    int64_t selectedKAddr = usedWorkspaceLen / sizeof(T1) + cubeBlockIdx * selectedKWorkspaceLen / sizeof(T1);
    usedWorkspaceLen += selectedKWorkspaceLen * usedCoreNum;
    int64_t mm1Addr = usedWorkspaceLen / sizeof(float) + cubeBlockIdx * mm12WorkspaceLen / sizeof(float);
    usedWorkspaceLen += mm12WorkspaceLen * usedCoreNum;
    int64_t pAddr = usedWorkspaceLen / sizeof(T1) + cubeBlockIdx * mm12WorkspaceLen / sizeof(T1);
    usedWorkspaceLen += mm12WorkspaceLen * usedCoreNum;

    int64_t mm2Addr = usedWorkspaceLen / sizeof(float) + cubeBlockIdx * mm12WorkspaceLen / sizeof(float);
    usedWorkspaceLen += mm12WorkspaceLen * usedCoreNum;
    int64_t dsAddr = usedWorkspaceLen / sizeof(T1) + cubeBlockIdx * mm12WorkspaceLen / sizeof(T1);
    usedWorkspaceLen += mm12WorkspaceLen * usedCoreNum;
    // post
    int64_t dqAddr = usedWorkspaceLen / sizeof(float);
    int64_t dkAddr = dqAddr + dqWorkspaceLen / sizeof(float);
    int64_t dvAddr = dkAddr + dkWorkspaceLen / sizeof(float);
    usedWorkspaceLen += dqWorkspaceLen + dkWorkspaceLen + dvWorkspaceLen;

    // scatter add
    uint64_t scatterBufferNum = tilingData->opInfo.enableOptimizedScatter ? SCATTER_BUFFER_NUM : PING_PONG_BUFFER;
    int64_t mm4ResAddr = usedWorkspaceLen / sizeof(float);
    int64_t mm5ResAddr = mm4ResAddr + MAX_CORE_NUM * selectedBlockCount * selectedBlockSizeDimDAlign * scatterBufferNum;
    usedWorkspaceLen += MAX_CORE_NUM * selectedBlockCount * selectedBlockSize * (dimDAlign + dimD2Align) *
                        scatterBufferNum * sizeof(float);

    mm1WorkspaceGm.SetGlobalBuffer((__gm__ float *)workspace + mm1Addr);
    mm2WorkspaceGm.SetGlobalBuffer((__gm__ float *)workspace + mm2Addr);
    pWorkspaceGm.SetGlobalBuffer((__gm__ T1 *)workspace + pAddr);
    dsWorkspaceGm.SetGlobalBuffer((__gm__ T1 *)workspace + dsAddr);
    dqWorkspaceGm.SetGlobalBuffer((__gm__ float *)workspace + dqAddr);
    dkWorkspaceGm.SetGlobalBuffer((__gm__ float *)workspace + dkAddr);
    dvWorkspaceGm.SetGlobalBuffer((__gm__ float *)workspace + dvAddr);
    selectedKWorkspaceGm.SetGlobalBuffer((__gm__ T1 *)workspace + selectedKAddr);

    mm4ResWorkspaceGm.SetGlobalBuffer((__gm__ float *)workspace + mm4ResAddr);
    mm5ResWorkspaceGm.SetGlobalBuffer((__gm__ float *)workspace + mm5ResAddr);
}

template <typename SFAGT>
__aicore__ inline void VecOp<SFAGT>::InitUB(TPipe *pipe)
{
    uint32_t dataSize = params.sftBaseM * dimDv;
    uint32_t ubOffset = 0;
    uint32_t rowsumUbOffset = 0;
    uint32_t totalUbSpace = 191 * 1024;

    // The optimized path keeps rowSum/max/sum for the complete G row and
    // reuses them across K blocks. The non-optimized path recalculates them
    // for every K block, so one aligned softmax tile is sufficient.
    uint32_t softmaxParamM =
        enableOptimizedScatter ? AlignUp(params.singleM, BLOCK_FP32) : AlignUp(params.sftBaseM, BLOCK_FP32);

    pipe->InitBuffer(vecQue, totalUbSpace);

    if (!enableOptimizedScatter) {
        uint32_t topkUbOffset = ubOffset;

        rowSumOutTensor = vecQue.GetWithOffset<float>(softmaxParamM * BLOCK_FP32, ubOffset);
        ubOffset += softmaxParamM * BLOCK_FP32 * sizeof(float);
        maxTensor = vecQue.GetWithOffset<float>(softmaxParamM * BLOCK_FP32, ubOffset);
        ubOffset += softmaxParamM * BLOCK_FP32 * sizeof(float);
        sumTensor = vecQue.GetWithOffset<float>(softmaxParamM * BLOCK_FP32, ubOffset);
        ubOffset += softmaxParamM * BLOCK_FP32 * sizeof(float);

        int32_t maxTmpSize = AlignUp(params.sftBaseM, BLOCK_FP32);
        maxTmp = vecQue.GetWithOffset<float>(maxTmpSize, ubOffset);
        ubOffset += maxTmpSize * sizeof(float);
        sumTmp = vecQue.GetWithOffset<float>(maxTmpSize, ubOffset);
        ubOffset += maxTmpSize * sizeof(float);
        rowsumUbOffset = ubOffset;

        attentionGradT1Tensor = vecQue.GetWithOffset<T1>(params.sftBaseM * dimDv, ubOffset);
        ubOffset += params.sftBaseM * dimDv * sizeof(T1);
        attentionGradFP32Tensor = vecQue.GetWithOffset<float>(params.sftBaseM * dimDv, ubOffset);
        ubOffset += params.sftBaseM * dimDv * sizeof(float);
        attentionT1Tensor = vecQue.GetWithOffset<T1>(params.sftBaseM * dimDv, ubOffset);
        ubOffset += params.sftBaseM * dimDv * sizeof(T1);
        attentionFP32Tensor = vecQue.GetWithOffset<float>(params.sftBaseM * dimDv, ubOffset);
        ubOffset += params.sftBaseM * dimDv * sizeof(float);

        int32_t sftDataSize = params.sftBaseM * params.sftBaseN;
        pTensor = vecQue.GetWithOffset<float>(sftDataSize, ubOffset);
        ubOffset += sftDataSize * sizeof(float);
        dPTensor = vecQue.GetWithOffset<float>(sftDataSize, ubOffset);
        sftgOutT1Tensor = vecQue.GetWithOffset<T1>(sftDataSize, ubOffset);
        ubOffset += sftDataSize * sizeof(float);

        sftOutT1Tensor = vecQue.GetWithOffset<T1>(sftDataSize, ubOffset);
        ubOffset += sftDataSize * sizeof(T1);

        helpTensor = vecQue.GetWithOffset<uint8_t>((totalUbSpace - rowsumUbOffset) / sizeof(uint8_t), rowsumUbOffset);
        scatterAddTensorK = vecQue.GetWithOffset<float>(NON_OPT_UB_ROW_SIZE * dimDAlign * 2, rowsumUbOffset);
        rowsumUbOffset += NON_OPT_UB_ROW_SIZE * dimDAlign * 2 * sizeof(float);
        scatterAddTensorV = vecQue.GetWithOffset<float>(NON_OPT_UB_ROW_SIZE * dimD2Align * 2, rowsumUbOffset);
        rowsumUbOffset += NON_OPT_UB_ROW_SIZE * dimD2Align * 2 * sizeof(float);

        maxSelCnt = selectedBlockSize <= 32 ? 2 : 1;
        gatherTensorPing = vecQue.GetWithOffset<T1>(NON_OPT_MAX_GATHER_SIZE * dimDqk, topkUbOffset);
        topkUbOffset += NON_OPT_MAX_GATHER_SIZE * dimDqk * sizeof(T1);
        gatherTensorPong = vecQue.GetWithOffset<T1>(NON_OPT_MAX_GATHER_SIZE * dimDqk, topkUbOffset);
        topkUbOffset += NON_OPT_MAX_GATHER_SIZE * dimDqk * sizeof(T1);
        gatherRopePing = vecQue.GetWithOffset<T1>(NON_OPT_MAX_GATHER_SIZE * dimRope, topkUbOffset);
        topkUbOffset += NON_OPT_MAX_GATHER_SIZE * dimRope * sizeof(T1);
        gatherRopePong = vecQue.GetWithOffset<T1>(NON_OPT_MAX_GATHER_SIZE * dimRope, topkUbOffset);

        uint32_t attentionShape[2] = {params.sftBaseM, static_cast<uint32_t>(dimDv)};
        uint32_t softmaxGradShape[2] = {params.sftBaseM, BLOCK_FP32};
        attentionGradFP32Tensor.SetShapeInfo(ShapeInfo(2, attentionShape, DataFormat::ND));
        attentionFP32Tensor.SetShapeInfo(ShapeInfo(2, attentionShape, DataFormat::ND));
        rowSumOutTensor.SetShapeInfo(ShapeInfo(2, softmaxGradShape, DataFormat::ND));
        rowSumOutTensor.SetSize(softmaxParamM * BLOCK_FP32);
        attentionGradT1Tensor.SetSize(dataSize);
        attentionGradFP32Tensor.SetSize(dataSize);
        attentionT1Tensor.SetSize(dataSize);
        attentionFP32Tensor.SetSize(dataSize);
        sumTensor.SetSize(softmaxParamM * BLOCK_FP32);
        maxTensor.SetSize(softmaxParamM * BLOCK_FP32);

        uint32_t softmaxShape[2] = {static_cast<uint32_t>(params.sftBaseM), BLOCK_FP32};
        sumTensor.SetShapeInfo(ShapeInfo(2, softmaxShape, DataFormat::ND));
        maxTensor.SetShapeInfo(ShapeInfo(2, softmaxShape, DataFormat::ND));

        uint32_t dstSoftShape[2] = {static_cast<uint32_t>(params.sftBaseM), static_cast<uint32_t>(params.sftBaseN)};
        pTensor.SetShapeInfo(ShapeInfo(2, dstSoftShape, DataFormat::ND));
        mte2WaitMte3Proc = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>());
        mte2WaitMte3 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>());
        mte3WaitMte2 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE3>());
        mte2WaitMte3Pong = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>());
        mte3WaitMte2Pong = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE3>());
        sWaitMte2 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_S>());
        vWaitMte2 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
        vWaitMte3 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>());
        mte3WaitV = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());
        vWaitMte2Pong = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
        return;
    }

    // rowsum out
    rowSumOutTensor = vecQue.GetWithOffset<float>(softmaxParamM * BLOCK_FP32, ubOffset);
    ubOffset += softmaxParamM * BLOCK_FP32 * sizeof(float);
    maxTensor = vecQue.GetWithOffset<float>(softmaxParamM * BLOCK_FP32, ubOffset);
    ubOffset += softmaxParamM * BLOCK_FP32 * sizeof(float);
    sumTensor = vecQue.GetWithOffset<float>(softmaxParamM * BLOCK_FP32, ubOffset);
    ubOffset += softmaxParamM * BLOCK_FP32 * sizeof(float);

    int32_t maxTmpSize = AlignUp(params.sftBaseM, BLOCK_FP32);
    maxTmp = vecQue.GetWithOffset<float>(maxTmpSize, ubOffset);
    ubOffset += maxTmpSize * sizeof(float);
    sumTmp = vecQue.GetWithOffset<float>(maxTmpSize, ubOffset);
    ubOffset += maxTmpSize * sizeof(float);
    rowsumUbOffset = ubOffset;

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
    ubOffset += sftDataSize * sizeof(float);
    dPTensor = vecQue.GetWithOffset<float>(sftDataSize, ubOffset);
    sftgOutT1Tensor = vecQue.GetWithOffset<T1>(sftDataSize, ubOffset);
    ubOffset += sftDataSize * sizeof(float);

    sftOutT1Tensor = vecQue.GetWithOffset<T1>(sftDataSize, ubOffset);
    ubOffset += sftDataSize * sizeof(T1);

    helpTensor = vecQue.GetWithOffset<uint8_t>((totalUbSpace - rowsumUbOffset) / sizeof(uint8_t), rowsumUbOffset);
    scatterAddTensorK = vecQue.GetWithOffset<float>(UB_ROW_SIZE * dimDAlign * 2, ubOffset);
    ubOffset += UB_ROW_SIZE * dimDAlign * 2 * sizeof(float);
    int64_t scatterVRowStride = KV_MERGE ? dimDAlign : dimD2Align;
    scatterAddTensorV = vecQue.GetWithOffset<float>(UB_ROW_SIZE * scatterVRowStride * 2, ubOffset);
    ubOffset += UB_ROW_SIZE * scatterVRowStride * 2 * sizeof(float);

    tempTensorK = vecQue.GetWithOffset<float>((UB_ROW_SIZE * 2 + 1) * dimDAlign, ubOffset);
    ubOffset += (UB_ROW_SIZE * 2 + 1) * dimDAlign * sizeof(float);
    tempTensorV = vecQue.GetWithOffset<float>((UB_ROW_SIZE * 2 + 1) * dimD2Align, ubOffset);
    ubOffset += (UB_ROW_SIZE * 2 + 1) * dimD2Align * sizeof(float);

    maxSelCnt = selectedBlockSize <= 32 ? 2 : 1;
    int64_t gatherDim = Max(dimDqk, dimDv);
    gatherTensorPing = vecQue.GetWithOffset<T1>(MAX_GATHER_SIZE * gatherDim, ubOffset);
    ubOffset += MAX_GATHER_SIZE * gatherDim * sizeof(T1);
    gatherTensorPong = vecQue.GetWithOffset<T1>(MAX_GATHER_SIZE * gatherDim, ubOffset);
    ubOffset += MAX_GATHER_SIZE * gatherDim * sizeof(T1);
    gatherRopePing = vecQue.GetWithOffset<T1>(MAX_GATHER_SIZE * dimRope, ubOffset);
    ubOffset += MAX_GATHER_SIZE * dimRope * sizeof(T1);
    gatherRopePong = vecQue.GetWithOffset<T1>(MAX_GATHER_SIZE * dimRope, ubOffset);
    ubOffset += MAX_GATHER_SIZE * dimRope * sizeof(T1);

    uint32_t attentionShape[2] = {params.sftBaseM, static_cast<uint32_t>(dimDv)};
    uint32_t softmaxGradShape[2] = {params.singleM, BLOCK_FP32};
    attentionGradFP32Tensor.SetShapeInfo(ShapeInfo(2, attentionShape, DataFormat::ND));
    attentionFP32Tensor.SetShapeInfo(ShapeInfo(2, attentionShape, DataFormat::ND));
    rowSumOutTensor.SetShapeInfo(ShapeInfo(2, softmaxGradShape, DataFormat::ND));
    rowSumOutTensor.SetSize(softmaxParamM * BLOCK_FP32);
    attentionGradT1Tensor.SetSize(dataSize);
    attentionGradFP32Tensor.SetSize(dataSize);
    attentionT1Tensor.SetSize(dataSize);
    attentionFP32Tensor.SetSize(dataSize);
    // params
    sumTensor.SetSize(softmaxParamM * BLOCK_FP32);
    maxTensor.SetSize(softmaxParamM * BLOCK_FP32);

    // set shape
    uint32_t softmaxShape[2] = {static_cast<uint32_t>(params.singleM), BLOCK_FP32};
    sumTensor.SetShapeInfo(ShapeInfo(2, softmaxShape, DataFormat::ND));
    maxTensor.SetShapeInfo(ShapeInfo(2, softmaxShape, DataFormat::ND));

    uint32_t dstSoftShape[2] = {static_cast<uint32_t>(params.sftBaseM), static_cast<uint32_t>(params.sftBaseN)};
    pTensor.SetShapeInfo(ShapeInfo(2, dstSoftShape, DataFormat::ND));
    mte2WaitMte3Proc = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>());
    mte3WaitMte2 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE3>());
    mte3WaitMte2Pong = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE3>());
    mte2WaitMte3 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>());
    mte2WaitMte3Pong = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>());
    sWaitMte2 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_S>());
    vWaitMte2 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
    vWaitMte3 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>());
    mte3WaitV = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());
    vWaitMte2Pong = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
}

template <typename SFAGT>
__aicore__ inline void VecOp<SFAGT>::AtomicClean()
{
    int64_t dkSize, dvSize;
    dkSize = dimTkv * dimN2 * dimDAlign;
    dvSize = dimTkv * dimN2 * dimD2Align;
    dkSize = (dkSize + BLOCK_INT32 - 1) / BLOCK_INT32 * BLOCK_INT32;
    dvSize = (dvSize + BLOCK_INT32 - 1) / BLOCK_INT32 * BLOCK_INT32;

    DumpGmZero(dkWorkspaceGm, dkSize);
    if constexpr (!KV_MERGE) {
        DumpGmZero(dvWorkspaceGm, dvSize);
    }
}

template <typename SFAGT>
__aicore__ inline void VecOp<SFAGT>::CleanDqRow(const RunInfo &runInfo)
{
    // Non-empty rows are initialized by the first non-atomic MM3 FIXP. Only rows
    // without a K task still need an explicit zero result, and one AIV sub-core is enough.
    if (subBlockIdx == 0) {
        InitOutput<float>(dqWorkspaceGm[runInfo.mm3OutGmOffset], dimG * dimDAlign, 0);
    }
}

template <typename SFAGT>
template <typename GM_TYPE>
__aicore__ inline void VecOp<SFAGT>::DumpGmZero(GlobalTensor<GM_TYPE> &gm, int64_t num)
{
    int64_t perSize = (num + tilingData->opInfo.castUsedCoreNum - 1) / tilingData->opInfo.castUsedCoreNum;
    int64_t coreNum = (num + perSize - 1) / perSize;
    int64_t tailSize = num - perSize * (coreNum - 1);
    int64_t initSize = perSize;

    if (vecBlockIdx == coreNum - 1) {
        initSize = tailSize;
    }

    if (vecBlockIdx < coreNum) {
        InitOutput<GM_TYPE>(gm[vecBlockIdx * perSize], initSize, 0);
    }
}

template <typename SFAGT>
__aicore__ inline void VecOp<SFAGT>::CalSub(LocalTensor<float> &dstTensor, LocalTensor<float> &srcTensor, int32_t baseM,
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

template <typename SFAGT>
__aicore__ inline void VecOp<SFAGT>::CalRowsumAndSftCopyIn(const int32_t loopIdx, const int64_t dyGmOffset,
                                                           const int64_t sumGmOffset, const int32_t processM,
                                                           const bool calcSoftmaxGradFront)
{
    DataCopyExtParams copyParams2{1, static_cast<uint32_t>(processM * sizeof(float)), 0, 0, 0};
    DataCopyPadExtParams<float> padParams2{false, 0, 0, 0};

    if (calcSoftmaxGradFront) {
        uint32_t dataSize = processM * dimDv;
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(dataSize * sizeof(T1)), 0, 0, 0};
        DataCopyPadExtParams<T1> padParams{false, 0, 0, 0};
        uint32_t softmaxParamOffset = enableOptimizedScatter ? loopIdx * params.sftBaseM * BLOCK_FP32 : 0;
        LocalTensor<float> curRowSumTensor = rowSumOutTensor[softmaxParamOffset];
        uint32_t alignedProcessM = AlignUp(processM, BLOCK_FP32);
        uint32_t curSoftmaxGradShape[2] = {static_cast<uint32_t>(processM), BLOCK_FP32};
        curRowSumTensor.SetShapeInfo(ShapeInfo(2, curSoftmaxGradShape, DataFormat::ND));
        curRowSumTensor.SetSize(alignedProcessM * BLOCK_FP32);

        DataCopyPad(attentionGradT1Tensor, attentionGradGm[dyGmOffset], copyParams, padParams);
        DataCopyPad(attentionT1Tensor, attentionGm[dyGmOffset], copyParams, padParams);
        SET_FLAG(MTE2, V, vWaitMte2);
        WAIT_FLAG(MTE2, V, vWaitMte2);

        Cast(attentionGradFP32Tensor, attentionGradT1Tensor, RoundMode::CAST_NONE, dataSize);
        Cast(attentionFP32Tensor, attentionT1Tensor, RoundMode::CAST_NONE, dataSize);
        PIPE_BARRIER(PIPE_V);
        SoftmaxGradFront<float, false>(curRowSumTensor, attentionGradFP32Tensor, attentionFP32Tensor, helpTensor,
                                       tilingData->softmaxGradTilingData);

        // [N2, T1, G] or [B, N2, S1, G] --> current [sftBaseM, 1].
        // The optimized path stores all tiles; the non-optimized path reuses tile 0.
        DataCopyPad(maxTmp, softmaxMaxGm[sumGmOffset], copyParams2, padParams2);
        DataCopyPad(sumTmp, softmaxSumGm[sumGmOffset], copyParams2, padParams2);
        SET_FLAG(MTE2, V, vWaitMte2);

        WAIT_FLAG(MTE2, V, vWaitMte2);
        LocalTensor<float> curMaxTensor = maxTensor[softmaxParamOffset];
        LocalTensor<float> curSumTensor = sumTensor[softmaxParamOffset];
        Brcb(curMaxTensor, maxTmp, CeilDiv(processM, BLOCK_FP32), {1, BLOCK_FP32});
        PIPE_BARRIER(PIPE_V);
        Brcb(curSumTensor, sumTmp, CeilDiv(processM, BLOCK_FP32), {1, BLOCK_FP32});
        PIPE_BARRIER(PIPE_V);
    }
}

template <typename SFAGT>
__aicore__ inline void VecOp<SFAGT>::CalAttenMsk(const int32_t processM, const int32_t actualSelS2Align,
                                                 const RunInfo &runInfo)
{ /*
   * 计算attenmask时对应blockSize需要保留的长度。
   * 例如：对于blockSize=64的块，需要保留0-9块，则attenMskRsvLen=10；
   */
    if (selectedBlockSize == 1 || !runInfo.isLastBasicBlock) {
        return;
    }
    int32_t attenMskRsv = 0;
    int32_t attenMskEnd = 0;
    int32_t attenMskStartIdx = -1; // 计算attenmask时attenmask开始计算的下标
    float scalar = *((float *)&ATTEN_MASK_SCALE);
    uint64_t mask[1];
    LocalTensor<float> tmpTensor;

    int64_t valid_col_end = runInfo.s1Index + runInfo.curS2 - runInfo.curS1;
    int32_t i = runInfo.blkCntOffset + runInfo.actualSelCntOffset - 1;
    int32_t topkIdx = topkIndicesGm[runInfo.indicesGmOffset].GetValue(i);
    // 处于对角线上的block
    if (topkIdx * selectedBlockSize <= valid_col_end && (topkIdx + 1) * selectedBlockSize > valid_col_end) {
        attenMskRsv = valid_col_end - (topkIdx * selectedBlockSize) + 1;
        attenMskEnd = (i == runInfo.blkCntOffset + runInfo.actualSelCntOffset - 1 && runInfo.isLastBasicBlock) ?
                          runInfo.lastBlockSize :
                          selectedBlockSize;
        attenMskStartIdx = i * selectedBlockSize + attenMskRsv - runInfo.blkCntOffset * selectedBlockSize;
    }

    if (attenMskStartIdx == -1) {
        return; // 默认mask掉的块为topk的无效值，不参与计算
    }

    int32_t selectIdx = (attenMskStartIdx - 1) / selectedBlockSize;
    int32_t selectBlkStartOffset = selectIdx * selectedBlockSize;
    int32_t selectBlkStartOffsetAlign = selectBlkStartOffset / 8 * 8;
    attenMskRsv += selectBlkStartOffset - selectBlkStartOffsetAlign;
    attenMskEnd += selectBlkStartOffset - selectBlkStartOffsetAlign;

    if (attenMskEnd - attenMskRsv > 0) {
        for (int32_t i = 0; i < processM; i++) { // selectedBlockSize need <= 64
            int32_t pOffset = i * actualSelS2Align;
            tmpTensor = pTensor[selectBlkStartOffsetAlign + pOffset];
            // attenMskRsv ~ attenMskEnd位置1
            uint64_t maskEnd = (attenMskEnd == 64) ? UINT64_MAX : (1ULL << attenMskEnd) - 1;
            uint64_t maskRsv = (attenMskRsv == 64) ? UINT64_MAX : (1ULL << attenMskRsv) - 1;
            mask[0] = maskEnd & ~maskRsv;
            Duplicate(tmpTensor, scalar, mask, 1, 0, 0);
        }
    }
}

template <typename SFAGT>
__aicore__ inline void VecOp<SFAGT>::CalSoftmax(const int32_t loopIdx, const int32_t processM, const int64_t mm12Addr,
                                                const int64_t mm345Addr, const RunInfo &runInfo)
{
    int64_t actualSelS2 = runInfo.isLastBasicBlock ?
                              (runInfo.actualSelCntOffset - 1) * selectedBlockSize + runInfo.lastBlockSize :
                              runInfo.actualSelCntOffset * selectedBlockSize;
    int64_t actualSelS2Align = AlignUp(actualSelS2, 8);
    int64_t dataSize = processM * actualSelS2Align;

    DataCopyPad(pTensor, mm1WorkspaceGm[mm12Addr],
                {static_cast<uint16_t>(processM), static_cast<uint32_t>(actualSelS2 * sizeof(float)),
                 static_cast<uint32_t>((params.sftBaseN - actualSelS2) * sizeof(float)), 0, 0},
                {false, 0, 0, 0});
    SET_FLAG(MTE2, V, vWaitMte2);
    WAIT_FLAG(MTE2, V, vWaitMte2);

    Muls(pTensor, pTensor, scaleValue, dataSize);
    PIPE_BARRIER(PIPE_V);

    if constexpr (ATTEN_ENABLE) {
        CalAttenMsk(processM, actualSelS2Align, runInfo);
        PIPE_BARRIER(PIPE_V);
    }

    uint32_t dstSoftShape[2] = {static_cast<uint32_t>(processM), static_cast<uint32_t>(actualSelS2Align)};
    pTensor.SetShapeInfo(ShapeInfo(2, dstSoftShape, DataFormat::ND));
    uint32_t softmaxParamOffset = enableOptimizedScatter ? loopIdx * params.sftBaseM * BLOCK_FP32 : 0;
    LocalTensor<float> curSumTensor = sumTensor[softmaxParamOffset];
    LocalTensor<float> curMaxTensor = maxTensor[softmaxParamOffset];
    uint32_t alignedProcessM = AlignUp(processM, BLOCK_FP32);
    uint32_t curSoftmaxParamShape[2] = {static_cast<uint32_t>(processM), BLOCK_FP32};
    curSumTensor.SetShapeInfo(ShapeInfo(2, curSoftmaxParamShape, DataFormat::ND));
    curMaxTensor.SetShapeInfo(ShapeInfo(2, curSoftmaxParamShape, DataFormat::ND));
    curSumTensor.SetSize(alignedProcessM * BLOCK_FP32);
    curMaxTensor.SetSize(alignedProcessM * BLOCK_FP32);
    SoftMaxShapeInfo softmaxShapeInfo{static_cast<uint32_t>(processM), static_cast<uint32_t>(actualSelS2Align),
                                      static_cast<uint32_t>(processM), static_cast<uint32_t>(actualSelS2)};
    SimpleSoftMax<float, true, false>(pTensor, curSumTensor, curMaxTensor, pTensor, helpTensor,
                                      tilingData->softmaxTilingData, softmaxShapeInfo);
    PIPE_BARRIER(PIPE_V);

    auto castS2Loop = CeilDiv(actualSelS2Align, 64);
    for (int i = 0; i < castS2Loop; ++i) {
        uint64_t mask = i == (castS2Loop - 1) ? actualSelS2 - (i * 64) : 64;
        Cast(sftOutT1Tensor[i * 64], pTensor[i * 64], RoundMode::CAST_ROUND, mask, processM,
             {1, 1, static_cast<uint8_t>(CeilDiv(actualSelS2Align, 16)),
              static_cast<uint8_t>(CeilDiv(actualSelS2Align, 8))});
    }
    SET_FLAG(V, MTE3, mte3WaitV);
    WAIT_FLAG(V, MTE3, mte3WaitV);

    DataCopyPad(pWorkspaceGm[mm345Addr], sftOutT1Tensor,
                {static_cast<uint16_t>(processM), static_cast<uint32_t>(actualSelS2 * sizeof(T1)), 0,
                 static_cast<uint32_t>((params.sftBaseN - actualSelS2) * sizeof(T1)), 0});
}

template <typename SFAGT>
__aicore__ inline void VecOp<SFAGT>::CalSoftmaxGrad(const int32_t loopIdx, const int32_t processM,
                                                    const int64_t mm12Addr, const int64_t mm345Addr,
                                                    const RunInfo &runInfo)
{
    int64_t actualSelS2 = runInfo.isLastBasicBlock ?
                              (runInfo.actualSelCntOffset - 1) * selectedBlockSize + runInfo.lastBlockSize :
                              runInfo.actualSelCntOffset * selectedBlockSize;
    int64_t actualSelS2Align = AlignUp(actualSelS2, 8);
    int64_t dataSize = processM * actualSelS2Align;
    DataCopyPad(dPTensor, mm2WorkspaceGm[mm12Addr],
                {static_cast<uint16_t>(processM), static_cast<uint32_t>(actualSelS2 * sizeof(float)),
                 static_cast<uint32_t>((params.sftBaseN - actualSelS2) * sizeof(float)), 0, 0},
                {false, 0, 0, 0});
    SET_FLAG(MTE2, V, vWaitMte2);
    WAIT_FLAG(MTE2, V, vWaitMte2);

    uint32_t softmaxParamOffset = enableOptimizedScatter ? loopIdx * params.sftBaseM * BLOCK_FP32 : 0;
    LocalTensor<float> curRowSumTensor = rowSumOutTensor[softmaxParamOffset];
    CalSub(dPTensor, curRowSumTensor, processM, actualSelS2Align);
    PIPE_BARRIER(PIPE_V);

    Mul(pTensor, pTensor, dPTensor, dataSize);
    PIPE_BARRIER(PIPE_V);

    auto castS2Loop = CeilDiv(actualSelS2Align, 64);
    for (int i = 0; i < castS2Loop; ++i) {
        uint64_t mask = i == (castS2Loop - 1) ? actualSelS2 - (i * 64) : 64;
        Cast(sftgOutT1Tensor[i * 64], pTensor[i * 64], RoundMode::CAST_ROUND, mask, processM,
             {1, 1, static_cast<uint8_t>(CeilDiv(actualSelS2Align, 16)),
              static_cast<uint8_t>(CeilDiv(actualSelS2Align, 8))});
    }
    SET_FLAG(V, MTE3, mte3WaitV);
    WAIT_FLAG(V, MTE3, mte3WaitV);

    DataCopyPad(dsWorkspaceGm[mm345Addr], sftgOutT1Tensor,
                {static_cast<uint16_t>(processM), static_cast<uint32_t>(actualSelS2 * sizeof(T1)), 0,
                 static_cast<uint32_t>((params.sftBaseN - actualSelS2) * sizeof(T1)), 0});
}

template <typename SFAGT>
__aicore__ inline void VecOp<SFAGT>::GatherKV(const int64_t n2Index, uint64_t currentS1Offset, const RunInfo &runInfo)
{
    if (runInfo.isSmallS2)
        return;
    if constexpr (IS_DETERMINISTIC) {
        GatherKVNonOptimized(n2Index, currentS1Offset, runInfo);
    } else {
        if (!enableOptimizedScatter) {
            GatherKVNonOptimized(n2Index, currentS1Offset, runInfo);
        } else {
            GatherKVOptimized(n2Index, currentS1Offset, runInfo);
        }
    }
}

template <typename SFAGT>
__aicore__ inline void VecOp<SFAGT>::GatherKVNonOptimized(const int64_t n2Index, uint64_t currentS1Offset,
                                                          const RunInfo &runInfo)
{
    uint64_t kSelectedWsAddr = runInfo.selectedKGmOffset;
    uint64_t s1Offset = IS_BSND ? currentS1Offset : currentS1Offset + runInfo.s1Index;
    uint64_t gmOffset = s1Offset * (dimN2 * selectedBlockCount) + n2Index * selectedBlockCount + runInfo.blkCntOffset;

    outParamK.blockCount = selectedBlockSize < 64 ? 2 * selectedBlockSize : selectedBlockSize;
    outParamRope.blockCount = selectedBlockSize < 64 ? 2 * selectedBlockSize : selectedBlockSize;

    uint64_t ropePlaneOffset = PER_LOOP_BLOCK_SIZE * dimDqk;
    event_t mte2WaitMte3EventId;
    event_t mte3WaitMte2EventId;

    SET_FLAG(MTE3, MTE2, mte2WaitMte3Pong);
    SET_FLAG(MTE3, MTE2, mte2WaitMte3);

    uint32_t s2Pair = CeilDiv(runInfo.actualSelCntOffset, 2) * 2;
    uint32_t firstVecEnd = (s2Pair / 2);
    uint32_t curBlk = subBlockIdx == 0 ? 0 : firstVecEnd;
    uint32_t curActualSelCntEnd = subBlockIdx == 0 ? firstVecEnd : runInfo.actualSelCntOffset;
    uint32_t curActualSelCntOffset = curActualSelCntEnd - curBlk;
    uint64_t outKOffset = subBlockIdx == 0 ? 0 : firstVecEnd * selectedBlockSize * dimDqk;
    uint64_t outRopeOffset = subBlockIdx == 0 ? 0 : firstVecEnd * selectedBlockSize * dimRope;
    uint32_t i;

    bool isLast = runInfo.isLastBasicBlock && subBlockIdx == 1;
    uint32_t curGatherSize = 0;
    for (i = curBlk; i < curBlk + curActualSelCntOffset / maxSelCnt * maxSelCnt; i += maxSelCnt) {
        int64_t keyOffset1 = topkIndicesGm.GetValue(gmOffset + i) * selectedBlockSize;
        int64_t keyOffset2 = topkIndicesGm.GetValue(gmOffset + i + 1) * selectedBlockSize;

        uint32_t s2OrgStride = keyOffset2 - keyOffset1 - selectedBlockSize;
        intriParamsKey.blockCount = 2;
        intriParamsKey.blockLen = selectedBlockSizeDqk * sizeof(T1);

        mte2WaitMte3EventId = mergePingPong ? mte2WaitMte3Pong : mte2WaitMte3;
        mte3WaitMte2EventId = mergePingPong ? mte3WaitMte2Pong : mte3WaitMte2;

        WAIT_FLAG(MTE3, MTE2, mte2WaitMte3EventId);

        intriParamsKey.srcStride = s2OrgStride * dimN2 * dimDqk * sizeof(T1);
        LocalTensor<T1> &gatherTensor = mergePingPong ? gatherTensorPing : gatherTensorPong;
        LocalTensor<T1> &gatherRopeTensor = mergePingPong ? gatherRopePing : gatherRopePong;

        bool isActualLast = isLast && i >= runInfo.actualSelCntOffset - 2;
        uint32_t curGatherSizeDqk = curGatherSize * dimDqk;
        if (keyOffset2 <= keyOffset1 || selectedBlockSize >= 64 || isActualLast) {
            intriParamsKey.blockCount = 1;
            DataCopyPad(gatherTensor[curGatherSizeDqk], keyGm[runInfo.keyGmOffset + keyOffset1 * dimN2 * dimDqk],
                        intriParamsKey, padParams);
            if (selectedBlockSize < 64) {
                intriParamsKey.blockLen =
                    isActualLast ? runInfo.lastBlockSize * dimDqk * sizeof(T1) : selectedBlockSizeDqk * sizeof(T1);
                DataCopyPad(gatherTensor[curGatherSizeDqk + selectedBlockSizeDqk],
                            keyGm[runInfo.keyGmOffset + keyOffset2 * dimN2 * dimDqk], intriParamsKey, padParams);
            }
        } else {
            DataCopyPad(gatherTensor[curGatherSizeDqk], keyGm[runInfo.keyGmOffset + keyOffset1 * dimN2 * dimDqk],
                        intriParamsKey, padParams);
        }

        if constexpr (HAS_ROPE) {
            intriParamsRope.blockCount = 2;
            intriParamsRope.srcStride = s2OrgStride * dimN2 * dimRope * sizeof(T1);
            intriParamsRope.blockLen = selectedBlockSizeDrope * sizeof(T1);

            uint32_t curGatherSizeDrope = curGatherSize * dimRope;
            if (keyOffset2 <= keyOffset1 || selectedBlockSize >= 64 || isActualLast) {
                intriParamsRope.blockCount = 1;
                DataCopyPad(gatherRopeTensor[curGatherSizeDrope],
                            keyRopeGm[runInfo.keyRopeGmOffset + keyOffset1 * dimN2 * dimRope], intriParamsRope,
                            padParams);
                if (selectedBlockSize < 64) {
                    intriParamsRope.blockLen = isActualLast ? runInfo.lastBlockSize * dimRope * sizeof(T1) :
                                                              selectedBlockSizeDrope * sizeof(T1);
                    DataCopyPad(gatherRopeTensor[curGatherSizeDrope + selectedBlockSizeDrope],
                                keyRopeGm[runInfo.keyRopeGmOffset + keyOffset2 * dimN2 * dimRope], intriParamsRope,
                                padParams);
                }
            } else {
                DataCopyPad(gatherRopeTensor[curGatherSizeDrope],
                            keyRopeGm[runInfo.keyRopeGmOffset + keyOffset1 * dimN2 * dimRope], intriParamsRope,
                            padParams);
            }
        }
        curGatherSize += maxSelCnt * selectedBlockSize;

        if (curGatherSize == NON_OPT_MAX_GATHER_SIZE) {
            SET_FLAG(MTE2, MTE3, mte3WaitMte2EventId);
            WAIT_FLAG(MTE2, MTE3, mte3WaitMte2EventId);

            outParamK.blockCount = curGatherSize;
            outParamRope.blockCount = curGatherSize;
            DataCopyPad(selectedKWorkspaceGm[kSelectedWsAddr + outKOffset], gatherTensor, outParamK);
            if constexpr (HAS_ROPE) {
                DataCopyPad(selectedKWorkspaceGm[kSelectedWsAddr + ropePlaneOffset + outRopeOffset], gatherRopeTensor,
                            outParamRope);
            }

            SET_FLAG(MTE3, MTE2, mte2WaitMte3EventId);
            mergePingPong = 1 - mergePingPong;
            outKOffset += curGatherSize * dimDqk;
            outRopeOffset += curGatherSize * dimRope;
            curGatherSize = 0;
        } else {
            SET_FLAG(MTE3, MTE2, mte2WaitMte3EventId);
        }
    }
    if (i < curActualSelCntEnd) {
        int64_t keyOffset1 = topkIndicesGm.GetValue(gmOffset + i) * selectedBlockSize;

        mte2WaitMte3EventId = mergePingPong ? mte2WaitMte3Pong : mte2WaitMte3;
        mte3WaitMte2EventId = mergePingPong ? mte3WaitMte2Pong : mte3WaitMte2;

        WAIT_FLAG(MTE3, MTE2, mte2WaitMte3EventId);

        LocalTensor<T1> &gatherTensor = mergePingPong ? gatherTensorPing : gatherTensorPong;
        LocalTensor<T1> &gatherRopeTensor = mergePingPong ? gatherRopePing : gatherRopePong;

        intriParamsKey.blockCount = 1;
        intriParamsRope.blockCount = 1;
        if (i == runInfo.actualSelCntOffset - 1 && isLast) {
            intriParamsKey.blockLen = runInfo.lastBlockSize * dimDqk * sizeof(T1);
            intriParamsRope.blockLen = runInfo.lastBlockSize * dimRope * sizeof(T1);
        }
        DataCopyPad(gatherTensor[curGatherSize * dimDqk], keyGm[runInfo.keyGmOffset + keyOffset1 * dimN2 * dimDqk],
                    intriParamsKey, padParams);

        if constexpr (HAS_ROPE) {
            DataCopyPad(gatherRopeTensor[curGatherSize * dimRope],
                        keyRopeGm[runInfo.keyRopeGmOffset + keyOffset1 * dimN2 * dimRope], intriParamsRope, padParams);
        }
        curGatherSize += selectedBlockSize;

        SET_FLAG(MTE3, MTE2, mte2WaitMte3EventId);
    }

    if (curGatherSize != 0) {
        LocalTensor<T1> &gatherTensor = mergePingPong ? gatherTensorPing : gatherTensorPong;
        LocalTensor<T1> &gatherRopeTensor = mergePingPong ? gatherRopePing : gatherRopePong;

        WAIT_FLAG(MTE3, MTE2, mte2WaitMte3EventId);

        SET_FLAG(MTE2, MTE3, mte3WaitMte2EventId);
        WAIT_FLAG(MTE2, MTE3, mte3WaitMte2EventId);

        outParamK.blockCount = curGatherSize;
        DataCopyPad(selectedKWorkspaceGm[kSelectedWsAddr + outKOffset], gatherTensor, outParamK);

        if constexpr (HAS_ROPE) {
            outParamRope.blockCount = curGatherSize;
            DataCopyPad(selectedKWorkspaceGm[kSelectedWsAddr + ropePlaneOffset + outRopeOffset], gatherRopeTensor,
                        outParamRope);
        }

        SET_FLAG(MTE3, MTE2, mte2WaitMte3EventId);
        mergePingPong = 1 - mergePingPong;
    }
    WAIT_FLAG(MTE3, MTE2, mte2WaitMte3Pong);
    WAIT_FLAG(MTE3, MTE2, mte2WaitMte3);
}

template <typename SFAGT>
__aicore__ inline void VecOp<SFAGT>::GatherKVOptimized(const int64_t n2Index, uint64_t currentS1Offset,
                                                       const RunInfo &runInfo)
{
    uint64_t kSelectedWsAddr = runInfo.selectedKGmOffset;
    uint64_t s1Offset = IS_BSND ? currentS1Offset : currentS1Offset + runInfo.s1Index;
    uint64_t gmOffset = s1Offset * (dimN2 * selectedBlockCount) + n2Index * selectedBlockCount + runInfo.blkCntOffset;

    outParamK.blockCount = selectedBlockSize < 64 ? 2 * selectedBlockSize : selectedBlockSize;
    outParamK.blockLen = dimDqk * sizeof(T1);
    outParamK.srcStride = 0;
    outParamK.dstStride = 0;
    outParamRope.blockCount = selectedBlockSize < 64 ? 2 * selectedBlockSize : selectedBlockSize;

    uint64_t ropePlaneOffset = PER_LOOP_BLOCK_SIZE * dimDqk;
    event_t mte2WaitMte3EventId;
    event_t mte3WaitMte2EventId;

    // ------------- MergeKv --------------
    uint32_t s2Pair = CeilDiv(runInfo.actualSelCntOffset, 2) * 2;
    uint32_t firstVecEnd = (s2Pair / 2);
    uint32_t curBlk = subBlockIdx == 0 ? 0 : firstVecEnd;
    uint32_t curActualSelCntEnd = subBlockIdx == 0 ? firstVecEnd : runInfo.actualSelCntOffset;
    uint32_t curActualSelCntOffset = curActualSelCntEnd - curBlk;
    uint64_t outKOffset = subBlockIdx == 0 ? 0 : firstVecEnd * selectedBlockSize * dimDqk;
    uint64_t outRopeOffset = subBlockIdx == 0 ? 0 : firstVecEnd * selectedBlockSize * dimRope;
    uint32_t i;

    if (selectedBlockSize >= MAX_GATHER_SIZE) {
        for (i = curBlk; i < curActualSelCntEnd; ++i) {
            int64_t keyOffset = topkIndicesGm.GetValue(gmOffset + i) * selectedBlockSize;
            uint32_t blockRows = selectedBlockSize;
            if (runInfo.isLastBasicBlock && i == runInfo.actualSelCntOffset - 1) {
                blockRows = static_cast<uint32_t>(runInfo.lastBlockSize);
            }

            for (uint32_t rowOffset = 0; rowOffset < blockRows; rowOffset += MAX_GATHER_SIZE) {
                uint32_t copyRows =
                    (blockRows - rowOffset) > MAX_GATHER_SIZE ? MAX_GATHER_SIZE : (blockRows - rowOffset);
                uint64_t kOutOffset = i * selectedBlockSize * dimDqk + rowOffset * dimDqk;
                uint64_t ropeOutOffset = i * selectedBlockSize * dimRope + rowOffset * dimRope;

                mte2WaitMte3EventId = mergePingPong ? runInfo.gatherMte2WaitMte3Pong : runInfo.gatherMte2WaitMte3;
                mte3WaitMte2EventId = mergePingPong ? mte3WaitMte2Pong : mte3WaitMte2;
                WAIT_FLAG(MTE3, MTE2, mte2WaitMte3EventId);

                LocalTensor<T1> &gatherTensor = mergePingPong ? gatherTensorPing : gatherTensorPong;
                LocalTensor<T1> &gatherRopeTensor = mergePingPong ? gatherRopePing : gatherRopePong;

                intriParamsKey.blockCount = copyRows;
                intriParamsKey.blockLen = dimDqk * sizeof(T1);
                intriParamsKey.srcStride = (dimN2 - 1) * dimDqk * sizeof(T1);
                intriParamsKey.dstStride = 0;
                DataCopyPad(gatherTensor, keyGm[runInfo.keyGmOffset + (keyOffset + rowOffset) * dimN2 * dimDqk],
                            intriParamsKey, padParams);

                if constexpr (HAS_ROPE) {
                    intriParamsRope.blockCount = copyRows;
                    intriParamsRope.blockLen = dimRope * sizeof(T1);
                    intriParamsRope.srcStride = (dimN2 - 1) * dimRope * sizeof(T1);
                    intriParamsRope.dstStride = 0;
                    DataCopyPad(gatherRopeTensor,
                                keyRopeGm[runInfo.keyRopeGmOffset + (keyOffset + rowOffset) * dimN2 * dimRope],
                                intriParamsRope, padParams);
                }

                SET_FLAG(MTE2, MTE3, mte3WaitMte2EventId);
                WAIT_FLAG(MTE2, MTE3, mte3WaitMte2EventId);

                outParamK.blockCount = copyRows;
                outParamK.blockLen = dimDqk * sizeof(T1);
                outParamK.srcStride = 0;
                outParamK.dstStride = 0;
                DataCopyPad(selectedKWorkspaceGm[kSelectedWsAddr + kOutOffset], gatherTensor, outParamK);

                if constexpr (HAS_ROPE) {
                    outParamRope.blockCount = copyRows;
                    outParamRope.blockLen = dimRope * sizeof(T1);
                    outParamRope.srcStride = 0;
                    outParamRope.dstStride = 0;
                    DataCopyPad(selectedKWorkspaceGm[kSelectedWsAddr + ropePlaneOffset + ropeOutOffset],
                                gatherRopeTensor, outParamRope);
                }

                SET_FLAG(MTE3, MTE2, mte2WaitMte3EventId);

                mergePingPong = 1 - mergePingPong;
            }
        }
        return;
    }

    bool isLast = runInfo.isLastBasicBlock && subBlockIdx == 1;

    uint32_t curGatherSize = 0;
    for (i = curBlk; i < curBlk + curActualSelCntOffset / maxSelCnt * maxSelCnt; i += maxSelCnt) {
        int64_t keyOffset1 = topkIndicesGm.GetValue(gmOffset + i) * selectedBlockSize;
        int64_t keyOffset2 = topkIndicesGm.GetValue(gmOffset + i + 1) * selectedBlockSize;

        uint32_t s2OrgStride = keyOffset2 - keyOffset1 - selectedBlockSize;
        intriParamsKey.blockCount = 2;
        intriParamsKey.blockLen = selectedBlockSizeDqk * sizeof(T1);

        if (curGatherSize == 0) {
            mte2WaitMte3EventId = mergePingPong ? runInfo.gatherMte2WaitMte3Pong : runInfo.gatherMte2WaitMte3;
            mte3WaitMte2EventId = mergePingPong ? mte3WaitMte2Pong : mte3WaitMte2;
            WAIT_FLAG(MTE3, MTE2, mte2WaitMte3EventId);
        }

        // CopyIn
        intriParamsKey.srcStride = s2OrgStride * dimN2 * dimDqk * sizeof(T1);
        LocalTensor<T1> &gatherTensor = mergePingPong ? gatherTensorPing : gatherTensorPong;
        LocalTensor<T1> &gatherRopeTensor = mergePingPong ? gatherRopePing : gatherRopePong;

        bool isActualLast = isLast && i >= runInfo.actualSelCntOffset - 2;
        uint32_t curGatherSizeDqk = curGatherSize * dimDqk;
        if (keyOffset2 <= keyOffset1 || selectedBlockSize >= 64 || isActualLast) {
            intriParamsKey.blockCount = 1;
            DataCopyPad(gatherTensor[curGatherSizeDqk], keyGm[runInfo.keyGmOffset + keyOffset1 * dimN2 * dimDqk],
                        intriParamsKey, padParams);
            if (selectedBlockSize < 64) {
                intriParamsKey.blockLen =
                    isActualLast ? runInfo.lastBlockSize * dimDqk * sizeof(T1) : selectedBlockSizeDqk * sizeof(T1);
                DataCopyPad(gatherTensor[curGatherSizeDqk + selectedBlockSizeDqk],
                            keyGm[runInfo.keyGmOffset + keyOffset2 * dimN2 * dimDqk], intriParamsKey, padParams);
            }
        } else {
            DataCopyPad(gatherTensor[curGatherSizeDqk], keyGm[runInfo.keyGmOffset + keyOffset1 * dimN2 * dimDqk],
                        intriParamsKey, padParams);
        }

        if constexpr (HAS_ROPE) {
            intriParamsRope.blockCount = 2;
            intriParamsRope.srcStride = s2OrgStride * dimN2 * dimRope * sizeof(T1);
            intriParamsRope.blockLen = selectedBlockSizeDrope * sizeof(T1);

            uint32_t curGatherSizeDrope = curGatherSize * dimRope;
            if (keyOffset2 <= keyOffset1 || selectedBlockSize >= 64 || isActualLast) {
                intriParamsRope.blockCount = 1;
                DataCopyPad(gatherRopeTensor[curGatherSizeDrope],
                            keyRopeGm[runInfo.keyRopeGmOffset + keyOffset1 * dimN2 * dimRope], intriParamsRope,
                            padParams);
                if (selectedBlockSize < 64) {
                    intriParamsRope.blockLen = isActualLast ? runInfo.lastBlockSize * dimRope * sizeof(T1) :
                                                              selectedBlockSizeDrope * sizeof(T1);
                    DataCopyPad(gatherRopeTensor[curGatherSizeDrope + selectedBlockSizeDrope],
                                keyRopeGm[runInfo.keyRopeGmOffset + keyOffset2 * dimN2 * dimRope], intriParamsRope,
                                padParams);
                }
            } else {
                DataCopyPad(gatherRopeTensor[curGatherSizeDrope],
                            keyRopeGm[runInfo.keyRopeGmOffset + keyOffset1 * dimN2 * dimRope], intriParamsRope,
                            padParams);
            }
        }
        curGatherSize += maxSelCnt * selectedBlockSize;

        if (curGatherSize == MAX_GATHER_SIZE) {
            SET_FLAG(MTE2, MTE3, mte3WaitMte2EventId);
            WAIT_FLAG(MTE2, MTE3, mte3WaitMte2EventId);

            outParamK.blockCount = curGatherSize;
            outParamRope.blockCount = curGatherSize;
            // CopyOut
            DataCopyPad(selectedKWorkspaceGm[kSelectedWsAddr + outKOffset], gatherTensor, outParamK);
            if constexpr (HAS_ROPE) {
                DataCopyPad(selectedKWorkspaceGm[kSelectedWsAddr + ropePlaneOffset + outRopeOffset], gatherRopeTensor,
                            outParamRope);
            }

            SET_FLAG(MTE3, MTE2, mte2WaitMte3EventId);
            mergePingPong = 1 - mergePingPong;
            outKOffset += curGatherSize * dimDqk;
            outRopeOffset += curGatherSize * dimRope;
            curGatherSize = 0;
        }
    }
    if (i < curActualSelCntEnd) {
        int64_t keyOffset1 = topkIndicesGm.GetValue(gmOffset + i) * selectedBlockSize;

        if (curGatherSize == 0) {
            mte2WaitMte3EventId = mergePingPong ? runInfo.gatherMte2WaitMte3Pong : runInfo.gatherMte2WaitMte3;
            mte3WaitMte2EventId = mergePingPong ? mte3WaitMte2Pong : mte3WaitMte2;
            WAIT_FLAG(MTE3, MTE2, mte2WaitMte3EventId);
        }

        // CopyIn
        LocalTensor<T1> &gatherTensor = mergePingPong ? gatherTensorPing : gatherTensorPong;
        LocalTensor<T1> &gatherRopeTensor = mergePingPong ? gatherRopePing : gatherRopePong;

        intriParamsKey.blockCount = 1;
        intriParamsRope.blockCount = 1;
        if (i == runInfo.actualSelCntOffset - 1 && isLast) {
            intriParamsKey.blockLen = runInfo.lastBlockSize * dimDqk * sizeof(T1);
            intriParamsRope.blockLen = runInfo.lastBlockSize * dimRope * sizeof(T1);
        }
        DataCopyPad(gatherTensor[curGatherSize * dimDqk], keyGm[runInfo.keyGmOffset + keyOffset1 * dimN2 * dimDqk],
                    intriParamsKey, padParams);

        if constexpr (HAS_ROPE) {
            DataCopyPad(gatherRopeTensor[curGatherSize * dimRope],
                        keyRopeGm[runInfo.keyRopeGmOffset + keyOffset1 * dimN2 * dimRope], intriParamsRope, padParams);
        }
        curGatherSize += selectedBlockSize;
    }

    if (curGatherSize != 0) {
        LocalTensor<T1> &gatherTensor = mergePingPong ? gatherTensorPing : gatherTensorPong;
        LocalTensor<T1> &gatherRopeTensor = mergePingPong ? gatherRopePing : gatherRopePong;

        SET_FLAG(MTE2, MTE3, mte3WaitMte2EventId);
        WAIT_FLAG(MTE2, MTE3, mte3WaitMte2EventId);

        outParamK.blockCount = curGatherSize;
        DataCopyPad(selectedKWorkspaceGm[kSelectedWsAddr + outKOffset], gatherTensor, outParamK);

        if constexpr (HAS_ROPE) {
            outParamRope.blockCount = curGatherSize;
            DataCopyPad(selectedKWorkspaceGm[kSelectedWsAddr + ropePlaneOffset + outRopeOffset], gatherRopeTensor,
                        outParamRope);
        }

        mergePingPong = 1 - mergePingPong;

        SET_FLAG(MTE3, MTE2, mte2WaitMte3EventId);
    }
}

template <typename SFAGT>
__aicore__ inline void VecOp<SFAGT>::Process(const RunInfo &runInfo)
{
    int32_t loopEnd = (params.singleM + params.sftBaseM - 1) / params.sftBaseM;
    int32_t firstLoopEnd = loopEnd / 2;
    int32_t substart = subBlockIdx == 0 ? 0 : firstLoopEnd;
    int32_t subLoopEnd = subBlockIdx == 0 ? firstLoopEnd : loopEnd;

    int32_t processM = params.sftBaseM;
    int32_t tailM = params.singleM % params.sftBaseM;
    int64_t dataSize = processM * params.sftBaseN;

    int64_t dyGmOffset = runInfo.dyGmOffset + subBlockIdx * (processM * dimDv * firstLoopEnd);
    int64_t sumGmOffset = runInfo.sumGmOffset + subBlockIdx * (processM * firstLoopEnd);
    int64_t mm12Addr = runInfo.mm12GmOffset + subBlockIdx * (dataSize * firstLoopEnd);
    int64_t mm345Addr = runInfo.mm345GmOffset + subBlockIdx * (dataSize * firstLoopEnd);

    for (int32_t i = substart; i < subLoopEnd; i++) {
        if (i == loopEnd - 1 && tailM != 0) {
            processM = tailM;
        }
        bool calcSoftmaxGradFront = !enableOptimizedScatter || runInfo.blkCntOffset == 0;
        CalRowsumAndSftCopyIn(i, dyGmOffset, sumGmOffset, processM, calcSoftmaxGradFront);
        dyGmOffset += (processM * dimDv);
        sumGmOffset += processM;

        CalSoftmax(i, processM, mm12Addr, mm345Addr, runInfo);

        SET_FLAG(MTE3, MTE2, mte2WaitMte3Proc);
        WAIT_FLAG(MTE3, MTE2, mte2WaitMte3Proc);

        CalSoftmaxGrad(i, processM, mm12Addr, mm345Addr, runInfo);

        if (i < subLoopEnd - 1) {
            SET_FLAG(MTE3, MTE2, mte2WaitMte3Proc);
            WAIT_FLAG(MTE3, MTE2, mte2WaitMte3Proc);
        }
        mm12Addr += dataSize;
        mm345Addr += dataSize;
    }
}

template <typename SFAGT>
__aicore__ inline void VecOp<SFAGT>::NzToNd(Nz2NdInfo &nz2NdInfo, const GlobalTensor<float> &bmmResGm,
                                            LocalTensor<float> &tempUb, LocalTensor<float> &bmmResUb,
                                            event_t eventIdMte2ToV, event_t backEventTmp, event_t backEvent,
                                            bool isBackEvent, int64_t dstLastAxis)
{
    int64_t dstAxis = dstLastAxis == 0 ? nz2NdInfo.ndLastAxis : dstLastAxis;
    // 1.将bmm1结果由GM搬至UB，每块数据在UB上间隔1个block，防止BANK冲突
    DataCopyParams dataCopyParams;
    int64_t nzFirstAxis = CeilDiv(nz2NdInfo.ndLastAxis, 16L);
    dataCopyParams.blockCount = nzFirstAxis;
    dataCopyParams.blockLen = nz2NdInfo.ndFirstAxisLoopSize * 2;
    dataCopyParams.srcStride = (nz2NdInfo.ndFirstAxisRealSize - nz2NdInfo.ndFirstAxisLoopSize) * 2;
    dataCopyParams.dstStride = 1;
    int64_t bmmResOffset = nz2NdInfo.loopIdx * nz2NdInfo.ndFirstAxisBaseSize * 16;
    int64_t innerLoop = nzFirstAxis / 8L;   // 36 / 8 = 4
    int64_t innerRemain = nzFirstAxis % 8L; // 36 % 8 = 4

    CopyRepeatParams repeatParams;
    repeatParams.srcStride = nz2NdInfo.ndFirstAxisLoopSize * 2 + 1;
    repeatParams.dstStride = 2;
    repeatParams.srcRepeatSize = 2;
    repeatParams.dstRepeatSize = dstAxis / 8;
    int32_t outerLoop = nz2NdInfo.ndFirstAxisLoopSize / repeatMaxTimes;
    int32_t outerRemain = nz2NdInfo.ndFirstAxisLoopSize % repeatMaxTimes;
    int32_t outerBmmOffset = repeatMaxTimes * dstAxis;
    int32_t outerTempOffset = repeatMaxTimes * 16;
    int64_t offsetJ = 128 * nz2NdInfo.ndFirstAxisLoopSize + 64;
    WaitFlag<HardEvent::V_MTE2>(backEventTmp);
    DataCopy(tempUb, bmmResGm[bmmResOffset], dataCopyParams);

    // 2.使用vcopy进行transpose，[S2/16, vec1S1Base * 16 + 8] -> [vec1S1Base, S2/16 , 16]
    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    if (isBackEvent) {
        WaitFlag<HardEvent::MTE3_V>(backEvent);
    }
    if (likely(outerRemain)) {
        for (int64_t i = 0; i < 2; ++i) {
            for (int64_t j = 0; j < innerLoop; ++j) {
                Copy(bmmResUb[outerLoop * outerBmmOffset + j * 128 + i * 8],
                     tempUb[outerLoop * outerTempOffset + j * offsetJ + i * 8], repeatMaxSize, outerRemain,
                     repeatParams); // repeatTime = 8, repeatSize = 64(个元素)
            }
            if (likely(innerRemain)) {
                Copy(bmmResUb[outerLoop * outerBmmOffset + innerLoop * 128 + i * 8],
                     tempUb[outerLoop * outerTempOffset + innerLoop * offsetJ + i * 8], innerRemain * 8, outerRemain,
                     repeatParams); // repeatTime = 8, repeatSize = 4 * 8 = 32
            }
        }
    }
    SetFlag<HardEvent::V_MTE2>(backEventTmp);
    uint32_t bmm1ResUbShape[] = {static_cast<uint32_t>(nz2NdInfo.ndFirstAxisLoopSize), static_cast<uint32_t>(dstAxis)};
    bmmResUb.SetShapeInfo(ShapeInfo(2, bmm1ResUbShape, DataFormat::ND));
}

template <typename SFAGT>
__aicore__ inline void VecOp<SFAGT>::ScatterAddUnDeter(const RunInfo &runInfo)
{
    LocalTensor<float> dkInUb;
    LocalTensor<float> dvInUb;
    LocalTensor<float> tmpK;
    LocalTensor<float> tmpV;
    int64_t tmpKPingPongOffset = (UB_ROW_SIZE * 2 + 1) * dimDAlign / 2;
    int64_t tmpVPingPongOffset = (UB_ROW_SIZE * 2 + 1) * dimD2Align / 2;

    GlobalTensor<float> dkOutGm = dkWorkspaceGm[runInfo.mm4OutGmOffset];
    GlobalTensor<float> dvOutGm = dvWorkspaceGm[runInfo.mm5OutGmOffset];
    int64_t s2RealSize =
        enableOptimizedScatter ? runInfo.actualSelCntOffset : Min(selectedBlockCount, runInfo.actualSelectedBlockCount);
    int64_t firstCoreKSize = s2RealSize / 2;
    int64_t currentCoreKSize = subBlockIdx == 0 ? firstCoreKSize : s2RealSize - firstCoreKSize;

    if (currentCoreKSize == 0) {
        return;
    }
    SetAtomicAdd<float>();

    bool useTailBlock = enableOptimizedScatter ? (subBlockIdx == 1 && runInfo.isLastBasicBlock) : (subBlockIdx == 1);
    int64_t actTotalRows = useTailBlock ? (currentCoreKSize - 1) * selectedBlockSize + runInfo.lastBlockSize :
                                          currentCoreKSize * selectedBlockSize;
    int64_t scatterRowSize = enableOptimizedScatter ? UB_ROW_SIZE : NON_OPT_UB_ROW_SIZE;
    int64_t scatterRowSizeDAlign = scatterRowSize * dimDAlign;
    int64_t scatterRowSizeD2Align = scatterRowSize * dimD2Align;
    int64_t maxLoops = CeilDiv(actTotalRows, scatterRowSize);
    int64_t tailRows = actTotalRows - (maxLoops - 1) * scatterRowSize;

    int64_t currentDkSrcOffset =
        runInfo.scatterTaskId * MAX_CORE_NUM * selectedBlockCount * selectedBlockSizeDimDAlign +
        cubeBlockIdx * selectedBlockCount * selectedBlockSizeDimDAlign;
    int64_t currentDvSrcOffset =
        runInfo.scatterTaskId * MAX_CORE_NUM * selectedBlockCount * selectedBlockSizeDimD2Align +
        cubeBlockIdx * selectedBlockCount * selectedBlockSizeDimD2Align;
    int64_t currentIndicesOffset = runInfo.indicesGmOffset + subBlockIdx * firstCoreKSize;
    if (enableOptimizedScatter) {
        currentDkSrcOffset +=
            runInfo.blkCntOffset * selectedBlockSizeDimDAlign + subBlockIdx * firstCoreKSize * selectedBlockSize * 16;
        currentDvSrcOffset +=
            runInfo.blkCntOffset * selectedBlockSizeDimD2Align + subBlockIdx * firstCoreKSize * selectedBlockSize * 16;
        currentIndicesOffset += runInfo.blkCntOffset;
    } else {
        currentDkSrcOffset += subBlockIdx * firstCoreKSize * selectedBlockSizeDimDAlign;
        currentDvSrcOffset += subBlockIdx * firstCoreKSize * selectedBlockSizeDimD2Align;
    }
    GlobalTensor<float> dkSrcGm = mm4ResWorkspaceGm[currentDkSrcOffset];
    GlobalTensor<float> dvSrcGm = mm5ResWorkspaceGm[currentDvSrcOffset];
    GlobalTensor<int32_t> indicesGm = topkIndicesGm[currentIndicesOffset];

    int64_t curSelBlk = 0;
    int64_t curRow = 0;
    int32_t s2Idx = 0;
    if (!runInfo.isSmallS2) {
        s2Idx = indicesGm.GetValue(curSelBlk);
    }
    int64_t curProcessRow = Min(scatterRowSize, selectedBlockSize);

    if (!enableOptimizedScatter) {
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(mte2WaitMte3);
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(mte2WaitMte3Pong);
        for (int64_t loop = 0; loop < maxLoops - 1; loop++) {
            event_t backEvent = pingPongIdx == 0 ? mte2WaitMte3 : mte2WaitMte3Pong;
            WaitFlag<AscendC::HardEvent::MTE3_MTE2>(backEvent);
            dkInUb = scatterAddTensorK[pingPongIdx * scatterRowSizeDAlign];
            dvInUb = scatterAddTensorV[pingPongIdx * scatterRowSizeD2Align];
            DataCopy(dkInUb, dkSrcGm[loop * scatterRowSizeDAlign], scatterRowSizeDAlign);
            event_t event = pingPongIdx == 0 ? vWaitMte2 : vWaitMte2Pong;
            SetFlag<AscendC::HardEvent::MTE2_V>(event);
            WaitFlag<AscendC::HardEvent::MTE2_V>(event);
            Muls(dkInUb, dkInUb, (float)tilingData->postTilingData.scaleValue, scatterRowSizeDAlign);
            DataCopy(dvInUb, dvSrcGm[loop * scatterRowSizeD2Align], scatterRowSizeD2Align);
            SetFlag<AscendC::HardEvent::MTE2_V>(event);
            WaitFlag<AscendC::HardEvent::MTE2_V>(event);

            if constexpr (KV_MERGE) {
                for (int64_t row = 0; row < scatterRowSize; row++) {
                    Add(dkInUb[row * dimDAlign], dkInUb[row * dimDAlign], dvInUb[row * dimD2Align], dimD2Align);
                }
            }
            SetFlag<AscendC::HardEvent::V_MTE3>(mte3WaitV);
            WaitFlag<AscendC::HardEvent::V_MTE3>(mte3WaitV);

            if (runInfo.isSmallS2) {
                int64_t baseRow = loop * scatterRowSize + subBlockIdx * firstCoreKSize * selectedBlockSize;
                DataCopy(dkOutGm[baseRow * dimDAlign], dkInUb, scatterRowSizeDAlign);
                if constexpr (!KV_MERGE) {
                    DataCopy(dvOutGm[baseRow * dimD2Align], dvInUb, scatterRowSizeD2Align);
                }
            } else {
                for (int64_t row = 0; row < scatterRowSize;) {
                    if (curRow / selectedBlockSize > curSelBlk) {
                        curSelBlk += 1;
                        s2Idx = indicesGm.GetValue(curSelBlk);
                    }
                    if (s2Idx >= 0) {
                        DataCopy(
                            dkOutGm[s2Idx * selectedBlockSize * dimDAlign + (curRow % selectedBlockSize) * dimDAlign],
                            dkInUb[row * dimDAlign], curProcessRow * dimDAlign);
                        if constexpr (!KV_MERGE) {
                            DataCopy(dvOutGm[s2Idx * selectedBlockSize * dimD2Align +
                                             (curRow % selectedBlockSize) * dimD2Align],
                                     dvInUb[row * dimD2Align], curProcessRow * dimD2Align);
                        }
                    }
                    row += curProcessRow;
                    curRow += curProcessRow;
                }
            }

            SetFlag<AscendC::HardEvent::MTE3_MTE2>(backEvent);
            pingPongIdx = 1 - pingPongIdx;
        }
        event_t backEvent = pingPongIdx == 0 ? mte2WaitMte3 : mte2WaitMte3Pong;
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(backEvent);
        dkInUb = scatterAddTensorK[pingPongIdx * scatterRowSizeDAlign];
        dvInUb = scatterAddTensorV[pingPongIdx * scatterRowSizeD2Align];
        DataCopy(dkInUb, dkSrcGm[(maxLoops - 1) * scatterRowSizeDAlign], tailRows * dimDAlign);
        event_t event = pingPongIdx == 0 ? vWaitMte2 : vWaitMte2Pong;
        SetFlag<AscendC::HardEvent::MTE2_V>(event);
        WaitFlag<AscendC::HardEvent::MTE2_V>(event);
        Muls(dkInUb, dkInUb, (float)tilingData->postTilingData.scaleValue, tailRows * dimDAlign);
        DataCopy(dvInUb, dvSrcGm[(maxLoops - 1) * scatterRowSizeD2Align], tailRows * dimD2Align);
        SetFlag<AscendC::HardEvent::MTE2_V>(event);
        WaitFlag<AscendC::HardEvent::MTE2_V>(event);

        if constexpr (KV_MERGE) {
            for (int64_t row = 0; row < tailRows; row++) {
                Add(dkInUb[row * dimDAlign], dkInUb[row * dimDAlign], dvInUb[row * dimD2Align], dimD2Align);
            }
        }
        SetFlag<AscendC::HardEvent::V_MTE3>(mte3WaitV);
        WaitFlag<AscendC::HardEvent::V_MTE3>(mte3WaitV);

        int64_t totalRound = CeilDiv(tailRows, curProcessRow);
        int64_t row = 0;
        if (runInfo.isSmallS2 && tailRows != 0) {
            int64_t baseRow = (maxLoops - 1) * scatterRowSize + subBlockIdx * firstCoreKSize * selectedBlockSize;
            DataCopy(dkOutGm[baseRow * dimDAlign], dkInUb, tailRows * dimDAlign);
            if constexpr (!KV_MERGE) {
                DataCopy(dvOutGm[baseRow * dimD2Align], dvInUb, tailRows * dimD2Align);
            }
        } else {
            for (int64_t loop = 0; loop < totalRound; loop++) {
                if (curRow / selectedBlockSize > curSelBlk) {
                    curSelBlk += 1;
                    s2Idx = indicesGm.GetValue(curSelBlk);
                }
                if (s2Idx >= 0) {
                    if (subBlockIdx == 1 && loop == totalRound - 1) {
                        curProcessRow = (runInfo.lastBlockSize % curProcessRow) ?
                                            runInfo.lastBlockSize % curProcessRow :
                                            curProcessRow;
                    }
                    DataCopy(dkOutGm[s2Idx * selectedBlockSize * dimDAlign + (curRow % selectedBlockSize) * dimDAlign],
                             dkInUb[row * dimDAlign], curProcessRow * dimDAlign);
                    if constexpr (!KV_MERGE) {
                        DataCopy(
                            dvOutGm[s2Idx * selectedBlockSize * dimD2Align + (curRow % selectedBlockSize) * dimD2Align],
                            dvInUb[row * dimD2Align], curProcessRow * dimD2Align);
                    }
                }
                row += curProcessRow;
                curRow += curProcessRow;
            }
        }
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(backEvent);
        pingPongIdx = 1 - pingPongIdx;
        SetAtomicNone();

        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte2WaitMte3);
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte2WaitMte3Pong);

        SetFlag<AscendC::HardEvent::MTE3_V>(vWaitMte3);
        WaitFlag<AscendC::HardEvent::MTE3_V>(vWaitMte3);
        return;
    }

    Nz2NdInfo nz2NdInfo;
    nz2NdInfo.ndFirstAxisRealSize = runInfo.isLastBasicBlock ?
                                        (runInfo.actualSelCntOffset - 1) * selectedBlockSize + runInfo.lastBlockSize :
                                        runInfo.actualSelCntOffset * selectedBlockSize;
    nz2NdInfo.ndFirstAxisBaseSize = UB_ROW_SIZE;
    nz2NdInfo.ndFirstAxisLoopSize = UB_ROW_SIZE;
    nz2NdInfo.ndLastAxis = dimDAlign;

    scatterParams.blockLen = curProcessRow * dimDAlign * sizeof(float);

    int64_t curUbRowSizeDAlign = ubRowSizeDAlign;
    int64_t scatterVPingPongOffset = KV_MERGE ? ubRowSizeDAlign : ubRowSizeD2Align;
    for (int64_t loop = 0; loop < maxLoops; loop++) {
        nz2NdInfo.loopIdx = loop;
        if (unlikely(loop == maxLoops - 1)) {
            nz2NdInfo.ndFirstAxisLoopSize = tailRows;
            curUbRowSizeDAlign = tailRows * dimDAlign;
        }
        dkInUb = scatterAddTensorK[pingPongIdx * ubRowSizeDAlign];
        dvInUb = scatterAddTensorV[pingPongIdx * scatterVPingPongOffset];
        tmpK = tempTensorK[pingPongIdx * tmpKPingPongOffset];
        tmpV = tempTensorV[pingPongIdx * tmpVPingPongOffset];

        event_t event = pingPongIdx == 0 ? vWaitMte2 : vWaitMte2Pong;
        event_t backEvent = pingPongIdx == 0 ? runInfo.scatterVWaitMte3 : runInfo.scatterVWaitMte3Pong;
        event_t backEventTmpK = pingPongIdx == 0 ? runInfo.scatterTmpKMte2WaitV : runInfo.scatterTmpKMte2WaitVPong;
        event_t backEventTmpV = pingPongIdx == 0 ? runInfo.scatterTmpVMte2WaitV : runInfo.scatterTmpVMte2WaitVPong;

        nz2NdInfo.ndLastAxis = dimD2Align;
        if constexpr (KV_MERGE) {
            Duplicate(dvInUb, 0.0f, curUbRowSizeDAlign);
            PIPE_BARRIER(PIPE_V);
        }
        NzToNd(nz2NdInfo, dvSrcGm, tmpV, dvInUb, event, backEventTmpV, backEvent, !KV_MERGE,
               KV_MERGE ? dimDAlign : dimD2Align);
        PIPE_BARRIER(PIPE_V);

        nz2NdInfo.ndLastAxis = dimDAlign;
        NzToNd(nz2NdInfo, dkSrcGm, tmpK, dkInUb, event, backEventTmpK, backEvent, KV_MERGE);
        PIPE_BARRIER(PIPE_V);
        Muls(dkInUb, dkInUb, (float)tilingData->postTilingData.scaleValue, curUbRowSizeDAlign);
        PIPE_BARRIER(PIPE_V);
        if constexpr (KV_MERGE) {
            Add(dkInUb, dkInUb, dvInUb, curUbRowSizeDAlign);
            PIPE_BARRIER(PIPE_V);
        }

        int64_t totalRows = loop < maxLoops - 1 ? UB_ROW_SIZE : tailRows;
        int64_t totalInnerLoop = CeilDiv(totalRows, curProcessRow);
        int64_t row = 0;
        for (int64_t innerLoop = 0; innerLoop < totalInnerLoop;) {
            if (!runInfo.isSmallS2 && curRow / selectedBlockSize > curSelBlk) {
                curSelBlk += 1;
                s2Idx = indicesGm.GetValue(curSelBlk);
            }
            bool doBatch = false;
            bool skipBatchForLast =
                (runInfo.isLastBasicBlock && subBlockIdx == 1 && innerLoop + 1 == totalInnerLoop - 1);
            int32_t s2Idx2 = -1;
            if (!runInfo.isSmallS2 && innerLoop + 1 < totalInnerLoop && !skipBatchForLast &&
                curProcessRow == selectedBlockSize) {
                s2Idx2 = indicesGm.GetValue(curSelBlk + 1);
                doBatch = (s2Idx >= 0 && s2Idx2 > s2Idx);
            }
            if (doBatch) {
                scatterParams.dstStride = (s2Idx2 - s2Idx - 1) * selectedBlockSizeDimDAlign * sizeof(float);
                SetFlag<AscendC::HardEvent::V_MTE3>(mte3WaitV);
                WaitFlag<AscendC::HardEvent::V_MTE3>(mte3WaitV);
                DataCopyPad(dkOutGm[s2Idx * selectedBlockSizeDimDAlign], dkInUb[row * dimDAlign], scatterParams);
                if constexpr (!KV_MERGE) {
                    scatterParams.blockLen = curProcessRow * dimD2Align * sizeof(float);
                    scatterParams.dstStride = (s2Idx2 - s2Idx - 1) * selectedBlockSizeDimD2Align * sizeof(float);
                    DataCopyPad(dvOutGm[s2Idx * selectedBlockSizeDimD2Align], dvInUb[row * dimD2Align], scatterParams);
                    scatterParams.blockLen = curProcessRow * dimDAlign * sizeof(float);
                }
                curSelBlk += 1;
                s2Idx = s2Idx2;
                row += 2 * curProcessRow;
                curRow += 2 * curProcessRow;
                innerLoop += 2;
            } else {
                if (s2Idx >= 0) {
                    if (unlikely(loop == maxLoops - 1 && runInfo.isLastBasicBlock && subBlockIdx == 1 &&
                                 innerLoop == totalInnerLoop - 1)) {
                        curProcessRow = (runInfo.lastBlockSize % curProcessRow) ?
                                            runInfo.lastBlockSize % curProcessRow :
                                            curProcessRow;
                    }
                    if (!runInfo.isSmallS2) {
                        SetFlag<AscendC::HardEvent::V_MTE3>(mte3WaitV);
                        WaitFlag<AscendC::HardEvent::V_MTE3>(mte3WaitV);
                        DataCopy(dkOutGm[s2Idx * selectedBlockSizeDimDAlign + (curRow % selectedBlockSize) * dimDAlign],
                                 dkInUb[row * dimDAlign], curProcessRow * dimDAlign);
                        if constexpr (!KV_MERGE) {
                            DataCopy(dvOutGm[s2Idx * selectedBlockSizeDimD2Align +
                                             (curRow % selectedBlockSize) * dimD2Align],
                                     dvInUb[row * dimD2Align], curProcessRow * dimD2Align);
                        }
                    }
                }
                row += curProcessRow;
                curRow += curProcessRow;
                innerLoop += 1;
            }
        }
        if (runInfo.isSmallS2) {
            SetFlag<AscendC::HardEvent::V_MTE3>(mte3WaitV);
            WaitFlag<AscendC::HardEvent::V_MTE3>(mte3WaitV);
            int64_t baseRow = loop * UB_ROW_SIZE + subBlockIdx * firstCoreKSize * selectedBlockSize +
                              runInfo.blkCntOffset * selectedBlockSize;
            DataCopy(dkOutGm[baseRow * dimDAlign], dkInUb, curUbRowSizeDAlign);
            if constexpr (!KV_MERGE) {
                DataCopy(dvOutGm[baseRow * dimD2Align], dvInUb, totalRows * dimD2Align);
            }
        }

        SetFlag<HardEvent::MTE3_V>(backEvent);
        pingPongIdx = 1 - pingPongIdx;
    }
    SetAtomicNone();
}

template <typename SFAGT>
__aicore__ inline void VecOp<SFAGT>::ScatterAddDeter(const RunInfo &runInfo)
{
    LocalTensor<float> dkInUb;
    LocalTensor<float> dvInUb;

    GlobalTensor<float> dkOutGm = dkWorkspaceGm[runInfo.mm4OutGmOffset];
    GlobalTensor<float> dvOutGm = dvWorkspaceGm[runInfo.mm5OutGmOffset];
    int64_t s2RealSize = Min(selectedBlockCount, runInfo.actualSelectedBlockCount);

    int64_t totalVec = (runInfo.s1End - runInfo.s1Begin) * 2;

    int64_t firstCoreKSize = s2RealSize / totalVec;
    int64_t currentCoreKSize =
        (vecBlockIdx == totalVec - 1) ? (s2RealSize % totalVec + firstCoreKSize) : firstCoreKSize;

    if (currentCoreKSize == 0) {
        return;
    }
    SetAtomicAdd<float>();
    int64_t cubeBlockIdxCalculate = runInfo.s1Index - runInfo.s1Begin;
    int64_t actTotalRows = (vecBlockIdx == totalVec - 1) ?
                               (currentCoreKSize - 1) * selectedBlockSize + runInfo.lastBlockSize :
                               currentCoreKSize * selectedBlockSize;
    int64_t maxLoops = CeilDiv(actTotalRows, UB_ROW_SIZE);
    int64_t tailRows = actTotalRows - (maxLoops - 1) * UB_ROW_SIZE;

    int64_t currentDkSrcOffset =
        runInfo.scatterTaskId * MAX_CORE_NUM * selectedBlockCount * selectedBlockSizeDimDAlign +
        cubeBlockIdxCalculate * selectedBlockCount * selectedBlockSizeDimDAlign +
        vecBlockIdx * firstCoreKSize * selectedBlockSizeDimDAlign;
    int64_t currentDvSrcOffset =
        runInfo.scatterTaskId * MAX_CORE_NUM * selectedBlockCount * selectedBlockSizeDimD2Align +
        cubeBlockIdxCalculate * selectedBlockCount * selectedBlockSizeDimD2Align +
        vecBlockIdx * firstCoreKSize * selectedBlockSizeDimD2Align;
    int64_t currentIndicesOffset = runInfo.indicesGmOffset + vecBlockIdx * firstCoreKSize;
    GlobalTensor<float> dkSrcGm = mm4ResWorkspaceGm[currentDkSrcOffset];
    GlobalTensor<float> dvSrcGm = mm5ResWorkspaceGm[currentDvSrcOffset];
    GlobalTensor<int32_t> indicesGm = topkIndicesGm[currentIndicesOffset];

    int64_t curSelBlk = 0;
    int64_t curRow = 0;
    int32_t s2Idx = 0;
    if (!runInfo.isSmallS2) {
        s2Idx = indicesGm.GetValue(curSelBlk);
    }
    int64_t curProcessRow = Min(UB_ROW_SIZE, selectedBlockSize);

    SetFlag<AscendC::HardEvent::MTE3_MTE2>(mte2WaitMte3);
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(mte2WaitMte3Pong);
    for (int64_t loop = 0; loop < maxLoops - 1; loop++) {
        event_t backEvent = pingPongIdx == 0 ? mte2WaitMte3 : mte2WaitMte3Pong;
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(backEvent);
        dkInUb = scatterAddTensorK[pingPongIdx * ubRowSizeDAlign];
        dvInUb = scatterAddTensorV[pingPongIdx * ubRowSizeD2Align];
        DataCopy(dkInUb, dkSrcGm[loop * ubRowSizeDAlign], ubRowSizeDAlign);
        event_t event = pingPongIdx == 0 ? vWaitMte2 : vWaitMte2Pong;
        SetFlag<AscendC::HardEvent::MTE2_V>(event);
        WaitFlag<AscendC::HardEvent::MTE2_V>(event);
        Muls(dkInUb, dkInUb, (float)tilingData->postTilingData.scaleValue, ubRowSizeDAlign);
        DataCopy(dvInUb, dvSrcGm[loop * ubRowSizeD2Align], ubRowSizeD2Align);
        SetFlag<AscendC::HardEvent::MTE2_V>(event);
        WaitFlag<AscendC::HardEvent::MTE2_V>(event);

        if constexpr (KV_MERGE) {
            for (int64_t row = 0; row < UB_ROW_SIZE; row++) {
                Add(dkInUb[row * dimDAlign], dkInUb[row * dimDAlign], dvInUb[row * dimD2Align], dimD2Align);
            }
        }
        SetFlag<AscendC::HardEvent::V_MTE3>(mte3WaitV);
        WaitFlag<AscendC::HardEvent::V_MTE3>(mte3WaitV);

        if (runInfo.isSmallS2) {
            int64_t baseRow = loop * UB_ROW_SIZE + vecBlockIdx * firstCoreKSize * selectedBlockSize;
            DataCopy(dkOutGm[baseRow * dimDAlign], dkInUb, ubRowSizeDAlign);
            if constexpr (!KV_MERGE) {
                DataCopy(dvOutGm[baseRow * dimD2Align], dvInUb, ubRowSizeD2Align);
            }
        } else {
            for (int64_t row = 0; row < UB_ROW_SIZE;) {
                if (curRow / selectedBlockSize > curSelBlk) {
                    curSelBlk += 1;
                    s2Idx = indicesGm.GetValue(curSelBlk);
                }
                if (s2Idx >= 0) {
                    DataCopy(dkOutGm[s2Idx * selectedBlockSize * dimDAlign + (curRow % selectedBlockSize) * dimDAlign],
                             dkInUb[row * dimDAlign], curProcessRow * dimDAlign);
                    if constexpr (!KV_MERGE) {
                        DataCopy(
                            dvOutGm[s2Idx * selectedBlockSize * dimD2Align + (curRow % selectedBlockSize) * dimD2Align],
                            dvInUb[row * dimD2Align], curProcessRow * dimD2Align);
                    }
                }
                row += curProcessRow;
                curRow += curProcessRow;
            }
        }
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(backEvent);
        pingPongIdx = 1 - pingPongIdx;
    }
    event_t backEvent = pingPongIdx == 0 ? mte2WaitMte3 : mte2WaitMte3Pong;
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(backEvent);
    dkInUb = scatterAddTensorK[pingPongIdx * ubRowSizeDAlign];
    dvInUb = scatterAddTensorV[pingPongIdx * ubRowSizeD2Align];
    DataCopy(dkInUb, dkSrcGm[(maxLoops - 1) * ubRowSizeDAlign], tailRows * dimDAlign);
    event_t event = pingPongIdx == 0 ? vWaitMte2 : vWaitMte2Pong;
    SetFlag<AscendC::HardEvent::MTE2_V>(event);
    WaitFlag<AscendC::HardEvent::MTE2_V>(event);
    Muls(dkInUb, dkInUb, (float)tilingData->postTilingData.scaleValue, tailRows * dimDAlign);
    DataCopy(dvInUb, dvSrcGm[(maxLoops - 1) * ubRowSizeD2Align], tailRows * dimD2Align);
    SetFlag<AscendC::HardEvent::MTE2_V>(event);
    WaitFlag<AscendC::HardEvent::MTE2_V>(event);

    if constexpr (KV_MERGE) {
        for (int64_t row = 0; row < tailRows; row++) {
            Add(dkInUb[row * dimDAlign], dkInUb[row * dimDAlign], dvInUb[row * dimD2Align], dimD2Align);
        }
    }
    SetFlag<AscendC::HardEvent::V_MTE3>(mte3WaitV);
    WaitFlag<AscendC::HardEvent::V_MTE3>(mte3WaitV);

    int64_t totalRound = CeilDiv(tailRows, curProcessRow);
    int64_t row = 0;
    if (runInfo.isSmallS2) {
        int64_t baseRow = (maxLoops - 1) * UB_ROW_SIZE + vecBlockIdx * firstCoreKSize * selectedBlockSize;
        DataCopy(dkOutGm[baseRow * dimDAlign], dkInUb, tailRows * dimDAlign);
        if constexpr (!KV_MERGE) {
            DataCopy(dvOutGm[baseRow * dimD2Align], dvInUb, tailRows * dimD2Align);
        }
    } else {
        for (int64_t loop = 0; loop < totalRound; loop++) {
            if (curRow / selectedBlockSize > curSelBlk) {
                curSelBlk += 1;
                s2Idx = indicesGm.GetValue(curSelBlk);
            }
            if (s2Idx >= 0) {
                if (vecBlockIdx == totalVec - 1 && loop == totalRound - 1) {
                    curProcessRow =
                        (runInfo.lastBlockSize % curProcessRow) ? runInfo.lastBlockSize % curProcessRow : curProcessRow;
                }
                DataCopy(dkOutGm[s2Idx * selectedBlockSize * dimDAlign + (curRow % selectedBlockSize) * dimDAlign],
                         dkInUb[row * dimDAlign], curProcessRow * dimDAlign);
                if constexpr (!KV_MERGE) {
                    DataCopy(
                        dvOutGm[s2Idx * selectedBlockSize * dimD2Align + (curRow % selectedBlockSize) * dimD2Align],
                        dvInUb[row * dimD2Align], curProcessRow * dimD2Align);
                }
            }
            row += curProcessRow;
            curRow += curProcessRow;
        }
    }
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(backEvent);
    pingPongIdx = 1 - pingPongIdx;
    SetAtomicNone();

    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte2WaitMte3);
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte2WaitMte3Pong);

    SetFlag<AscendC::HardEvent::MTE3_V>(vWaitMte3);
    WaitFlag<AscendC::HardEvent::MTE3_V>(vWaitMte3);
}

template <typename SFAGT>
__aicore__ inline void VecOp<SFAGT>::ScatterAdd(const RunInfo &runInfo)
{
    if constexpr (IS_DETERMINISTIC) {
        ScatterAddDeter(runInfo);
    } else {
        ScatterAddUnDeter(runInfo);
    }
}

} // namespace SFAG_BASIC
