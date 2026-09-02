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
 * \file fused_causal_conv1d_cut_bh.h
 * \brief FusedCausalConv1dCutBH kernel implementation
 */

#ifndef CAUSAL_CONV1D_CUT_BH_H
#define CAUSAL_CONV1D_CUT_BH_H

#include "kernel_operator.h"
#include "fused_causal_conv1d_cut_bh_struct.h"
#include "vf/fused_causal_conv1d_state_single_tail.h"
#include "vf/fused_causal_conv1d_no_state_single_tail.h"
#include "vf/fused_causal_conv1d_no_state_double_tail.h"
#include "vf/fused_causal_conv1d_128dim.h"

using namespace AscendC;

// ========== 常量定义 ==========
constexpr int32_t DOUBLE_BUFFER = 2;    // 双Buffer数量，用于重叠数据搬运和计算
constexpr int32_t ALIGN_BYTES = 32;     // 32字节对齐，用于DataCopyParams的stride和对齐计算
constexpr int32_t MAX_SEQUENCE_LEN = 8; // F4: m最大为7，seqLen最大为m+1=8
constexpr int32_t CAST_REP_SIZE = 64;

template <typename T>
class FusedCausalConv1dCutBH {
public:
    __aicore__ inline FusedCausalConv1dCutBH(TPipe *pipe)
        : pipe_(pipe){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR convStates, GM_ADDR queryStartLoc,
                                GM_ADDR cacheIndices, GM_ADDR numAcceptedToken, GM_ADDR numComputedTokens,
                                GM_ADDR blockIdxFirstScheduledToken, GM_ADDR blockIdxLastScheduledToken,
                                GM_ADDR initialStateIdx, GM_ADDR y, const FusedCausalConv1dCutBHTilingData *tilingData);

    __aicore__ inline void Process();

private:
    // ========== Init子函数 ==========
    __aicore__ inline void InitParams(const FusedCausalConv1dCutBHTilingData *tilingData);
    __aicore__ inline void InitQueues();

    // ========== 三阶段流水线函数 ==========
    __aicore__ inline void CopyIn(int32_t batchLoop, int32_t dimLoop, uint16_t blockCount, int64_t xOffset);
    __aicore__ inline void Compute(int32_t batchLoop, int32_t dimLoop, const LocalTensor<int32_t> &indicesLocal,
                                   const LocalTensor<int32_t> &acceptTokenLocal,
                                   const LocalTensor<int32_t> &queryStartLocLocal,
                                   const LocalTensor<int32_t> &numComputedTokensLocal,
                                   const LocalTensor<int32_t> &blockIdxFirstLocal,
                                   const LocalTensor<int32_t> &blockIdxLastLocal,
                                   const LocalTensor<int32_t> &initialStateIdxLocal, uint16_t yBlockCount,
                                   int64_t yOffset, int64_t xOffset);

    __aicore__ inline void UpdateConvStates(const LocalTensor<T> &xLocal, const LocalTensor<T> &convStatesLocal,
                                            int32_t curBatchUbOffset, int64_t writeCacheLine, int32_t curBatchSeq,
                                            int32_t offset);

    __aicore__ inline void WriteApcPrefixCache(const LocalTensor<T> &xLocal, const LocalTensor<T> &convStatesLocalOut,
                                               const LocalTensor<int32_t> &indicesLocal,
                                               const LocalTensor<int32_t> &blockIdxFirstLocal,
                                               const LocalTensor<int32_t> &blockIdxLastLocal, int32_t curBatchIdx,
                                               int32_t curBatchUbOffset, int32_t curBatchSeq, int32_t numComputed);

    template <HardEvent event>
    __aicore__ inline void SetWaitFlag(HardEvent evt)
    {
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(evt));
        SetFlag<event>(eventId);
        WaitFlag<event>(eventId);
    }

    // ========== Global Memory指针 ==========
    GlobalTensor<T> xGm;                                 // 输入序列 [batch, seqLen, dim]
    GlobalTensor<T> weightGm;                            // 卷积核 [K, dim]
    GlobalTensor<T> convStatesGm;                        // 输入cache state [-1, K-1+m, dim]
    GlobalTensor<int32_t> cacheIndicesGm;                // cache索引：非APC=[batch]，APC=[batch, maxNumBlocks]
    GlobalTensor<int32_t> acceptTokenNumGm;              // 接受的token数 [batch]
    GlobalTensor<int32_t> queryStartLocGm;               // query起始位置 [batch+1]
    GlobalTensor<T> yGm;                                 // 输出序列 [batch, seqLen, dim]
    GlobalTensor<int32_t> numComputedTokensGm;           // [batch] 已处理token总数
    GlobalTensor<int32_t> blockIdxFirstScheduledTokenGm; // [batch] APC：第一个调度token的block索引
    GlobalTensor<int32_t> blockIdxLastScheduledTokenGm;  // [batch] APC：最后一个调度token的block索引
    GlobalTensor<int32_t> initialStateIdxGm;             // [batch] APC：读取初始state的block索引

    // ========== UB队列（Unified Buffer中的数据缓存） ==========
    TQue<QuePosition::VECIN, 1> cacheQueueIn;
    TQue<QuePosition::VECOUT, 1> cacheQueueOut;
    TQue<QuePosition::VECIN, 1> xWeightQueue; // x和weight合并队列（前半段x，后半段weight）
    TQue<QuePosition::VECOUT, 1> yQueue;
    TQue<QuePosition::VECIN, 1> indicesQueue;      // APC 2D indices 专用队列（DB，消除数据竞争）
    TBuf<TPosition::VECCALC> indicesBuf;           // cache索引：非APC 1D=[batch]
    TBuf<TPosition::VECCALC> acceptTokenBuf;       // acceptToken（始终）
    TBuf<TPosition::VECCALC> queryStartLocBuf;     // queryStartLoc（2D输入时）
    TBuf<TPosition::VECCALC> numComputedTokensBuf; // numComputedTokens（APC或convMode时）
    TBuf<TPosition::VECCALC> initialStateIdxBuf;   // APC: initialStateIdx
    TBuf<TPosition::VECCALC> blockIdxFirstBuf;     // APC: blockIdxFirst
    TBuf<TPosition::VECCALC> blockIdxLastBuf;      // APC: blockIdxLast
    TPipe *pipe_;                                  // Pipeline管理对象

    // ========== Tiling参数（二维切分：Dim方向 × Batch方向） ==========
    // 核间切分参数
    int64_t dimCoreCnt_;       // dim方向的核数
    int64_t dimMainCoreCnt_;   // dim方向的主核个数
    int64_t mainCoredimLen_;   // 主核处理的Dim大小（128对齐），尾核可能含不足128的余数元素
    int64_t tailCoredimLen_;   // 尾核处理的Dim大小
    int64_t batchMainCoreCnt_; // batch方向的主核个数
    int64_t mainCoreBatchNum_; // 主核batch个数
    int64_t tailCoreBatchNum_; // 尾核batch个数

    // 核内UB切分参数
    int64_t ubMainFactorBS_;  // 主UB循环处理的batch数
    int64_t ubTailFactorBS_;  // 尾UB循环处理的batch数
    int64_t ubMainFactorDim_; // 主UB循环处理的dim数
    int64_t ubTailFactorDim_; // 尾UB循环处理的dim数
    int64_t loopNumBS_;       // Batch方向的循环次数
    int64_t loopNumDim_;      // Dim方向的循环次数

    // ========== Shape参数 ==========
    int64_t batchSize_;  // Batch大小
    int64_t seqLen_;     // 序列长度（m+1）
    int64_t cuSeqLen_;   // 累积序列长度（2D输入用）
    int64_t dim_;        // 特征维度
    int64_t kernelSize_; // 卷积核大小（K=3）
    int64_t stateLen_;   // cache_state第二维 (K-1+m)
    int64_t xInputMode_; // 输入模式：0=3D, 1=2D

    // ========== 运行时计算参数 ==========
    int32_t blockIdx_;          // 当前核的索引
    int32_t batchIdx_;          // 当前核在Batch维度的索引
    int32_t dimIdx_;            // 当前核在Dim维度的索引
    int32_t firstBatchIdx_;     // 当前核处理的起始batch索引
    int32_t coreBatchNum_;      // 当前核处理的batch数量
    int32_t dimOffset_;         // 当前核处理的Dim起始偏移（元素）
    int32_t coreDimLen_;        // 当前核处理的Dim大小
    int64_t hasAcceptTokenNum_; // 是否提供了acceptTokenNum输入
    int64_t dimSum_;            // x每个token间的stride
    int64_t cacheLenSum_;       // cache每个token间的stride
    int64_t cacheBatchLenSum_;  // cache每个batch间的stride
    int32_t batchNumInLoop_;
    int32_t dimSizeInLoop_;
    int32_t dimOffsetPerLoop_;
    int64_t isResidualConnection_;
    int8_t padSlotId_;
    int64_t yDim_;
    int64_t maxQueryLen_;
    int64_t apcEnable_;            // APC 是否开启（F1）
    int64_t blockSize_;            // APC block 大小（F1）
    int64_t maxNumBlocks_;         // cacheIndices 第二维大小（F1）
    int64_t hasCacheIndices_;      // cacheIndices 是否有效（F2）
    int64_t inplace_;              // 是否原地更新（F3）
    int64_t convMode_;             // 0=Qwen3，1=Pangu v2（F5 convMode）
    int64_t hasNumComputedTokens_; // numComputedTokens 是否有效
    int32_t xWeightQueueSize_;     // xWeightQueue 单 buffer 大小（x部分 + weight部分）
    int32_t xPartSize_;            // xWeightQueue 中 x 部分的元素个数（用于计算weight偏移）
};

template <typename T>
__aicore__ inline void FusedCausalConv1dCutBH<T>::Init(GM_ADDR x, GM_ADDR weight, GM_ADDR convStates,
                                                       GM_ADDR queryStartLoc, GM_ADDR cacheIndices,
                                                       GM_ADDR numAcceptedToken, GM_ADDR numComputedTokens,
                                                       GM_ADDR blockIdxFirstScheduledToken,
                                                       GM_ADDR blockIdxLastScheduledToken, GM_ADDR initialStateIdx,
                                                       GM_ADDR y, const FusedCausalConv1dCutBHTilingData *tilingData)
{
    InitParams(tilingData);

    // === 设置Global Memory buffers ===
    if (xInputMode_ == 0) {
        xGm.SetGlobalBuffer((__gm__ T *)x, batchSize_ * seqLen_ * dimSum_);
        if (inplace_) {
            yGm.SetGlobalBuffer((__gm__ T *)x, batchSize_ * seqLen_ * yDim_);
        } else {
            yGm.SetGlobalBuffer((__gm__ T *)y, batchSize_ * seqLen_ * yDim_);
        }
    } else {
        xGm.SetGlobalBuffer((__gm__ T *)x, cuSeqLen_ * dimSum_);
        if (inplace_) {
            yGm.SetGlobalBuffer((__gm__ T *)x, cuSeqLen_ * yDim_);
        } else {
            yGm.SetGlobalBuffer((__gm__ T *)y, cuSeqLen_ * yDim_);
        }
    }
    weightGm.SetGlobalBuffer((__gm__ T *)weight, kernelSize_ * dim_);
    convStatesGm.SetGlobalBuffer((__gm__ T *)convStates);

    // F1/F2: 根据 APC 模式绑定 cacheIndices GM
    if (apcEnable_) {
        cacheIndicesGm.SetGlobalBuffer((__gm__ int32_t *)cacheIndices, batchSize_ * maxNumBlocks_);
        blockIdxFirstScheduledTokenGm.SetGlobalBuffer((__gm__ int32_t *)blockIdxFirstScheduledToken, batchSize_);
        blockIdxLastScheduledTokenGm.SetGlobalBuffer((__gm__ int32_t *)blockIdxLastScheduledToken, batchSize_);
        initialStateIdxGm.SetGlobalBuffer((__gm__ int32_t *)initialStateIdx, batchSize_);
    } else {
        cacheIndicesGm.SetGlobalBuffer((__gm__ int32_t *)cacheIndices, batchSize_);
    }
    // numComputedTokens 以 hasNumComputedTokens_ 为准（与 APC/convMode 无关）
    if (hasNumComputedTokens_) {
        numComputedTokensGm.SetGlobalBuffer((__gm__ int32_t *)numComputedTokens, batchSize_);
    }

    if (hasAcceptTokenNum_ == 1) {
        acceptTokenNumGm.SetGlobalBuffer((__gm__ int32_t *)numAcceptedToken, batchSize_);
    }
    if (xInputMode_ == 1) {
        queryStartLocGm.SetGlobalBuffer((__gm__ int32_t *)queryStartLoc, batchSize_ + 1);
    }

    InitQueues();
}

template <typename T>
__aicore__ inline void FusedCausalConv1dCutBH<T>::InitParams(const FusedCausalConv1dCutBHTilingData *tilingData)
{
    // === 核间切分参数（二维：Dim方向 × Batch方向） ===
    dimCoreCnt_ = tilingData->dimCoreCnt;
    dimMainCoreCnt_ = tilingData->dimMainCoreCnt;
    mainCoredimLen_ = tilingData->mainCoredimLen;
    tailCoredimLen_ = tilingData->tailCoredimLen;
    batchMainCoreCnt_ = tilingData->batchMainCoreCnt;
    mainCoreBatchNum_ = tilingData->mainCoreBatchNum;
    tailCoreBatchNum_ = tilingData->tailCoreBatchNum;

    // === shape参数 ===
    batchSize_ = tilingData->batchSize;
    seqLen_ = tilingData->seqLen;
    cuSeqLen_ = tilingData->cuSeqLen;
    dim_ = tilingData->dim;
    kernelSize_ = tilingData->kernelSize;
    stateLen_ = tilingData->stateLen;
    hasAcceptTokenNum_ = tilingData->hasAcceptTokenNum;
    xInputMode_ = tilingData->xInputMode;
    isResidualConnection_ = tilingData->residualConnection;
    padSlotId_ = tilingData->padSlotId;
    apcEnable_ = tilingData->apcEnable;
    blockSize_ = tilingData->blockSize;
    maxNumBlocks_ = tilingData->maxNumBlocks;
    hasCacheIndices_ = tilingData->hasCacheIndices;
    inplace_ = tilingData->inplace;
    convMode_ = tilingData->convMode;
    hasNumComputedTokens_ = tilingData->hasNumComputedTokens;
    maxQueryLen_ = tilingData->maxQueryLen;

    // === stride参数 ===
    dimSum_ = tilingData->xStride;
    cacheLenSum_ = tilingData->cacheStride1;
    cacheBatchLenSum_ = tilingData->cacheStride0;
    yDim_ = (inplace_) ? dimSum_ : dim_;

    // === 当前核的参数计算 ===
    blockIdx_ = GetBlockIdx();
    batchIdx_ = blockIdx_ / dimCoreCnt_;           // Batch方向索引
    dimIdx_ = blockIdx_ - batchIdx_ * dimCoreCnt_; // Dim方向索引

    // dim 方向：多主核/多尾核策略
    //   dimIdx_ < dimMainCoreCnt_  → 大核（每核 mainCoredimLen_ 元素）
    //   dimIdx_ >= dimMainCoreCnt_ → 小核（每核 tailCoredimLen_ 元素）
    //   dimIdx_ == dimCoreCnt_-1   → 最后一个 dim 核，额外处理 dimRemainderElems 余数元素
    bool isLastDimCore = (dimIdx_ == dimCoreCnt_ - 1);
    if (dimIdx_ < dimMainCoreCnt_) {
        // 大核
        loopNumDim_ = tilingData->loopNumDim;
        coreDimLen_ = mainCoredimLen_;
        ubMainFactorDim_ = tilingData->ubMainFactorDim;
        ubTailFactorDim_ = tilingData->ubTailFactorDim;
        dimOffset_ = dimIdx_ * mainCoredimLen_;
    } else {
        // 小核：偏移 = 所有大核占用的 dim 量 + 当前小核在小核中的序号 × tailCoredimLen_
        int64_t tailIdx = dimIdx_ - dimMainCoreCnt_;
        loopNumDim_ = tilingData->tailBlockloopNumDim;
        coreDimLen_ = tailCoredimLen_;
        ubMainFactorDim_ = tilingData->tailBlockubFactorDim;
        ubTailFactorDim_ = tilingData->tailBlockubTailFactorDim;
        dimOffset_ = dimMainCoreCnt_ * mainCoredimLen_ + tailIdx * tailCoredimLen_;
    }
    // 最后一个 dim 核：在普通 dim 大小基础上追加余数元素，使用专用 UB 参数
    if (isLastDimCore && tilingData->dimRemainderElems > 0) {
        coreDimLen_ += tilingData->dimRemainderElems;
        loopNumDim_ = tilingData->lastCoreloopNumDim;
        ubMainFactorDim_ = tilingData->lastCoreubMainFactorDim;
        ubTailFactorDim_ = tilingData->lastCoreubTailFactorDim;
    }

    // batch 方向：多主核/多尾核策略
    //   batchIdx_ < batchMainCoreCnt_  → 大核（每核 mainCoreBatchNum_ 个 batch）
    //   batchIdx_ >= batchMainCoreCnt_ → 小核（每核 tailCoreBatchNum_ 个 batch）
    if (batchIdx_ < batchMainCoreCnt_) {
        loopNumBS_ = tilingData->loopNumBS;
        ubMainFactorBS_ = tilingData->ubMainFactorBS;
        ubTailFactorBS_ = tilingData->ubTailFactorBS;
        coreBatchNum_ = mainCoreBatchNum_;
        firstBatchIdx_ = batchIdx_ * mainCoreBatchNum_;
    } else {
        int64_t batchTailIdx = batchIdx_ - batchMainCoreCnt_;
        loopNumBS_ = tilingData->tailBlockloopNumBS;
        ubMainFactorBS_ = tilingData->tailBlockubFactorBS;
        ubTailFactorBS_ = tilingData->tailBlockubTailFactorBS;
        coreBatchNum_ = tailCoreBatchNum_;
        firstBatchIdx_ = batchMainCoreCnt_ * mainCoreBatchNum_ + batchTailIdx * tailCoreBatchNum_;
    }
}

template <typename T>
__aicore__ inline void FusedCausalConv1dCutBH<T>::InitQueues()
{
    // xWeightQueue: x和weight合并，前半段x，后半段weight
    int32_t xQueueSize = ubMainFactorBS_ * seqLen_ * ubMainFactorDim_ * sizeof(T);
    if (xInputMode_ == 1) {
        xQueueSize = ubMainFactorBS_ * maxQueryLen_ * ubMainFactorDim_ * sizeof(T);
    }
    int32_t weightQueueSize = kernelSize_ * ubMainFactorDim_ * sizeof(T);
    xPartSize_ = xQueueSize / sizeof(T);
    xWeightQueueSize_ = xQueueSize + weightQueueSize;
    pipe_->InitBuffer(xWeightQueue, 1, xWeightQueueSize_);
    pipe_->InitBuffer(yQueue, 1, xQueueSize);

    // cacheQueue: 存储cache state（stateLen行）
    int32_t cacheQueueSize = stateLen_ * ubMainFactorDim_ * sizeof(T);
    pipe_->InitBuffer(cacheQueueIn, DOUBLE_BUFFER, cacheQueueSize);
    pipe_->InitBuffer(cacheQueueOut, DOUBLE_BUFFER, cacheQueueSize);

    // acceptTokenBuf: 始终分配
    int32_t batchBufSize = (batchSize_ * sizeof(int32_t) + ALIGN_BYTES - 1) / ALIGN_BYTES * ALIGN_BYTES;
    pipe_->InitBuffer(acceptTokenBuf, batchBufSize);

    // indicesBuf：非APC且hasCacheIndices时分配 1D [batch] 大小
    if (!apcEnable_ && hasCacheIndices_) {
        int32_t indicesBufSize = (batchSize_ * sizeof(int32_t) + ALIGN_BYTES - 1) / ALIGN_BYTES * ALIGN_BYTES;
        pipe_->InitBuffer(indicesBuf, indicesBufSize);
    }

    // indicesQueue：APC 2D indices 专用，DB，每次 batch 循环搬入单个 batch 的 [maxNumBlocks] 行
    if (apcEnable_) {
        int32_t indicesQueueSize = (maxNumBlocks_ * sizeof(int32_t) + ALIGN_BYTES - 1) / ALIGN_BYTES * ALIGN_BYTES;
        pipe_->InitBuffer(indicesQueue, DOUBLE_BUFFER, indicesQueueSize);
    }

    // queryStartLocBuf: 仅2D输入时分配
    if (xInputMode_ == 1) {
        int32_t queryStartLocQueueSize =
            ((batchSize_ + 1) * sizeof(int32_t) + ALIGN_BYTES - 1) / ALIGN_BYTES * ALIGN_BYTES;
        pipe_->InitBuffer(queryStartLocBuf, queryStartLocQueueSize);
    }

    // numComputedTokensBuf：以 numComputedTokens 是否提供为准
    if (hasNumComputedTokens_) {
        pipe_->InitBuffer(numComputedTokensBuf, batchBufSize);
    }
    if (apcEnable_) {
        pipe_->InitBuffer(initialStateIdxBuf, batchBufSize);
        pipe_->InitBuffer(blockIdxFirstBuf, batchBufSize);
        pipe_->InitBuffer(blockIdxLastBuf, batchBufSize);
    }
}

template <typename T>
__aicore__ inline void FusedCausalConv1dCutBH<T>::Process()
{
    DataCopyPadParams padParams{false, 0, 0, 0};
    uint32_t indicesBlockLen = batchSize_ * sizeof(int32_t);
    DataCopyParams batchCopyParams;
    batchCopyParams.blockCount = 1;
    batchCopyParams.blockLen = indicesBlockLen;
    batchCopyParams.srcStride = 0;
    batchCopyParams.dstStride = 0;

    // === acceptTokenBuf：始终分配，无输入时填充1 ===
    LocalTensor<int32_t> acceptTokenLocal = acceptTokenBuf.Get<int32_t>();
    Duplicate(acceptTokenLocal, static_cast<int32_t>(1), batchSize_);
    SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2); // 确保Duplicate(V)完成后再做MTE2
    if (hasAcceptTokenNum_ == 1) {
        DataCopyPad(acceptTokenLocal, acceptTokenNumGm, batchCopyParams, padParams);
    }

    LocalTensor<int32_t> indicesLocal;
    if (!apcEnable_ && hasCacheIndices_) {
        // 非APC且有cacheIndices：一次性将全部 batch 的 1D 索引从 GM 搬入 UB
        indicesLocal = indicesBuf.Get<int32_t>();
        DataCopyPad(indicesLocal, cacheIndicesGm, batchCopyParams, padParams);
    }

    // 2D输入时搬运 queryStartLoc
    LocalTensor<int32_t> queryStartLocLocal;
    if (xInputMode_ == 1) {
        queryStartLocLocal = queryStartLocBuf.Get<int32_t>();
        uint32_t queryStartLocBlockLen = (batchSize_ + 1) * sizeof(int32_t);
        DataCopyParams queryStartLocCopyParams;
        queryStartLocCopyParams.blockCount = 1;
        queryStartLocCopyParams.blockLen = queryStartLocBlockLen;
        queryStartLocCopyParams.srcStride = 0;
        queryStartLocCopyParams.dstStride = 0;
        DataCopyPad(queryStartLocLocal, queryStartLocGm, queryStartLocCopyParams, padParams);
    }

    // numComputedTokens：以 hasNumComputedTokens_ 为准搬运
    LocalTensor<int32_t> numComputedTokensLocal;
    if (hasNumComputedTokens_) {
        numComputedTokensLocal = numComputedTokensBuf.Get<int32_t>();
        DataCopyPad(numComputedTokensLocal, numComputedTokensGm, batchCopyParams, padParams);
    }

    LocalTensor<int32_t> initialStateIdxLocal;
    LocalTensor<int32_t> blockIdxFirstLocal;
    LocalTensor<int32_t> blockIdxLastLocal;
    if (apcEnable_) {
        initialStateIdxLocal = initialStateIdxBuf.Get<int32_t>();
        DataCopyPad(initialStateIdxLocal, initialStateIdxGm, batchCopyParams, padParams);

        blockIdxFirstLocal = blockIdxFirstBuf.Get<int32_t>();
        DataCopyPad(blockIdxFirstLocal, blockIdxFirstScheduledTokenGm, batchCopyParams, padParams);

        blockIdxLastLocal = blockIdxLastBuf.Get<int32_t>();
        DataCopyPad(blockIdxLastLocal, blockIdxLastScheduledTokenGm, batchCopyParams, padParams);
    }

    // 等待所有常驻数据预搬运完成
    SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);

    // === 主双重循环（BS方向 × Dim方向）===
    for (int32_t batchLoop = 0; batchLoop < loopNumBS_; batchLoop++) {
        for (int32_t dimLoop = 0; dimLoop < loopNumDim_; dimLoop++) {
            batchNumInLoop_ = (batchLoop == loopNumBS_ - 1) ? ubTailFactorBS_ : ubMainFactorBS_;
            dimSizeInLoop_ = (dimLoop == loopNumDim_ - 1) ? ubTailFactorDim_ : ubMainFactorDim_;
            dimOffsetPerLoop_ = dimLoop * ubMainFactorDim_;
            int32_t firstBatchIdxInLoop = firstBatchIdx_ + batchLoop * ubMainFactorBS_;
            uint16_t blockCount = batchNumInLoop_ * seqLen_;
            int64_t startSeqIdx = firstBatchIdxInLoop * seqLen_;
            int64_t yOffset = firstBatchIdxInLoop * seqLen_ * yDim_ + dimOffset_ + dimOffsetPerLoop_;
            if (xInputMode_ == 1) {
                blockCount = queryStartLocLocal.GetValue(firstBatchIdxInLoop + batchNumInLoop_) -
                             queryStartLocLocal.GetValue(firstBatchIdxInLoop);
                startSeqIdx = queryStartLocLocal.GetValue(firstBatchIdxInLoop) - queryStartLocLocal.GetValue(0);
                yOffset = startSeqIdx * yDim_ + dimOffset_ + dimOffsetPerLoop_;
            }
            int64_t xOffset = startSeqIdx * dimSum_ + dimOffset_ + dimOffsetPerLoop_;
            CopyIn(batchLoop, dimLoop, blockCount, xOffset);
            Compute(batchLoop, dimLoop, indicesLocal, acceptTokenLocal, queryStartLocLocal, numComputedTokensLocal,
                    blockIdxFirstLocal, blockIdxLastLocal, initialStateIdxLocal, blockCount, yOffset, xOffset);
        }
    }
}

template <typename T>
__aicore__ inline void FusedCausalConv1dCutBH<T>::CopyIn(int32_t batchLoop, int32_t dimLoop, uint16_t blockCount,
                                                         int64_t xOffset)
{
    LocalTensor<T> xWeightLocal = xWeightQueue.AllocTensor<T>();
    LocalTensor<T> xLocal = xWeightLocal[0];
    LocalTensor<T> weightLocal = xWeightLocal[xPartSize_];

    // === 拷贝x数据：[blockCount, dimSizeInLoop] ===
    DataCopyPadParams padParams{false, 0, 0, 0};
    DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = blockCount;
    dataCopyParams.blockLen = dimSizeInLoop_ * sizeof(T);
    dataCopyParams.srcStride = (dimSum_ - dimSizeInLoop_) * sizeof(T);
    dataCopyParams.dstStride = 0;
    DataCopyPad(xLocal, xGm[xOffset], dataCopyParams, padParams);

    // === 拷贝weight数据：[K, dimSizeInLoop] ===
    int32_t weightOffset = dimOffset_ + dimOffsetPerLoop_;
    DataCopyParams weightCopyParams;
    weightCopyParams.blockCount = kernelSize_;
    weightCopyParams.blockLen = dimSizeInLoop_ * sizeof(T);
    weightCopyParams.srcStride = (dim_ - dimSizeInLoop_) * sizeof(T);
    weightCopyParams.dstStride = 0;
    DataCopyPad(weightLocal, weightGm[weightOffset], weightCopyParams, padParams);

    xWeightQueue.EnQue(xWeightLocal);
}

template <typename T>
__aicore__ inline void FusedCausalConv1dCutBH<T>::Compute(
    int32_t batchLoop, int32_t dimLoop, const LocalTensor<int32_t> &indicesLocal,
    const LocalTensor<int32_t> &acceptTokenLocal, const LocalTensor<int32_t> &queryStartLocLocal,
    const LocalTensor<int32_t> &numComputedTokensLocal, const LocalTensor<int32_t> &blockIdxFirstLocal,
    const LocalTensor<int32_t> &blockIdxLastLocal, const LocalTensor<int32_t> &initialStateIdxLocal,
    uint16_t yBlockCount, int64_t yOffset, int64_t xOffset)
{
    LocalTensor<T> xWeightLocal = xWeightQueue.DeQue<T>();
    LocalTensor<T> xLocal = xWeightLocal[0];
    LocalTensor<T> weightLocal = xWeightLocal[xPartSize_];
    LocalTensor<T> yLocal = yQueue.AllocTensor<T>();

    int32_t batchNumPerLoop = batchLoop * ubMainFactorBS_;
    int64_t yUbOffset = 0;
    DataCopyPadParams padParams{false, 0, 0, 0};

    DataCopyParams cacheCopyParams;
    // blockCount 在 batch 循环内按 offset 动态设置
    cacheCopyParams.blockLen = dimSizeInLoop_ * sizeof(T);
    cacheCopyParams.srcStride = (cacheLenSum_ - dimSizeInLoop_) * sizeof(T);
    cacheCopyParams.dstStride = 0;

    for (int32_t b = 0; b < batchNumInLoop_; b++) {
        int32_t curBatchIdx = firstBatchIdx_ + batchNumPerLoop + b;
        // F1: APC模式下，通过 indicesQueue 加载当前 batch 的 2D block 索引（DB，消除数据竞争）
        LocalTensor<int32_t> indicesQueueLocal;
        if (apcEnable_) {
            indicesQueueLocal = indicesQueue.AllocTensor<int32_t>();
            DataCopyParams indices2DCopyParams;
            indices2DCopyParams.blockCount = 1;
            indices2DCopyParams.blockLen = maxNumBlocks_ * sizeof(int32_t);
            indices2DCopyParams.srcStride = 0;
            indices2DCopyParams.dstStride = 0;
            DataCopyPad(indicesQueueLocal, cacheIndicesGm[curBatchIdx * maxNumBlocks_], indices2DCopyParams, padParams);
            SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
        }
        LocalTensor<T> convStatesLocalIn = cacheQueueIn.AllocTensor<T>();
        // ========== F1/F2: 统一 read/write cache 索引计算 ==========
        int64_t readCacheLine = curBatchIdx;
        int64_t writeCacheLine = curBatchIdx;

        if (apcEnable_) {
            // APC 开启：从 indicesQueueLocal 中取 block 物理行号
            int32_t initStateIdx = initialStateIdxLocal.GetValue(curBatchIdx);
            int32_t idxLast = blockIdxLastLocal.GetValue(curBatchIdx);
            readCacheLine = indicesQueueLocal.GetValue(initStateIdx);
            writeCacheLine = indicesQueueLocal.GetValue(idxLast);
        } else {
            if (hasCacheIndices_) {
                readCacheLine = indicesLocal.GetValue(curBatchIdx);
                writeCacheLine = indicesLocal.GetValue(curBatchIdx);
            }
        }

        // ========== 计算当前 batch 的序列长度和 UB 偏移 ==========
        int32_t curBatchSeq = seqLen_;
        int32_t curBatchUbOffset = b * curBatchSeq * dimSizeInLoop_;
        if (xInputMode_ == 1) {
            curBatchSeq = queryStartLocLocal.GetValue(curBatchIdx + 1) - queryStartLocLocal.GetValue(curBatchIdx);
            curBatchUbOffset =
                (queryStartLocLocal.GetValue(curBatchIdx) - queryStartLocLocal.GetValue(curBatchIdx - b)) *
                dimSizeInLoop_;
        }

        // pad_slot_id 仅在 hasCacheIndices_ 有效时生效（APC 模式不涉及 padSlotId）
        if (readCacheLine == padSlotId_) {
            cacheQueueIn.FreeTensor(convStatesLocalIn);
            if (apcEnable_) {
                indicesQueue.FreeTensor(indicesQueueLocal);
            }
            yUbOffset += curBatchSeq * dimSizeInLoop_;
            continue;
        }

        // ========== F1: 计算 conv_states 读取偏移 ==========
        int32_t acceptToken = acceptTokenLocal.GetValue(curBatchIdx);
        int32_t numComputed = hasNumComputedTokens_ ? numComputedTokensLocal.GetValue(curBatchIdx) : -1;
        int32_t offset = 0;

        if (hasAcceptTokenNum_) {
            offset = acceptToken - 1;
        } else {
            offset = stateLen_ - (kernelSize_ - 1);
        }

        // 动态设置搬运行数：offset + K-1
        cacheCopyParams.blockCount = (numComputed == 0) ? 0 : (offset + kernelSize_ - 1);

        // 从 GM 搬入 conv_states（或置零初始化）
        if (numComputed == 0) {
            Duplicate(convStatesLocalIn, static_cast<T>(0), (kernelSize_ - 1) * dimSizeInLoop_);
        } else {
            int32_t convStatesGmOffset = readCacheLine * cacheBatchLenSum_ + dimOffset_ + dimOffsetPerLoop_;
            DataCopyPad(convStatesLocalIn, convStatesGm[convStatesGmOffset], cacheCopyParams, padParams);
        }
        cacheQueueIn.EnQue<T>(convStatesLocalIn);
        convStatesLocalIn = cacheQueueIn.DeQue<T>();

        LocalTensor<T> convStatesLocalOut = cacheQueueOut.AllocTensor<T>();
        int32_t cacheRowsCopied = (numComputed == 0) ? (kernelSize_ - 1) : (offset + kernelSize_ - 1);
        DataCopy(convStatesLocalOut, convStatesLocalIn, cacheRowsCopied * dimSizeInLoop_);
        cacheQueueOut.EnQue<T>(convStatesLocalOut);

        // ========== F5: 尾对齐机制更新 running cache ==========
        convStatesLocalOut = cacheQueueOut.DeQue<T>();
        // 传入完整 stateLen_ 行的 convStatesLocalOut，offset 作为参数传入，由 UpdateConvStates 内部处理坐标偏移
        UpdateConvStates(xLocal, convStatesLocalOut, curBatchUbOffset, writeCacheLine, curBatchSeq,
                         (numComputed == 0) ? 0 : offset);

        // ========== F1: APC prefix cache 写入 ==========
        if (apcEnable_ && hasNumComputedTokens_) {
            LocalTensor<T> convStatesLocalOutSlice = convStatesLocalOut[offset * dimSizeInLoop_];
            WriteApcPrefixCache(xLocal, convStatesLocalOutSlice, indicesQueueLocal, blockIdxFirstLocal,
                                blockIdxLastLocal, curBatchIdx, curBatchUbOffset, curBatchSeq, numComputed);
            indicesQueue.FreeTensor(indicesQueueLocal);
        }
        cacheQueueOut.FreeTensor<T>(convStatesLocalOut);

        // ========== 卷积计算（情况A：seq[0, K-2]含cache；情况B：seq[K-1, end]纯x）==========
        uint32_t convStateLen = (curBatchSeq < 2) ? curBatchSeq : kernelSize_ - 1;
        LocalTensor<T> xSlice = xLocal[curBatchUbOffset];
        uint16_t dimRem = dimSizeInLoop_ - dimSizeInLoop_ / (CAST_REP_SIZE * 2) * (CAST_REP_SIZE * 2);
        if (convStateLen > 0) {
            LocalTensor<T> y1Slice = yLocal[yUbOffset];
            LocalTensor<T> convStatesSlice =
                (numComputed == 0) ? convStatesLocalIn : convStatesLocalIn[offset * dimSizeInLoop_];
            uint16_t dimRem = dimSizeInLoop_ - dimSizeInLoop_ / CAST_REP_SIZE * CAST_REP_SIZE;
            if (dimRem == 0) {
                Conv1dNeedStateBH(xSlice, weightLocal, convStatesSlice, y1Slice, convStateLen, dimSizeInLoop_,
                                  isResidualConnection_);
            } else {
                Conv1dNeedStateSingleTailBH(xSlice, weightLocal, convStatesSlice, y1Slice, convStateLen, dimSizeInLoop_,
                                            isResidualConnection_);
            }
        }
        cacheQueueIn.FreeTensor<T>(convStatesLocalIn);

        int16_t blockCount = curBatchSeq - convStateLen;
        if (blockCount > 0) {
            uint8_t xSLen = static_cast<uint8_t>(blockCount);
            LocalTensor<T> y2Slice = yLocal[yUbOffset + convStateLen * dimSizeInLoop_];
            uint16_t dimRem = dimSizeInLoop_ - dimSizeInLoop_ / (2 * CAST_REP_SIZE) * (2 * CAST_REP_SIZE);
            if (dimRem == 0) {
                Conv1dNoNeedState(xSlice, weightLocal, y2Slice, xSLen, dimSizeInLoop_, isResidualConnection_);
            } else if (dimRem <= CAST_REP_SIZE) {
                Conv1dNoStateSingleTail(xSlice, weightLocal, y2Slice, xSLen, dimSizeInLoop_, isResidualConnection_);
            } else {
                Conv1dNoStateDoubleTail(xSlice, weightLocal, y2Slice, xSLen, dimSizeInLoop_, isResidualConnection_);
            }
        }
        // ========== F3 convMode: Pangu v2 前 width-1 token 置零 ==========
        if (convMode_ == 1 && hasNumComputedTokens_) {
            int32_t lastResetIdx = (kernelSize_ - 1) - numComputed;
            if (lastResetIdx > 0) {
                SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
                lastResetIdx = lastResetIdx < curBatchSeq ? lastResetIdx : curBatchSeq;
                if (isResidualConnection_ == 1)
                    DataCopy(yLocal[yUbOffset], xLocal[yUbOffset], lastResetIdx * dimSizeInLoop_);
                else {
                    Duplicate(yLocal[yUbOffset], static_cast<T>(0), lastResetIdx * dimSizeInLoop_);
                }
            }
        }
        yUbOffset += curBatchSeq * dimSizeInLoop_;
    }

    // === 写出 yLocal ===
    yQueue.EnQue<T>(yLocal);
    yLocal = yQueue.DeQue<T>();

    DataCopyParams yParams;
    yParams.blockCount = yBlockCount;
    yParams.blockLen = dimSizeInLoop_ * sizeof(T);
    yParams.srcStride = 0;
    yParams.dstStride = (yDim_ - dimSizeInLoop_) * sizeof(T);
    // 写 y GM
    DataCopyPad(yGm[yOffset], yLocal, yParams);
    xWeightQueue.FreeTensor(xWeightLocal);
    yQueue.FreeTensor(yLocal);
}

// ========== F5: UpdateConvStates（尾对齐机制重构）==========
template <typename T>
__aicore__ inline void FusedCausalConv1dCutBH<T>::UpdateConvStates(const LocalTensor<T> &xLocal,
                                                                   const LocalTensor<T> &convStatesLocal,
                                                                   int32_t curBatchUbOffset, int64_t writeCacheLine,
                                                                   int32_t curBatchSeq, int32_t offset)
{
    // padded_input 总长 = (offset + K-1) + seq_len
    int64_t paddedLen = offset + (kernelSize_ - 1) + curBatchSeq;
    // 实际写入行数（尾对齐：最多写 stateLen_ 行）
    int64_t cacheLen = paddedLen < stateLen_ ? paddedLen : stateLen_;
    int64_t xRows = cacheLen < (int64_t)curBatchSeq ? cacheLen : (int64_t)curBatchSeq;
    int64_t cacheRows = cacheLen - xRows; // 可能为0

    // 写入 GM 的起始偏移（尾对齐：从 stateLen_ - cacheLen 行开始写）
    int64_t dstBaseGmOffset =
        writeCacheLine * cacheBatchLenSum_ + (stateLen_ - cacheLen) * cacheLenSum_ + dimOffset_ + dimOffsetPerLoop_;
    uint32_t blockLen = dimSizeInLoop_ * sizeof(T);
    uint32_t dstStride = (cacheLenSum_ - dimSizeInLoop_) * sizeof(T);

    // Step1：拷贝旧 cache 末尾 cacheRows 行（跳过头部 (K-1-cacheRows) 行）
    if (cacheRows > 0) {
        int32_t srcCacheSkip = offset + (kernelSize_ - 1) - cacheRows;
        DataCopyParams cacheToGmParams;
        cacheToGmParams.blockCount = cacheRows;
        cacheToGmParams.blockLen = blockLen;
        cacheToGmParams.srcStride = 0;
        cacheToGmParams.dstStride = dstStride;
        DataCopyPad(convStatesGm[dstBaseGmOffset], convStatesLocal[srcCacheSkip * dimSizeInLoop_], cacheToGmParams);
    }

    // Step2：拷贝 xLocal 末尾 xRows 行
    int32_t xStartRowInUb = curBatchUbOffset + (curBatchSeq - xRows) * dimSizeInLoop_;
    int64_t xDstGmOffset = dstBaseGmOffset + cacheRows * cacheLenSum_;
    DataCopyParams xToGmParams;
    xToGmParams.blockCount = xRows;
    xToGmParams.blockLen = blockLen;
    xToGmParams.srcStride = 0;
    xToGmParams.dstStride = dstStride;
    DataCopyPad(convStatesGm[xDstGmOffset], xLocal[xStartRowInUb], xToGmParams);
}

// ========== F1: APC prefix cache 写入 ==========
template <typename T>
__aicore__ inline void FusedCausalConv1dCutBH<T>::WriteApcPrefixCache(
    const LocalTensor<T> &xLocal, const LocalTensor<T> &convStatesLocalOut, const LocalTensor<int32_t> &indicesLocal,
    const LocalTensor<int32_t> &blockIdxFirstLocal, const LocalTensor<int32_t> &blockIdxLastLocal, int32_t curBatchIdx,
    int32_t curBatchUbOffset, int32_t curBatchSeq, int32_t numComputed)
{
    int32_t idxFirst = blockIdxFirstLocal.GetValue(curBatchIdx);
    int32_t idxLast = blockIdxLastLocal.GetValue(curBatchIdx);
    int32_t nBlockFill = idxLast - idxFirst;

    // seq_completed_offset: 上一次调度最后一个 token 在新调度中的相对位置
    int32_t completedOffsetToken = numComputed % blockSize_;
    int32_t seqCompletedOffset = blockSize_ - completedOffsetToken;

    // lastFullBlockTokenIndex: seq_x 坐标系中最后完整 block 的末尾索引
    int32_t seqEndOffset = (curBatchSeq - seqCompletedOffset) % blockSize_;
    int32_t lastFullBlockTokenIndex = curBatchSeq - seqEndOffset;
    if (seqEndOffset == 0) {
        lastFullBlockTokenIndex -= blockSize_;
    }

    DataCopyPadParams padParams{false, 0, 0, 0};
    uint32_t blockLen = dimSizeInLoop_ * sizeof(T);
    uint32_t dstStride = (cacheLenSum_ - dimSizeInLoop_) * sizeof(T);

    for (int32_t chunk = 0; chunk < nBlockFill; chunk++) {
        int32_t boundaryIdx = lastFullBlockTokenIndex - (nBlockFill - chunk - 1) * blockSize_;
        int64_t prefixCacheLine = indicesLocal.GetValue(idxFirst + chunk);
        int64_t prefixGmOffset = prefixCacheLine * cacheBatchLenSum_ + (stateLen_ - kernelSize_ + 1) * cacheLenSum_ +
                                 dimOffset_ + dimOffsetPerLoop_;
        // 分两段处理：
        //   cacheRows: 落在 cached_state 区域 [0, W-1) 的行数
        //   xRows:     落在 seq_x 区域 [W-1, ...) 的行数
        int32_t cacheRows = (kernelSize_ - 1 - boundaryIdx > 0) ? kernelSize_ - 1 - boundaryIdx : 0;
        int32_t xRows = kernelSize_ - 1 - cacheRows;
        DataCopyParams apcParams;
        apcParams.blockLen = blockLen;
        apcParams.srcStride = 0;
        apcParams.dstStride = dstStride;

        // --- 段1：从 convStatesLocalOut 拷贝 cacheRows 行 ---
        if (cacheRows > 0) {
            apcParams.blockCount = cacheRows;
            int32_t cacheUbOffset = boundaryIdx * dimSizeInLoop_;
            DataCopyPad(convStatesGm[prefixGmOffset], convStatesLocalOut[cacheUbOffset], apcParams);
        }

        // --- 段2：从 xLocal 拷贝 xRows 行 ---
        if (xRows > 0) {
            apcParams.blockCount = xRows;
            int64_t xUbOffset = (cacheRows > 0) ? curBatchUbOffset :
                                                  curBatchUbOffset + (boundaryIdx - kernelSize_ + 1) * dimSizeInLoop_;
            int64_t xDstGmOffset = prefixGmOffset + cacheRows * cacheLenSum_;
            DataCopyPad(convStatesGm[xDstGmOffset], xLocal[xUbOffset], apcParams);
        }
    }
}

#endif // CAUSAL_CONV1D_CUT_BH_H
