/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef LIGHTNING_INDEXER_KL_LOSS_KERNEL_H
#define LIGHTNING_INDEXER_KL_LOSS_KERNEL_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lightning_indexer_kl_loss_tiling_data.h"
#include "lightning_indexer_kl_loss_tiling_key.h"

namespace NsLightningIndexerKLLoss {

using namespace AscendC;

constexpr uint32_t HALF_BLOCK_SIZE = 16;
constexpr uint32_t HALF_VECTOR_SIZE = 128;
constexpr uint32_t FLOAT_BLOCK_SIZE = 8;
constexpr uint32_t FLOAT_VECTOR_SIZE = 64;
constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t VECTOR_SIZE = 8;

template <typename T, bool isDeterministic, bool weightType>
class LightningIndexerKLLoss {
public:
    __aicore__ inline LightningIndexerKLLoss(){};
    __aicore__ inline void Init(GM_ADDR targetScore, GM_ADDR indexProbs, GM_ADDR loss, GM_ADDR workspace,
                                const LightningIndexerKLLossTilingData *tilingData, TPipe *pipe);
    __aicore__ inline void InitWorkspaceDet(GM_ADDR workspace);
    __aicore__ inline void InitHalfBufs();
    __aicore__ inline void Process();
    __aicore__ inline void WriteBackNonDet();
    __aicore__ inline void WriteBackDet();

private:
    // 纯 MTE2 搬运
    __aicore__ inline void CopyIn(uint32_t rowOffset, uint32_t curTileLen);
    // 纯 Vector 计算（含 setflag/waitflag）
    __aicore__ inline void Compute(uint32_t tileElems, uint32_t curTileLen);
    // 纯 MTE3 搬运累加
    __aicore__ inline void CopyOut();

    TPipe *pipePtr;
    TBuf<TPosition::VECCALC> tmpBuf_;
    GM_ADDR userWorkspace_;

    GlobalTensor<T> inputGMTargetScore;
    GlobalTensor<T> inputGMIndexProbs;
    GlobalTensor<float> perCoreSumWS;
    GlobalTensor<T> outputGMLoss;

    // UB buffer LocalTensor（使用 TBuf 分配，Init 中初始化）
    LocalTensor<float> ubTargetScoreIn;
    LocalTensor<float> ubIndexProbsIn;
    LocalTensor<float> ubReduceSum;
    LocalTensor<float> ubLogP;
    LocalTensor<float> ubLogY;
    LocalTensor<float> ubOut;
    LocalTensor<float> ubTmp;
    // half 版本：指向 float buffer 后半段的 half 原始数据
    LocalTensor<T> ubHalfTarget;
    LocalTensor<T> ubHalfIndexProbs;

    DataCopyExtParams copyParams;
    DataCopyPadExtParams<T> padParams;

    // Y 和 y 独立流水线 EventID
    int32_t eventMTE2YToV_; // MTE2 → Vector: Y 搬运完成
    int32_t eventMTE2yToV_; // MTE2 → Vector: y 搬运完成
    int32_t eventVToMTE2Y_; // Vector → MTE2: Y 的 UB 已释放
    int32_t eventVToMTE2y_; // Vector → MTE2: y 的 UB 已释放
    int32_t eventVToMTE3_;
    int32_t eventMTE3ToV_;

    int64_t totalLength_;
    int64_t K_;
    int64_t KAligned_;
    int64_t halfKAligned_;
    int64_t formerNum_;
    int64_t formerTileNum_;
    int64_t tailTileNum_;
    int64_t tileLength_;
    float eps_;
    int64_t coreNum_;
    uint32_t blockIdx_;
    int64_t totalTileNum_;
    int64_t blocksPerCore_;
};

template <typename T, bool isDeterministic, bool weightType>
__aicore__ inline void LightningIndexerKLLoss<T, isDeterministic, weightType>::Init(
    GM_ADDR targetScore, GM_ADDR indexProbs, GM_ADDR loss, GM_ADDR workspace,
    const LightningIndexerKLLossTilingData *tilingData, TPipe *pipe)
{
    totalLength_ = tilingData->totalLength;
    K_ = tilingData->K;
    KAligned_ = tilingData->KAligned;
    halfKAligned_ = (K_ + HALF_BLOCK_SIZE - 1) / HALF_BLOCK_SIZE * HALF_BLOCK_SIZE;
    formerNum_ = tilingData->formerNum;
    formerTileNum_ = tilingData->formerTileNum;
    tailTileNum_ = tilingData->tailTileNum;
    tileLength_ = tilingData->tileLength;
    eps_ = tilingData->eps;
    coreNum_ = tilingData->coreNum;
    // 预计算循环控制参数
    blockIdx_ = GetBlockIdx();
    blocksPerCore_ = (blockIdx_ < formerNum_) ? formerTileNum_ : tailTileNum_;

    inputGMTargetScore.SetGlobalBuffer((__gm__ T *)targetScore, totalLength_ * K_);
    inputGMIndexProbs.SetGlobalBuffer((__gm__ T *)indexProbs, totalLength_ * K_);
    outputGMLoss.SetGlobalBuffer((__gm__ T *)loss, 1);
    if (std::is_same_v<T, float> && !isDeterministic) {
        if (blockIdx_ == 0) {
            outputGMLoss.SetValue(0, (T)0);
            DataCacheCleanAndInvalid<T, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(outputGMLoss);
        }
    } else {
        InitWorkspaceDet(workspace);
    }

    // 分配 UB buffer：7 块连续 LocalTensor，总大小 = tileLen_ * (5 * KAligned_ + 8) + 8
    uint32_t tileElemsAligned = tileLength_ * KAligned_;
    int64_t totalUbElems = tileLength_ * (5 * KAligned_ + 8) + 8;

    pipePtr = pipe;
    pipePtr->InitBuffer(tmpBuf_, totalUbElems * sizeof(float));
    uint32_t ubOffset = 0;
    ubTargetScoreIn = tmpBuf_.GetWithOffset<float>(tileElemsAligned, ubOffset);
    ubOffset += tileElemsAligned * sizeof(float);
    ubIndexProbsIn = tmpBuf_.GetWithOffset<float>(tileElemsAligned, ubOffset);
    ubOffset += tileElemsAligned * sizeof(float);
    ubReduceSum = tmpBuf_.GetWithOffset<float>(FLOAT_BLOCK_SIZE * tileLength_, ubOffset);
    ubOffset += FLOAT_BLOCK_SIZE * tileLength_ * sizeof(float);
    ubLogP = tmpBuf_.GetWithOffset<float>(tileElemsAligned, ubOffset);
    ubOffset += tileElemsAligned * sizeof(float);
    ubLogY = tmpBuf_.GetWithOffset<float>(tileElemsAligned, ubOffset);
    ubOffset += tileElemsAligned * sizeof(float);
    // ubOut: 8 个元素对齐
    ubOut = tmpBuf_.GetWithOffset<float>(FLOAT_BLOCK_SIZE, ubOffset);
    ubOffset += BLOCK_SIZE;
    ubTmp = tmpBuf_.GetWithOffset<float>(tileElemsAligned, ubOffset);

    // 分配流水线同步 EventID
    eventMTE2YToV_ = static_cast<int32_t>(pipePtr->AllocEventID<HardEvent::MTE2_V>());
    eventMTE2yToV_ = static_cast<int32_t>(pipePtr->AllocEventID<HardEvent::MTE2_V>());
    eventVToMTE2Y_ = static_cast<int32_t>(pipePtr->AllocEventID<HardEvent::V_MTE2>());
    eventVToMTE2y_ = static_cast<int32_t>(pipePtr->AllocEventID<HardEvent::V_MTE2>());
    eventVToMTE3_ = static_cast<int32_t>(pipePtr->AllocEventID<HardEvent::V_MTE3>());
    eventMTE3ToV_ = static_cast<int32_t>(pipePtr->AllocEventID<HardEvent::MTE3_V>());

    copyParams.blockLen = K_ * sizeof(float);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    copyParams.rsv = 0;

    padParams.isPad = true;
    padParams.leftPadding = 0;
    padParams.rightPadding = KAligned_ - K_;
    padParams.paddingValue = 0;
}

/* ---------- InitWorkspaceDet: 初始化 deterministic workspace ---------- */
template <typename T, bool isDeterministic, bool weightType>
__aicore__ inline void LightningIndexerKLLoss<T, isDeterministic, weightType>::InitWorkspaceDet(GM_ADDR workspace)
{
    perCoreSumWS.SetGlobalBuffer((__gm__ float *)(workspace) + blockIdx_ * FLOAT_BLOCK_SIZE, FLOAT_BLOCK_SIZE);
    Fill(perCoreSumWS, FLOAT_BLOCK_SIZE, (float)0);
    userWorkspace_ = workspace;
}

/* ---------- InitHalfBufs: half/bf16 版本 ubHalf 指向 float buffer 后半段 ---------- */
template <typename T, bool isDeterministic, bool weightType>
__aicore__ inline void LightningIndexerKLLoss<T, isDeterministic, weightType>::InitHalfBufs()
{
    uint32_t tileElemsAligned = tileLength_ * KAligned_ * 2;
    uint32_t floatDword = tileLength_ * KAligned_ * sizeof(float);
    uint32_t halfOff = (KAligned_ * 2 - halfKAligned_) * sizeof(T);
    ubHalfTarget = tmpBuf_.GetWithOffset<T>(tileElemsAligned, halfOff);
    ubHalfIndexProbs = tmpBuf_.GetWithOffset<T>(tileElemsAligned, floatDword + halfOff);
    copyParams.blockLen = K_ * sizeof(T);
    copyParams.dstStride = (KAligned_ * 2 - halfKAligned_) / HALF_BLOCK_SIZE;
    padParams.rightPadding = halfKAligned_ - K_;
}

template <typename T, bool isDeterministic, bool weightType>
__aicore__ inline void LightningIndexerKLLoss<T, isDeterministic, weightType>::CopyIn(uint32_t rowOffset,
                                                                                      uint32_t curTileLen)
{
    uint64_t offset = (uint64_t)rowOffset * (uint64_t)K_;
    copyParams.blockCount = curTileLen;
    if constexpr (std::is_same_v<T, float>) {
        WaitFlag<HardEvent::V_MTE2>(eventVToMTE2Y_);
        DataCopyPad(ubIndexProbsIn, inputGMIndexProbs[offset], copyParams, padParams);
        SetFlag<HardEvent::MTE2_V>(eventMTE2YToV_);

        WaitFlag<HardEvent::V_MTE2>(eventVToMTE2y_);
        DataCopyPad(ubTargetScoreIn, inputGMTargetScore[offset], copyParams, padParams);
        SetFlag<HardEvent::MTE2_V>(eventMTE2yToV_);
    } else {
        WaitFlag<HardEvent::V_MTE2>(eventVToMTE2Y_);
        DataCopyPad(ubHalfIndexProbs, inputGMIndexProbs[offset], copyParams, padParams);
        SetFlag<HardEvent::MTE2_V>(eventMTE2YToV_);

        WaitFlag<HardEvent::V_MTE2>(eventVToMTE2y_);
        DataCopyPad(ubHalfTarget, inputGMTargetScore[offset], copyParams, padParams);
        SetFlag<HardEvent::MTE2_V>(eventMTE2yToV_);
    }
}

/* ---------- Compute: Vector 全部计算，含 setflag/waitflag ---------- */
template <typename T, bool isDeterministic, bool weightType>
__aicore__ inline void LightningIndexerKLLoss<T, isDeterministic, weightType>::Compute(uint32_t tileElems,
                                                                                       uint32_t curTileLen)
{
    uint32_t repeat, remain;
    constexpr uint32_t MAX_REPEAT_TIME = 255;
    WaitFlag<HardEvent::MTE2_V>(eventMTE2YToV_);
    if constexpr (!std::is_same_v<T, float>) {
        repeat = KAligned_ / FLOAT_VECTOR_SIZE;
        remain = KAligned_ % FLOAT_VECTOR_SIZE;
        UnaryRepeatParams repeatParams(1, 1, 8, 4);
        for (uint32_t i = 0; i < curTileLen; ++i) {
            Cast(ubIndexProbsIn[i * KAligned_], ubHalfIndexProbs[i * KAligned_ * 2], RoundMode::CAST_NONE,
                 FLOAT_VECTOR_SIZE, repeat, repeatParams);
            if (remain) {
                Cast(ubIndexProbsIn[i * KAligned_ + repeat * FLOAT_VECTOR_SIZE],
                     ubHalfIndexProbs[i * KAligned_ * 2 + repeat * FLOAT_VECTOR_SIZE], RoundMode::CAST_NONE, remain, 1,
                     repeatParams);
            }
        }
    }
    Adds(ubLogY, ubIndexProbsIn, eps_, tileElems);
    SetFlag<HardEvent::V_MTE2>(eventVToMTE2Y_);
    Ln(ubLogY, ubLogY, tileElems);

    WaitFlag<HardEvent::MTE2_V>(eventMTE2yToV_);
    if constexpr (!std::is_same_v<T, float>) {
        UnaryRepeatParams repeatParams(1, 1, 8, 4);
        for (uint32_t i = 0; i < curTileLen; ++i) {
            Cast(ubTargetScoreIn[i * KAligned_], ubHalfTarget[i * KAligned_ * 2], RoundMode::CAST_NONE,
                 FLOAT_VECTOR_SIZE, repeat, repeatParams);
            if (remain) {
                Cast(ubTargetScoreIn[i * KAligned_ + repeat * FLOAT_VECTOR_SIZE],
                     ubHalfTarget[i * KAligned_ * 2 + repeat * FLOAT_VECTOR_SIZE], RoundMode::CAST_NONE, remain, 1,
                     repeatParams);
            }
        }
    }

    // CalcLogP: log(clamp_min(y/sum(y), eps)) * y
    repeat = K_ / FLOAT_VECTOR_SIZE;
    remain = K_ % FLOAT_VECTOR_SIZE;
    uint32_t reduceOff = (curTileLen * (FLOAT_BLOCK_SIZE - 1)) / FLOAT_BLOCK_SIZE * FLOAT_BLOCK_SIZE;
    for (int i = 0; i < curTileLen; ++i) {
        if (remain && !repeat) {
            ReduceSum(ubReduceSum[reduceOff + i], ubTargetScoreIn[i * KAligned_], ubTmp, remain, 1, 8);
            continue;
        }
        ReduceSum(ubReduceSum[reduceOff + i], ubTargetScoreIn[i * KAligned_], ubTmp, FLOAT_VECTOR_SIZE, repeat, 8);
        if (remain) {
            ReduceSum(ubLogP[i], ubTargetScoreIn[i * KAligned_ + repeat * FLOAT_VECTOR_SIZE], ubTmp, remain, 1, 8);
        }
    }
    if (remain && repeat) {
        repeat = curTileLen / FLOAT_VECTOR_SIZE;
        remain = curTileLen % FLOAT_VECTOR_SIZE;
        Add(ubReduceSum[reduceOff], ubReduceSum[reduceOff], ubLogP, FLOAT_VECTOR_SIZE, repeat, {1, 1, 1, 8, 8, 8});
        if (remain) {
            Add(ubReduceSum[reduceOff + repeat * FLOAT_VECTOR_SIZE],
                ubReduceSum[reduceOff + repeat * FLOAT_VECTOR_SIZE], ubLogP[repeat * FLOAT_VECTOR_SIZE], remain, 1,
                {1, 1, 1, 8, 8, 8});
        }
    }
    repeat = curTileLen / FLOAT_VECTOR_SIZE;
    remain = curTileLen % FLOAT_VECTOR_SIZE;
    Adds(ubReduceSum[reduceOff], ubReduceSum[reduceOff], eps_, FLOAT_VECTOR_SIZE, repeat, {1, 1, 8, 8});
    if (remain) {
        Adds(ubReduceSum[reduceOff + repeat * FLOAT_VECTOR_SIZE], ubReduceSum[reduceOff + repeat * FLOAT_VECTOR_SIZE],
             eps_, remain, 1, {1, 1, 8, 8});
    }

    repeat = (curTileLen + 7) / 8;
    Brcb(ubReduceSum, ubReduceSum[reduceOff], repeat, {1, 8});
    repeat = KAligned_ / FLOAT_VECTOR_SIZE;
    remain = KAligned_ % FLOAT_VECTOR_SIZE;
    BinaryRepeatParams divParams{1, 1, 0, 8, 8, 0};
    for (uint32_t i = 0; i < curTileLen; ++i) {
        Div(ubTmp[i * KAligned_], ubTargetScoreIn[i * KAligned_], ubReduceSum[i * FLOAT_BLOCK_SIZE], FLOAT_VECTOR_SIZE,
            repeat, divParams);
        if (remain) {
            Div(ubTmp[i * KAligned_ + repeat * FLOAT_VECTOR_SIZE],
                ubTargetScoreIn[i * KAligned_ + repeat * FLOAT_VECTOR_SIZE], ubReduceSum[i * FLOAT_BLOCK_SIZE], remain,
                1, divParams);
        }
    }
    ClampMin(ubLogP, ubTmp, eps_, tileElems);
    Ln(ubLogP, ubLogP, tileElems);
    Sub(ubLogP, ubLogP, ubLogY, tileElems);
    // 根据 weightType 选择外层权重:
    //   weightType=0 ('logits'): result = log_ratio * ubTargetScoreIn (原始 y)
    //   weightType=1 ('probs'):  result = log_ratio * p, p = ubTargetScoreIn / (ubReduceSum + eps)

    if constexpr (weightType) {
        Mul(ubLogP, ubLogP, ubTmp, tileElems);
    } else {
        Mul(ubLogP, ubLogP, ubTargetScoreIn, tileElems);
    }
    SetFlag<HardEvent::V_MTE2>(eventVToMTE2y_);

    WaitFlag<HardEvent::MTE3_V>(eventMTE3ToV_);
    Duplicate(ubOut, (float)0, 8);
    repeat = tileElems / FLOAT_VECTOR_SIZE;
    remain = tileElems % FLOAT_VECTOR_SIZE;
    if (remain && !repeat) {
        ReduceSum(ubOut, ubLogP, ubTmp, remain, 1, 8);
    } else {
        ReduceSum(ubOut, ubLogP, ubTmp, FLOAT_VECTOR_SIZE, repeat, 8);
        if (remain) {
            ReduceSum(ubReduceSum, ubLogP[repeat * FLOAT_VECTOR_SIZE], ubTmp, remain, 1, 8);
            Add(ubOut, ubOut, ubReduceSum, 1, 1, {1, 1, 1, 8, 8, 8});
        }
    }
    SetFlag<HardEvent::V_MTE3>(eventVToMTE3_);
}

template <typename T, bool isDeterministic, bool weightType>
__aicore__ inline void LightningIndexerKLLoss<T, isDeterministic, weightType>::CopyOut()
{
    // 等待 Vector 完成 ReduceSum，ubOut 已就绪
    WaitFlag<HardEvent::V_MTE3>(eventVToMTE3_);
    SetAtomicAdd<float>();
    // DumpTensor(ubOut, 111, 8);
    // DumpTensor(outputGMLoss, 222, 8);
    if constexpr (std::is_same_v<T, float> && !isDeterministic) {
        DataCopyPad(outputGMLoss, ubOut, {1, sizeof(float), 0, 0});
    } else {
        DataCopyPad(perCoreSumWS, ubOut, {1, sizeof(float), 0, 0});
    }
    DisableDmaAtomic();
    // 通知 Vector ubOut 已读取，可安全写入下一轮
    SetFlag<HardEvent::MTE3_V>(eventMTE3ToV_);
}

/* ---------- Process: 主循环，isHalf 区分 half/fp32 分支 ---------- */
template <typename T, bool isDeterministic, bool weightType>
__aicore__ inline void LightningIndexerKLLoss<T, isDeterministic, weightType>::Process()
{
    uint32_t tileId, rowOffset, curTileLen, tileElems;
    rowOffset = blockIdx_ * tileLength_;
    uint32_t step = coreNum_ * tileLength_;
    // 预置所有 flag，使第一轮各 WaitFlag 直接通过
    SetFlag<HardEvent::V_MTE2>(eventVToMTE2Y_);
    SetFlag<HardEvent::V_MTE2>(eventVToMTE2y_);
    SetFlag<HardEvent::MTE3_V>(eventMTE3ToV_);
    for (int64_t b = 0; b < blocksPerCore_; b++) {
        curTileLen = (rowOffset + tileLength_ <= totalLength_) ? tileLength_ : (totalLength_ - rowOffset);
        CopyIn(rowOffset, curTileLen);
        tileElems = curTileLen * KAligned_;
        Compute(tileElems, curTileLen);
        CopyOut();
        rowOffset += step;
    }

    WaitFlag<HardEvent::V_MTE2>(eventVToMTE2Y_);
    WaitFlag<HardEvent::V_MTE2>(eventVToMTE2y_);
    WaitFlag<HardEvent::MTE3_V>(eventMTE3ToV_);

    pipePtr->ReleaseEventID<HardEvent::MTE2_V>(eventMTE2YToV_);
    pipePtr->ReleaseEventID<HardEvent::MTE2_V>(eventMTE2yToV_);
    pipePtr->ReleaseEventID<HardEvent::V_MTE2>(eventVToMTE2Y_);
    pipePtr->ReleaseEventID<HardEvent::V_MTE2>(eventVToMTE2y_);
    pipePtr->ReleaseEventID<HardEvent::V_MTE3>(eventVToMTE3_);
    pipePtr->ReleaseEventID<HardEvent::MTE3_V>(eventMTE3ToV_);
}

/* ---------- WriteBackDet: 确定性结果写回 ---------- */
template <typename T, bool isDeterministic, bool weightType>
__aicore__ inline void LightningIndexerKLLoss<T, isDeterministic, weightType>::WriteBackDet()
{
    SyncAll();
    if (blockIdx_ == 0) {
        GlobalTensor<float> ws;
        uint32_t count = coreNum_ * FLOAT_BLOCK_SIZE;
        LocalTensor<float> ubCoreSum = tmpBuf_.Get<float>(count);
        ws.SetGlobalBuffer((__gm__ float *)userWorkspace_, count);
        DataCopy(ubCoreSum, ws, count);
        SetFlag<HardEvent::MTE2_V>(eventMTE2YToV_);
        WaitFlag<HardEvent::MTE2_V>(eventMTE2YToV_);
        ReduceSum(ubOut, ubCoreSum, ubTmp, count);

        if constexpr (std::is_same_v<T, float>) {
            SetFlag<HardEvent::V_MTE3>(eventVToMTE3_);
            WaitFlag<HardEvent::V_MTE3>(eventVToMTE3_);
            DataCopyPad(outputGMLoss, ubOut, {1, sizeof(float), 0, 0});
        } else {
            LocalTensor<T> ubOutHalf = tmpBuf_.GetWithOffset<T>(HALF_BLOCK_SIZE, count * sizeof(float));
            Cast(ubOutHalf, ubOut, RoundMode::CAST_RINT, FLOAT_BLOCK_SIZE);
            SetFlag<HardEvent::V_MTE3>(eventVToMTE3_);
            WaitFlag<HardEvent::V_MTE3>(eventVToMTE3_);
            DataCopyPad(outputGMLoss, ubOutHalf, {1, sizeof(T), 0, 0});
        }
    }
}

} // namespace NsLightningIndexerKLLoss
#endif // LIGHTNING_INDEXER_KL_LOSS_KERNEL_H
