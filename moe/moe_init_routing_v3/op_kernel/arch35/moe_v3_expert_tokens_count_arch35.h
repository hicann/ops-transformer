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
 * \file moe_v3_expert_tokens_count_arch35.h
 * \brief
 */
#ifndef MOE_V3_EXPERT_TOKENS_COUNT_H_REGBASE_ARCH35
#define MOE_V3_EXPERT_TOKENS_COUNT_H_REGBASE_ARCH35

#include "moe_v3_common.h"
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"

namespace MoeInitRoutingV3 {
using namespace AscendC;

constexpr int64_t CUMSUM_MODE = 0LL;
constexpr int64_t COUNT_MODE = 1LL;
constexpr int64_t KEY_VALUE_MODE = 2LL;
constexpr int64_t KEY_VALUE_MODE_DIM_NUM = 2LL;

class ExpertTokensCount {
public:
    __aicore__ inline ExpertTokensCount(){};
    __aicore__ inline void Init(GM_ADDR expandedRowIdx, GM_ADDR expertTokensCount, GM_ADDR workspace,
                                const MoeInitRoutingV3Arch35TilingData* tilingData, TPipe* tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyOut();

    __aicore__ inline void expertCountCopyIn();
    __aicore__ inline void expertCountCompute();
    __aicore__ inline void expertCountCopyOut();
    __aicore__ inline void InitBasicParams(const MoeInitRoutingV3Arch35TilingData* tilingData, TPipe* tPipe);

private:
    GlobalTensor<int32_t> sortedExpertIdxGm_;
    GlobalTensor<int32_t> expertCountTempGm_;
    GlobalTensor<int64_t> expertTokensCountGm_;
    GlobalTensor<int32_t> expertTotalCountGm_;
    GlobalTensor<int32_t> expandedRowIdxGm_;
    GlobalTensor<int32_t> expertIdxValueGm_;
    TPipe* pipe_;

    TQue<QuePosition::VECIN, 1> sortedExpertIdxInQueue_;
    TQue<QuePosition::VECOUT, 1> expertCountOutToTempQueue_;
    TQue<QuePosition::VECIN, 1> expertCountTempInQueue_;
    TQue<QuePosition::VECOUT, 1> expertIdxCountOutQueue_;
    TQue<QuePosition::VECOUT, 1> expertTotalCountQueue_;
    TBuf<TPosition::VECCALC> reduceFloatBuf_;

    const MoeV3Arch35ExpertTokensCountTilingData* expertTokensCountTilingData_;
    int64_t blockIdx_;
    int64_t needCoreNum_;
    int64_t perCoreElements_;
    int64_t curCoreElements_ = 0;
    int64_t expertStart_ = 0;
    int64_t expertEnd_ = 0;
    int64_t actualExpertNum_ = 0;
    int64_t coreLoopsNum_ = 0;
    int64_t perCorePerLoopElements_ = 0;
    int64_t perCoreLastLoopElements_ = 0;
    int64_t actualExpertTotalNum_ = 0;
    int64_t expertNum_ = 0;
    int64_t useFp32Vector_ = 0;
    int64_t expertTokensNumType_ = 0;
    int64_t expertCountElements_ = 0;
    int64_t coreNum_ = 0;
    int64_t dropPadMode_ = 0;
    int64_t needExpertTotalCount_ = 0;
    int64_t useGatherCopy_ = 0;
    int64_t totalLength_ = 0;
    bool expertTokensNumFlag_ = true;
    int32_t finalExpertId_ = -1;
    int32_t expertTokenValue_ = 0;
};

__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_THREAD_NUM) inline void ComputeExpertTotalOnlySimt(
    int64_t elementNum, int32_t expertStart, int32_t expertEnd, __gm__ int32_t* sortedExpertId,
    __gm__ int32_t* expertTotalCount)
{
    if (threadIdx.x != 0) {
        return;
    }

    if (elementNum == 0) {
        expertTotalCount[0] = 0;
        return;
    }
    int32_t lastExpertId = sortedExpertId[elementNum - 1];
    if (lastExpertId >= expertStart && lastExpertId < expertEnd) {
        expertTotalCount[0] = static_cast<int32_t>(elementNum);
        return;
    }
    int32_t firstExpertId = sortedExpertId[0];
    if (firstExpertId < expertStart || firstExpertId >= expertEnd) {
        expertTotalCount[0] = 0;
        return;
    }
    int64_t left = 1;
    int64_t right = elementNum - 1;
    while (left < right) {
        int64_t mid = left + (right - left) / 2;
        int32_t expertId = sortedExpertId[mid];
        if (expertId >= expertStart && expertId < expertEnd) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    expertTotalCount[0] = static_cast<int32_t>(left);
}

__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_THREAD_NUM) inline void ComputeExpertBoundaryOnlySimt(
    int64_t elementNum, int32_t expertStart, int32_t expertEnd, __gm__ int32_t* sortedExpertId,
    __gm__ int32_t* expertBoundary)
{
    if (threadIdx.x != 0) {
        return;
    }
    int32_t lastExpert = elementNum > 0 ? sortedExpertId[elementNum - 1] : -1;
    int64_t left = 0;
    int64_t right = elementNum;
    if (lastExpert >= expertStart && lastExpert < expertEnd) {
        while (left < right) {
            int64_t mid = left + (right - left) / 2;
            if (sortedExpertId[mid] < lastExpert) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
    } else {
        left = elementNum;
    }
    expertBoundary[0] = lastExpert;
    expertBoundary[1] = static_cast<int32_t>(elementNum - left);
}

__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_THREAD_NUM) inline void ComputeExpertFirstIndexSimt(
    int32_t elementNum, int32_t expertStart, int32_t expertEnd, __gm__ int32_t* sortedExpertIdGmAddr,
    __ubuf__ int32_t* expertFirstIndexLocalAddr)
{
    for (auto i = static_cast<int32_t>(threadIdx.x); i < elementNum; i += static_cast<int32_t>(blockDim.x)) {
        auto currExpertId = sortedExpertIdGmAddr[i];
        if (currExpertId < expertStart) {
            continue;
        }
        if (currExpertId >= expertEnd) {
            break;
        }
        auto prevExpertId = (i == 0 ? -1 : sortedExpertIdGmAddr[i - 1]);
        if (currExpertId != prevExpertId) {
            expertFirstIndexLocalAddr[currExpertId - expertStart] = i;
        }
    }
}

__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_THREAD_NUM) inline void ComputeExpertCountOutSimt(
    int32_t elementNum, int32_t expertStart, int32_t expertEnd, __gm__ int32_t* sortedExpertIdGmAddr,
    __ubuf__ int32_t* expertFirstIndexLocalAddr, __ubuf__ int32_t* expertCountOutLocalAddr)
{
    for (auto i = static_cast<int32_t>(threadIdx.x); i < elementNum; i += static_cast<int32_t>(blockDim.x)) {
        auto currExpertId = sortedExpertIdGmAddr[i];
        if (currExpertId < expertStart) {
            continue;
        }
        if (currExpertId >= expertEnd) {
            break;
        }
        if (i == elementNum - 1 || currExpertId != sortedExpertIdGmAddr[i + 1]) {
            expertCountOutLocalAddr[currExpertId - expertStart] =
                i + 1 - expertFirstIndexLocalAddr[currExpertId - expertStart];
        }
    }
}

__aicore__ inline void ExpertTokensCount::InitBasicParams(const MoeInitRoutingV3Arch35TilingData* tilingData,
                                                          TPipe* tPipe)
{
    pipe_ = tPipe;
    expertTokensCountTilingData_ = &(tilingData->expertTokensCountTilingDataOp);
    blockIdx_ = GetBlockIdx();
    needCoreNum_ = expertTokensCountTilingData_->needCoreNum;
    perCoreElements_ = expertTokensCountTilingData_->perCoreElements;
    expertStart_ = tilingData->expertStart;
    expertEnd_ = tilingData->expertEnd;
    actualExpertNum_ = tilingData->actualExpertNum;
    expertNum_ = tilingData->expertNum;
    expertTokensNumType_ = tilingData->expertTokensNumType;
    coreNum_ = tilingData->coreNum;
    dropPadMode_ = tilingData->dropPadMode;
    useGatherCopy_ = tilingData->useGatherCopy;
    totalLength_ = tilingData->n * tilingData->k;
    expertTokensNumFlag_ = tilingData->expertTokensNumFlag;
    needExpertTotalCount_ = static_cast<int64_t>(dropPadMode_ != DROP_PAD_MODE);
    // Single truth from the host: the FP32 vector reduce is enabled only when the 2^24 precision
    // gate holds AND the reduce buffer fits the UB budget the host reserved for it.
    useFp32Vector_ = expertTokensCountTilingData_->expertTokensCountVecReduce;

    if (blockIdx_ == needCoreNum_ - 1) {
        curCoreElements_ = expertTokensCountTilingData_->lastCoreElements;
        coreLoopsNum_ = expertTokensCountTilingData_->lastCoreLoops;
        perCorePerLoopElements_ = expertTokensCountTilingData_->lastCorePerLoopElements;
        perCoreLastLoopElements_ = expertTokensCountTilingData_->lastCoreLastLoopElements;
    } else {
        curCoreElements_ = expertTokensCountTilingData_->perCoreElements;
        coreLoopsNum_ = expertTokensCountTilingData_->perCoreLoops;
        perCorePerLoopElements_ = expertTokensCountTilingData_->perCorePerLoopElements;
        perCoreLastLoopElements_ = expertTokensCountTilingData_->perCoreLastLoopElements;
    }
    if (expertTokensNumType_ == KEY_VALUE_MODE) {
        expertCountElements_ = ((actualExpertNum_ + 1) < expertNum_) ? (actualExpertNum_ + 1) * KEY_VALUE_MODE_DIM_NUM :
                                                                       expertNum_ * KEY_VALUE_MODE_DIM_NUM;
    } else {
        expertCountElements_ = actualExpertNum_;
    }
}

__aicore__ inline void ExpertTokensCount::Init(GM_ADDR expandedRowIdx, GM_ADDR expertTokensCount, GM_ADDR workspace,
                                               const MoeInitRoutingV3Arch35TilingData* tilingData, TPipe* tPipe)
{
    InitBasicParams(tilingData, tPipe);

    sortedExpertIdxGm_.SetGlobalBuffer((__gm__ int32_t*)workspace + blockIdx_ * perCoreElements_, curCoreElements_);
    if (expertTokensNumFlag_) {
        expertTokensCountGm_.SetGlobalBuffer((__gm__ int64_t*)expertTokensCount, expertCountElements_);
    }
    expertCountTempGm_.SetGlobalBuffer(
        (__gm__ int32_t*)workspace + Align(tilingData->n * tilingData->k, sizeof(int32_t)) * 2, actualExpertNum_);
    if (needExpertTotalCount_) {
        expertTotalCountGm_.SetGlobalBuffer((__gm__ int32_t*)workspace +
                                                Align(tilingData->n * tilingData->k, sizeof(int32_t)) * 2 +
                                                Align(actualExpertNum_, sizeof(int32_t)),
                                            1);
    }
    if (dropPadMode_ == DROP_PAD_MODE) {
        expertIdxValueGm_.SetGlobalBuffer((__gm__ int32_t*)workspace +
                                              Align(tilingData->n * tilingData->k, sizeof(int32_t)) * 2 +
                                              Align(actualExpertNum_, sizeof(int32_t)),
                                          coreNum_ * EXPERT_ID_VALUE_NUM);
    }

    expandedRowIdxGm_.SetGlobalBuffer((__gm__ int32_t*)expandedRowIdx + blockIdx_ * perCoreElements_);
    if ((tilingData->rowIdxType == GATHER) && (blockIdx_ < needCoreNum_)) {
        InitGlobalMemory(expandedRowIdxGm_, curCoreElements_, -1);
        SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
        SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
    }
    if (useGatherCopy_ && tilingData->rowIdxType && (blockIdx_ < needCoreNum_)) {
        int64_t totalLength = Align(tilingData->n * tilingData->k, sizeof(int32_t));
        GlobalTensor<int32_t> gatherRowIdxInitGm;
        gatherRowIdxInitGm.SetGlobalBuffer((__gm__ int32_t*)workspace + totalLength + blockIdx_ * perCoreElements_,
                                           curCoreElements_);
        InitGlobalMemory(gatherRowIdxInitGm, curCoreElements_, static_cast<int32_t>(-1));
        SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
        SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
    }

    if (!expertTokensNumFlag_) {
        return;
    }
    int64_t sortedExpertIdxInLen = Max(perCorePerLoopElements_, perCoreLastLoopElements_);
    pipe_->InitBuffer(sortedExpertIdxInQueue_, 1, AlignBytes(sortedExpertIdxInLen, sizeof(int32_t)));
    pipe_->InitBuffer(expertCountOutToTempQueue_, 1, AlignBytes(actualExpertNum_, sizeof(int32_t)));
    pipe_->InitBuffer(expertCountTempInQueue_, 1, AlignBytes(actualExpertNum_, sizeof(int32_t)));
    pipe_->InitBuffer(expertIdxCountOutQueue_, 1, AlignBytes(expertCountElements_, sizeof(int64_t)));
    if (needExpertTotalCount_) {
        pipe_->InitBuffer(expertTotalCountQueue_, 1, AlignBytes(1, sizeof(int32_t)));
    }
    if (expertTokensNumType_ == COUNT_MODE && needExpertTotalCount_ && useFp32Vector_) {
        pipe_->InitBuffer(reduceFloatBuf_, AlignBytes(actualExpertNum_, sizeof(float)));
    }
}

__aicore__ inline void ExpertTokensCount::Process()
{
    if (!expertTokensNumFlag_) {
        auto* sortedExpertId = (__gm__ int32_t*)sortedExpertIdxGm_.GetPhyAddr();
        if (needExpertTotalCount_ && blockIdx_ == 0) {
            asc_vf_call<ComputeExpertTotalOnlySimt>(dim3{SIMT_THREAD_NUM, 1, 1}, totalLength_, expertStart_, expertEnd_,
                                                    sortedExpertId, (__gm__ int32_t*)expertTotalCountGm_.GetPhyAddr());
        } else if (!needExpertTotalCount_) {
            int64_t elements = blockIdx_ < needCoreNum_ ? curCoreElements_ : 0;
            asc_vf_call<ComputeExpertBoundaryOnlySimt>(
                dim3{SIMT_THREAD_NUM, 1, 1}, elements, expertStart_, expertEnd_, sortedExpertId,
                (__gm__ int32_t*)expertIdxValueGm_.GetPhyAddr() + blockIdx_ * EXPERT_ID_VALUE_NUM);
        }
        // Also completes the common index initialization before phases 3 and 5.
        SyncAll();
        return;
    }
    if (blockIdx_ < needCoreNum_) {
        LocalTensor<int32_t> expertCountOutLocal = expertCountOutToTempQueue_.AllocTensor<int32_t>();
        Duplicate(expertCountOutLocal, 0, actualExpertNum_);

        __gm__ int32_t* sortedExpertIdxGmAddr = (__gm__ int32_t*)sortedExpertIdxGm_.GetPhyAddr();
        __ubuf__ int32_t* expertCountOutLocalAddr = (__ubuf__ int32_t*)expertCountOutLocal.GetPhyAddr();

        asc_vf_call<ComputeExpertFirstIndexSimt>(dim3{SIMT_THREAD_NUM, 1, 1}, curCoreElements_, expertStart_,
                                                 expertEnd_, sortedExpertIdxGmAddr, expertCountOutLocalAddr);
        asc_vf_call<ComputeExpertCountOutSimt>(dim3{SIMT_THREAD_NUM, 1, 1}, curCoreElements_, expertStart_, expertEnd_,
                                               sortedExpertIdxGmAddr, expertCountOutLocalAddr, expertCountOutLocalAddr);

        expertCountOutToTempQueue_.EnQue<int32_t>(expertCountOutLocal);

        if (dropPadMode_ == DROP_PAD_MODE && curCoreElements_ > 0) {
            SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
            finalExpertId_ = sortedExpertIdxGm_.GetValue(curCoreElements_ - 1);
            if (finalExpertId_ >= expertStart_ && finalExpertId_ < expertEnd_) {
                expertTokenValue_ = expertCountOutLocal.GetValue(finalExpertId_ - expertStart_);
            }
        }

        CopyOut();
    }

    SyncAll();

    if (blockIdx_ == 0) {
        expertCountCopyIn();
        expertCountCompute();
        expertCountCopyOut();
    }
    if (needExpertTotalCount_) {
        SyncAll();
    }
}

__aicore__ inline void ExpertTokensCount::CopyOut()
{
    LocalTensor<int32_t> expertCountOutLocal = expertCountOutToTempQueue_.DeQue<int32_t>();

    DataCopyExtParams copyParams{static_cast<uint16_t>(1), static_cast<uint32_t>((actualExpertNum_) * sizeof(int32_t)),
                                 0, 0, 0};
    SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
    SetAtomicAdd<int32_t>();
    DataCopyPad(expertCountTempGm_, expertCountOutLocal, copyParams);
    SetAtomicNone();
    SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);

    if (dropPadMode_ == DROP_PAD_MODE) {
        expertCountOutLocal.SetValue(0, finalExpertId_);
        expertCountOutLocal.SetValue(1, expertTokenValue_);
        DataCopyExtParams expertIdxCopyParams{static_cast<uint16_t>(1),
                                              static_cast<uint32_t>(EXPERT_ID_VALUE_NUM * sizeof(int32_t)), 0, 0, 0};
        SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
        DataCopyPad(expertIdxValueGm_[blockIdx_ * EXPERT_ID_VALUE_NUM], expertCountOutLocal, expertIdxCopyParams);
    }
    expertCountOutToTempQueue_.FreeTensor(expertCountOutLocal);
}

__aicore__ inline void ExpertTokensCount::expertCountCopyIn()
{
    LocalTensor<int32_t> expertCountTempInLocal = expertCountTempInQueue_.AllocTensor<int32_t>();

    DataCopyExtParams dataCopyParams{static_cast<uint16_t>(1),
                                     static_cast<uint32_t>((actualExpertNum_) * sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams dataCopyPadParams{false, 0, 0, 0};
    DataCopyPad(expertCountTempInLocal, expertCountTempGm_, dataCopyParams, dataCopyPadParams);
    expertCountTempInQueue_.EnQue(expertCountTempInLocal);
}

__aicore__ inline void ExpertTokensCount::expertCountCompute()
{
    LocalTensor<int32_t> expertCountTempInLocal = expertCountTempInQueue_.DeQue<int32_t>();
    LocalTensor<int64_t> expertCountOutLocal = expertIdxCountOutQueue_.AllocTensor<int64_t>();
    event_t eventIDMte2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(eventIDMte2ToS);
    WaitFlag<HardEvent::MTE2_S>(eventIDMte2ToS);
    if (expertTokensNumType_ == KEY_VALUE_MODE) {
        int64_t expertOffset = 0;
        Duplicate(expertCountOutLocal.ReinterpretCast<int32_t>(), static_cast<int32_t>(0),
                  static_cast<int32_t>(expertCountElements_ * KEY_VALUE_MODE));
        event_t eventIDVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventIDVToS);
        WaitFlag<HardEvent::V_S>(eventIDVToS);
        for (int64_t i = 0; i < actualExpertNum_; i++) {
            int64_t expertCount = static_cast<int64_t>(expertCountTempInLocal.GetValue(i));
            if (expertCount != 0) {
                expertCountOutLocal.SetValue(expertOffset * KEY_VALUE_MODE_DIM_NUM, i + expertStart_);
                expertCountOutLocal.SetValue(expertOffset * KEY_VALUE_MODE_DIM_NUM + 1, expertCount);
                expertOffset++;
                actualExpertTotalNum_ += expertCount;
            }
        }
        expertCountElements_ = Min(expertCountElements_, static_cast<int64_t>((expertOffset + 1) * KEY_VALUE_MODE));
    } else if (expertTokensNumType_ == COUNT_MODE) {
        Cast(expertCountOutLocal, expertCountTempInLocal, RoundMode::CAST_NONE, actualExpertNum_);
        PipeBarrier<PIPE_V>();
        if (needExpertTotalCount_ && useFp32Vector_) {
            // FP32 accumulation is exact on this gated path: every per-expert count and every
            // partial sum is a non-negative integer <= FP32_EXACT_INT_MAX (2^24), which FP32
            // represents and adds without rounding error.
            LocalTensor<float> reduceFloatLocal = reduceFloatBuf_.Get<float>();
            Cast(reduceFloatLocal, expertCountTempInLocal, RoundMode::CAST_NONE, actualExpertNum_);
            PipeBarrier<PIPE_V>();
            ReduceSum(reduceFloatLocal, reduceFloatLocal, reduceFloatLocal, static_cast<int32_t>(actualExpertNum_));
            event_t eventIDVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(eventIDVToS);
            WaitFlag<HardEvent::V_S>(eventIDVToS);
            actualExpertTotalNum_ = static_cast<int64_t>(reduceFloatLocal.GetValue(0));
        } else if (needExpertTotalCount_) {
            for (int64_t i = 0; i < actualExpertNum_; i++) {
                actualExpertTotalNum_ += static_cast<int64_t>(expertCountTempInLocal.GetValue(i));
            }
        }
    } else if (expertTokensNumType_ == CUMSUM_MODE) {
        int64_t cumsumCount = 0;
        for (int64_t i = 0; i < actualExpertNum_; i++) {
            int64_t expertCount = static_cast<int64_t>(expertCountTempInLocal.GetValue(i));
            cumsumCount += expertCount;
            expertCountOutLocal.SetValue(i, cumsumCount);
            actualExpertTotalNum_ += expertCount;
        }
    }
    event_t eventIDSToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(eventIDSToMte3);
    WaitFlag<HardEvent::S_MTE3>(eventIDSToMte3);
    expertIdxCountOutQueue_.EnQue<int64_t>(expertCountOutLocal);
    if (needExpertTotalCount_) {
        LocalTensor<int32_t> expertTotalCountLocal = expertTotalCountQueue_.AllocTensor<int32_t>();
        expertTotalCountLocal.SetValue(0, static_cast<int32_t>(actualExpertTotalNum_));
        expertTotalCountQueue_.EnQue<int32_t>(expertTotalCountLocal);
    }
    expertCountTempInQueue_.FreeTensor(expertCountTempInLocal);
}

__aicore__ inline void ExpertTokensCount::expertCountCopyOut()
{
    LocalTensor<int64_t> expertCountOutLocal = expertIdxCountOutQueue_.DeQue<int64_t>();
    DataCopyExtParams copyParams{static_cast<uint16_t>(1),
                                 static_cast<uint32_t>(expertCountElements_ * sizeof(int64_t)), 0, 0, 0};
    DataCopyPad(expertTokensCountGm_, expertCountOutLocal, copyParams);
    if (needExpertTotalCount_) {
        LocalTensor<int32_t> expertTotalCountLocal = expertTotalCountQueue_.DeQue<int32_t>();
        copyParams.blockLen = sizeof(int32_t);
        DataCopyPad(expertTotalCountGm_, expertTotalCountLocal, copyParams);
        expertTotalCountQueue_.FreeTensor(expertTotalCountLocal);
    }
    expertIdxCountOutQueue_.FreeTensor(expertCountOutLocal);
}

} // namespace MoeInitRoutingV3
#endif // MOE_V3_EXPERT_TOKENS_COUNT_H_REGBASE_ARCH35
