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
 * \file moe_update_expert.h
 * \brief
 */

#ifndef MOE_UPDATE_EXPERT_H
#define MOE_UPDATE_EXPERT_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "adv_api/reduce/sum.h"
#include "kernel_tiling/kernel_tiling.h"
#include "moe_update_expert_tiling.h"
#if __has_include("../common/mc2_kernel_utils.h")
#include "../common/mc2_kernel_utils.h"
#else
#include "../../common/op_kernel/mc2_kernel_utils.h"
#endif

namespace MoeUpdateExpertNamespace {
constexpr uint32_t UB_ALIGN = 32U;
constexpr int32_t FLOAT_BYTES = 4;
constexpr int32_t ALIGN_256_BYTES = 256;
constexpr uint8_t BUFFER_NUM = 2;

using namespace AscendC;

#define UPDATE_EXPERT_DECLARE typename DataType, typename ScalesDataType, bool EnablePruning
#define UPDATE_EXPERT_ARGS DataType, ScalesDataType, EnablePruning

template <UPDATE_EXPERT_DECLARE>
class MoeUpdateExpert {
public:
    __aicore__ inline MoeUpdateExpert() {}
    __aicore__ inline void Init(GM_ADDR expertIdsGM, GM_ADDR eplbTableGM, GM_ADDR expertScalesGM,
                                GM_ADDR pruningThresholdGM, GM_ADDR activeMaskGM, GM_ADDR balancedExpertIdsOutGM,
                                GM_ADDR balancedActiveMaskOutGM, GM_ADDR workspaceGM,
                                const MoeUpdateExpertTilingData *tilingData, TPipe *tPipe);
    __aicore__ inline void Process();
    __aicore__ inline void ProcessPruning(int32_t tokenStart, int32_t tokenEnd);

private:
    __aicore__ inline void CalculateUnroll(int32_t start, int32_t computeLen);
    __aicore__ inline void Calculate(int32_t start, int32_t end, int32_t computeLen);
    __aicore__ inline void CalculateUnrollRest(int32_t start, int32_t loopNum, int32_t restNum);
    __aicore__ inline void GetCoreRange(int32_t coreIndex, int32_t coreNum, int32_t totalData, int32_t &start,
                                        int32_t &end);
    __aicore__ inline void InitPruning();
    __aicore__ inline void ProcessEachTokenPruning(int32_t tokenIdx);

    static constexpr uint32_t UNROLL_NUM = 4U;
    static constexpr uint32_t BUFFER_SIZE = 512U;

    GM_ADDR expertIdsGM_ = nullptr;
    GM_ADDR eplbTableGM_ = nullptr;
    GM_ADDR expertScalesGM_ = nullptr;
    GM_ADDR pruningThresholdGM_ = nullptr;
    GM_ADDR activeMaskGM_ = nullptr;
    GM_ADDR balancedExpertIdsOutGM_ = nullptr;
    GM_ADDR balancedActiveMaskOutGM_ = nullptr;

    GlobalTensor<DataType> expertIdsTensor_;
    GlobalTensor<DataType> balancedExpertIdsTensor_;
    GlobalTensor<int32_t> eplbTableTensor_;
    // 剪枝相关成员
    GlobalTensor<ScalesDataType> expertScalesTensor_;
    GlobalTensor<float> preTuningThresholdTensor_;
    GlobalTensor<uint8_t> activeMaskInTensor_;
    GlobalTensor<bool> balancedActiveMaskTensor_;

    TQue<QuePosition::VECIN, 1> expertScalesQueue_;
    TBuf<> localScalesFp32Buf_;
    TBuf<> localScaleSumBuf_;
    TBuf<> localThresholdsBuf_;
    TBuf<> localThresholdProductsBuf_;
    TBuf<> localActiveMaskBuf_;
    TBuf<> localOriginActiveMaskBuf_;
    TBuf<> localSelectedActiveMaskHalfBuf_;
    TQue<QuePosition::VECOUT, 1> resultMaskUint8Queue_;
    TBuf<> localAllFalseMaskUint8Buf_;
    TBuf<> localCompareResultBuf_;

    TPipe *tPipe_{nullptr};
    TBuf<> buffer_;
    LocalTensor<DataType> outTensor_;
    LocalTensor<float> localScalesFp32_;
    LocalTensor<float> localScaleSum_;
    LocalTensor<float> localThresholds_;
    LocalTensor<float> localThresholdProducts_;
    LocalTensor<half> localActiveMask_;
    LocalTensor<uint8_t> localOriginActiveMask_;
    LocalTensor<half> localSelectedActiveMaskHalf_;
    LocalTensor<uint8_t> localAllFalseMaskUint8_;
    LocalTensor<uint8_t> localCompareResult_;

    int32_t bs_ = 0;
    int32_t k_ = 0;
    int32_t moeExpertNum_ = 0;
    int32_t f_ = 0;
    uint32_t aivCoreNum_ = 0U;
    int64_t localRankId_ = 0;
    int64_t worldSize_ = 0;
    int32_t balanceMode_ = 0; // 负载均衡模式 (0: rank_id, 1: token_id)
    bool isActiveMask_ = 0;   // 0代表没有active_mask, 1代表传参active_mask
};

template <UPDATE_EXPERT_DECLARE>
__aicore__ inline void MoeUpdateExpert<UPDATE_EXPERT_ARGS>::Init(GM_ADDR expertIdsGM, GM_ADDR eplbTableGM,
                                                                 GM_ADDR expertScalesGM, GM_ADDR pruningThresholdGM,
                                                                 GM_ADDR activeMaskGM, GM_ADDR balancedExpertIdsOutGM,
                                                                 GM_ADDR balancedActiveMaskOutGM, GM_ADDR workspaceGM,
                                                                 const MoeUpdateExpertTilingData *tilingData,
                                                                 TPipe *tPipe)
{
    tPipe_ = tPipe;

    expertIdsGM_ = expertIdsGM;
    eplbTableGM_ = eplbTableGM;
    balancedExpertIdsOutGM_ = balancedExpertIdsOutGM;

    bs_ = tilingData->bs;
    k_ = tilingData->k;
    moeExpertNum_ = tilingData->moeExpertNum;
    f_ = tilingData->f;
    aivCoreNum_ = tilingData->aivCoreNum;
    balanceMode_ = tilingData->balanceMode;
    if (balanceMode_ == 0) {
        localRankId_ = tilingData->localRankId;
        worldSize_ = tilingData->worldSize;
    }

    expertIdsTensor_.SetGlobalBuffer((__gm__ DataType *)expertIdsGM_);
    balancedExpertIdsTensor_.SetGlobalBuffer((__gm__ DataType *)balancedExpertIdsOutGM_);
    eplbTableTensor_.SetGlobalBuffer((__gm__ int32_t *)eplbTableGM_);

    // 剪枝相关张量初始化
    if constexpr (EnablePruning) {
        int32_t newKCount =
            ((k_ * FLOAT_BYTES - 1 + ALIGN_256_BYTES) / ALIGN_256_BYTES * ALIGN_256_BYTES) / FLOAT_BYTES;
        tPipe_->InitBuffer(expertScalesQueue_, BUFFER_NUM, k_ * sizeof(ScalesDataType));
        tPipe_->InitBuffer(localScalesFp32Buf_, newKCount * sizeof(float));
        tPipe_->InitBuffer(localScaleSumBuf_, newKCount * sizeof(float));
        tPipe_->InitBuffer(localThresholdsBuf_, k_ * sizeof(float));
        tPipe_->InitBuffer(localThresholdProductsBuf_, newKCount * sizeof(float));
        tPipe_->InitBuffer(localActiveMaskBuf_, k_ * sizeof(half));
        tPipe_->InitBuffer(localOriginActiveMaskBuf_, bs_ * sizeof(uint8_t));
        tPipe_->InitBuffer(localSelectedActiveMaskHalfBuf_, k_ * sizeof(half));
        tPipe_->InitBuffer(resultMaskUint8Queue_, BUFFER_NUM, k_ * sizeof(uint8_t));
        tPipe_->InitBuffer(localAllFalseMaskUint8Buf_, k_ * sizeof(uint8_t));
        tPipe_->InitBuffer(localCompareResultBuf_, newKCount * sizeof(uint8_t));

        isActiveMask_ = bool(tilingData->isActiveMask);
        if (isActiveMask_ == 1) {
            activeMaskGM_ = activeMaskGM;
            activeMaskInTensor_.SetGlobalBuffer((__gm__ uint8_t *)activeMaskGM_);
        }
        expertScalesGM_ = expertScalesGM;
        pruningThresholdGM_ = pruningThresholdGM;
        balancedActiveMaskOutGM_ = balancedActiveMaskOutGM;
        expertScalesTensor_.SetGlobalBuffer((__gm__ ScalesDataType *)expertScalesGM_);
        preTuningThresholdTensor_.SetGlobalBuffer((__gm__ float *)pruningThresholdGM_);
        balancedActiveMaskTensor_.SetGlobalBuffer((__gm__ bool *)balancedActiveMaskOutGM_);
    }
}

template <UPDATE_EXPERT_DECLARE>
__aicore__ inline void MoeUpdateExpert<UPDATE_EXPERT_ARGS>::Calculate(int32_t start, int32_t end, int32_t computeLen)
{
    DataCopyExtParams dataCopyParams;
    dataCopyParams = {1, computeLen * sizeof(DataType), 0, 0, 0};
    for (int32_t i = start; i < end; i++) {
        DataCacheCleanAndInvalid<DataType, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(expertIdsTensor_[i]);
        DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(eplbTableTensor_[i]);
        DataType expertId = expertIdsTensor_.GetValue(i);
        DataType tableOffset = expertId * f_;
        DataType newExpertId = 0;
        int32_t tokenId = i / k_;

        if (eplbTableTensor_.GetValue(tableOffset) == 1) {
            newExpertId = static_cast<DataType>(eplbTableTensor_.GetValue(tableOffset + 1));
        } else {
            if (balanceMode_ == 0) { // rank_id 分发模式
                int64_t modeValue = Ceil(worldSize_, eplbTableTensor_.GetValue(tableOffset));
                int64_t rankId = localRankId_ / modeValue + 1;
                newExpertId = static_cast<DataType>(eplbTableTensor_.GetValue(tableOffset + rankId));
            } else { // token_id 分发模式
                int64_t expertCount = eplbTableTensor_.GetValue(tableOffset);
                int64_t selectedId = tokenId % expertCount + 1; // 使用 tokenId 取模
                newExpertId = eplbTableTensor_.GetValue(tableOffset + selectedId);
            }
        }
        outTensor_.SetValue(i - start, newExpertId);
    }
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopyPad(balancedExpertIdsTensor_[start], outTensor_, dataCopyParams);
}

template <UPDATE_EXPERT_DECLARE>
__aicore__ inline void MoeUpdateExpert<UPDATE_EXPERT_ARGS>::CalculateUnrollRest(int32_t start, int32_t loopNum,
                                                                                int32_t restNum)
{
    SyncFunc<AscendC::HardEvent::MTE3_S>();
    DataCacheCleanAndInvalid<DataType, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
        expertIdsTensor_[start + loopNum * UNROLL_NUM]);
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
        eplbTableTensor_[start + loopNum * UNROLL_NUM]);
    DataCopyExtParams unrollRestDataCopyParams;
    unrollRestDataCopyParams = {1, restNum * sizeof(DataType), 0, 0, 0};

    for (int32_t i = 0; i < restNum; i++) {
        DataType expertId = expertIdsTensor_.GetValue(start + loopNum * UNROLL_NUM + i);
        DataType tableOffset = expertId * f_;
        DataType newExpertId = 0;
        int32_t tokenId = (start + loopNum * static_cast<int32_t>(UNROLL_NUM) + i) / k_;

        if (eplbTableTensor_.GetValue(tableOffset) == 1) {
            newExpertId = static_cast<DataType>(eplbTableTensor_.GetValue(tableOffset + 1));
        } else {
            if (balanceMode_ == 0) { // rank_id 分发模式
                int64_t modeValue = Ceil(worldSize_, eplbTableTensor_.GetValue(tableOffset));
                int64_t rankId = localRankId_ / modeValue + 1;
                newExpertId = static_cast<DataType>(eplbTableTensor_.GetValue(tableOffset + rankId));
            } else { // token_id 分发模式
                int64_t expertCount = eplbTableTensor_.GetValue(tableOffset);
                int64_t selectedId = tokenId % expertCount + 1; // 使用 tokenId 取模
                newExpertId = eplbTableTensor_.GetValue(tableOffset + selectedId);
            }
        }
        outTensor_.SetValue(i, newExpertId);
    }
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopyPad(balancedExpertIdsTensor_[start + loopNum * UNROLL_NUM], outTensor_, unrollRestDataCopyParams);
}

template <UPDATE_EXPERT_DECLARE>
__aicore__ inline void MoeUpdateExpert<UPDATE_EXPERT_ARGS>::CalculateUnroll(int32_t start, int32_t computeLen)
{
    int32_t loopNum = computeLen / static_cast<int32_t>(UNROLL_NUM);
    int32_t restNum = computeLen % static_cast<int32_t>(UNROLL_NUM);
    DataCopyExtParams unrollDataCopyParams;
    unrollDataCopyParams = {1, UNROLL_NUM * sizeof(DataType), 0, 0, 0};

    for (int32_t i = 0; i < loopNum; i++) {
        if (i > 0) {
            SyncFunc<AscendC::HardEvent::MTE3_S>();
        }
        DataCacheCleanAndInvalid<DataType, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
            expertIdsTensor_[start + i * UNROLL_NUM]);
        DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
            eplbTableTensor_[start + i * UNROLL_NUM]);

        DataType expertId0 = expertIdsTensor_.GetValue(start + i * UNROLL_NUM);
        DataType expertId1 = expertIdsTensor_.GetValue(start + i * UNROLL_NUM + 1);
        DataType expertId2 = expertIdsTensor_.GetValue(start + i * UNROLL_NUM + 2);
        DataType expertId3 = expertIdsTensor_.GetValue(start + i * UNROLL_NUM + 3);

        DataType tableOffsetList[UNROLL_NUM] = {expertId0 * f_, expertId1 * f_, expertId2 * f_, expertId3 * f_};
        DataType newExpertIdList[UNROLL_NUM] = {0};
        for (int32_t j = 0; j < UNROLL_NUM; j++) {
            if (eplbTableTensor_.GetValue(tableOffsetList[j]) == 1) {
                newExpertIdList[j] = static_cast<DataType>(eplbTableTensor_.GetValue(tableOffsetList[j] + 1));
            } else {
                if (balanceMode_ == 0) { // rank_id 模式
                    int64_t modeValue = Ceil(worldSize_, eplbTableTensor_.GetValue(tableOffsetList[j]));
                    int64_t rankId = localRankId_ / modeValue + 1;
                    newExpertIdList[j] = static_cast<DataType>(eplbTableTensor_.GetValue(tableOffsetList[j] + rankId));
                } else {                                                                       // token_id 模式
                    int32_t tokenId = (start + i * static_cast<int32_t>(UNROLL_NUM) + j) / k_; // 计算tokenId
                    int64_t expertCount = eplbTableTensor_.GetValue(tableOffsetList[j]);
                    int64_t selectedId = tokenId % expertCount + 1;
                    newExpertIdList[j] =
                        static_cast<DataType>(eplbTableTensor_.GetValue(tableOffsetList[j] + selectedId));
                }
            }
            outTensor_.SetValue(j, newExpertIdList[j]);
        }
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        DataCopyPad(balancedExpertIdsTensor_[start + i * UNROLL_NUM], outTensor_, unrollDataCopyParams);
    }

    if (restNum != 0) {
        CalculateUnrollRest(start, loopNum, restNum);
    }
}

template <UPDATE_EXPERT_DECLARE>
__aicore__ inline void MoeUpdateExpert<UPDATE_EXPERT_ARGS>::InitPruning()
{
    localScalesFp32_ = localScalesFp32Buf_.Get<float>();
    localScaleSum_ = localScaleSumBuf_.Get<float>();
    localThresholds_ = localThresholdsBuf_.Get<float>();
    localThresholdProducts_ = localThresholdProductsBuf_.Get<float>();
    localActiveMask_ = localActiveMaskBuf_.Get<half>();
    localOriginActiveMask_ = localOriginActiveMaskBuf_.Get<uint8_t>();
    localSelectedActiveMaskHalf_ = localSelectedActiveMaskHalfBuf_.Get<half>();
    localAllFalseMaskUint8_ = localAllFalseMaskUint8Buf_.Get<uint8_t>();
    localCompareResult_ = localCompareResultBuf_.Get<uint8_t>();
}

template <UPDATE_EXPERT_DECLARE>
__aicore__ inline void MoeUpdateExpert<UPDATE_EXPERT_ARGS>::ProcessEachTokenPruning(int32_t tokenIdx)
{
    LocalTensor<ScalesDataType> localScales = expertScalesQueue_.AllocTensor<ScalesDataType>();
    DataCopyPadExtParams<ScalesDataType> padParamsScalesDataType{false, 0, 0, 0};
    DataCopyPad(localScales, expertScalesTensor_[tokenIdx * k_],
                {1, static_cast<uint32_t>(k_ * sizeof(ScalesDataType)), 0, 0, 0}, padParamsScalesDataType);
    expertScalesQueue_.EnQue(localScales);
    localScales = expertScalesQueue_.DeQue<ScalesDataType>();
    if constexpr (!std::is_same_v<ScalesDataType, float>) {
        Cast(localScalesFp32_, localScales, RoundMode::CAST_NONE, k_);
    } else {
        localScalesFp32_ = localScales;
    }

    uint32_t innerSumParams = (static_cast<uint32_t>(k_) * static_cast<uint32_t>(sizeof(float)) + UB_ALIGN - 1) /
                              UB_ALIGN * UB_ALIGN / static_cast<uint32_t>(sizeof(float));
    SumParams sumParams{1, innerSumParams, static_cast<uint32_t>(k_)};
    PipeBarrier<PIPE_V>();
    Sum(localScaleSum_, localScalesFp32_, sumParams);
    SyncFunc<AscendC::HardEvent::V_S>();
    float scaleSum = localScaleSum_.GetValue(0);
    PipeBarrier<PIPE_V>();
    // === 计算阈值乘积和生成比较掩码 ===
    Muls(localThresholdProducts_, localThresholds_, scaleSum, k_);
    PipeBarrier<PIPE_V>();
    int32_t newCalCount = ((k_ * FLOAT_BYTES - 1 + ALIGN_256_BYTES) / ALIGN_256_BYTES * ALIGN_256_BYTES) / FLOAT_BYTES;
    Compare(localCompareResult_, localScalesFp32_, localThresholdProducts_, CMPMODE::GE, newCalCount);
    expertScalesQueue_.FreeTensor(localScales);

    // === 生成最终激活掩码 ===
    Duplicate<half>(localActiveMask_, 1, k_);
    PipeBarrier<PIPE_V>();
    Select(localSelectedActiveMaskHalf_, localCompareResult_, localActiveMask_, static_cast<half>(0),
           SELMODE::VSEL_TENSOR_SCALAR_MODE, k_);
    PipeBarrier<PIPE_V>();
    LocalTensor<uint8_t> localresultMaskUint8 = resultMaskUint8Queue_.AllocTensor<uint8_t>();
    Cast(localresultMaskUint8, localSelectedActiveMaskHalf_, RoundMode::CAST_NONE, k_);
    resultMaskUint8Queue_.EnQue(localresultMaskUint8);
    localresultMaskUint8 = resultMaskUint8Queue_.DeQue<uint8_t>();
    LocalTensor<bool> localSelectedActiveMaskBool = localresultMaskUint8.template ReinterpretCast<bool>();
    // === 保存结果 ===
    DataCopyPad(balancedActiveMaskTensor_[tokenIdx * k_], localSelectedActiveMaskBool,
                {1, static_cast<uint32_t>(k_ * sizeof(bool)), 0, 0, 0});
    resultMaskUint8Queue_.FreeTensor(localresultMaskUint8);
}

template <UPDATE_EXPERT_DECLARE>
__aicore__ inline void MoeUpdateExpert<UPDATE_EXPERT_ARGS>::ProcessPruning(int32_t tokenStart, int32_t tokenEnd)
{
    InitPruning();
    DataCopyPadExtParams<float> padParamsFloat{false, 0, 0, 0};
    DataCopyPadExtParams<uint8_t> padParamsUint8{false, 0, 0, 0};
    // 预加载专家阈值（所有token共享）
    DataCopyPad(localThresholds_, preTuningThresholdTensor_, {1, static_cast<uint32_t>(k_ * sizeof(float)), 0, 0, 0},
                padParamsFloat);
    if (isActiveMask_ == 1) {
        DataCopyPad(localOriginActiveMask_, activeMaskInTensor_[tokenStart],
                    {1, static_cast<uint32_t>((tokenEnd - tokenStart) * sizeof(uint8_t)), 0, 0, 0}, padParamsUint8);
    }

    Duplicate<half>(localActiveMask_, 0, k_);
    PipeBarrier<PIPE_V>();
    Cast(localAllFalseMaskUint8_, localActiveMask_, RoundMode::CAST_NONE, k_);
    PipeBarrier<PIPE_V>();
    LocalTensor<bool> localAllFalseMaskBool = localAllFalseMaskUint8_.template ReinterpretCast<bool>();

    SyncFunc<AscendC::HardEvent::MTE2_V>();
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    for (int32_t tokenIdx = tokenStart; tokenIdx < tokenEnd; tokenIdx++) {
        bool active = true;
        if (isActiveMask_ == 1) {
            active = localOriginActiveMask_.GetValue(tokenIdx - tokenStart);
        }

        // 如果token不激活，直接设置全false掩码
        if (!active) {
            SyncFunc<AscendC::HardEvent::V_MTE3>();
            DataCopyPad(balancedActiveMaskTensor_[tokenIdx * k_], localAllFalseMaskBool,
                        {1, static_cast<uint32_t>(k_ * sizeof(bool)), 0, 0, 0});
            continue;
        }

        ProcessEachTokenPruning(tokenIdx);
    }
}

template <UPDATE_EXPERT_DECLARE>
__aicore__ inline void MoeUpdateExpert<UPDATE_EXPERT_ARGS>::GetCoreRange(int32_t coreIndex, int32_t coreNum,
                                                                         int32_t totalData, int32_t &start,
                                                                         int32_t &end)
{
    int32_t perCore = totalData / coreNum;
    int32_t remainder = totalData % coreNum;

    if (coreIndex < remainder) {
        start = (perCore + 1) * coreIndex;
        end = start + perCore + 1;
    } else {
        start = (perCore + 1) * remainder + perCore * (coreIndex - remainder);
        end = start + perCore;
    }
    end = end < totalData ? end : totalData;
};

template <UPDATE_EXPERT_DECLARE>
__aicore__ inline void MoeUpdateExpert<UPDATE_EXPERT_ARGS>::Process()
{
    // === 独立剪枝处理流程 ===
    int32_t aivIndex = GetBlockIdx();
    int32_t startIndex;
    int32_t endIndex;
    if constexpr (EnablePruning) {
        GetCoreRange(aivIndex, aivCoreNum_, bs_, startIndex, endIndex);
        ProcessPruning(startIndex, endIndex);
    }

    // === 主体计算分核逻辑 ===
    int32_t totalData = bs_ * k_;
    GetCoreRange(aivIndex, aivCoreNum_, totalData, startIndex, endIndex);
    int32_t computeLen = endIndex - startIndex;

    tPipe_->InitBuffer(buffer_, BUFFER_SIZE);
    outTensor_ = buffer_.Get<DataType>();

    if (computeLen >= UNROLL_NUM) {
        CalculateUnroll(startIndex, computeLen);
    } else {
        Calculate(startIndex, endIndex, computeLen);
    }
}
} // namespace MoeUpdateExpertNamespace
#endif
