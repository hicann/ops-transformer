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
 * \file apply_rotary_pos_emb_grad_bab.h
 * \brief
 *
 * 核间: B × S 二维网格分核
 * 核内: tile-interleaved Q/K 交叉处理, cos/sin 部分积 UB 内累加
 */

#ifndef APPLY_ROTARY_POS_EMB_GRAD_BAB_H
#define APPLY_ROTARY_POS_EMB_GRAD_BAB_H

#include "apply_rotary_pos_emb_grad_common.h"
#include "apply_rotary_pos_emb_grad_tiling_data.h"

namespace ApplyRotaryPosEmbGrad {
using namespace AscendC;

template <typename T>
class ApplyRotaryPosEmbGradBAB {
public:
    __aicore__ inline ApplyRotaryPosEmbGradBAB(TPipe *pipe, const ApplyRopeGradRegbaseParams *tiling)
        : pipe_(pipe),
          tilingData_(tiling)
    {}
    __aicore__ inline void Init(GM_ADDR gradQueryEmbed, GM_ADDR gradKeyEmbed, GM_ADDR cos, GM_ADDR sin, GM_ADDR query,
                                GM_ADDR key, GM_ADDR gradQueryOut, GM_ADDR gradKeyOut, GM_ADDR cosWs, GM_ADDR sinWs);
    __aicore__ inline void Process();

private:
    constexpr static int32_t bufferNum = 2;

    const ApplyRopeGradRegbaseParams *tilingData_;
    TPipe *pipe_;

    GlobalTensor<T> gradQGm_;
    GlobalTensor<T> gradKGm_;
    GlobalTensor<T> cosGm_;
    GlobalTensor<T> sinGm_;
    GlobalTensor<T> queryGm_;
    GlobalTensor<T> keyGm_;
    GlobalTensor<T> gradQOutGm_;
    GlobalTensor<T> gradKOutGm_;
    GlobalTensor<float> cosWsGm_;
    GlobalTensor<float> sinWsGm_;

    TQue<QuePosition::VECIN, bufferNum> gradQInQue_;
    TQue<QuePosition::VECIN, bufferNum> gradKInQue_;
    TQue<QuePosition::VECIN, bufferNum> cosInQue_;
    TQue<QuePosition::VECIN, bufferNum> sinInQue_;
    TQue<QuePosition::VECOUT, bufferNum> gradQOutQue_;
    TQue<QuePosition::VECOUT, bufferNum> gradKOutQue_;

    TQue<QuePosition::VECIN, bufferNum> queryInQue_;
    TQue<QuePosition::VECIN, bufferNum> keyInQue_;
    TQue<QuePosition::VECOUT, bufferNum> gradCosOutQue_;
    TQue<QuePosition::VECOUT, bufferNum> gradSinOutQue_;

    int64_t blockIdx_ = 0;
    int64_t dAlign_ = 0;
    int64_t partialDAlign_ = 0;
    int64_t dHalfSize_ = 0; // D/2 (元素个数)
    int64_t bIdx_ = 0;
    int64_t sIdx_ = 0;
    int64_t bNum_ = 0;
    int64_t sNum_ = 0;
    int64_t ubFactorS_ = 0;
    int64_t ubFactorN_ = 0;

private:
    __aicore__ inline void PrePareParams();
    __aicore__ inline void ProcessNLoop(uint32_t bIdx, uint32_t sIdx, uint32_t currSNum);
    __aicore__ inline void ProcessN(const LocalTensor<T> &sinTensor, const LocalTensor<T> &cosTensor, int64_t gmOffset,
                                    int64_t nPath, int64_t currXN, uint32_t currSNum, LocalTensor<float> &cosPartial,
                                    LocalTensor<float> &sinPartial, bool isQPath, bool isAccumulate);
};

// =================================================================
// Init
// =================================================================
template <typename T>
__aicore__ inline void ApplyRotaryPosEmbGradBAB<T>::Init(GM_ADDR gradQueryEmbed, GM_ADDR gradKeyEmbed, GM_ADDR cos,
                                                         GM_ADDR sin, GM_ADDR query, GM_ADDR key, GM_ADDR gradQueryOut,
                                                         GM_ADDR gradKeyOut, GM_ADDR cosWs, GM_ADDR sinWs)
{
    if (GetBlockIdx() >= tilingData_->usedCoreNum) {
        return;
    }
    this->blockIdx_ = GetBlockIdx();
    this->dHalfSize_ = tilingData_->d / HALF_COEF;
    this->dAlign_ = Ops::Base::CeilAlign<int64_t>(dHalfSize_, BLOCK_TYPE_SIZE / sizeof(T)) * HALF_COEF;
    this->partialDAlign_ = Ops::Base::CeilAlign<int64_t>(dHalfSize_, BLOCK_TYPE_SIZE / sizeof(float)) * HALF_COEF;
    ubFactorN_ = tilingData_->ubFactorN;
    ubFactorS_ = tilingData_->ubFactorS;

    this->gradQGm_.SetGlobalBuffer((__gm__ T *)gradQueryEmbed);
    this->gradKGm_.SetGlobalBuffer((__gm__ T *)gradKeyEmbed);
    this->cosGm_.SetGlobalBuffer((__gm__ T *)cos);
    this->sinGm_.SetGlobalBuffer((__gm__ T *)sin);
    this->queryGm_.SetGlobalBuffer((__gm__ T *)query);
    this->keyGm_.SetGlobalBuffer((__gm__ T *)key);
    this->gradQOutGm_.SetGlobalBuffer((__gm__ T *)gradQueryOut);
    this->gradKOutGm_.SetGlobalBuffer((__gm__ T *)gradKeyOut);
    this->cosWsGm_.SetGlobalBuffer((__gm__ float *)cosWs);
    this->sinWsGm_.SetGlobalBuffer((__gm__ float *)sinWs);

    gradQGm_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    gradKGm_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);

    this->pipe_->InitBuffer(cosInQue_, bufferNum, ubFactorS_ * dAlign_ * sizeof(T));
    this->pipe_->InitBuffer(sinInQue_, bufferNum, ubFactorS_ * dAlign_ * sizeof(T));
    this->pipe_->InitBuffer(gradQInQue_, bufferNum, ubFactorS_ * ubFactorN_ * dAlign_ * sizeof(T));
    this->pipe_->InitBuffer(gradKInQue_, bufferNum, ubFactorS_ * ubFactorN_ * dAlign_ * sizeof(T));
    this->pipe_->InitBuffer(gradQOutQue_, bufferNum, ubFactorS_ * ubFactorN_ * dAlign_ * sizeof(T));
    this->pipe_->InitBuffer(gradKOutQue_, bufferNum, ubFactorS_ * ubFactorN_ * dAlign_ * sizeof(T));

    if (tilingData_->dCosFlag) {
        this->pipe_->InitBuffer(queryInQue_, bufferNum, ubFactorS_ * ubFactorN_ * dAlign_ * sizeof(T));
        this->pipe_->InitBuffer(keyInQue_, bufferNum, ubFactorS_ * ubFactorN_ * dAlign_ * sizeof(T));
        this->pipe_->InitBuffer(gradCosOutQue_, bufferNum, ubFactorS_ * ubFactorN_ * partialDAlign_ * sizeof(float));
        this->pipe_->InitBuffer(gradSinOutQue_, bufferNum, ubFactorS_ * ubFactorN_ * partialDAlign_ * sizeof(float));
    }
}

// =================================================================
// PrePareParams
// =================================================================
template <typename T>
__aicore__ inline void ApplyRotaryPosEmbGradBAB<T>::PrePareParams()
{
    bIdx_ = blockIdx_ % tilingData_->blockNumB;
    sIdx_ = blockIdx_ / tilingData_->blockNumB;
    bNum_ = tilingData_->blockFactorB;
    sNum_ = tilingData_->blockFactorS;
    if (bIdx_ == tilingData_->blockNumB - 1 && tilingData_->b % tilingData_->blockFactorB != 0) {
        bNum_ = tilingData_->b % tilingData_->blockFactorB;
    }
    if (sIdx_ == tilingData_->blockNumS - 1 && tilingData_->s % tilingData_->blockFactorS != 0) {
        sNum_ = tilingData_->s % tilingData_->blockFactorS;
    }
}

// =================================================================
// Process
// =================================================================
template <typename T>
__aicore__ inline void ApplyRotaryPosEmbGradBAB<T>::Process()
{
    if (GetBlockIdx() >= tilingData_->usedCoreNum) {
        return;
    }
    PrePareParams();
    uint32_t bIdxStart = bIdx_ * tilingData_->blockFactorB;
    uint32_t sIdxStart = sIdx_ * tilingData_->blockFactorS;
    uint32_t sLoopCnt = Ops::Base::CeilDiv(sNum_, ubFactorS_);
    for (uint32_t bIdx = bIdxStart; bIdx < bIdxStart + bNum_; bIdx++) {
        for (uint32_t loopIdx = 0; loopIdx < sLoopCnt; loopIdx++) {
            uint32_t currSNum = (loopIdx != sLoopCnt - 1) ? ubFactorS_ : sNum_ - loopIdx * ubFactorS_;
            ProcessNLoop(bIdx, sIdxStart + loopIdx * ubFactorS_, currSNum);
        }
    }
}

// =================================================================
// ProcessNLoop — 搬 cos/sin, tile-interleaved Q/K
//
// 同一 tile 内 Q 算完 cosPartial→UB → K 直接在 UB 累加 → 写一次 GM
// =================================================================
template <typename T>
__aicore__ inline void ApplyRotaryPosEmbGradBAB<T>::ProcessNLoop(uint32_t bIdx, uint32_t sIdx, uint32_t currSNum)
{
    int64_t bSStride = bIdx * tilingData_->s + sIdx;
    int64_t maxN = max(tilingData_->nQ, tilingData_->nK);
    int64_t cosWsBase = bSStride * maxN * tilingData_->d;

    LocalTensor<T> sinTensor = sinInQue_.AllocTensor<T>();
    LocalTensor<T> cosTensor = cosInQue_.AllocTensor<T>();
    DataCopyExtParams copyCosParams = {static_cast<uint16_t>(currSNum * HALF_COEF),
                                       static_cast<uint32_t>(dHalfSize_ * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(sinTensor, sinGm_[sIdx * tilingData_->d], copyCosParams, padParams);
    DataCopyPad(cosTensor, cosGm_[sIdx * tilingData_->d], copyCosParams, padParams);
    sinInQue_.EnQue(sinTensor);
    cosInQue_.EnQue(cosTensor);
    sinTensor = sinInQue_.DeQue<T>();
    cosTensor = cosInQue_.DeQue<T>();

    for (uint32_t idxN = 0; idxN < tilingData_->ubLoopNumN; idxN++) {
        int64_t currQN = (idxN == tilingData_->ubLoopNumN - 1) ?
                             min(static_cast<int64_t>(tilingData_->ubTailFactorN),
                                 tilingData_->nQ - static_cast<int64_t>(idxN) * ubFactorN_) :
                             min(ubFactorN_, tilingData_->nQ - static_cast<int64_t>(idxN) * ubFactorN_);
        int64_t currKN = (idxN == tilingData_->ubLoopNumN - 1) ?
                             min(static_cast<int64_t>(tilingData_->ubTailFactorN),
                                 tilingData_->nK - static_cast<int64_t>(idxN) * ubFactorN_) :
                             min(ubFactorN_, tilingData_->nK - static_cast<int64_t>(idxN) * ubFactorN_);
        if (currQN <= 0 && currKN <= 0)
            break;

        LocalTensor<float> cosPartial, sinPartial;
        if (tilingData_->dCosFlag) {
            cosPartial = gradCosOutQue_.AllocTensor<float>();
            sinPartial = gradSinOutQue_.AllocTensor<float>();
        }

        // Q/K tile：先 SET head 数更多的一路（覆盖整个 cos/sin 部分积范围），
        // 再 ACCUMULATE 较小的一路，避免 nQ<nK 时 K 累加读到未初始化区域
        if (currQN >= currKN) {
            if (currQN > 0) {
                int64_t qBaseOffset = bSStride * tilingData_->nQ * tilingData_->d + idxN * ubFactorN_ * tilingData_->d;
                ProcessN(sinTensor, cosTensor, qBaseOffset, tilingData_->nQ, currQN, currSNum, cosPartial, sinPartial,
                         true, false);
            }
            if (currKN > 0) {
                int64_t kBaseOffset = bSStride * tilingData_->nK * tilingData_->d + idxN * ubFactorN_ * tilingData_->d;
                ProcessN(sinTensor, cosTensor, kBaseOffset, tilingData_->nK, currKN, currSNum, cosPartial, sinPartial,
                         false, true);
            }
        } else {
            if (currKN > 0) {
                int64_t kBaseOffset = bSStride * tilingData_->nK * tilingData_->d + idxN * ubFactorN_ * tilingData_->d;
                ProcessN(sinTensor, cosTensor, kBaseOffset, tilingData_->nK, currKN, currSNum, cosPartial, sinPartial,
                         false, false);
            }
            if (currQN > 0) {
                int64_t qBaseOffset = bSStride * tilingData_->nQ * tilingData_->d + idxN * ubFactorN_ * tilingData_->d;
                ProcessN(sinTensor, cosTensor, qBaseOffset, tilingData_->nQ, currQN, currSNum, cosPartial, sinPartial,
                         true, true);
            }
        }

        // Write cos/sin to workspace (Q+K already accumulated in UB)
        if (tilingData_->dCosFlag && (currQN > 0 || currKN > 0)) {
            gradCosOutQue_.EnQue(cosPartial);
            gradSinOutQue_.EnQue(sinPartial);
            cosPartial = gradCosOutQue_.DeQue<float>();
            sinPartial = gradSinOutQue_.DeQue<float>();
            int64_t wsOffset = cosWsBase + idxN * ubFactorN_ * tilingData_->d;
            uint32_t wsCurrN = static_cast<uint32_t>(max(currQN, currKN));

            DataCopyExtParams copyWsParams = {static_cast<uint16_t>(currSNum * wsCurrN * HALF_COEF),
                                              static_cast<uint32_t>(dHalfSize_ * sizeof(float)), 0, 0, 0};
            DataCopyPad(cosWsGm_[wsOffset], cosPartial, copyWsParams);
            DataCopyPad(sinWsGm_[wsOffset], sinPartial, copyWsParams);
            gradCosOutQue_.FreeTensor(cosPartial);
            gradSinOutQue_.FreeTensor(sinPartial);
        }
    }

    sinInQue_.FreeTensor(sinTensor);
    cosInQue_.FreeTensor(cosTensor);
}

// =================================================================
// ProcessN — 单 tile 处理, CopyIn → Compute → CopyOut
//
// isQPath:     true=Q 路, false=K 路 (选择 grad/x 对应 GM 与队列)
// isAccumulate: true=cosPartial/sinPartial 累加模式, false=SET 初始化模式
// =================================================================
template <typename T>
__aicore__ inline void ApplyRotaryPosEmbGradBAB<T>::ProcessN(
    const LocalTensor<T> &sinTensor, const LocalTensor<T> &cosTensor, int64_t gmOffset, int64_t nPath, int64_t currXN,
    uint32_t currSNum, LocalTensor<float> &cosPartial, LocalTensor<float> &sinPartial, bool isQPath, bool isAccumulate)
{
    auto &gradXGm = isQPath ? gradQGm_ : gradKGm_;
    auto &gradXOutGm = isQPath ? gradQOutGm_ : gradKOutGm_;
    auto &gradXInQue = isQPath ? gradQInQue_ : gradKInQue_;
    auto &gradXOutQue = isQPath ? gradQOutQue_ : gradKOutQue_;
    auto &xGm = isQPath ? queryGm_ : keyGm_;
    auto &xInQue = isQPath ? queryInQue_ : keyInQue_;

    if (tilingData_->dCosFlag) {
        ProcessGradTile<T, true, bufferNum>(sinTensor, cosTensor, gradXGm, gradXOutGm, xGm, gradXInQue, gradXOutQue,
                                            xInQue, gmOffset, currXN, currSNum, cosPartial, sinPartial, isAccumulate,
                                            tilingData_->d, dAlign_, partialDAlign_, ubFactorN_);
    } else {
        ProcessGradTile<T, false, bufferNum>(sinTensor, cosTensor, gradXGm, gradXOutGm, xGm, gradXInQue, gradXOutQue,
                                             xInQue, gmOffset, currXN, currSNum, cosPartial, sinPartial, isAccumulate,
                                             tilingData_->d, dAlign_, partialDAlign_, ubFactorN_);
    }
}
} // namespace ApplyRotaryPosEmbGrad
#endif // APPLY_ROTARY_POS_EMB_GRAD_BAB_H
