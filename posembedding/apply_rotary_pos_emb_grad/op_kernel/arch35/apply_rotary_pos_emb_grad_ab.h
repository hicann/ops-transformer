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
 * \file apply_rotary_pos_emb_grad_ab.h
 * \brief
 *
 * 核间: BS 合并轴 × N 轴 二维分核 (blockIdx -> (blockDimBS, blockDimN))
 * 核内: 外层 BS 循环搬 cos/sin(Q/K 双路复用), 内层 N 循环搬 grad、算 dx
 *   - cosb>1: bs=b*s, 每行 N=nQ/nK
 *   - cosb==1: bs=s, B 合并进 N, 每行 N=b*nQ/b*nK (tiling 已把统一后的 totalN 填入 nQ/nK)
 */

#ifndef APPLY_ROTARY_POS_EMB_GRAD_AB_H
#define APPLY_ROTARY_POS_EMB_GRAD_AB_H

#include "apply_rotary_pos_emb_grad_common.h"
#include "apply_rotary_pos_emb_grad_tiling_data.h"

namespace ApplyRotaryPosEmbGrad {
using namespace AscendC;

template <typename T, bool kDcosFlag>
class ApplyRotaryPosEmbGradAB {
public:
    __aicore__ inline ApplyRotaryPosEmbGradAB(TPipe *pipe, const ApplyRopeGradRegbaseABParams *tiling)
        : pipe_(pipe),
          tilingData_(tiling)
    {}
    __aicore__ inline void Init(GM_ADDR gradQueryEmbed, GM_ADDR gradKeyEmbed, GM_ADDR cos, GM_ADDR sin, GM_ADDR query,
                                GM_ADDR key, GM_ADDR gradQueryOut, GM_ADDR gradKeyOut, GM_ADDR cosWs, GM_ADDR sinWs);
    __aicore__ inline void Process();

private:
    constexpr static int32_t bufferNum = 2;

    const ApplyRopeGradRegbaseABParams *tilingData_;
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

    int64_t dAlign_ = 0;
    int64_t partialDAlign_ = 0; // float partial 的 UB 行距
    int64_t ubFactorBS_ = 0;
    int64_t ubFactorN_ = 0;

private:
    __aicore__ inline void ProcessNLoop(int64_t bsAbsIdx, int64_t nStart, uint32_t currBS, int64_t nBlockCount);
    __aicore__ inline void ProcessN(const LocalTensor<T> &sinTensor, const LocalTensor<T> &cosTensor,
                                    GlobalTensor<T> &gradXGm, GlobalTensor<T> &gradXOutGm, GlobalTensor<T> &xGm,
                                    TQue<QuePosition::VECIN, bufferNum> &gradXInQue,
                                    TQue<QuePosition::VECOUT, bufferNum> &gradXOutQue,
                                    TQue<QuePosition::VECIN, bufferNum> &xInQue, int64_t gradGmOffset, int64_t currXN,
                                    uint32_t currBS, LocalTensor<float> &cosPartial, LocalTensor<float> &sinPartial,
                                    bool isAccumulate);
};

// =================================================================
// Init
// =================================================================
template <typename T, bool kDcosFlag>
__aicore__ inline void ApplyRotaryPosEmbGradAB<T, kDcosFlag>::Init(GM_ADDR gradQueryEmbed, GM_ADDR gradKeyEmbed,
                                                                   GM_ADDR cos, GM_ADDR sin, GM_ADDR query, GM_ADDR key,
                                                                   GM_ADDR gradQueryOut, GM_ADDR gradKeyOut,
                                                                   GM_ADDR cosWs, GM_ADDR sinWs)
{
    if (GetBlockIdx() >= tilingData_->usedCoreNum) {
        return;
    }

    dAlign_ = tilingData_->dAlign;
    ubFactorN_ = tilingData_->ubFactorN;
    ubFactorBS_ = tilingData_->ubFactorBS;

    gradQGm_.SetGlobalBuffer((__gm__ T *)gradQueryEmbed);
    gradKGm_.SetGlobalBuffer((__gm__ T *)gradKeyEmbed);
    cosGm_.SetGlobalBuffer((__gm__ T *)cos);
    sinGm_.SetGlobalBuffer((__gm__ T *)sin);
    queryGm_.SetGlobalBuffer((__gm__ T *)query);
    keyGm_.SetGlobalBuffer((__gm__ T *)key);
    gradQOutGm_.SetGlobalBuffer((__gm__ T *)gradQueryOut);
    gradKOutGm_.SetGlobalBuffer((__gm__ T *)gradKeyOut);
    cosWsGm_.SetGlobalBuffer((__gm__ float *)cosWs);
    sinWsGm_.SetGlobalBuffer((__gm__ float *)sinWs);

    gradQGm_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    gradKGm_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);

    // UB: cos/sin 各 ubFactorBS 行; grad/dx 各 ubFactorBS*ubFactorN 行 (双缓冲)
    pipe_->InitBuffer(cosInQue_, bufferNum, ubFactorBS_ * dAlign_ * sizeof(T));
    pipe_->InitBuffer(sinInQue_, bufferNum, ubFactorBS_ * dAlign_ * sizeof(T));
    pipe_->InitBuffer(gradQInQue_, bufferNum, ubFactorBS_ * ubFactorN_ * dAlign_ * sizeof(T));
    pipe_->InitBuffer(gradKInQue_, bufferNum, ubFactorBS_ * ubFactorN_ * dAlign_ * sizeof(T));
    pipe_->InitBuffer(gradQOutQue_, bufferNum, ubFactorBS_ * ubFactorN_ * dAlign_ * sizeof(T));
    pipe_->InitBuffer(gradKOutQue_, bufferNum, ubFactorBS_ * ubFactorN_ * dAlign_ * sizeof(T));

    if constexpr (kDcosFlag) {
        partialDAlign_ =
            Ops::Base::CeilAlign<int64_t>(tilingData_->d / HALF_COEF, BLOCK_TYPE_SIZE / sizeof(float)) * HALF_COEF;
        pipe_->InitBuffer(queryInQue_, bufferNum, ubFactorBS_ * ubFactorN_ * dAlign_ * sizeof(T));
        pipe_->InitBuffer(keyInQue_, bufferNum, ubFactorBS_ * ubFactorN_ * dAlign_ * sizeof(T));
        pipe_->InitBuffer(gradCosOutQue_, bufferNum, ubFactorBS_ * ubFactorN_ * partialDAlign_ * sizeof(float));
        pipe_->InitBuffer(gradSinOutQue_, bufferNum, ubFactorBS_ * ubFactorN_ * partialDAlign_ * sizeof(float));
    }
}

// =================================================================
// Process — 外层 BS 循环, 每段搬一次 cos/sin, 内层交给 ProcessNLoop
// =================================================================
template <typename T, bool kDcosFlag>
__aicore__ inline void ApplyRotaryPosEmbGradAB<T, kDcosFlag>::Process()
{
    if (GetBlockIdx() >= tilingData_->usedCoreNum) {
        return;
    }
    int64_t blockDimN = GetBlockIdx() % tilingData_->blockNumN;
    int64_t blockDimBS = GetBlockIdx() / tilingData_->blockNumN;
    int64_t bsBlockCount =
        (blockDimBS == tilingData_->blockNumBS - 1) ? tilingData_->blockTailBS : tilingData_->blockFactorBS;
    int64_t nBlockCount =
        (blockDimN == tilingData_->blockNumN - 1) ? tilingData_->blockTailN : tilingData_->blockFactorN;
    int64_t bsStart = blockDimBS * tilingData_->blockFactorBS;
    int64_t nStart = blockDimN * tilingData_->blockFactorN;
    uint32_t bsLoopCnt = Ops::Base::CeilDiv(bsBlockCount, ubFactorBS_);
    for (uint32_t bsLoop = 0; bsLoop < bsLoopCnt; bsLoop++) {
        uint32_t currBS = (bsLoop != bsLoopCnt - 1) ? static_cast<uint32_t>(ubFactorBS_) :
                                                      static_cast<uint32_t>(bsBlockCount - bsLoop * ubFactorBS_);
        int64_t bsAbsIdx = bsStart + bsLoop * ubFactorBS_;
        ProcessNLoop(bsAbsIdx, nStart, currBS, nBlockCount);
    }
}

// =================================================================
// ProcessNLoop — 搬 cos/sin (被 Q/K 复用), 内层 N 循环 tile-interleaved Q/K
// =================================================================
template <typename T, bool kDcosFlag>
__aicore__ inline void ApplyRotaryPosEmbGradAB<T, kDcosFlag>::ProcessNLoop(int64_t bsAbsIdx, int64_t nStart,
                                                                           uint32_t currBS, int64_t nBlockCount)
{
    int64_t d = tilingData_->d;
    int64_t dHalf = d / HALF_COEF;
    int64_t dHalfByteSize = dHalf * sizeof(T);

    // 搬入本 BS 段的 cos/sin (currBS 行 × d)
    LocalTensor<T> sinTensor = sinInQue_.AllocTensor<T>();
    LocalTensor<T> cosTensor = cosInQue_.AllocTensor<T>();
    DataCopyExtParams copyCosParams = {static_cast<uint16_t>(currBS * HALF_COEF), static_cast<uint32_t>(dHalfByteSize),
                                       0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(sinTensor, sinGm_[bsAbsIdx * d], copyCosParams, padParams);
    DataCopyPad(cosTensor, cosGm_[bsAbsIdx * d], copyCosParams, padParams);
    sinInQue_.EnQue(sinTensor);
    cosInQue_.EnQue(cosTensor);
    sinTensor = sinInQue_.DeQue<T>();
    cosTensor = cosInQue_.DeQue<T>();

    uint32_t nLoopCnt = Ops::Base::CeilDiv(nBlockCount, ubFactorN_);
    for (uint32_t nLoop = 0; nLoop < nLoopCnt; nLoop++) {
        int64_t absN = nStart + nLoop * ubFactorN_;
        // Q/K 各自按统一后的 totalN(nQ/nK) 裁尾
        int64_t currQN = (absN < tilingData_->nQ) ? min(ubFactorN_, tilingData_->nQ - absN) : 0;
        int64_t currKN = (absN < tilingData_->nK) ? min(ubFactorN_, tilingData_->nK - absN) : 0;
        if (currQN <= 0 && currKN <= 0) {
            break;
        }

        LocalTensor<float> cosPartial, sinPartial;
        if constexpr (kDcosFlag) {
            cosPartial = gradCosOutQue_.AllocTensor<float>();
            sinPartial = gradSinOutQue_.AllocTensor<float>();
        }

        // Q/K tile：先 SET head 数更多的一路（覆盖整个 cos/sin 部分积范围），
        // 再 ACCUMULATE 较小的一路，避免 nQ<nK 时 K 累加读到未初始化区域 (与 BAB 一致)
        if (currQN >= currKN) {
            if (currQN > 0) {
                int64_t qGmOffset = bsAbsIdx * tilingData_->nQ * d + absN * d;
                ProcessN(sinTensor, cosTensor, gradQGm_, gradQOutGm_, queryGm_, gradQInQue_, gradQOutQue_, queryInQue_,
                         qGmOffset, currQN, currBS, cosPartial, sinPartial, false);
            }
            if (currKN > 0) {
                int64_t kGmOffset = bsAbsIdx * tilingData_->nK * d + absN * d;
                ProcessN(sinTensor, cosTensor, gradKGm_, gradKOutGm_, keyGm_, gradKInQue_, gradKOutQue_, keyInQue_,
                         kGmOffset, currKN, currBS, cosPartial, sinPartial, true);
            }
        } else {
            if (currKN > 0) {
                int64_t kGmOffset = bsAbsIdx * tilingData_->nK * d + absN * d;
                ProcessN(sinTensor, cosTensor, gradKGm_, gradKOutGm_, keyGm_, gradKInQue_, gradKOutQue_, keyInQue_,
                         kGmOffset, currKN, currBS, cosPartial, sinPartial, false);
            }
            if (currQN > 0) {
                int64_t qGmOffset = bsAbsIdx * tilingData_->nQ * d + absN * d;
                ProcessN(sinTensor, cosTensor, gradQGm_, gradQOutGm_, queryGm_, gradQInQue_, gradQOutQue_, queryInQue_,
                         qGmOffset, currQN, currBS, cosPartial, sinPartial, true);
            }
        }

        // 写出 cos/sin 部分积到 workspace (Q+K 已在 UB 内累加), 供 Phase2 reduce
        if constexpr (kDcosFlag) {
            if (currQN > 0 || currKN > 0) {
                gradCosOutQue_.EnQue(cosPartial);
                gradSinOutQue_.EnQue(sinPartial);
                cosPartial = gradCosOutQue_.DeQue<float>();
                sinPartial = gradSinOutQue_.DeQue<float>();
                // 写回范围须覆盖 SET 路的完整区间 (nQ<nK 时 K 先 SET 覆盖 [0,currKN)), 与 BAB 一致
                uint32_t wsCurrN = static_cast<uint32_t>(max(currQN, currKN));
                // workspace 部分积偏移（按统一 maxN 线性，与 Phase2 reduce 对齐）
                int64_t maxN = max(tilingData_->nQ, tilingData_->nK);
                int64_t wsOffset = bsAbsIdx * maxN * d + absN * d;
                DataCopyExtParams copyWsParams = {static_cast<uint16_t>(currBS * wsCurrN * HALF_COEF),
                                                  static_cast<uint32_t>(dHalf * sizeof(float)), 0, 0, 0};
                DataCopyPad(cosWsGm_[wsOffset], cosPartial, copyWsParams);
                DataCopyPad(sinWsGm_[wsOffset], sinPartial, copyWsParams);
                gradCosOutQue_.FreeTensor(cosPartial);
                gradSinOutQue_.FreeTensor(sinPartial);
            }
        }
    }

    sinInQue_.FreeTensor(sinTensor);
    cosInQue_.FreeTensor(cosTensor);
}

// =================================================================
// ProcessN — 单 tile 单路径: CopyIn grad (+x) -> Compute -> CopyOut dx
//   isAccumulate: true=cosPartial/sinPartial 累加模式, false=SET 初始化模式
// =================================================================
template <typename T, bool kDcosFlag>
__aicore__ inline void ApplyRotaryPosEmbGradAB<T, kDcosFlag>::ProcessN(
    const LocalTensor<T> &sinTensor, const LocalTensor<T> &cosTensor, GlobalTensor<T> &gradXGm,
    GlobalTensor<T> &gradXOutGm, GlobalTensor<T> &xGm, TQue<QuePosition::VECIN, bufferNum> &gradXInQue,
    TQue<QuePosition::VECOUT, bufferNum> &gradXOutQue, TQue<QuePosition::VECIN, bufferNum> &xInQue,
    int64_t gradGmOffset, int64_t currXN, uint32_t currBS, LocalTensor<float> &cosPartial,
    LocalTensor<float> &sinPartial, bool isAccumulate)
{
    ProcessGradTile<T, kDcosFlag, bufferNum>(sinTensor, cosTensor, gradXGm, gradXOutGm, xGm, gradXInQue, gradXOutQue,
                                             xInQue, gradGmOffset, currXN, currBS, cosPartial, sinPartial, isAccumulate,
                                             tilingData_->d, dAlign_, partialDAlign_, ubFactorN_);
}
} // namespace ApplyRotaryPosEmbGrad
#endif // APPLY_ROTARY_POS_EMB_GRAD_AB_H
