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
 * \file mhc_pre_backward_kernel_arch22.h
 * \brief MhcPreBackward kernel for arch22 (A2/ascend910b)
 *        Migrated from mhc_pre_sinkhorn_backward arch22, with Sinkhorn code removed
 *        and phi shape changed from N^2+2N to N!+2N
 *
 * \details 公式映射（对应 aclnnMhcPreBackward.md 计算公式）:
 *
 *  正向数据流:
 *    x_rs = x * gamma
 *    H_mix = x_rs @ phi^T                         → [B,S, N!+2N]
 *    H_mix_tmp = H_mix * inv_rms                  (RMSNorm)
 *    H_pre  = Sigmoid(α_pre  * H_mix_tmp[:N]  + bias_pre ) + eps
 *    H_post = Sigmoid(α_post * H_mix_tmp[N:2N] + bias_post) * 2
 *    H_res  = α_res * H_mix_tmp[2N:] + bias_res
 *    H_in   = sum_i( x[:,:,i,:] * H_pre[:,:,i] )
 *
 *  反向计算流程（函数 → 公式编号）:
 *    GetHcScaleAndHcBase()    → 加载 α=[α_pre, α_post, α_res] 和 bias
 *    ComputeGradPre()         → [1] 输出组合梯度: H_pre_grad = Reduce(H_in_grad ⊙ x, -1)
 *                               同时保存 x*grad_y 到 workspace 供 phi_grad 使用
 *    ComputeGradHHat2()       → [2] Sigmoid门控反向(H_pre):  H_pre_2_grad = H_pre_grad ⊙ s ⊙ (1-s)
 *                               [3] Sigmoid门控反向(H_post): H_post_2_grad = H_post_grad⊙(H_post⊙(1-H_post/2))
 *                               [4] 残差连接反向(H_res):     H_res_2_grad = H_res_grad · α_res
 *                               [5] RMSNorm Fusion反向:      H_mix_grad = H_mix_tmp_grad · inv_rms
 *                                                           inv_rms_grad = sum(H_mix_tmp_grad ⊙ H_mix)
 *                               [8] RMS归一化梯度:           x_rs_grad_inv = -(inv_rms_grad·inv_rms³/(N·D))·x_rs
 *    ComputeGradX1()          → [1] x_grad_vec3 = H_in_grad × H_pre
 *                               [8] x_grad_vec1 = Reshape(x_rs_grad_inv + x_rs_grad)
 *                               [6] x_grad_mm   = (H_mix_grad @ phi) * gamma  (AIC侧ProcessMatmul1)
 *                               最终: x_grad = x_grad_vec3 + x_grad_vec1 + x_grad_mm
 *    ComputeScaleBias()       → [2][3][4] bias_grad = sum(H_2_grad), α_grad = sum(H_2_grad ⊙ H_1)
 *    ProcessMatmul1() (AIC)   → [6] x_rs_grad = H_mix_grad @ phi
 *    ProcessMatmul2() (AIC)   → [6] phi_grad = G^T @ X
 *
 *  fusionSize_ = N! + 2N  (A2 使用排列数 N! 替代 N²)
 *  UB 布局: scaleLocal_ 按 block(8 elem) 组织: [pre(8), post(8), res_block0(8), res_block1(8), ...]
 */

#ifndef MHC_PRE_BACKWARD_OP_KERNEL_ARCH22_KERNEL_H
#define MHC_PRE_BACKWARD_OP_KERNEL_ARCH22_KERNEL_H

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "op_kernel/math_util.h"
#include "mhc_pre_backward_key_arch22.h"
#include "mhc_pre_backward_data_arch22.h"

using namespace AscendC;

namespace {
constexpr int32_t BYTE_SIZE_PER_BLOCK = 32;
constexpr int32_t ELEMENTS_SIZE_PER_BLOCK = BYTE_SIZE_PER_BLOCK / sizeof(float);
constexpr int32_t BYTE_SIZE_PER_REPEAT = 256;
constexpr int32_t ELEMENTS_SIZE_PER_REPEAT = 256 / sizeof(float);
constexpr int32_t REPEAT_LENTH = ELEMENTS_SIZE_PER_REPEAT;
constexpr int32_t BLOCK_PER_REPEAT = 8;
constexpr uint64_t MASK_POST_SCALE[] = {0b0000000000000000000000000000000000000000000000000000000011110000};
constexpr int32_t PING_PONG_NUM = 2;
constexpr int32_t PRE_POST_NUM = 2;
constexpr int32_t DOUBLE_RATIO = 2;

constexpr int32_t INNER_SPILT_NUM = 8;

constexpr MatmulConfig MHC_PRE_GRAD_MM1_CFG = GetMDLConfig(true, false, 0, false, false, false, false);
constexpr MatmulConfig MHC_PRE_GRAD_MM2_CFG = GetMDLConfig(true, false, 0, false, false, false, false);

template <typename T>
__aicore__ inline void kahanCustom(LocalTensor<T> &inputTensor, LocalTensor<T> sumTensorList[2], const int32_t len,
                                   int32_t &outPos)
{
    LocalTensor<T> sumTensor = sumTensorList[outPos];
    LocalTensor<T> eTensor = sumTensorList[1 - outPos];
    PipeBarrier<PIPE_V>();
    Sub(inputTensor, inputTensor, eTensor, len);
    PipeBarrier<PIPE_V>();
    Add(eTensor, inputTensor, sumTensor, len);
    PipeBarrier<PIPE_V>();
    Sub(sumTensor, eTensor, sumTensor, len);
    PipeBarrier<PIPE_V>();
    Sub(sumTensor, sumTensor, inputTensor, len);
    PipeBarrier<PIPE_V>();
    outPos = 1 - outPos;
}
} // namespace

template <typename TYPE_X, typename T, bool DETERMINISTIC>
class MhcPreBackwardKernelArch22 {
public:
    using A0Type = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>;
    using A1Type = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T, true>;
    using BType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>;
    using CType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>;

    matmul::MatmulImpl<A0Type, BType, CType, CType, MHC_PRE_GRAD_MM1_CFG> mm1_;
    matmul::MatmulImpl<A1Type, BType, CType, CType, MHC_PRE_GRAD_MM2_CFG> mm2_;

    __aicore__ inline MhcPreBackwardKernelArch22() = default;

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR phi, GM_ADDR h_pre, GM_ADDR h_post, GM_ADDR grad_h_in,
                                GM_ADDR grad_h_post, GM_ADDR grad_h_res, GM_ADDR alpha, GM_ADDR h_mix, GM_ADDR inv_rms,
                                GM_ADDR grad_x, GM_ADDR grad_phi, GM_ADDR grad_alpha, GM_ADDR grad_bias,
                                GM_ADDR workspace, const MhcPreBackwardArch22TilingData *tilingData, TPipe *pipe,
                                GM_ADDR gamma, GM_ADDR grad_x_post_optional, GM_ADDR grad_gamma)
    {
        pipe_ = pipe;
        blkIdx_ = GetBlockIdx();

        InitTiling(tilingData);
        InitGM(x, phi, h_pre, h_post, grad_h_in, grad_h_post, grad_h_res, alpha, h_mix, inv_rms, grad_x, grad_phi,
               grad_alpha, grad_bias, workspace, gamma, grad_x_post_optional, grad_gamma);
        InitGradPreStageBuffer();
    }
    __aicore__ inline void Process();

protected:
    int64_t blkIdx_ = 0, aivNum_ = 0, aicNum_ = 0;
    TPipe *pipe_;
    int64_t batchSize_ = 0, seqLength_ = 0, totalTasks_ = 0, totalTasksAligned_ = 0, BSN = 0;
    int64_t c_ = 0, n_ = 0, factN_ = 0, fusionSize_ = 0, isBSNN_ = 0;
    int64_t c0_ = 0, c1_ = 0, cTail_ = 0, c0RepeatTime_ = 0, cTailAlign_ = 0, cTailBlockStride_ = 0, c1Align_ = 0;
    int64_t tileCoreBS_ = 0;
    int64_t ubSize_ = 0;
    int64_t mm1K_ = 0, mm1M_ = 0, mm1N_ = 0;
    int64_t mm2K_ = 0, mm2M_ = 0, mm2N_ = 0;
    int64_t tileRepeatTimes_ = 0;
    float eps_ = 1e-6f;
    int64_t needAdd = 0;
    event_t eventIdVToMTE3XCast;
    event_t eventIdMTE2ToV;
    event_t eventIdVToMTE2;

    T hcScalePre_, hcScalePost_, hcScaleRes_;

    TQue<QuePosition::VECIN, 1> inputXInQueue;
    TQue<QuePosition::VECIN, 1> inputGradQueue;
    TQue<QuePosition::VECOUT, 1> OutQueue;

    TBuf<TPosition::VECCALC> fusedGradHPre2AndGradHPost2Buf_, gradRsqrtBuf_, gradBiasBuf_, onesBuf_, ScaleBuf_,
        hcBaseBuf_, tempBuf_;

    LocalTensor<T> dBiasLocal_, gradRsqrtLocal_, dPrePostTempLocal_;
    LocalTensor<T> xCastLocal_, gradYCastLocal_, gradXCastLocal_;
    LocalTensor<T> scaleLocal_, dScaleLocal_, dBiasLocalTemp_, dScaleLocalTemp_;
    LocalTensor<T> dBiasLocalList_[2], dpreLocalList_[2];
    int32_t scalePos_ = 0, biasPos_ = 0;
    int32_t onceTask_ = 0;
    LocalTensor<T> gradHResLocal_;
    LocalTensor<T> hcBaseLocal_;
    LocalTensor<T> preBrcbLocal_, dRsqrtBrcbLocal_, rsqrtbrcbLocal_, tmpLocal_, hat2Scale, dhatBeforeNormLocal,
        dhatLocal_, rsqrtTempLocal_, gradXCubeLocal_;
    LocalTensor<T> hatLocal;
    LocalTensor<T> onesLocal_;
    LocalTensor<T> gammaLocal_;

    bool withGamma_ = false;
    bool withGradXPost_ = false;
    int64_t gammaUseCoreNum_ = 0;

    GlobalTensor<T> gammaGlobal_;
    GlobalTensor<T> gradGammaGlobal_;
    GlobalTensor<T> gradGammaWSGlobal_;
    GlobalTensor<TYPE_X> gradXPostGlobal_;

    GlobalTensor<TYPE_X> xGlobal_, gradYGlobal_;
    GlobalTensor<T> preGlobal_;
    GlobalTensor<TYPE_X> gradXGlobal_;
    GlobalTensor<T> gradPostGlobal_;
    GlobalTensor<T> hcScaleGlobal_;
    GlobalTensor<T> hPostGlobal_;
    GlobalTensor<T> rsqrtGlobal_;
    GlobalTensor<T> gradHcScaleGlobal_, gradHcBaseGlobal_;
    GlobalTensor<T> h2Global_;
    GlobalTensor<T> gradHResGlobal_;
    GlobalTensor<T> gradH2Global_;
    GlobalTensor<T> gradWeightGlobal_;
    GlobalTensor<T> weightGlobal_;
    GlobalTensor<T> gradHcBaseWSGlobal_;
    GlobalTensor<T> gradHcScaleWSGlobal_;
    GlobalTensor<T> gradWeightWSGlobal_;
    GlobalTensor<T> xWorkspaceGlobal_;
    GlobalTensor<T> gradXCubeGlobal_;

private:
    __aicore__ inline int64_t Factorial(int64_t n)
    {
        if (n <= 1) {
            return 1;
        }
        int64_t result = 1;
        for (int64_t i = 2; i <= n; i++) {
            result *= i;
        }
        return result;
    }

    __aicore__ inline void InitTiling(const MhcPreBackwardArch22TilingData *tilingData)
    {
        batchSize_ = tilingData->batchSize;
        seqLength_ = tilingData->seqLength;
        aivNum_ = tilingData->aivNum;
        aicNum_ = tilingData->aivNum / DOUBLE_RATIO;
        c_ = tilingData->c;
        n_ = tilingData->n;
        c0_ = tilingData->c0;
        c1_ = tilingData->c1;
        BSN = batchSize_ * seqLength_ * n_;
        c1Align_ = Ops::Base::CeilDiv(c_, c0_);
        cTail_ = max((c_ - (c1Align_ - 1) * c0_), static_cast<int64_t>(0));

        cTailAlign_ = AlignUp(cTail_, ELEMENTS_SIZE_PER_BLOCK);
        cTailBlockStride_ = c0_ / ELEMENTS_SIZE_PER_BLOCK - cTailAlign_ / ELEMENTS_SIZE_PER_BLOCK;
        ubSize_ = tilingData->ubSize;
        eps_ = tilingData->eps;
        tileCoreBS_ = tilingData->tileSize;

        // 从 tiling data 获取 fusionSize, 反推 factN (支持 N!+2N 和 N²+2N)
        fusionSize_ = tilingData->fusionSize;
        factN_ = fusionSize_ - PRE_POST_NUM * n_;
        isBSNN_ = tilingData->isBSNN;
        withGamma_ = (tilingData->withGamma != 0);
        withGradXPost_ = (tilingData->withGradXPost != 0);

        c0RepeatTime_ = c0_ / ELEMENTS_SIZE_PER_REPEAT;
        totalTasks_ = batchSize_ * seqLength_;
        totalTasksAligned_ = AlignUp(totalTasks_, aivNum_ * tileCoreBS_);
        gammaUseCoreNum_ = Ops::Base::CeilDiv(n_ * c_, aivNum_ * tileCoreBS_);
        if ASCEND_IS_AIC {
            mm1K_ = fusionSize_;
            mm1M_ = tileCoreBS_ * 2;
            mm1N_ = n_ * c_;

            mm2K_ = batchSize_ * seqLength_;
            mm2M_ = fusionSize_;
            mm2N_ = n_ * c_;
        }
    }

    __aicore__ inline void InitGM(GM_ADDR x, GM_ADDR phi, GM_ADDR h_pre, GM_ADDR h_post, GM_ADDR grad_h_in,
                                  GM_ADDR grad_h_post, GM_ADDR grad_h_res, GM_ADDR alpha, GM_ADDR h_mix,
                                  GM_ADDR inv_rms, GM_ADDR grad_x, GM_ADDR grad_phi, GM_ADDR grad_alpha,
                                  GM_ADDR grad_bias, GM_ADDR workspace, GM_ADDR gamma, GM_ADDR grad_x_post_optional,
                                  GM_ADDR grad_gamma)
    {
        xGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ TYPE_X *>(x));
        gradWeightGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(grad_phi));
        if (withGamma_) {
            gammaGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(gamma));
            gradGammaGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(grad_gamma));
        }
        if (withGradXPost_) {
            gradXPostGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ TYPE_X *>(grad_x_post_optional));
        }
        int64_t workspaceOffset = 0;
        gradH2Global_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(workspace) + workspaceOffset);
        workspaceOffset += batchSize_ * seqLength_ * fusionSize_;
        xWorkspaceGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(workspace) + workspaceOffset);
        workspaceOffset += batchSize_ * seqLength_ * (n_ * c_);
        gradXCubeGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(workspace) + workspaceOffset);
        workspaceOffset += batchSize_ * seqLength_ * (n_ * c_);
        if constexpr (DETERMINISTIC == true) {
            gradWeightWSGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(workspace) + workspaceOffset);
            workspaceOffset += aicNum_ * fusionSize_ * (n_ * c_);
        }

        weightGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(phi));
        gradXGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ TYPE_X *>(grad_x));

        if ASCEND_IS_AIV {
            preGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(h_pre));
            hPostGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(h_post));
            gradYGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ TYPE_X *>(grad_h_in));

            gradPostGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(grad_h_post));

            hcScaleGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(alpha));

            rsqrtGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(inv_rms));
            gradHcScaleGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(grad_alpha));
            gradHcBaseGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(grad_bias));
            if constexpr (DETERMINISTIC == true) {
                gradHcScaleWSGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(workspace) + workspaceOffset);
                workspaceOffset += aivNum_ * (ELEMENTS_SIZE_PER_BLOCK);
                gradHcBaseWSGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(workspace) + workspaceOffset);
                workspaceOffset += aivNum_ * fusionSize_;
                if (withGamma_) {
                    gradGammaWSGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(workspace) + workspaceOffset);
                    workspaceOffset += aivNum_ * (n_ * c_);
                }
            }
            h2Global_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(h_mix));

            gradHResGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(grad_h_res));
            if constexpr (DETERMINISTIC == false) {
                if (blkIdx_ == aivNum_ - 1) {
                    InitOutput<T>(gradHcBaseGlobal_, fusionSize_, 0);
                    InitOutput<T>(gradHcScaleGlobal_, 3, 0);
                }
                for (int64_t taskOffset = blkIdx_ * tileCoreBS_; taskOffset < n_ * c_;
                     taskOffset += aivNum_ * tileCoreBS_) {
                    int32_t tileTaskCount =
                        min(static_cast<int32_t>(tileCoreBS_), static_cast<int32_t>(n_ * c_ - taskOffset));
                    InitOutput<T>(gradWeightGlobal_[taskOffset * fusionSize_], fusionSize_ * tileTaskCount, 0);
                }
                SyncAll<true>();
            }
        }
    }

    __aicore__ inline void InitGradPreStageBuffer()
    {
        if ASCEND_IS_AIV {
            pipe_->InitBuffer(fusedGradHPre2AndGradHPost2Buf_, tileCoreBS_ * n_ * 2 * sizeof(float));
            pipe_->InitBuffer(gradRsqrtBuf_, tileCoreBS_ * fusionSize_ * sizeof(float) * 2);
            pipe_->InitBuffer(gradBiasBuf_, 2 * 2 * tileCoreBS_ * fusionSize_ * sizeof(float));
            pipe_->InitBuffer(onesBuf_, tileCoreBS_ * n_ * 2 * sizeof(float));
            pipe_->InitBuffer(hcBaseBuf_, fusionSize_ * sizeof(float));
            pipe_->InitBuffer(ScaleBuf_, BYTE_SIZE_PER_BLOCK * 2);
            pipe_->InitBuffer(inputXInQueue, 2, tileCoreBS_ * n_ * c0_ * sizeof(float) / 4);
            pipe_->InitBuffer(inputGradQueue, 1, tileCoreBS_ * n_ * ELEMENTS_SIZE_PER_BLOCK * sizeof(float));
            pipe_->InitBuffer(OutQueue, 2, tileCoreBS_ * n_ * c0_ * sizeof(float) / 8);

            onceTask_ = tileCoreBS_ / INNER_SPILT_NUM;
            // tempBuf_ 两阶段复用 (offset=0 重置), 取较大值
            // 阶段1 (ComputeGradX1): preBrcbLocal_ + dRsqrtBrcbLocal_ + gradYCastLocal_ + gradXCastLocal_ + xCastLocal_
            auto ubSizeStage1 = Ops::Base::CeilDiv(tileCoreBS_, static_cast<int64_t>(ELEMENTS_SIZE_PER_BLOCK)) *
                                    ELEMENTS_SIZE_PER_REPEAT * sizeof(float) +
                                Ops::Base::CeilDiv(tileCoreBS_ * n_, static_cast<int64_t>(ELEMENTS_SIZE_PER_BLOCK)) *
                                    ELEMENTS_SIZE_PER_REPEAT * sizeof(float) +
                                onceTask_ * n_ * c0_ * sizeof(float) * 2 + onceTask_ * c0_ * sizeof(float);
            // 阶段2 (ComputeGradPre): dhatLocal_ + rsqrtbrcbLocal_ + hat2Scale + dhatBeforeNormLocal + hatLocal +
            auto brcbAlignS2 = Ops::Base::CeilDiv(tileCoreBS_, static_cast<int64_t>(ELEMENTS_SIZE_PER_BLOCK));
            auto ubSizeStage2 = tileCoreBS_ * fusionSize_ * sizeof(float) * 2 +
                                brcbAlignS2 * ELEMENTS_SIZE_PER_BLOCK * fusionSize_ * sizeof(float) +
                                tileCoreBS_ * fusionSize_ * sizeof(float) * 4;
            auto ubSizeRemain = ubSizeStage1 > ubSizeStage2 ? ubSizeStage1 : ubSizeStage2;
            if (withGamma_) {
                ubSizeRemain += n_ * c0_ * sizeof(float);
            }

            pipe_->InitBuffer(tempBuf_, ubSizeRemain);
            int64_t totalSize = tileCoreBS_ * n_ * 2 * sizeof(float) + tileCoreBS_ * fusionSize_ * sizeof(float) * 2 +
                                2 * 2 * tileCoreBS_ * fusionSize_ * sizeof(float) +
                                tileCoreBS_ * n_ * 2 * sizeof(float) + fusionSize_ * sizeof(float) +
                                BYTE_SIZE_PER_BLOCK * 2 + tileCoreBS_ * n_ * c0_ * sizeof(float) / 4 * 2 +
                                tileCoreBS_ * n_ * ELEMENTS_SIZE_PER_BLOCK * sizeof(float) +
                                tileCoreBS_ * n_ * c0_ * sizeof(float) / 8 * 2 + ubSizeRemain;
            dPrePostTempLocal_ = fusedGradHPre2AndGradHPost2Buf_.Get<T>();
            hcBaseLocal_ = hcBaseBuf_.Get<T>();
            onesLocal_ = onesBuf_.Get<T>();
            scaleLocal_ = ScaleBuf_.Get<T>();
            gradRsqrtLocal_ = gradRsqrtBuf_.Get<T>();
            dBiasLocal_ = gradBiasBuf_.Get<T>();
            dScaleLocal_ = dBiasLocal_[tileCoreBS_ * fusionSize_];
            dBiasLocalTemp_ = dScaleLocal_[tileCoreBS_ * fusionSize_];
            dScaleLocalTemp_ = dBiasLocalTemp_[tileCoreBS_ * fusionSize_];
            dBiasLocalList_[0] = dBiasLocal_;
            dBiasLocalList_[1] = dBiasLocalTemp_;

            int32_t offset = 0;
            int32_t brcbAlign = static_cast<int32_t>(
                Ops::Base::CeilDiv(tileCoreBS_ * n_, static_cast<int64_t>(ELEMENTS_SIZE_PER_BLOCK)));
            preBrcbLocal_ = tempBuf_.GetWithOffset<T>(brcbAlign * ELEMENTS_SIZE_PER_REPEAT, offset);
            offset += brcbAlign * ELEMENTS_SIZE_PER_REPEAT * sizeof(float);
            brcbAlign =
                static_cast<int32_t>(Ops::Base::CeilDiv(tileCoreBS_, static_cast<int64_t>(ELEMENTS_SIZE_PER_BLOCK)));
            dRsqrtBrcbLocal_ = tempBuf_.GetWithOffset<T>(brcbAlign * ELEMENTS_SIZE_PER_REPEAT, offset);
            offset += brcbAlign * ELEMENTS_SIZE_PER_REPEAT * sizeof(float);
            gradYCastLocal_ = tempBuf_.GetWithOffset<T>(onceTask_ * c0_, offset);
            offset += onceTask_ * c0_ * sizeof(float);
            gradXCastLocal_ = tempBuf_.GetWithOffset<T>(onceTask_ * n_ * c0_, offset);
            offset += onceTask_ * n_ * c0_ * sizeof(float);
            xCastLocal_ = tempBuf_.GetWithOffset<T>(onceTask_ * n_ * c0_, offset);
            if (withGamma_) {
                offset += onceTask_ * n_ * c0_ * sizeof(float);
                gammaLocal_ = tempBuf_.GetWithOffset<T>(n_ * c0_, offset);
            }
            offset = 0;
            tmpLocal_ = gradXCastLocal_;
            dpreLocalList_[0] = tmpLocal_;
            dpreLocalList_[1] = tmpLocal_[onceTask_ * n_ * ELEMENTS_SIZE_PER_REPEAT];

            offset = 0;
            // dhatLocal_ and related buffers use fusionSize_ (= N! + 2N)
            dhatLocal_ = tempBuf_.GetWithOffset<T>(tileCoreBS_ * fusionSize_ * 2, offset);
            offset += tileCoreBS_ * fusionSize_ * sizeof(float) * 2;
            brcbAlign =
                static_cast<int32_t>(Ops::Base::CeilDiv(tileCoreBS_, static_cast<int64_t>(ELEMENTS_SIZE_PER_BLOCK)));
            rsqrtbrcbLocal_ = tempBuf_.GetWithOffset<T>(brcbAlign * ELEMENTS_SIZE_PER_BLOCK * fusionSize_, offset);
            offset += brcbAlign * ELEMENTS_SIZE_PER_BLOCK * fusionSize_ * sizeof(float);
            hat2Scale = tempBuf_.GetWithOffset<T>(tileCoreBS_ * fusionSize_, offset);
            offset += tileCoreBS_ * fusionSize_ * sizeof(float);
            dhatBeforeNormLocal = tempBuf_.GetWithOffset<T>(tileCoreBS_ * fusionSize_, offset);
            offset += tileCoreBS_ * fusionSize_ * sizeof(float);
            hatLocal = tempBuf_.GetWithOffset<T>(tileCoreBS_ * fusionSize_, offset);
            offset += tileCoreBS_ * fusionSize_ * sizeof(float);
            rsqrtTempLocal_ = tempBuf_.GetWithOffset<T>(tileCoreBS_ * fusionSize_, offset);

            Duplicate(onesLocal_, 1.f, tileCoreBS_ * n_ * 2);
            Duplicate(dBiasLocal_, 0.f, 2 * 2 * tileCoreBS_ * fusionSize_);
        }
    }

    __aicore__ inline void ComputeGradPre(const int64_t taskOffset, const int32_t tileTaskCount, const int32_t innerId);
    __aicore__ inline void ComputeGradHHat2(const int64_t taskOffset, const int32_t tileTaskCount);
    __aicore__ inline void ComputeGradX1(const int64_t taskOffset, const int32_t tileTaskCount, const int32_t innerId);
    __aicore__ inline void GetHcScaleAndHcBase();
    __aicore__ inline void ProcessMatmul1(const int64_t taskOffset, const int32_t mm1M);
    __aicore__ inline void ProcessMatmul2(const int64_t taskOffset, const int32_t mm2K);
    __aicore__ inline void ComputeScaleBias();
    __aicore__ inline void ComputeDeterministic(GlobalTensor<float> &inputGm, GlobalTensor<float> &outputGm,
                                                const int64_t dimR, const int64_t dimA);
};

template <typename TYPE_X, typename T, bool DETERMINISTIC>
__aicore__ inline void MhcPreBackwardKernelArch22<TYPE_X, T, DETERMINISTIC>::GetHcScaleAndHcBase()
{
    // 加载 alpha = [α_pre, α_post, α_res]，对应公式中的缩放参数
    // hcScaleGlobal_ 即输入 alpha，shape=(3)
    hcScalePre_ = hcScaleGlobal_.GetValue(0);  // α_pre
    hcScalePost_ = hcScaleGlobal_.GetValue(1); // α_post
    hcScaleRes_ = hcScaleGlobal_.GetValue(2);  // α_res

    event_t eventIDSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIDSToV);
    WaitFlag<HardEvent::S_V>(eventIDSToV);

    // 构建 scaleLocal_: 将 α 值按 fusionSize_ 的 block 布局展开
    // UB 布局: [pre+post(8), res_block0(8), res_block1(8), ...] fusionSize=32
    //   block 0  (offset 0):  [α_pre(N), α_post(N)] packed → 正向 Sigmoid(α_pre*H_pre_1+bias_pre),
    //   Sigmoid(α_post*H_post_1+bias_post)*2 block 1..(offset 8):  α_res   → 正向 α_res * H_res_1 + bias_res
    // pre pre pre pre post post post post res res res res res res res res
    Duplicate(scaleLocal_[8], hcScaleRes_, 8); // res blocks: α_res
    Duplicate(scaleLocal_, hcScalePost_, 8);   // post block: α_post
    PipeBarrier<PIPE_V>();

    Duplicate(scaleLocal_, hcScalePre_, 4); // pre block: α_pre (N=4)
}

template <typename TYPE_X, typename T, bool DETERMINISTIC>
__aicore__ inline void MhcPreBackwardKernelArch22<TYPE_X, T, DETERMINISTIC>::ComputeGradHHat2(
    const int64_t taskOffset, const int32_t tileTaskCount)
{
    // ============================================================================
    // 本函数实现公式 [4][3][2][5][8] 的反向计算，核心流程:
    //   1. [公式4] 残差连接反向(H_res): 加载 grad_h_res，直接作为 H_res_2_grad (无 sigmoid)
    //   2. 加载 grad_h_post, h_mix, inv_rms
    //   3. 合并 H_pre_grad 和 H_post_grad (post 乘 2，因为正向 H_post = sigmoid(...)*2)
    //   4. 重建 sigmoid 输入: α * H_mix * inv_rms + bias
    //   5. [公式2][3] Sigmoid 门控反向: s*(1-s) * grad → H_pre_2_grad, H_post_2_grad
    //   6. [公式5] RMSNorm Fusion 反向: H_mix_grad = H_mix_tmp_grad * inv_rms
    //                                   inv_rms_grad = sum(H_mix_tmp_grad ⊙ H_mix)
    //   7. [公式8] RMS归一化梯度: x_rs_grad_inv = -(inv_rms_grad * inv_rms³ / (N*D))
    //   8. 输出 H_mix_grad 供 AIC 侧 matmul 使用
    // ============================================================================

    // --- [公式4] 残差连接反向(H_res) ---
    // 正向: H_res = α_res * H_res_1 + bias_res
    // 反向: H_res_2_grad = H_res_grad (α_res 乘法在后续 scaleLocal_ 中体现)
    // grad_h_res 支持 BSNN(B,S,N,N) 和 BSN!(B,S,N!) 两种布局
    gradHResLocal_ = inputGradQueue.AllocTensor<T>();
    // BSNN(B,S,N,N) 和 BSN!(B,S,N!) 均使用扁平加载: factN_ = N²(BSNN) 或 N!(BSN!)
    DataCopyPad(gradHResLocal_, gradHResGlobal_[taskOffset * factN_],
                {static_cast<uint16_t>(1), static_cast<uint32_t>(tileTaskCount * factN_ * sizeof(T)), 0, 0, 0},
                {false, 0, 0, 0});
    inputGradQueue.EnQue(gradHResLocal_);
    inputGradQueue.DeQue();

    // 将 grad_h_res 拷贝到 dhatLocal_ 的 res 区段 (从 block 1 开始)
    // dhatLocal_ 布局: [pre+post(8), res_block0(8), res_block1(8), ...] = fusionSize_ 元素
    int32_t resBlocks = static_cast<int32_t>(factN_ / ELEMENTS_SIZE_PER_BLOCK);
    for (int32_t loopIdN = 0; loopIdN < resBlocks; loopIdN += 1) {
        Copy(dhatLocal_[(1 + loopIdN) * ELEMENTS_SIZE_PER_BLOCK], gradHResLocal_[loopIdN * ELEMENTS_SIZE_PER_BLOCK],
             ELEMENTS_SIZE_PER_REPEAT, tileRepeatTimes_,
             {static_cast<uint16_t>(fusionSize_ / ELEMENTS_SIZE_PER_BLOCK), static_cast<uint16_t>(resBlocks),
              static_cast<uint16_t>(fusionSize_), static_cast<uint16_t>(resBlocks * 8)});
    }

    inputGradQueue.FreeTensor(gradHResLocal_);

    // --- 加载 grad_h_post, h_mix, inv_rms ---
    auto gradHPostLocal_ = inputXInQueue.AllocTensor<T>();
    auto hat2LocalTemp = gradHPostLocal_[tileCoreBS_ * ELEMENTS_SIZE_PER_BLOCK];                           // h_mix
    auto rsqrtLocal_ = gradHPostLocal_[tileCoreBS_ * ELEMENTS_SIZE_PER_BLOCK + tileCoreBS_ * fusionSize_]; // inv_rms

    // grad_h_post → H_post_grad，shape [tileTaskCount, N]
    DataCopyPad(gradHPostLocal_, gradPostGlobal_[taskOffset * n_],
                {static_cast<uint16_t>(tileTaskCount), static_cast<uint32_t>(n_ * sizeof(T)), 0, 0, 0},
                {true, static_cast<uint8_t>(ELEMENTS_SIZE_PER_BLOCK - n_), 0, 0});

    // h_mix → H_mix (phi^T @ x_rs 的结果)，shape [tileTaskCount, fusionSize_]
    DataCopyPad(hat2LocalTemp, h2Global_[taskOffset * fusionSize_],
                {static_cast<uint16_t>(tileTaskCount), static_cast<uint32_t>(fusionSize_ * sizeof(T)), 0, 0, 0},
                {false, 0, 0, 0});
    // inv_rms → inv_rms，shape [tileTaskCount]
    DataCopyPad(rsqrtLocal_, rsqrtGlobal_[taskOffset],
                {static_cast<uint16_t>(1), static_cast<uint32_t>(tileTaskCount * sizeof(T)), 0, 0, 0},
                {false, 0, 0, 0});
    inputXInQueue.EnQue(gradHPostLocal_);
    inputXInQueue.DeQue();

    PipeBarrier<PIPE_V>();

    // --- 合并 H_pre_grad 和 H_post_grad ---
    // dPrePostTempLocal_ 已包含 H_pre_grad (来自 ComputeGradPre)
    // Axpy: dPrePostTempLocal_ += gradHPost * 2
    //   H_post 正向有 *2，故反向梯度乘 2: H_post_grad_effective = H_post_grad * 2
    // 合并后 dPrePostTempLocal_ = [H_pre_grad, H_post_grad * 2]，shape [tileTaskCount, 2N]
    Axpy(dPrePostTempLocal_, gradHPostLocal_, float(2), tileTaskCount * 2 * n_);

    // 广播 inv_rms 到 fusionSize_ 维: rsqrtbrcbLocal_ [tileTaskCount, fusionSize_]
    const uint32_t srcShape[2] = {static_cast<uint32_t>(tileTaskCount), 1};
    const uint32_t dstShape[2] = {static_cast<uint32_t>(tileTaskCount), static_cast<uint32_t>(fusionSize_)};
    PipeBarrier<PIPE_V>();

    AscendC::Broadcast<float, 2, 1>(rsqrtbrcbLocal_, rsqrtLocal_, dstShape, srcShape);

    PipeBarrier<PIPE_V>();
    // --- 重建 sigmoid 输入: α * H_mix * inv_rms + bias ---
    // hat2Scale = α * H_mix  (scaleLocal_ = [α_pre, α_post, α_res, ...], hat2LocalTemp = H_mix)
    // 对应正向: α_pre * H_pre_1, α_post * H_post_1, α_res * H_res_1
    // [适配] 循环处理所有 block: block 0 用 scaleLocal_(pre+post), block 1+ 用 scaleLocal_[8](res)
    int32_t totalBlocks = static_cast<int32_t>(fusionSize_ / ELEMENTS_SIZE_PER_BLOCK);
    for (int32_t blk = 0; blk < totalBlocks; blk++) {
        auto scaleSrc = (blk == 0) ? scaleLocal_ : scaleLocal_[ELEMENTS_SIZE_PER_BLOCK];
        Mul(hat2Scale[blk * ELEMENTS_SIZE_PER_BLOCK], scaleSrc, hat2LocalTemp[blk * ELEMENTS_SIZE_PER_BLOCK],
            ELEMENTS_SIZE_PER_REPEAT, tileRepeatTimes_,
            {static_cast<uint8_t>(totalBlocks), 0, static_cast<uint8_t>(totalBlocks), static_cast<uint8_t>(fusionSize_),
             0, static_cast<uint8_t>(fusionSize_)});
    }
    PipeBarrier<PIPE_V>();
    eventIdVToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID<HardEvent::V_MTE2>());
    SetFlag<HardEvent::V_MTE2>(eventIdVToMTE2);
    WaitFlag<HardEvent::V_MTE2>(eventIdVToMTE2);
    // 新增: 加载正向输出 h_pre 、 h_post和反向的grad_h_res
    DataCopyPad(hatLocal, preGlobal_[taskOffset * n_],
                {static_cast<uint16_t>(tileTaskCount), static_cast<uint32_t>(n_ * sizeof(T)), 0, 0, 0},
                {true, 0, static_cast<uint8_t>(ELEMENTS_SIZE_PER_BLOCK - n_), eps_});

    DataCopyPad(hatLocal[ELEMENTS_SIZE_PER_BLOCK * tileTaskCount], hPostGlobal_[taskOffset * n_],
                {static_cast<uint16_t>(tileTaskCount), static_cast<uint32_t>(n_ * sizeof(T)), 0, 0, 0},
                {true, static_cast<uint8_t>(ELEMENTS_SIZE_PER_BLOCK - n_), 0, 0});

    eventIdMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID<HardEvent::MTE2_V>());
    SetFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);

    // 计算真正的 sigmoid 输出 s
    // pre 区段:  s = H_pre - eps  (正向 H_pre = sigmoid(x) + eps)
    // post 区段: s = H_post / 2   (正向 H_post = 2 * sigmoid(x))
    Adds(hatLocal, hatLocal, float(-eps_), ELEMENTS_SIZE_PER_BLOCK * tileTaskCount);
    Muls(hatLocal[ELEMENTS_SIZE_PER_BLOCK * tileTaskCount], hatLocal[ELEMENTS_SIZE_PER_BLOCK * tileTaskCount],
         float(0.5), ELEMENTS_SIZE_PER_BLOCK * tileTaskCount);
    PipeBarrier<PIPE_V>();
    Add(hatLocal, hatLocal, hatLocal[ELEMENTS_SIZE_PER_BLOCK * tileTaskCount], ELEMENTS_SIZE_PER_BLOCK * tileTaskCount);
    PipeBarrier<PIPE_V>();

    auto hatLocalTemp = dhatBeforeNormLocal;

    // Sigmoid 导数: s * (1 - s)，其中 s = sigmoid(hatLocal)
    // hatLocalTemp = s - s² = s * (1 - s)
    Mul(hatLocalTemp, hatLocal, hatLocal, ELEMENTS_SIZE_PER_REPEAT, tileRepeatTimes_, {1, 1, 1, 8, 8, 8});
    PipeBarrier<PIPE_V>();
    Sub(hatLocalTemp, hatLocal, hatLocalTemp, ELEMENTS_SIZE_PER_REPEAT, tileRepeatTimes_, {1, 1, 1, 8, 8, 8});
    PipeBarrier<PIPE_V>();

    // --- [公式2][3] H_pre_2_grad = H_pre_grad ⊙ s ⊙ (1-s), H_post_2_grad = H_post_grad ⊙ s ⊙ (1-s) ---
    // dhatLocal_ = dPrePostTempLocal_ * hatLocalTemp
    //   pre 区段:  H_pre_2_grad  = H_pre_grad  * s * (1-s)
    //   post 区段: H_post_2_grad = H_post_grad * 2 * s * (1-s)  (已含 *2)
    //   res 区段:  保持 grad_h_res 不变 (res 无 sigmoid)
    Mul(dhatLocal_, dPrePostTempLocal_, hatLocalTemp, ELEMENTS_SIZE_PER_REPEAT, tileRepeatTimes_,
        {static_cast<uint8_t>(fusionSize_ / ELEMENTS_SIZE_PER_BLOCK), 1, 1, static_cast<uint8_t>(fusionSize_), 8, 8});
    PipeBarrier<PIPE_V>();

    // --- [公式5] RMSNorm Fusion 反向 ---
    // inv_rms_grad = sum_last_dim(H_mix_tmp_grad ⊙ H_mix)
    // H_mix_tmp_grad = [H_pre_1_grad, H_post_1_grad, H_res_1_grad]
    //                 = [H_pre_2_grad * α_pre, H_post_2_grad * α_post, H_res_2_grad * α_res]
    // 注意: hat2Scale = α * H_mix, dhatLocal_ = H_2_grad (含 sigmoid 导数)
    // 所以 gradRsqrtLocal_ = H_2_grad * (α * H_mix) = H_1_grad * H_mix = H_mix_tmp_grad * H_mix
    Mul(gradRsqrtLocal_, dhatLocal_, hat2Scale, fusionSize_ * tileTaskCount);
    PipeBarrier<PIPE_V>();
    // inv_rms_grad = sum(gradRsqrtLocal_, dim=fusionSize_)
    // srcRepStride = fusionSize_/8 (datablock 单位), 参考 sinkhorn 算子一次调用
    WholeReduceSum(rsqrtTempLocal_, gradRsqrtLocal_, fusionSize_, tileTaskCount, 1, 1,
                   static_cast<int32_t>(fusionSize_ / ELEMENTS_SIZE_PER_BLOCK));
    PipeBarrier<PIPE_V>();

    // --- [公式8] RMS归一化梯度计算 ---
    // x_rs_grad_inv = -(inv_rms_grad * inv_rms³ / (N*D)) * x_rs
    // 此处先计算 gradRsqrtLocal_ = -(inv_rms_grad * inv_rms³) / (N*D)
    // rsqrtTempLocal_ = inv_rms_grad * inv_rms
    Mul(rsqrtTempLocal_, rsqrtTempLocal_, rsqrtLocal_, tileTaskCount);
    PipeBarrier<PIPE_V>();

    // rsqrtLocal_ = inv_rms²
    Mul(rsqrtLocal_, rsqrtLocal_, rsqrtLocal_, tileTaskCount);
    PipeBarrier<PIPE_V>();
    // rsqrtTempLocal_ = inv_rms_grad * inv_rms³
    Mul(rsqrtTempLocal_, rsqrtTempLocal_, rsqrtLocal_, tileTaskCount);

    PipeBarrier<PIPE_V>();
    // gradRsqrtLocal_ = -(inv_rms_grad * inv_rms³) / (N*D)
    // 后续在 ComputeGradX1 中: x_rs_grad_inv = gradRsqrtLocal_ * x_rs (通过 MulAddDst)
    Muls(gradRsqrtLocal_, rsqrtTempLocal_, float(-1) / (n_ * c_), tileTaskCount);

    // --- [公式5] H_mix_grad = H_mix_tmp_grad * inv_rms ---
    // dhatBeforeNormLocal = inv_rms * H_2_grad = inv_rms * H_pre_2_grad (不含 α)
    // 后续 dhat2Local = α * dhatBeforeNormLocal = α * inv_rms * H_2_grad = inv_rms * H_1_grad = H_mix_grad
    Mul(dhatBeforeNormLocal, rsqrtbrcbLocal_, dhatLocal_, fusionSize_ * tileTaskCount);
    PipeBarrier<PIPE_V>();

    // --- 累加 α_grad 和 bias_grad 的中间结果 ---
    // α_grad = sum(H_2_grad * H_1)，其中 H_1 = H_mix (before norm)
    // dhatLocal_[tileCoreBS_*fusionSize_] = hat2LocalTemp * dhatBeforeNormLocal
    //   = H_mix * (inv_rms * H_2_grad) = H_mix * H_mix_tmp_grad (不含 α)
    //   后续在 ComputeScaleBias 中乘以 α 得到 α_grad
    Duplicate(dhatLocal_[tileTaskCount * fusionSize_], 0.f, (tileCoreBS_ - tileTaskCount) * fusionSize_);
    Duplicate(dhatLocal_[(tileCoreBS_ + tileTaskCount) * fusionSize_], 0.f,
              (tileCoreBS_ - tileTaskCount) * fusionSize_);
    Mul(dhatLocal_[tileCoreBS_ * fusionSize_], hat2LocalTemp, dhatBeforeNormLocal, fusionSize_ * tileTaskCount);
    PipeBarrier<PIPE_V>();

    // Kahan 累加跨 tile 的 α_grad / bias_grad 中间结果
    kahanCustom(dhatLocal_, dBiasLocalList_, tileCoreBS_ * fusionSize_ * 2, scalePos_);

    auto dhat2Local = OutQueue.AllocTensor<float>();
    inputXInQueue.FreeTensor(gradHPostLocal_);

    // --- 输出 H_mix_grad 供 AIC 侧 matmul 使用 ---
    // dhat2Local = α * dhatBeforeNormLocal = α * inv_rms * H_2_grad = inv_rms * H_1_grad = H_mix_grad
    // H_mix_grad 将被 AIC 侧 ProcessMatmul1 用于: x_rs_grad = H_mix_grad @ phi

    // [适配] 循环处理所有 block: block 0 用 scaleLocal_(pre+post), block 1+ 用 scaleLocal_[8](res)
    {
        int32_t totalBlocks = static_cast<int32_t>(fusionSize_ / ELEMENTS_SIZE_PER_BLOCK);
        for (int32_t blk = 0; blk < totalBlocks; blk++) {
            auto scaleSrc = (blk == 0) ? scaleLocal_ : scaleLocal_[ELEMENTS_SIZE_PER_BLOCK];
            Mul(dhat2Local[blk * ELEMENTS_SIZE_PER_BLOCK], scaleSrc, dhatBeforeNormLocal[blk * ELEMENTS_SIZE_PER_BLOCK],
                ELEMENTS_SIZE_PER_REPEAT, tileRepeatTimes_,
                {static_cast<uint8_t>(totalBlocks), 0, static_cast<uint8_t>(totalBlocks),
                 static_cast<uint8_t>(fusionSize_), 0, static_cast<uint8_t>(fusionSize_)});
        }
    }

    OutQueue.EnQue(dhat2Local);
    dhat2Local = OutQueue.DeQue<T>();
    // 写入 gradH2Global_ (workspace)，AIC 核心读取此数据做 matmul
    DataCopyPad(
        gradH2Global_[taskOffset * fusionSize_], dhat2Local,
        {static_cast<uint16_t>(1), static_cast<uint32_t>(tileTaskCount * fusionSize_ * sizeof(float)), 0, 0, 0});

    OutQueue.FreeTensor(dhat2Local);
}

template <typename TYPE_X, typename T, bool DETERMINISTIC>
__aicore__ inline void MhcPreBackwardKernelArch22<TYPE_X, T, DETERMINISTIC>::ComputeGradX1(const int64_t taskOffset,
                                                                                           const int32_t tileTaskCount,
                                                                                           const int32_t innerId)
{
    // ============================================================================
    // 本函数实现最终 x_grad 的组装，对应公式:
    //   [公式1] x_grad_vec3 = H_in_grad × H_pre         (输出组合梯度反向)
    //   [公式8] x_grad_vec1 = x_rs_grad_inv = -(inv_rms_grad·inv_rms³/(N·D))·x_rs  (RMS归一化梯度)
    //   [公式6] x_grad_mm   = (H_mix_grad @ phi) * gamma  (矩阵乘法反向，AIC侧ProcessMatmul1)
    //   最终: x_grad = x_grad_vec3 + x_grad_vec1 + x_grad_mm
    //   注意: gamma_grad 和 grad_x_post 融合在 host 侧或其它路径处理
    // ============================================================================

    PipeBarrier<PIPE_V>();

    for (int32_t loopIdC = 0; loopIdC < c1Align_; loopIdC += 1) {
        // 按 c0_ 分块遍历 D 维度
        int64_t copyLen = c0_;
        bool isPad = false;
        uint8_t padLen = 0;
        int64_t ubAlignC = c0_;
        if (loopIdC == c1_) {
            isPad = true;
            copyLen = cTail_;
            ubAlignC = cTailAlign_;
            padLen = static_cast<uint8_t>(cTailAlign_ - cTail_);
        }
        auto xLocal_ = inputXInQueue.AllocTensor<TYPE_X>();
        auto gradXCubeLocal_ =
            xLocal_.template ReinterpretCast<float>()[onceTask_ * n_ * c0_ / 2]; // x_grad_mm (AIC结果)
        auto gradYLocal_ = xLocal_[onceTask_ * n_ * c0_ * 3];                    // grad_h_in

        // 加载 grad_h_in (H_in_grad)，shape [tileTaskCount, copyLen]
        DataCopyPad(gradYLocal_, gradYGlobal_[taskOffset * c_ + c0_ * loopIdC],
                    {static_cast<uint16_t>(tileTaskCount), static_cast<uint32_t>(copyLen * sizeof(TYPE_X)),
                     static_cast<uint32_t>((c_ - copyLen) * sizeof(TYPE_X)), 0, 0},
                    {isPad, 0, padLen, 0});
        // 加载 x，shape [n_*tileTaskCount, copyLen]
        DataCopyPad(xLocal_, xGlobal_[taskOffset * n_ * c_ + c0_ * loopIdC],
                    {static_cast<uint16_t>(n_ * tileTaskCount), static_cast<uint32_t>(copyLen * sizeof(TYPE_X)),
                     static_cast<uint32_t>((c_ - copyLen) * sizeof(TYPE_X)), 0, 0},
                    {isPad, 0, padLen, 0});
        // 加载 x_grad_mm (来自 AIC 侧 ProcessMatmul1 的 matmul 结果)，shape [n_*tileTaskCount, copyLen]
        DataCopyPad(gradXCubeLocal_, gradXCubeGlobal_[taskOffset * n_ * c_ + c0_ * loopIdC],
                    {static_cast<uint16_t>(n_ * tileTaskCount), static_cast<uint32_t>(copyLen * sizeof(T)),
                     static_cast<uint32_t>((c_ - copyLen) * sizeof(T)), 0, 0},
                    {isPad, 0, padLen, 0});

        inputXInQueue.EnQue(xLocal_);
        inputXInQueue.DeQue();
        PipeBarrier<PIPE_V>();
        // Cast 到 fp32
        Cast(gradYCastLocal_, gradYLocal_, RoundMode::CAST_NONE, ubAlignC * tileTaskCount);
        Cast(xCastLocal_, xLocal_, RoundMode::CAST_NONE, ubAlignC * n_ * tileTaskCount);

        PipeBarrier<PIPE_V>();
        uint8_t blkStride1 = static_cast<uint8_t>(ubAlignC / ELEMENTS_SIZE_PER_BLOCK);
        uint8_t blkStride2 = static_cast<uint8_t>(n_ * ubAlignC / ELEMENTS_SIZE_PER_BLOCK);

        // --- [公式1] x_grad_vec3 = H_in_grad × H_pre ---
        // preBrcbLocal_ 已在 Process() 中从 h_pre 广播得到
        // gradXCastLocal_[n] = grad_y * H_pre[n]  (对每个 batch, 每个维度 D)
        for (int32_t loopIdN = 0; loopIdN < n_; loopIdN += 1) {
            for (int32_t loopOffsetC0 = 0; loopOffsetC0 < copyLen; loopOffsetC0 += ELEMENTS_SIZE_PER_REPEAT) {
                uint64_t mask =
                    min(static_cast<uint64_t>(ELEMENTS_SIZE_PER_REPEAT), static_cast<uint64_t>(copyLen - loopOffsetC0));
                Mul(gradXCastLocal_[loopIdN * ubAlignC + loopOffsetC0], gradYCastLocal_[loopOffsetC0],
                    preBrcbLocal_[loopIdN * ELEMENTS_SIZE_PER_BLOCK + innerId * n_ * ELEMENTS_SIZE_PER_BLOCK], mask,
                    tileTaskCount, {1, 1, 0, blkStride2, blkStride1, static_cast<uint8_t>(n_)});
            }
        }

        PipeBarrier<PIPE_V>();
        // --- [公式8] x_grad_vec1 = x_rs_grad_inv = -(inv_rms_grad·inv_rms³/(N·D))·x_rs ---
        // dRsqrtBrcbLocal_ 已在 Process() 中从 gradRsqrtLocal_ 广播得到
        // dRsqrtBrcbLocal_ = -(inv_rms_grad * inv_rms³) / (N*D)  (来自 ComputeGradHHat2)
        // MulAddDst: gradXCastLocal_ += x * dRsqrtBrcbLocal_
        //   即 x_grad = x_grad_vec3 + x * (-(inv_rms_grad·inv_rms³/(N·D)))
        //             = x_grad_vec3 + x_rs_grad_inv
        for (int32_t loopIdN = 0; loopIdN < n_; loopIdN += 1) {
            for (int32_t loopOffsetC0 = 0; loopOffsetC0 < copyLen; loopOffsetC0 += ELEMENTS_SIZE_PER_REPEAT) {
                uint64_t mask =
                    min(static_cast<uint64_t>(ELEMENTS_SIZE_PER_REPEAT), static_cast<uint64_t>(copyLen - loopOffsetC0));
                MulAddDst(gradXCastLocal_[loopIdN * ubAlignC + loopOffsetC0],
                          xCastLocal_[loopIdN * ubAlignC + loopOffsetC0],
                          dRsqrtBrcbLocal_[innerId * ELEMENTS_SIZE_PER_BLOCK], mask, tileTaskCount,
                          {1, 1, 0, blkStride2, blkStride2, 1});
            }
        }

        PipeBarrier<PIPE_V>();

        // --- [gamma] gradGamma 计算 + gamma 乘法 ---
        if (withGamma_) {
            // 加载 gamma
            DataCopyPad(gammaLocal_, gammaGlobal_[c0_ * loopIdC],
                        {static_cast<uint16_t>(n_), static_cast<uint32_t>(copyLen * sizeof(float)),
                         static_cast<uint32_t>((c_ - copyLen) * sizeof(float)), 0, 0},
                        {isPad, 0, padLen, 0});
            eventIdMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID<HardEvent::MTE2_V>());
            SetFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);

            // gradGamma = sum_BS(x * x_grad_mm), 复用 xCastLocal_
            Mul(xCastLocal_, xCastLocal_, gradXCubeLocal_, ubAlignC * n_ * tileTaskCount);
            PipeBarrier<PIPE_V>();

            // 跨 tileTaskCount 归约 (BS 维度)
            for (int32_t bs = 1; bs < tileTaskCount; bs++) {
                Add(xCastLocal_, xCastLocal_, xCastLocal_[bs * n_ * ubAlignC], n_ * ubAlignC);
                PipeBarrier<PIPE_V>();
            }
            eventIdVToMTE3XCast = static_cast<event_t>(GetTPipePtr()->FetchEventID<HardEvent::V_MTE3>());
            SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3XCast);
            WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3XCast);
            // V→MTE3 同步，写回 gradGammaWSGlobal_
            // 第一次外层迭代的第一次内层调用直接写入，后续使用 AtomicAdd 累加

            if (innerId == 0 && taskOffset == blkIdx_ * tileCoreBS_) {
                DataCopyPad(gradGammaWSGlobal_[blkIdx_ * (n_ * c_) + c0_ * loopIdC], xCastLocal_,
                            {static_cast<uint16_t>(n_), static_cast<uint32_t>(copyLen * sizeof(float)), 0,
                             static_cast<uint32_t>((c_ - copyLen) * sizeof(float)), 0});
            } else {
                SetAtomicAdd<T>();
                DataCopyPad(gradGammaWSGlobal_[blkIdx_ * (n_ * c_) + c0_ * loopIdC], xCastLocal_,
                            {static_cast<uint16_t>(n_), static_cast<uint32_t>(copyLen * sizeof(float)), 0,
                             static_cast<uint32_t>((c_ - copyLen) * sizeof(float)), 0});
                SetAtomicNone();
            }
            PipeBarrier<PIPE_V>();

            // gradXCubeLocal_ *= gamma (x_grad_mm *= gamma)
            for (int32_t loopIdN = 0; loopIdN < n_; loopIdN += 1) {
                for (int32_t loopOffsetC0 = 0; loopOffsetC0 < copyLen; loopOffsetC0 += ELEMENTS_SIZE_PER_REPEAT) {
                    uint64_t mask = min(static_cast<uint64_t>(ELEMENTS_SIZE_PER_REPEAT),
                                        static_cast<uint64_t>(copyLen - loopOffsetC0));
                    Mul(gradXCubeLocal_[loopIdN * ubAlignC + loopOffsetC0],
                        gradXCubeLocal_[loopIdN * ubAlignC + loopOffsetC0],
                        gammaLocal_[loopIdN * ubAlignC + loopOffsetC0], mask, tileTaskCount,
                        {1, 1, 1, blkStride2, blkStride2, 0});
                }
            }
            PipeBarrier<PIPE_V>();
            auto eventIdMTE3toV = static_cast<event_t>(GetTPipePtr()->FetchEventID<HardEvent::MTE3_V>());
            SetFlag<HardEvent::MTE3_V>(eventIdMTE3toV);
            WaitFlag<HardEvent::MTE3_V>(eventIdMTE3toV);
        }

        // --- [公式6] x_grad = x_grad_vec3 + x_grad_vec1 + x_grad_mm ---
        // gradXCubeLocal_ = x_grad_mm (来自 AIC 侧 H_mix_grad @ phi，已含 gamma 融合)
        // gradXCastLocal_ = x_grad_vec3 + x_grad_vec1
        // 最终: gradXCastLocal_ = x_grad_mm + (x_grad_vec3 + x_grad_vec1)
        // tileTaskCount = 2
        // copyLen = 256 128
        Add(gradXCastLocal_, gradXCubeLocal_, gradXCastLocal_, ubAlignC * n_ * tileTaskCount);
        PipeBarrier<PIPE_V>();
        inputXInQueue.FreeTensor(xLocal_);

        // --- [可选] grad_x_post 累加 ---
        if (withGradXPost_) {
            auto gradXPostXLocal = inputXInQueue.AllocTensor<TYPE_X>();
            DataCopyPad(gradXPostXLocal, gradXPostGlobal_[taskOffset * n_ * c_ + c0_ * loopIdC],
                        {static_cast<uint16_t>(n_ * tileTaskCount), static_cast<uint32_t>(copyLen * sizeof(TYPE_X)),
                         static_cast<uint32_t>((c_ - copyLen) * sizeof(TYPE_X)), 0, 0},
                        {isPad, 0, padLen, 0});
            inputXInQueue.EnQue(gradXPostXLocal);
            gradXPostXLocal = inputXInQueue.DeQue<TYPE_X>();
            Cast(xCastLocal_, gradXPostXLocal, RoundMode::CAST_NONE, ubAlignC * n_ * tileTaskCount);
            PipeBarrier<PIPE_V>();
            Add(gradXCastLocal_, gradXCastLocal_, xCastLocal_, ubAlignC * n_ * tileTaskCount);
            PipeBarrier<PIPE_V>();
            inputXInQueue.FreeTensor(gradXPostXLocal);
        }

        // Cast 回原始精度并写出
        auto gradXLocalOut = OutQueue.AllocTensor<TYPE_X>();
        Cast(gradXLocalOut, gradXCastLocal_, RoundMode::CAST_RINT, ubAlignC * n_ * tileTaskCount);
        OutQueue.EnQue(gradXLocalOut);
        gradXLocalOut = OutQueue.DeQue<TYPE_X>();

        DataCopyPad(gradXGlobal_[taskOffset * n_ * c_ + c0_ * loopIdC], gradXLocalOut,
                    {static_cast<uint16_t>(n_ * tileTaskCount), static_cast<uint32_t>(copyLen * sizeof(TYPE_X)), 0,
                     static_cast<uint32_t>((c_ - copyLen) * sizeof(TYPE_X)), 0});
        OutQueue.FreeTensor(gradXLocalOut);

        PipeBarrier<PIPE_V>();
    }
}

template <typename TYPE_X, typename T, bool DETERMINISTIC>
__aicore__ inline void MhcPreBackwardKernelArch22<TYPE_X, T, DETERMINISTIC>::ComputeDeterministic(
    GlobalTensor<float> &inputGm, GlobalTensor<float> &outputGm, const int64_t dimR, const int64_t dimA)
{
    int64_t queMaxLen = tileCoreBS_ * n_ * c0_ / 8;
    int64_t totalLen = dimR * dimA;
    for (int64_t taskOffset = blkIdx_ * queMaxLen; taskOffset < dimA; taskOffset += aivNum_ * queMaxLen) {
        int64_t tileTaskCount = min(static_cast<int64_t>(queMaxLen), static_cast<int64_t>(dimA - taskOffset));
        if (tileTaskCount > 0) {
            auto localOut = OutQueue.AllocTensor<float>();
            Duplicate(localOut, 0.f, tileTaskCount);
            for (int64_t dimRId = 0; dimRId < dimR; dimRId += 1) {
                auto localIn = inputXInQueue.AllocTensor<float>();

                DataCopyPad(localIn, inputGm[dimRId * dimA + taskOffset],
                            {static_cast<uint16_t>(1), static_cast<uint32_t>(tileTaskCount * sizeof(float)), 0, 0, 0},
                            {false, 0, 0, 0});
                inputXInQueue.EnQue(localIn);
                inputXInQueue.DeQue();
                PipeBarrier<PIPE_V>();
                Add(localOut, localOut, localIn, tileTaskCount);
                PipeBarrier<PIPE_V>();
                inputXInQueue.FreeTensor(localIn);
            }
            OutQueue.EnQue(localOut);
            localOut = OutQueue.DeQue<float>();
            DataCopyPad(outputGm[taskOffset], localOut,
                        {static_cast<uint16_t>(1), static_cast<uint32_t>(tileTaskCount * sizeof(float)), 0, 0, 0});
            OutQueue.FreeTensor(localOut);
        }
    }
}

template <typename TYPE_X, typename T, bool DETERMINISTIC>
__aicore__ inline void MhcPreBackwardKernelArch22<TYPE_X, T, DETERMINISTIC>::ComputeGradPre(const int64_t taskOffset,
                                                                                            const int32_t tileTaskCount,
                                                                                            const int32_t innerId)
{
    // [公式1] 输出组合梯度反向:
    //   正向: H_in = sum_i( x[:,:,i,:] * H_pre[:,:,i] )
    //   反向: H_pre_grad = Reduce(H_in_grad.unsqueeze(-2) ⊙ x, dim=-1)  → [B,S,N]
    // 同时: 保存 x * grad_y 到 xWorkspaceGlobal_ 供 AIC 侧 phi_grad 计算使用
    //         (phi_grad = G^T @ X, 其中 X = reshape(x * grad_y))

    // Kahan 求和双 buffer 初始化
    Duplicate(dpreLocalList_[0], 0.f, tileTaskCount * n_ * ELEMENTS_SIZE_PER_REPEAT);
    Duplicate(dpreLocalList_[1], 0.f, tileTaskCount * n_ * ELEMENTS_SIZE_PER_REPEAT);
    int32_t outPos = 0;
    for (int32_t loopIdC = 0; loopIdC < c1Align_; loopIdC += 1) {
        // 按 c0_ 分块遍历 D 维度
        int64_t copyLen = c0_;
        bool isPad = false;
        uint8_t padLen = 0;
        int64_t ubAlignC = c0_;
        if (loopIdC == c1_) {
            isPad = false;
            copyLen = cTail_;
            ubAlignC = cTailAlign_;
            padLen = static_cast<uint8_t>(cTailAlign_ - cTail_);
        }
        auto xLocal_ = inputXInQueue.AllocTensor<TYPE_X>();

        auto gradYLocal_ = xLocal_[onceTask_ * n_ * c0_];

        // 加载 grad_h_in (即 H_in_grad)，shape [tileTaskCount, copyLen]
        DataCopyPad(gradYLocal_, gradYGlobal_[taskOffset * c_ + c0_ * loopIdC],
                    {static_cast<uint16_t>(tileTaskCount), static_cast<uint32_t>(copyLen * sizeof(TYPE_X)),
                     static_cast<uint32_t>((c_ - copyLen) * sizeof(TYPE_X)), 0, 0},
                    {isPad, 0, padLen, 0});

        // 加载 x，shape [n_*tileTaskCount, copyLen]
        DataCopyPad(xLocal_, xGlobal_[taskOffset * n_ * c_ + c0_ * loopIdC],
                    {static_cast<uint16_t>(n_ * tileTaskCount), static_cast<uint32_t>(copyLen * sizeof(TYPE_X)),
                     static_cast<uint32_t>((c_ - copyLen) * sizeof(TYPE_X)), 0, 0},
                    {isPad, 0, padLen, 0});

        inputXInQueue.EnQue(xLocal_);
        inputXInQueue.DeQue();
        PipeBarrier<PIPE_V>();
        auto xCastOutLocal = OutQueue.AllocTensor<float>();

        // Cast x 和 grad_y 到 fp32
        Cast(xCastOutLocal, xLocal_, RoundMode::CAST_NONE, ubAlignC * n_ * tileTaskCount);
        Cast(gradYCastLocal_, gradYLocal_, RoundMode::CAST_NONE, ubAlignC * tileTaskCount);
        inputXInQueue.FreeTensor(xLocal_);

        PipeBarrier<PIPE_V>();
        uint8_t blkStride = static_cast<uint8_t>(ubAlignC / ELEMENTS_SIZE_PER_BLOCK);
        uint8_t blkStride3 = static_cast<uint8_t>(n_ * ubAlignC / ELEMENTS_SIZE_PER_BLOCK);

        // 计算 x ⊙ H_in_grad: 对每个 n, xCastLocal_[n] = x[n] * grad_y
        // 结果同时保存到 xCastOutLocal 供 phi_grad 使用 (即 X = x * grad_y)
        for (int32_t loopIdN = 0; loopIdN < n_; loopIdN += 1) {
            for (int32_t loopOffsetC0 = 0; loopOffsetC0 < copyLen; loopOffsetC0 += ELEMENTS_SIZE_PER_REPEAT) {
                uint64_t mask =
                    min(static_cast<uint64_t>(ELEMENTS_SIZE_PER_REPEAT), static_cast<uint64_t>(copyLen - loopOffsetC0));
                Mul(xCastLocal_[loopIdN * ubAlignC + loopOffsetC0], xCastOutLocal[loopIdN * ubAlignC + loopOffsetC0],
                    gradYCastLocal_[loopOffsetC0], mask, tileTaskCount, {1, 1, 1, blkStride3, blkStride3, blkStride});
            }
        }
        OutQueue.EnQue(xCastOutLocal);
        xCastOutLocal = OutQueue.DeQue<T>();

        // gamma: x_rs = x * gamma, 保存到 workspace 供 AIC 侧 phi_grad = G^T @ x_rs
        if (withGamma_) {
            DataCopyPad(gammaLocal_, gammaGlobal_[c0_ * loopIdC],
                        {static_cast<uint16_t>(n_), static_cast<uint32_t>(copyLen * sizeof(float)),
                         static_cast<uint32_t>((c_ - copyLen) * sizeof(float)), 0, 0},
                        {isPad, 0, padLen, 0});
            eventIdMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID<HardEvent::MTE2_V>());
            SetFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);

            for (int32_t loopIdN = 0; loopIdN < n_; loopIdN += 1) {
                for (int32_t loopOffsetC0 = 0; loopOffsetC0 < copyLen; loopOffsetC0 += ELEMENTS_SIZE_PER_REPEAT) {
                    uint64_t mask = min(static_cast<uint64_t>(ELEMENTS_SIZE_PER_REPEAT),
                                        static_cast<uint64_t>(copyLen - loopOffsetC0));
                    Mul(xCastOutLocal[loopIdN * ubAlignC + loopOffsetC0],
                        xCastOutLocal[loopIdN * ubAlignC + loopOffsetC0],
                        gammaLocal_[loopIdN * ubAlignC + loopOffsetC0], mask, tileTaskCount,
                        {1, 1, 1, blkStride3, blkStride3, 0});
                }
            }
            PipeBarrier<PIPE_V>();
            eventIdVToMTE3XCast = static_cast<event_t>(GetTPipePtr()->FetchEventID<HardEvent::V_MTE3>());
            SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3XCast);
            WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3XCast);
        }

        // 保存 x (或 x*gamma) 到 workspace，供 AIC 侧 ProcessMatmul2 计算 phi_grad = G^T @ X
        DataCopyPad(xWorkspaceGlobal_[taskOffset * n_ * c_ + c0_ * loopIdC], xCastOutLocal,
                    {static_cast<uint16_t>(n_ * tileTaskCount), static_cast<uint32_t>(copyLen * sizeof(float)), 0,
                     static_cast<uint32_t>((c_ - copyLen) * sizeof(float)), 0});
        OutQueue.FreeTensor(xCastOutLocal);

        PipeBarrier<PIPE_V>();

        // 对 D 维度做 reduce sum: H_pre_grad = sum_D(x * grad_y)
        // 手动树形归约 (c0_=256 时分 4 段: 0+128+64+192 → 0+64 → 0)
        int64_t reduceLen = ubAlignC;
        if (ubAlignC == c0_) {
            Add(xCastLocal_[64], xCastLocal_[64], xCastLocal_[128 + 64], ELEMENTS_SIZE_PER_REPEAT, tileTaskCount * n_,
                {1, 1, 1, blkStride, blkStride, blkStride});
            Add(xCastLocal_, xCastLocal_, xCastLocal_[128], ELEMENTS_SIZE_PER_REPEAT, tileTaskCount * n_,
                {1, 1, 1, blkStride, blkStride, blkStride});
            PipeBarrier<PIPE_V>();

            Add(xCastLocal_, xCastLocal_, xCastLocal_[64], ELEMENTS_SIZE_PER_REPEAT, tileTaskCount * n_,
                {1, 1, 1, blkStride, blkStride, blkStride});
        } else {
            // cTail 不足 c0_ 时的分块归约
            if (cTail_ - (128 + 64) > 0) {
                uint64_t mask = min(static_cast<uint64_t>(cTail_ - (128 + 64)), static_cast<uint64_t>(REPEAT_LENTH));
                Add(xCastLocal_[64], xCastLocal_[64], xCastLocal_[128 + 64], mask, tileTaskCount * n_,
                    {1, 1, 1, blkStride, blkStride, blkStride});
            }
            if (cTail_ - (128) > 0) {
                uint64_t mask = min(static_cast<uint64_t>(cTail_ - (128)), static_cast<uint64_t>(REPEAT_LENTH));
                Add(xCastLocal_, xCastLocal_, xCastLocal_[128], mask, tileTaskCount * n_,
                    {1, 1, 1, blkStride, blkStride, blkStride});
            }
            PipeBarrier<PIPE_V>();
            if (cTail_ - (64) > 0) {
                uint64_t mask = min(static_cast<uint64_t>(cTail_ - (64)), static_cast<uint64_t>(REPEAT_LENTH));
                Add(xCastLocal_, xCastLocal_, xCastLocal_[64], mask, tileTaskCount * n_,
                    {1, 1, 1, blkStride, blkStride, blkStride});
            }
        }
        PipeBarrier<PIPE_V>();

        uint64_t mask = min(static_cast<uint64_t>(cTail_), static_cast<uint64_t>(REPEAT_LENTH));
        PipeBarrier<PIPE_V>();

        // Kahan 求和: 跨 c0_ 分块累加 H_pre_grad
        LocalTensor<T> sumTensor = dpreLocalList_[outPos];
        LocalTensor<T> eTensor = dpreLocalList_[1 - outPos];
        int64_t len = tileTaskCount * n_ * ELEMENTS_SIZE_PER_REPEAT;
        auto inputTensor = gradYCastLocal_;
        PipeBarrier<PIPE_V>();
        Sub(inputTensor, xCastLocal_, eTensor, mask, tileTaskCount * n_, {1, 1, 1, 8, blkStride, 8});
        PipeBarrier<PIPE_V>();
        Add(eTensor, inputTensor, sumTensor, len);
        PipeBarrier<PIPE_V>();
        Sub(sumTensor, eTensor, sumTensor, len);
        PipeBarrier<PIPE_V>();
        Sub(sumTensor, sumTensor, inputTensor, len);
        PipeBarrier<PIPE_V>();

        outPos = 1 - outPos;
    }
    PipeBarrier<PIPE_V>();
    PipeBarrier<PIPE_V>();

    // 最终归约: H_pre_grad[n] = WholeReduceSum(xCastLocal_[n], dim=tileTaskCount)
    // 结果存入 dPrePostTempLocal_，后续在 ComputeGradHHat2 中作为 H_pre_grad 使用
    for (int32_t loopIdN = 0; loopIdN < n_; loopIdN += 1) {
        WholeReduceSum(dPrePostTempLocal_[loopIdN + innerId * n_ * 2], dpreLocalList_[outPos][loopIdN * REPEAT_LENTH],
                       REPEAT_LENTH, tileTaskCount, n_ * 2, 1, n_ * 8);
    }
}

template <typename TYPE_X, typename T, bool DETERMINISTIC>
__aicore__ inline void MhcPreBackwardKernelArch22<TYPE_X, T, DETERMINISTIC>::ComputeScaleBias()
{
    // ============================================================================
    // 本函数实现公式 [2][3][4] 中 bias_grad 和 α_grad 的最终归约输出:
    //   [公式2] bias_pre_grad  = sum_{b,s}(H_pre_2_grad)   → [N]
    //          α_pre_grad      = sum_{b,s,n}(H_pre_2_grad * H_pre_1)
    //   [公式3] bias_post_grad = sum_{b,s}(H_post_2_grad)  → [N]
    //          α_post_grad     = sum_{b,s,n}(H_post_2_grad * H_post_1)
    //   [公式4] bias_res_grad  = sum_{b,s}(H_res_grad)     → [N!]
    //          α_res_grad      = sum_{b,s}(H_res_grad * H_res_2)
    //
    // 数据布局说明:
    //   dBiasLocal   = sum(H_2_grad)，shape [tileCoreBS_, fusionSize_]
    //   dScaleLocal  = sum(H_2_grad * H_1)，shape [tileCoreBS_, fusionSize_]
    //   两者在 ComputeGradHHat2 中通过 kahanCustom 跨 tile 累加
    // ============================================================================

    auto dBiasLocal = dBiasLocalList_[scalePos_];
    auto dScaleLocal = dBiasLocal[tileCoreBS_ * fusionSize_];

    // α_grad: 将 res 区段的多个 block 归约到第一个 res block (block 1)
    // dScaleLocal 布局: [pre+post(8), res_block0(8), res_block1(8), ...]
    // [适配] 循环从最后一个 res block 向前累加, 支持任意 res block 数
    {
        int32_t totalBlocks = static_cast<int32_t>(fusionSize_ / ELEMENTS_SIZE_PER_BLOCK);
        int32_t resBlocks = static_cast<int32_t>(factN_ / ELEMENTS_SIZE_PER_BLOCK);
        for (int32_t i = resBlocks - 1; i > 0; i--) {
            Add(dScaleLocal[(i)*ELEMENTS_SIZE_PER_BLOCK], dScaleLocal[(i)*ELEMENTS_SIZE_PER_BLOCK],
                dScaleLocal[(1 + i) * ELEMENTS_SIZE_PER_BLOCK], ELEMENTS_SIZE_PER_REPEAT, tileRepeatTimes_,
                {static_cast<uint8_t>(totalBlocks), static_cast<uint8_t>(totalBlocks),
                 static_cast<uint8_t>(totalBlocks), static_cast<uint8_t>(fusionSize_),
                 static_cast<uint8_t>(fusionSize_), static_cast<uint8_t>(fusionSize_)});
            PipeBarrier<PIPE_V>();
        }
    }

    // bias_grad / α_grad: 跨 batch*seq 做树形归约 (tileCoreBS_ → 1)
    for (int32_t bsCount = tileCoreBS_ / 2; bsCount > 0; bsCount = bsCount / 2) {
        Add(dBiasLocal, dBiasLocal, dBiasLocal[bsCount * fusionSize_], bsCount * fusionSize_);
        Add(dScaleLocal, dScaleLocal, dScaleLocal[bsCount * fusionSize_], bsCount * fusionSize_);
        PipeBarrier<PIPE_V>();
    }
    eventIdVToMTE3XCast = static_cast<event_t>(GetTPipePtr()->FetchEventID<HardEvent::V_MTE3>());
    SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3XCast);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3XCast);
    LocalTensor<T> dscaleOut = tempBuf_.GetWithOffset<T>(ELEMENTS_SIZE_PER_BLOCK, 0);

    // --- 输出 bias_grad ---
    // grad_bias = [bias_pre_grad(N), bias_post_grad(N), bias_res_grad(N!)]
    // 即 dBiasLocal 的前 fusionSize_ 个元素
    if constexpr (DETERMINISTIC == false) {
        SetAtomicAdd<T>();
        DataCopyPad(gradHcBaseGlobal_, dBiasLocal,
                    {static_cast<uint16_t>(1), static_cast<uint32_t>(fusionSize_ * sizeof(T)), 0, 0, 0});
    } else {
        DataCopyPad(gradHcBaseWSGlobal_[blkIdx_ * fusionSize_], dBiasLocal,
                    {static_cast<uint16_t>(1), static_cast<uint32_t>(fusionSize_ * sizeof(T)), 0, 0, 0});
    }
    PipeBarrier<PIPE_V>();

    // --- 输出 α_grad ---
    // grad_alpha = [α_pre_grad, α_post_grad, α_res_grad]，shape (3)
    // α_pre_grad  = WholeReduceSum(dScaleLocal[0:4],  mask=4)           → dscaleOut[0]
    // α_post_grad = WholeReduceSum(dScaleLocal[0:8],  mask=POST_SCALE)  → dscaleOut[1] (mask取post的4个元素)
    // α_res_grad  = WholeReduceSum(dScaleLocal[8:16], mask=8)           → dscaleOut[2] (res归约到post block)
    // α_res_grad2 = WholeReduceSum(dScaleLocal[0:8],  mask=8)           → dscaleOut[3] (备用)
    WholeReduceSum(dscaleOut, dScaleLocal, 4, 1, 1, 3, 8);
    WholeReduceSum(dscaleOut[1], dScaleLocal, MASK_POST_SCALE, 1, 1, 3, 8);
    WholeReduceSum(dscaleOut[2], dScaleLocal[8], 8, 1, 1, 3, 8);
    WholeReduceSum(dscaleOut[3], dScaleLocal, 8, 1, 1, 3, 8);

    eventIdVToMTE3XCast = static_cast<event_t>(GetTPipePtr()->FetchEventID<HardEvent::V_MTE3>());
    SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3XCast);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3XCast);
    if constexpr (DETERMINISTIC == false) {
        DataCopyPad(gradHcScaleGlobal_, dscaleOut,
                    {static_cast<uint16_t>(1), static_cast<uint32_t>((3) * sizeof(T)), 0, 0, 0});
        SetAtomicNone();
    } else {
        DataCopyPad(gradHcScaleWSGlobal_[blkIdx_ * (3)], dscaleOut,
                    {static_cast<uint16_t>(1), static_cast<uint32_t>((3) * sizeof(T)), 0, 0, 0});
    }
}

template <typename TYPE_X, typename T, bool DETERMINISTIC>
__aicore__ inline void MhcPreBackwardKernelArch22<TYPE_X, T, DETERMINISTIC>::Process()
{
    // ============================================================================
    // 主调度函数，AIV/AIC 双核并行:
    //   AIV (Vector 核心): 执行公式 [1][2][3][4][5][8] 的逐元素和归约计算
    //   AIC (Cube 核心):   执行公式 [6] 的两个 matmul (ProcessMatmul1/2)
    //
    // 数据流:
    //   AIV: ComputeGradPre → workspace(x*grad_y) → ComputeGradHHat2 → workspace(H_mix_grad)
    //   AIV → AIC: CrossCoreSetFlag 通知 AIC 数据就绪
    //   AIC: ProcessMatmul1 (x_rs_grad = H_mix_grad @ phi)
    //        ProcessMatmul2 (phi_grad = G^T @ X)
    //   AIC → AIV: CrossCoreSetFlag 通知 AIV matmul 完成
    //   AIV: ComputeGradX1 (组装最终 x_grad = vec3 + vec1 + mm)
    // ============================================================================

    if ASCEND_IS_AIV {
        // --- AIV 侧: 逐元素计算 ---
        GetHcScaleAndHcBase();

        int8_t ping = 0;
        for (int64_t taskOffset = blkIdx_ * tileCoreBS_; taskOffset < totalTasksAligned_;
             taskOffset += aivNum_ * tileCoreBS_) {
            int32_t tileTaskCount =
                min(static_cast<int32_t>(tileCoreBS_), static_cast<int32_t>(totalTasks_ - taskOffset));
            tileRepeatTimes_ = Ops::Base::CeilDiv(static_cast<int64_t>(tileTaskCount) * 2 * n_,
                                                  static_cast<int64_t>(ELEMENTS_SIZE_PER_REPEAT));
            if (tileTaskCount > 0) {
                int32_t innerId = 0;
                Duplicate(dPrePostTempLocal_, 0.f, tileCoreBS_ * n_ * 2);
                // [公式1] 计算输出组合梯度 H_pre_grad，同时保存 x*grad_y 到 workspace
                for (int64_t taskOffsetInner = 0; taskOffsetInner < tileTaskCount; taskOffsetInner += onceTask_) {
                    int32_t tileTaskCountInner =
                        min(static_cast<int32_t>(onceTask_), static_cast<int32_t>(tileTaskCount - taskOffsetInner));

                    ComputeGradPre(taskOffset + taskOffsetInner, tileTaskCountInner, taskOffsetInner);
                    innerId++;
                }
                // [公式2][3][4][5][8] Sigmoid门控/残差/RMSNorm 反向，输出 H_mix_grad 到 workspace
                ComputeGradHHat2(taskOffset, tileTaskCount);
            }
            // 通知 AIC: workspace 数据就绪，可以开始 matmul
            CrossCoreSetFlag<0x2, PIPE_MTE3>(8);
            CrossCoreWaitFlag<0x2>(9);
            ping = (ping + 1) % 10;
            if (tileTaskCount > 0) {
                int32_t innerId = 0;
                for (int64_t taskOffsetInner = 0; taskOffsetInner < tileTaskCount; taskOffsetInner += onceTask_) {
                    int32_t tileTaskCountInner =
                        min(static_cast<int32_t>(onceTask_), static_cast<int32_t>(tileTaskCount - taskOffsetInner));

                    // 广播 h_pre 到 D 维度，供 ComputeGradX1 计算 x_grad_vec3
                    int32_t brcbAlign = static_cast<int32_t>(Ops::Base::CeilDiv(
                        static_cast<int64_t>(tileTaskCount) * n_, static_cast<int64_t>(ELEMENTS_SIZE_PER_BLOCK)));
                    int32_t offset = 0;
                    auto preLocal_ = inputGradQueue.AllocTensor<T>();
                    DataCopyPad(
                        preLocal_, preGlobal_[taskOffset * n_],
                        {static_cast<uint16_t>(1), static_cast<uint32_t>(tileTaskCount * n_ * sizeof(T)), 0, 0, 0},
                        {false, 0, 0, 0});
                    inputGradQueue.EnQue(preLocal_);
                    inputGradQueue.DeQue();

                    const uint32_t srcShape[2] = {static_cast<uint32_t>(tileTaskCount * n_), 1};
                    const uint32_t dstShape[2] = {static_cast<uint32_t>(tileTaskCount * n_), ELEMENTS_SIZE_PER_BLOCK};
                    Brcb(preBrcbLocal_, preLocal_, brcbAlign, {static_cast<uint8_t>(1), static_cast<uint8_t>(8)});
                    inputGradQueue.FreeTensor(preLocal_);

                    offset += brcbAlign * 8 * sizeof(float);
                    brcbAlign = static_cast<int32_t>(Ops::Base::CeilDiv(static_cast<int64_t>(tileTaskCount),
                                                                        static_cast<int64_t>(ELEMENTS_SIZE_PER_BLOCK)));

                    offset += brcbAlign * 8 * sizeof(float);

                    // 广播 x_rs_grad_inv 到 D 维度，供 ComputeGradX1 计算 x_grad_vec1
                    const uint32_t srcRsqrtShape[2] = {static_cast<uint32_t>(tileTaskCount), 1};
                    const uint32_t dstRsqrtShape[2] = {static_cast<uint32_t>(tileTaskCount), ELEMENTS_SIZE_PER_BLOCK};
                    Brcb(dRsqrtBrcbLocal_, gradRsqrtLocal_, brcbAlign,
                         {static_cast<uint8_t>(1), static_cast<uint8_t>(8)});
                    // [公式1][8][6] 组装最终 x_grad = vec3 + vec1 + mm
                    ComputeGradX1(taskOffset + taskOffsetInner, tileTaskCountInner, taskOffsetInner);
                    innerId++;
                }
            }
        }
        // [公式2][3][4] bias_grad 和 α_grad 最终归约
        tileRepeatTimes_ = Ops::Base::CeilDiv(tileCoreBS_ * 2 * n_, static_cast<int64_t>(ELEMENTS_SIZE_PER_REPEAT));
        ComputeScaleBias();
    }

    if ASCEND_IS_AIC {
        // --- AIC 侧: Cube matmul 计算 ---
        int8_t ping = 0;
        for (int64_t taskOffset = blkIdx_ * 2 * tileCoreBS_; taskOffset < totalTasksAligned_;
             taskOffset += aicNum_ * 2 * tileCoreBS_) {
            int32_t tileTaskCount =
                min(static_cast<int32_t>(2 * tileCoreBS_), static_cast<int32_t>(totalTasks_ - taskOffset));
            // 等待 AIV 侧 workspace 数据就绪
            CrossCoreWaitFlag<0x2>(8);

            if (tileTaskCount > 0) {
                // [公式6] x_rs_grad = H_mix_grad @ phi   (ProcessMatmul1)
                ProcessMatmul1(taskOffset, tileTaskCount);
                // [公式6] phi_grad = G^T @ X             (ProcessMatmul2)
                ProcessMatmul2(taskOffset, tileTaskCount);
            }
            // 通知 AIV: matmul 结果就绪
            AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(9);
            ping = (ping + 1) % 10;
        }
    }
    // --- Deterministic 模式: 多核归约 ---
    if constexpr (DETERMINISTIC == true) {
        SyncAll<false>();
        if ASCEND_IS_AIV {
            int64_t useCoreNum = min(Ops::Base::CeilDiv(totalTasks_, tileCoreBS_), aivNum_);
            ComputeDeterministic(gradHcBaseWSGlobal_, gradHcBaseGlobal_, useCoreNum, fusionSize_);
            ComputeDeterministic(gradHcScaleWSGlobal_, gradHcScaleGlobal_, useCoreNum, 3);
            if (withGamma_) {
                ComputeDeterministic(gradGammaWSGlobal_, gradGammaGlobal_, useCoreNum, n_ * c_);
            }
            useCoreNum = Ops::Base::CeilDiv(useCoreNum, static_cast<int64_t>(2));
            ComputeDeterministic(gradWeightWSGlobal_, gradWeightGlobal_, useCoreNum, fusionSize_ * (n_ * c_));
        }
    }
}

template <typename TYPE_X, typename T, bool DETERMINISTIC>
__aicore__ inline void MhcPreBackwardKernelArch22<TYPE_X, T, DETERMINISTIC>::ProcessMatmul1(const int64_t taskOffset,
                                                                                            const int32_t mm1M)
{
    // [公式6] x_rs_grad = H_mix_grad @ phi
    //   A = gradH2Global_ (H_mix_grad)，shape [M, K] = [tileTaskCount, fusionSize_]
    //   B = weightGlobal_ (phi)，shape [K, N] = [fusionSize_, n_*c_]
    //   C = gradXCubeGlobal_ (x_rs_grad)，shape [M, N] = [tileTaskCount, n_*c_]
    //   gamma 融合在 host 侧或后续处理中
    if (mm1M <= 0)
        return;

    mm1_.SetTensorA(gradH2Global_[taskOffset * mm1K_]);
    mm1_.SetTensorB(weightGlobal_);
    mm1_.SetOrgShape(mm1M, mm1N_, mm1K_);
    mm1_.SetSingleShape(mm1M, mm1N_, mm1K_);
    mm1_.template IterateAll<true>(gradXCubeGlobal_[taskOffset * (n_ * c_)]);
    mm1_.End();
}

template <typename TYPE_X, typename T, bool DETERMINISTIC>
__aicore__ inline void MhcPreBackwardKernelArch22<TYPE_X, T, DETERMINISTIC>::ProcessMatmul2(const int64_t taskOffset,
                                                                                            const int32_t mm2K)
{
    // [公式6] phi_grad = G^T @ X
    //   A = gradH2Global_ (H_mix_grad)^T，shape [M, K] = [fusionSize_, tileTaskCount]
    //   B = xWorkspaceGlobal_ (X = x * grad_y)，shape [K, N] = [tileTaskCount, n_*c_]
    //   C = gradWeightGlobal_ / gradWeightWSGlobal_ (phi_grad)，shape [M, N] = [fusionSize_, n_*c_]
    if (mm2K <= 0)
        return;
    mm2_.SetTensorA(gradH2Global_[taskOffset * mm2M_], true); // [B * S, 2N + N!]
    mm2_.SetTensorB(xWorkspaceGlobal_[taskOffset * mm2N_]);   // [B * S, N * D]
    mm2_.SetOrgShape(mm2M_, mm2N_, mm2K);
    mm2_.SetSingleShape(mm2M_, mm2N_, mm2K);
    if constexpr (DETERMINISTIC == true) {
        mm2_.template IterateAll<true>(gradWeightWSGlobal_[blkIdx_ * (mm2M_ * mm2N_)], needAdd);
        needAdd = 1;
    } else {
        mm2_.template IterateAll<true>(gradWeightGlobal_, needAdd);
        needAdd = 1;
    }
    mm2_.End();
}

#endif // MHC_PRE_BACKWARD_OP_KERNEL_ARCH22_KERNEL_H
