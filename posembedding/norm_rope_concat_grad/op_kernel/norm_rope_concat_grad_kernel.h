/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file norm_rope_concat_grad_kernel.h
 * \brief
 */
#ifndef NORM_ROPE_CONCAT_GRAD_KERNEL_H
#define NORM_ROPE_CONCAT_GRAD_KERNEL_H

#include "norm_rope_concat_grad_base_kernel.h"
using namespace AscendC;

namespace nrcg {
template <NormType normType>
class NormOperationBackward : public NormOperation<normType> {
public:
    __aicore__ inline NormOperationBackward(TBufPool<TPosition::VECCALC, BUF_ID_SIZE> &bufPool, GM_ADDR x,
                                            GM_ADDR weight, GM_ADDR bias, GM_ADDR mean, GM_ADDR rstd, GM_ADDR xGrad,
                                            GM_ADDR weightGrad, GM_ADDR biasGrad, float eps, float scale,
                                            uint32_t normDim, uint32_t normNum, uint32_t alignedNormDim,
                                            uint32_t alignedNormNum, uint32_t affineBlockNums, uint32_t avgHeadsLow)
        : NormOperation<normType>(x, weight, bias, mean, rstd, eps, scale, normDim, normNum, alignedNormDim,
                                  alignedNormNum),
          affineBlockNums_(affineBlockNums),
          avgHeadsLow_(avgHeadsLow)
    {
        xGradGM_.SetGlobalBuffer((__gm__ DTYPE_QUERY *)xGrad);
        outParams_.blockLen = normDim * sizeof(DTYPE_QUERY);
        bufPool.InitBuffer(inGradQue_, DOUBLE_BUFFER, normNum * alignedNormDim * sizeof(DTYPE_QUERY));

        if constexpr (normType == NormType::NONE) {
            return;
        }

        bufPool.InitBuffer(inQue_, DOUBLE_BUFFER, alignedNormDim * normNum * sizeof(DTYPE_QUERY)); // inputx que
        inParams_.blockLen = normDim * sizeof(DTYPE_QUERY);
        padParams_.rightPadding = alignedNormDim - normDim;

        bufPool.InitBuffer(normQue_, DOUBLE_BUFFER, NUM_TWO * alignedNormNum * sizeof(float)); // mean + rstd que

        uint32_t bufSize = 0;
        // if affine
        if constexpr (normType == NormType::LAYER_NORM_AFFINE || normType == NormType::LAYER_NORM_AFFINE_ACROSS_HEADS) {
            bufSize += normNum * alignedNormDim + NUM_TWO * alignedNormDim; // weight
        }

        bufSize += NUM_TWO * normNum * alignedNormDim + NUM_TWO * alignedNormNum; // x + share_+ mean + rstd
        bufPool.InitBuffer(buf_, bufSize * sizeof(float));
        inX_ = buf_.Get<float>();
        share_ = inX_[normNum * alignedNormDim];
        mean_ = inX_[NUM_TWO * normNum * alignedNormDim];
        rstd_ = inX_[NUM_TWO * normNum * alignedNormDim + alignedNormNum];
        if constexpr (normType == NormType::LAYER_NORM_AFFINE || normType == NormType::LAYER_NORM_AFFINE_ACROSS_HEADS) {
            weight_ = inX_[NUM_TWO * normNum * alignedNormDim + NUM_TWO * alignedNormNum];
            sumWeightGradBufLocal_ = inX_[NUM_THREE * normNum * alignedNormDim + NUM_TWO * alignedNormNum];
            sumBiasGradBufLocal_ =
                inX_[NUM_THREE * normNum * alignedNormDim + NUM_TWO * alignedNormNum + alignedNormDim];

            bufPool.InitBuffer(weightQue_, SINGLE_BUFFER, alignedNormDim * sizeof(DTYPE_QUERY));
            weightParams_.blockLen = normDim * sizeof(DTYPE_QUERY);
            padWeightParams_.rightPadding = alignedNormDim - normDim;

            weightGradGM_.SetGlobalBuffer((__gm__ DTYPE_QUERY *)weightGrad);
            biasGradGM_.SetGlobalBuffer((__gm__ DTYPE_QUERY *)biasGrad);
            outWeightParams_.blockLen = NUM_TWO * alignedNormDim * sizeof(float);

            blockIdx_ = GetBlockIdx();
            affineOffset_ = blockIdx_ * affineBlockNums * avgHeadsLow * NUM_TWO * alignedNormDim;
        }
    }
    __aicore__ inline void Process(const LocalTensor<float> &x, const GlobalTensor<float> &workspaceAffineGM,
                                   int64_t inOffset, int64_t normOffset, uint32_t heads);
    __aicore__ inline void Prepare();
    __aicore__ inline void PostProcess(const LocalTensor<float> &x, const GlobalTensor<float> &workspaceAffineGM,
                                       const GlobalTensor<float> &workspaceGM, uint32_t workspaceId);

private:
    __aicore__ inline void CopyIn(int64_t inOffset, int64_t normOffset, uint32_t heads);
    __aicore__ inline void Compute(const LocalTensor<float> &x, uint32_t heads);
    __aicore__ inline void CopyOut(const LocalTensor<float> &x, const GlobalTensor<float> &workspaceAffineGM,
                                   int64_t inOffset, uint32_t heads);
    __aicore__ inline void AffineAccSum(const LocalTensor<float> &x, const GlobalTensor<float> &workspaceAffineGM);
    __aicore__ inline void BinarySum(const LocalTensor<float> &x, int64_t rows, int64_t cols);

private:
    GlobalTensor<DTYPE_QUERY> xGradGM_;
    GlobalTensor<DTYPE_QUERY> weightGradGM_;
    GlobalTensor<DTYPE_QUERY> biasGradGM_;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER> inGradQue_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> inQue_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> normQue_;
    TQue<QuePosition::VECIN, SINGLE_BUFFER> weightQue_;
    TBuf<QuePosition::VECCALC> buf_;
    LocalTensor<float> inX_;
    LocalTensor<float> share_;
    LocalTensor<float> mean_;
    LocalTensor<float> rstd_;
    LocalTensor<float> weight_;
    LocalTensor<float> sumWeightGradBufLocal_;
    LocalTensor<float> sumBiasGradBufLocal_;
    DataCopyExtParams inParams_{0, 0, 0, 0, 0};
    DataCopyPadExtParams<DTYPE_QUERY> padParams_{true, 0, 0, 0};
    DataCopyExtParams meanParams_{1, 0, 0, 0, 0};
    DataCopyPadExtParams<float> padMeanParams_{true, 0, 0, 0};
    DataCopyExtParams weightParams_{1, 0, 0, 0, 0};
    DataCopyPadExtParams<DTYPE_QUERY> padWeightParams_{true, 0, 0, 0};
    DataCopyExtParams outParams_{0, 0, 0, 0, 0};
    DataCopyExtParams outWeightParams_{1, 0, 0, 0, 0};
    uint32_t blockIdx_;
    uint32_t affineArray_[MAX_AFFINE_BLOCK_NUM] = {0};
    uint32_t affineBlockNums_;
    uint32_t avgHeadsLow_;
    uint32_t affineOffset_;
};

template <NormType normType>
__aicore__ inline void NormOperationBackward<normType>::Prepare()
{
    // if affine, copy weight
    if constexpr (normType == NormType::LAYER_NORM_AFFINE || normType == NormType::LAYER_NORM_AFFINE_ACROSS_HEADS) {
        LocalTensor<DTYPE_QUERY> bufWeight = weightQue_.AllocTensor<DTYPE_QUERY>();
        // copy gm weight
        DataCopyPad(bufWeight, this->weightGm_, weightParams_, padWeightParams_);
        weightQue_.EnQue(bufWeight);
        LocalTensor<DTYPE_QUERY> bufWeightD = weightQue_.DeQue<DTYPE_QUERY>();
        if constexpr (sizeof(DTYPE_QUERY) == sizeof(float)) {
            Adds(inX_, bufWeightD.ReinterpretCast<float>(), 0.0f, this->alignedNormDim_);
        } else { // cast to float
            Cast(inX_, bufWeightD, RoundMode::CAST_NONE, this->alignedNormDim_);
        }
        PipeBarrier<PIPE_V>();
        weightQue_.FreeTensor(bufWeightD);
        uint32_t shapeND[2] = {this->normNum_, this->alignedNormDim_};
        uint32_t shapeD[2] = {1, this->alignedNormDim_};
        LocalTensor<uint8_t> shareBufCast_ = share_.ReinterpretCast<uint8_t>();
        BroadCast<float, 2, 0>(weight_, inX_, shapeND, shapeD, shareBufCast_);
        Duplicate(sumWeightGradBufLocal_, 0.0f, this->alignedNormDim_);
        Duplicate(sumBiasGradBufLocal_, 0.0f, this->alignedNormDim_);
        PipeBarrier<PIPE_V>();
    }
}

template <NormType normType>
__aicore__ inline void NormOperationBackward<normType>::Process(const LocalTensor<float> &x,
                                                                const GlobalTensor<float> &workspaceAffineGM,
                                                                int64_t inOffset, int64_t normOffset, uint32_t heads)
{
    CopyIn(inOffset, normOffset, heads);
    Compute(x, heads);
    CopyOut(x, workspaceAffineGM, inOffset, heads);
}

template <NormType normType>
__aicore__ inline void NormOperationBackward<normType>::CopyIn(int64_t inOffset, int64_t normOffset, uint32_t heads)
{
    if constexpr (normType == NormType::NONE) {
        return;
    }

    LocalTensor<DTYPE_QUERY> input = inQue_.AllocTensor<DTYPE_QUERY>();
    inParams_.blockCount = heads;
    DataCopyPad(input, this->xGm_[inOffset], inParams_, padParams_);
    inQue_.EnQue(input);

    // copy gm mean, rstd
    LocalTensor<float> norm = normQue_.AllocTensor<float>();
    meanParams_.blockLen = heads * sizeof(float);
    DataCopyPad(norm, this->meanGm_[normOffset], meanParams_, padMeanParams_);
    DataCopyPad(norm[this->alignedNormNum_], this->rstdGm_[normOffset], meanParams_, padMeanParams_);
    normQue_.EnQue(norm);
}

template <NormType normType>
__aicore__ inline void NormOperationBackward<normType>::BinarySum(const LocalTensor<float> &x, int64_t rows,
                                                                  int64_t cols)
{
    while (rows > 1) {
        uint32_t halfTimes = rows / 2;
        bool left = rows % 2;
        Add(x, x, x[halfTimes * cols], halfTimes * cols);
        PipeBarrier<PIPE_V>();
        if (left) {
            Add(x, x, x[(rows - 1) * cols], cols);
            PipeBarrier<PIPE_V>();
        }
        rows = halfTimes;
    }
}

template <NormType normType>
__aicore__ inline void NormOperationBackward<normType>::Compute(const LocalTensor<float> &gradOut, uint32_t heads)
{
    if constexpr (normType == NormType::NONE) {
        // EnQue
        LocalTensor<DTYPE_QUERY> xGrad = inGradQue_.AllocTensor<DTYPE_QUERY>();
        if (sizeof(DTYPE_QUERY) == sizeof(float)) {
            Adds(xGrad.ReinterpretCast<float>(), gradOut, 0.0f, heads * this->alignedNormDim_);
        } else {
            Cast(xGrad, gradOut, RoundMode::CAST_ROUND, heads * this->alignedNormDim_);
        }
        PipeBarrier<PIPE_V>();
        inGradQue_.EnQue(xGrad);
        return;
    }
    // get x tensor
    LocalTensor<DTYPE_QUERY> input = inQue_.DeQue<DTYPE_QUERY>();
    uint32_t size = heads * this->alignedNormDim_;
    if constexpr (sizeof(DTYPE_QUERY) != sizeof(float)) {
        Cast(inX_, input, RoundMode::CAST_NONE, size);
    } else {
        Adds(inX_, input.ReinterpretCast<float>(), 0.0f, size);
    }
    inQue_.FreeTensor(input);

    // get mean rstd tensor
    LocalTensor<float> norm = normQue_.DeQue<float>();
    Adds(mean_, norm, 0.0f, this->alignedNormNum_);
    Adds(rstd_, norm[this->alignedNormNum_], 0.0f, this->alignedNormNum_);
    PipeBarrier<PIPE_V>();
    normQue_.FreeTensor(norm);

    uint32_t shapeND[2] = {heads, this->alignedNormDim_};
    uint32_t shapeD[2] = {1, this->alignedNormDim_};
    uint32_t shapeN[2] = {heads, 1};
    SumParams sumParams{heads, this->alignedNormDim_, this->normDim_};
    LocalTensor<float> tempND = gradOut[this->normNum_ * this->alignedNormDim_];
    LocalTensor<uint8_t> shareBufCast_ =
        (gradOut[2 * this->normNum_ * this->alignedNormDim_]).template ReinterpretCast<uint8_t>();

    BroadCast<float, 2, 1, false>(tempND, mean_, shapeND, shapeN, shareBufCast_);
    PipeBarrier<PIPE_V>();
    Sub(inX_, inX_, tempND, heads * this->alignedNormDim_);
    PipeBarrier<PIPE_V>();

    BroadCast<float, 2, 1, false>(tempND, rstd_, shapeND, shapeN, shareBufCast_);
    PipeBarrier<PIPE_V>();
    Mul(inX_, inX_, tempND, heads * this->alignedNormDim_);
    PipeBarrier<PIPE_V>();

    if constexpr (normType == NormType::LAYER_NORM_AFFINE || normType == NormType::LAYER_NORM_AFFINE_ACROSS_HEADS) {
        Adds(tempND, gradOut, 0.0f, heads * this->alignedNormDim_);
        PipeBarrier<PIPE_V>();
        BinarySum(tempND, heads, this->alignedNormDim_);
        Adds(sumBiasGradBufLocal_, tempND, 0.0f, this->alignedNormDim_);
        PipeBarrier<PIPE_V>();

        // gradOut[headnum, headdim], mean[headnum, 1], rstd[headnum, 1] --> grad weight[headdim]
        Mul(tempND, inX_, gradOut, heads * this->alignedNormDim_);
        PipeBarrier<PIPE_V>();
        BinarySum(tempND, heads, this->alignedNormDim_);
        Adds(sumWeightGradBufLocal_, tempND, 0.0f, this->alignedNormDim_);
        PipeBarrier<PIPE_V>();
    }

    // gradOut, x, mean, rstd, weight --> grad x
    // part1, dx
    LocalTensor<float> dx_hat = tempND;
    if constexpr (normType == NormType::LAYER_NORM_AFFINE || normType == NormType::LAYER_NORM_AFFINE_ACROSS_HEADS) {
        Mul(dx_hat, gradOut, weight_, heads * this->alignedNormDim_);
    } else {
        Adds(dx_hat, gradOut, 0.0f, heads * this->alignedNormDim_);
    }
    PipeBarrier<PIPE_V>();
    Adds(gradOut, dx_hat, 0.0f, heads * this->alignedNormDim_);
    PipeBarrier<PIPE_V>();

    // part2, du
    LocalTensor<float> sum_dx_hat = mean_;
    Sum(sum_dx_hat, dx_hat, shareBufCast_, sumParams);
    PipeBarrier<PIPE_V>();
    Muls(sum_dx_hat, sum_dx_hat, -this->scale_, heads);
    PipeBarrier<PIPE_V>();
    BroadCast<float, 2, 1, false>(share_, sum_dx_hat, shapeND, shapeN, shareBufCast_);
    PipeBarrier<PIPE_V>();
    Add(gradOut, gradOut, share_, heads * this->alignedNormDim_);
    PipeBarrier<PIPE_V>();

    // part3, dσ
    LocalTensor<float> sum_dx_hat_x_centered = mean_;
    Mul(dx_hat, dx_hat, inX_, heads * this->alignedNormDim_);
    PipeBarrier<PIPE_V>();
    Sum(sum_dx_hat_x_centered, dx_hat, shareBufCast_, sumParams);
    PipeBarrier<PIPE_V>();
    Muls(sum_dx_hat_x_centered, sum_dx_hat_x_centered, -this->scale_, heads);
    PipeBarrier<PIPE_V>();
    BroadCast<float, 2, 1, false>(share_, sum_dx_hat_x_centered, shapeND, shapeN, shareBufCast_);
    PipeBarrier<PIPE_V>();
    MulAddDst(gradOut, inX_, share_, heads * this->alignedNormDim_);
    PipeBarrier<PIPE_V>();

    // rstd_ * (du + du + dσ)
    BroadCast<float, 2, 1, false>(share_, rstd_, shapeND, shapeN, shareBufCast_);
    PipeBarrier<PIPE_V>();
    Mul(gradOut, gradOut, share_, heads * this->alignedNormDim_);
    PipeBarrier<PIPE_V>();

    // EnQue
    LocalTensor<DTYPE_QUERY> xGrad = inGradQue_.AllocTensor<DTYPE_QUERY>();
    if (sizeof(DTYPE_QUERY) == sizeof(float)) {
        Adds(xGrad.ReinterpretCast<float>(), gradOut, 0.0f, heads * this->alignedNormDim_);
    } else {
        Cast(xGrad, gradOut, RoundMode::CAST_ROUND, heads * this->alignedNormDim_);
    }
    PipeBarrier<PIPE_V>();
    inGradQue_.EnQue(xGrad);
}

template <NormType normType>
__aicore__ inline void NormOperationBackward<normType>::CopyOut(const LocalTensor<float> &x,
                                                                const GlobalTensor<float> &workspaceAffineGM,
                                                                int64_t inOffset, uint32_t heads)
{
    LocalTensor<DTYPE_QUERY> xGrad = inGradQue_.DeQue<DTYPE_QUERY>();
    outParams_.blockCount = heads;
    DataCopyPad(this->xGradGM_[inOffset], xGrad, outParams_);
    inGradQue_.FreeTensor(xGrad);

    if constexpr (normType == NormType::LAYER_NORM_AFFINE || normType == NormType::LAYER_NORM_AFFINE_ACROSS_HEADS) {
        DataCopyPad(workspaceAffineGM[affineOffset_ + affineArray_[0] * NUM_TWO * this->alignedNormDim_],
                    sumWeightGradBufLocal_, outWeightParams_);
        MTE3ToMTE2Sync();
        AffineAccSum(x, workspaceAffineGM);
    }
}

template <NormType normType>
__aicore__ inline void NormOperationBackward<normType>::AffineAccSum(const LocalTensor<float> &x,
                                                                     const GlobalTensor<float> &workspaceAffineGM)
{
    affineArray_[0] += 1;
    for (size_t i = 0; i < affineBlockNums_; ++i) {
        if (affineArray_[i] == avgHeadsLow_ && i != (affineBlockNums_ - 1)) {
            // Sum up when the affine block is full
            DataCopyExtParams normParams_{1, avgHeadsLow_ * NUM_TWO * this->alignedNormDim_ * 4, 0, 0, 0};
            DataCopyPadExtParams<float> padNormParams_{true, 0, 0, 0};
            DataCopyPad(
                x,
                workspaceAffineGM[(blockIdx_ * affineBlockNums_ + i) * avgHeadsLow_ * NUM_TWO * this->alignedNormDim_],
                normParams_, padNormParams_);
            MTE2ToVSync();

            BinarySum(x, avgHeadsLow_, NUM_TWO * this->alignedNormDim_);
            VToMTE3Sync();

            DataCopyExtParams affinePushParams_{1, NUM_TWO * this->alignedNormDim_ * 4, 0, 0, 0};
            DataCopyPad(workspaceAffineGM[(blockIdx_ * affineBlockNums_ + i + 1) * avgHeadsLow_ * NUM_TWO *
                                              this->alignedNormDim_ +
                                          affineArray_[i + 1] * NUM_TWO * this->alignedNormDim_],
                        x, affinePushParams_);
            MTE3ToMTE2Sync();

            affineArray_[i] = 0;
            affineArray_[i + 1] += 1;
        } else {
            return;
        }
    }
    return;
}

template <NormType normType>
__aicore__ inline void NormOperationBackward<normType>::PostProcess(const LocalTensor<float> &x,
                                                                    const GlobalTensor<float> &workspaceAffineGM,
                                                                    const GlobalTensor<float> &workspaceGM,
                                                                    uint32_t workspaceId)
{
    if constexpr (normType != NormType::LAYER_NORM_AFFINE && normType != NormType::LAYER_NORM_AFFINE_ACROSS_HEADS) {
        return;
    }

    for (size_t i = 0; i < affineBlockNums_; ++i) {
        if (affineArray_[i] != 0) {
            // Finally sum up even if the affine block is not full
            DataCopyExtParams normParams_{1, affineArray_[i] * NUM_TWO * this->alignedNormDim_ * 4, 0, 0, 0};
            DataCopyPadExtParams<float> padNormParams_{true, 0, 0, 0};
            DataCopyPad(
                x,
                workspaceAffineGM[(blockIdx_ * affineBlockNums_ + i) * avgHeadsLow_ * NUM_TWO * this->alignedNormDim_],
                normParams_, padNormParams_);
            MTE2ToVSync();

            BinarySum(x, affineArray_[i], NUM_TWO * this->alignedNormDim_);
            VToMTE3Sync();

            DataCopyExtParams affinePushParams_{1, NUM_TWO * this->alignedNormDim_ * 4, 0, 0, 0};
            if (i == (affineBlockNums_ - 1)) {
                uint32_t start_offest = workspaceId * NUM_TWO * this->alignedNormDim_;
                DataCopyPad(workspaceGM[start_offest + blockIdx_ * TOTAL_NORM_NUM * this->alignedNormDim_], x,
                            affinePushParams_);
            } else {
                DataCopyPad(workspaceAffineGM[(blockIdx_ * affineBlockNums_ + i + 1) * avgHeadsLow_ * NUM_TWO *
                                                  this->alignedNormDim_ +
                                              affineArray_[i + 1] * NUM_TWO * this->alignedNormDim_],
                            x, affinePushParams_);
                affineArray_[i + 1] += 1;
            }
            MTE3ToMTE2Sync();

            affineArray_[i] = 0;
        }
    }
    return;
}

template <RopeType ropeType>
class RopeOperationBackward : public RopeOperation<ropeType, false> {
public:
    __aicore__ inline RopeOperationBackward(TBufPool<TPosition::VECCALC, BUF_ID_SIZE> &bufPool, GM_ADDR xGradOut,
                                            GM_ADDR ropeSin, GM_ADDR ropeCos, GM_ADDR y, uint32_t actualSeq,
                                            uint32_t totalSeq, uint32_t startSeq, uint32_t ropeDim, uint32_t ropeNum,
                                            uint32_t alignedRopeDim)
        : RopeOperation<ropeType, false>(bufPool, ropeSin, ropeCos, y, actualSeq, startSeq, ropeDim, alignedRopeDim)
    {
        xGradOutGm_.SetGlobalBuffer((__gm__ DTYPE_QUERY *)xGradOut);
        bufPool.InitBuffer(outGradQue_, DOUBLE_BUFFER, ropeNum * alignedRopeDim * sizeof(DTYPE_QUERY));
        outGradParams_.blockLen = ropeDim * sizeof(DTYPE_QUERY);
        padParams_.rightPadding = alignedRopeDim - ropeDim;
        outGradParams_.srcStride = (totalSeq - 1) * ropeDim * sizeof(DTYPE_QUERY);
    }
    __aicore__ inline void Process(const LocalTensor<float> &outGrad, int64_t outGradOffset, uint32_t heads);

private:
    __aicore__ inline void CopyIn(int64_t outGradOffset, uint32_t heads);
    __aicore__ inline void Compute(const LocalTensor<float> &outGrad, uint32_t heads);

private:
    GlobalTensor<DTYPE_QUERY> xGradOutGm_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> outGradQue_;
    DataCopyPadExtParams<DTYPE_QUERY> padParams_{true, 0, 0, 0};
    DataCopyExtParams outGradParams_{0, 0, 0, 0, 0};
};

template <RopeType ropeType>
__aicore__ inline void RopeOperationBackward<ropeType>::Process(const LocalTensor<float> &outGrad,
                                                                int64_t outGradOffset, uint32_t heads)
{
    CopyIn(outGradOffset, heads);
    Compute(outGrad, heads);
}

template <RopeType ropeType>
__aicore__ inline void RopeOperationBackward<ropeType>::CopyIn(int64_t outGradOffset, uint32_t heads)
{
    // copy gmOutGrad
    LocalTensor<DTYPE_QUERY> outGradLocal = outGradQue_.AllocTensor<DTYPE_QUERY>();
    outGradParams_.blockCount = heads;
    DataCopyPad(outGradLocal, this->xGradOutGm_[outGradOffset], outGradParams_, padParams_);
    outGradQue_.EnQue(outGradLocal);
}

template <RopeType ropeType>
__aicore__ inline void RopeOperationBackward<ropeType>::Compute(const LocalTensor<float> &outGrad, uint32_t heads)
{
    LocalTensor<DTYPE_QUERY> outGradLocal = outGradQue_.DeQue<DTYPE_QUERY>();
    uint32_t size = heads * this->alignedRopeDim_;

    if (sizeof(DTYPE_QUERY) < sizeof(float)) {
        Cast(outGrad, outGradLocal, RoundMode::CAST_NONE, size);
    } else {
        Adds(outGrad, outGradLocal.ReinterpretCast<float>(), 0.0f, size);
    }
    PipeBarrier<PIPE_V>();
    outGradQue_.FreeTensor(outGradLocal);

    if constexpr (ropeType == RopeType::NONE) {
        return;
    }

    if (this->isActive_) {
        LocalTensor<float> temp0 = outGrad[size];
        for (uint32_t i = 0; i < heads; ++i) {
            this->Rotate(outGrad[i * this->alignedRopeDim_], temp0[i * this->alignedRopeDim_]);
            PipeBarrier<PIPE_V>();
            this->Collect(outGrad[i * this->alignedRopeDim_], temp0[i * this->alignedRopeDim_]);
            PipeBarrier<PIPE_V>();
        }
        PipeBarrier<PIPE_V>();
    }
}

template <NormType normType, NormType addedNormType, RopeType ropeType, ConcatOrder concatOrder>
class NormRopeConcatGrad {
public:
    __aicore__ inline NormRopeConcatGrad() = default;

    __aicore__ inline NormRopeConcatGrad(
        GM_ADDR grad_query_output, GM_ADDR grad_key_output, GM_ADDR grad_value_output, GM_ADDR query, GM_ADDR key,
        GM_ADDR encoder_query, GM_ADDR encoder_key, GM_ADDR norm_query_weight, GM_ADDR norm_query_mean,
        GM_ADDR norm_query_rstd, GM_ADDR norm_key_weight, GM_ADDR norm_key_mean, GM_ADDR norm_key_rstd,
        GM_ADDR norm_added_query_weight, GM_ADDR norm_added_query_mean, GM_ADDR norm_added_query_rstd,
        GM_ADDR norm_added_key_weight, GM_ADDR norm_added_key_mean, GM_ADDR norm_added_key_rstd, GM_ADDR rope_sin,
        GM_ADDR rope_cos, GM_ADDR grad_query, GM_ADDR grad_key, GM_ADDR grad_value, GM_ADDR grad_encoderquery,
        GM_ADDR grad_encoderkey, GM_ADDR grad_encodervalue, GM_ADDR grad_norm_query_weight,
        GM_ADDR grad_norm_query_bias, GM_ADDR grad_norm_key_weight, GM_ADDR grad_norm_key_bias,
        GM_ADDR grad_norm_added_query_weight, GM_ADDR grad_norm_added_query_bias, GM_ADDR grad_norm_added_key_weight,
        GM_ADDR grad_norm_added_key_bias, GM_ADDR workspace)
        : gradQueryOutput_(grad_query_output),
          gradKeyOutput_(grad_key_output),
          gradValueOutput_(grad_value_output),
          query_(query),
          key_(key),
          encoderQuery_(encoder_query),
          encoderKey_(encoder_key),
          normQueryWeight_(norm_query_weight),
          normQueryMean_(norm_query_mean),
          normQueryRstd_(norm_query_rstd),
          normKeyWeight_(norm_key_weight),
          normKeyMean_(norm_key_mean),
          normKeyRstd_(norm_key_rstd),
          normAddedQueryWeight_(norm_added_query_weight),
          normAddedQueryMean_(norm_added_query_mean),
          normAddedQueryRstd_(norm_added_query_rstd),
          normAddedKeyWeight_(norm_added_key_weight),
          normAddedKeyMean_(norm_added_key_mean),
          normAddedKeyRstd_(norm_added_key_rstd),
          ropeSin_(rope_sin),
          ropeCos_(rope_cos),
          gradQuery_(grad_query),
          gradKey_(grad_key),
          gradValue_(grad_value),
          gradEncoderQuery_(grad_encoderquery),
          gradEncoderKey_(grad_encoderkey),
          gradEncoderValue_(grad_encodervalue),
          gradNormQueryWeight_(grad_norm_query_weight),
          gradNormQueryBias_(grad_norm_query_bias),
          gradNormKeyWeight_(grad_norm_key_weight),
          gradNormKeyBias_(grad_norm_key_bias),
          gradNormAddedQueryWeight_(grad_norm_added_query_weight),
          gradNormAddedQueryBias_(grad_norm_added_query_bias),
          gradNormAddedKeyWeight_(grad_norm_added_key_weight),
          gradNormAddedKeyBias_(grad_norm_added_key_bias),
          workspace_(workspace)
    {}

    __aicore__ inline void Init(TPipe *pipe, const NormRopeConcatGradTilingData &tilingData);
    __aicore__ inline void Prepare();
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessQuery();
    __aicore__ inline void ProcessKey();
    __aicore__ inline void ProcessValue();

    __aicore__ inline void ProcessQK(TBufPool<TPosition::VECCALC, BUF_ID_SIZE> &bufPool, GM_ADDR xGradOut, GM_ADDR x,
                                     GM_ADDR weight, GM_ADDR bias, GM_ADDR mean, GM_ADDR rstd, GM_ADDR xGrad,
                                     GM_ADDR weightGrad, GM_ADDR biasGrad, uint32_t xSeq, uint32_t totalSeq,
                                     uint32_t outputSeq);

    __aicore__ inline void ProcessEncoderQK(TBufPool<TPosition::VECCALC, BUF_ID_SIZE> &bufPool, GM_ADDR xGradOut,
                                            GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR mean, GM_ADDR rstd,
                                            GM_ADDR xGrad, GM_ADDR weightGrad, GM_ADDR biasGrad, uint32_t xSeq,
                                            uint32_t totalSeq, uint32_t outputSeq);

    __aicore__ inline void ProcessV(TBufPool<TPosition::VECCALC, BUF_ID_SIZE> &bufPool, GM_ADDR x, GM_ADDR y,
                                    uint32_t xSeq, uint32_t totalSeq, uint32_t outputSeq);

    __aicore__ inline void GetSeqRange(uint32_t &startSeq, uint32_t &endSeq, uint32_t totalSeq, uint32_t usedCore);
    __aicore__ inline void PostProcess();
    __aicore__ inline void OneSumProcess(const GlobalTensor<float> &gradNormGm,
                                         const GlobalTensor<float> &workspaceOneGm, uint32_t seq);

private:
    TPipe *pipe_;
    uint32_t blockIdx_;
    uint32_t querySeq_;
    uint32_t keySeq_;
    uint32_t valueSeq_;
    uint32_t encoderQuerySeq_;
    uint32_t encoderKeySeq_;
    uint32_t encoderValueSeq_;
    uint32_t totalQuerySeq_;
    uint32_t totalKeySeq_;
    uint32_t totalValueSeq_;
    uint32_t ropeActualSeq_;
    uint32_t batch_;
    uint32_t headNum_;
    uint32_t headDim_;
    uint32_t usedCore_;
    uint32_t splitHeadNum_;
    uint32_t avgHeads_;
    uint32_t tailHeads_;
    uint32_t normDim_;
    uint32_t alignedNormDim_;
    uint32_t alignedNormNum_;
    uint32_t ropeDim_;
    uint32_t alignedRopeDim_;
    GM_ADDR gradQueryOutput_;
    GM_ADDR gradKeyOutput_;
    GM_ADDR gradValueOutput_;
    GM_ADDR query_;
    GM_ADDR key_;
    GM_ADDR encoderQuery_;
    GM_ADDR encoderKey_;
    GM_ADDR normQueryWeight_;
    GM_ADDR normQueryBias_;
    GM_ADDR normQueryMean_;
    GM_ADDR normQueryRstd_;
    GM_ADDR normKeyWeight_;
    GM_ADDR normKeyBias_;
    GM_ADDR normKeyMean_;
    GM_ADDR normKeyRstd_;
    GM_ADDR normAddedQueryWeight_;
    GM_ADDR normAddedQueryBias_;
    GM_ADDR normAddedQueryMean_;
    GM_ADDR normAddedQueryRstd_;
    GM_ADDR normAddedKeyWeight_;
    GM_ADDR normAddedKeyBias_;
    GM_ADDR normAddedKeyMean_;
    GM_ADDR normAddedKeyRstd_;
    GM_ADDR ropeSin_;
    GM_ADDR ropeCos_;
    GM_ADDR gradQuery_;
    GM_ADDR gradKey_;
    GM_ADDR gradValue_;
    GM_ADDR gradEncoderQuery_;
    GM_ADDR gradEncoderKey_;
    GM_ADDR gradEncoderValue_;
    GM_ADDR gradNormQueryWeight_;
    GM_ADDR gradNormQueryBias_;
    GM_ADDR gradNormKeyWeight_;
    GM_ADDR gradNormKeyBias_;
    GM_ADDR gradNormAddedQueryWeight_;
    GM_ADDR gradNormAddedQueryBias_;
    GM_ADDR gradNormAddedKeyWeight_;
    GM_ADDR gradNormAddedKeyBias_;
    GM_ADDR workspace_;
    float eps_, scale_;
    TBuf<TPosition::VECCALC> buf_; // buffer of norm output(rope input), [result | tmp]
    TBufPool<TPosition::VECCALC, BUF_ID_SIZE> queryPool_;
    TBufPool<TPosition::VECCALC, BUF_ID_SIZE> keyPool_;
    TBufPool<TPosition::VECCALC, BUF_ID_SIZE> valuePool_;
    TBufPool<TPosition::VECCALC, BUF_ID_SIZE> encoderQueryPool_;
    TBufPool<TPosition::VECCALC, BUF_ID_SIZE> encoderKeyPool_;
    TBufPool<TPosition::VECCALC, BUF_ID_SIZE> encoderValuePool_;
    TBufPool<TPosition::VECCALC, BUF_ID_SIZE> postPool_;

    GlobalTensor<DTYPE_QUERY> gradNormQueryWeightGM_;
    GlobalTensor<DTYPE_QUERY> gradNormQueryBiasGM_;
    GlobalTensor<DTYPE_QUERY> gradNormKeyWeightGM_;
    GlobalTensor<DTYPE_QUERY> gradNormKeyBiasGM_;
    GlobalTensor<DTYPE_QUERY> gradNormAddedQueryWeightGM_;
    GlobalTensor<DTYPE_QUERY> gradNormAddedQueryBiasGM_;
    GlobalTensor<DTYPE_QUERY> gradNormAddedKeyWeightGM_;
    GlobalTensor<DTYPE_QUERY> gradNormAddedKeyBiasGM_;
    GlobalTensor<float> workspaceGM_;
    GlobalTensor<float> workspaceAffineGM_;
    TQue<QuePosition::VECIN, SINGLE_BUFFER> workspaceGetQue_;
    TQue<QuePosition::VECOUT, SINGLE_BUFFER> workspacePushQue_;
    TBuf<QuePosition::VECCALC> workspaceBuf_;
    LocalTensor<float> workspaceShareBuf_;
    uint32_t workspaceId_ = 0;
    uint32_t affineBlockNums_ = 0;
    uint32_t avgHeadsLow_ = 2;
    uint64_t affineOffset_;
    DataCopyExtParams workspaceParams_{1, 0, 0, 0, 0};
    DataCopyPadExtParams<float> padWorkspaceParams_{true, 0, 0, 0};

    LocalTensor<float> x;
    LocalTensor<float> workspaceAffineGetBuf;
};

template <NormType normType, NormType addedNormType, RopeType ropeType, ConcatOrder concatOrder>
__aicore__ inline void NormRopeConcatGrad<normType, addedNormType, ropeType, concatOrder>::Init(
    TPipe *pipe, const NormRopeConcatGradTilingData &tilingData)
{
    pipe_ = pipe;
    blockIdx_ = GetBlockIdx();
    querySeq_ = tilingData.querySeq;
    keySeq_ = tilingData.keySeq;
    valueSeq_ = tilingData.valueSeq;
    encoderQuerySeq_ = tilingData.encoderQuerySeq;
    encoderKeySeq_ = tilingData.encoderKeySeq;
    encoderValueSeq_ = tilingData.encoderValueSeq;
    totalQuerySeq_ = tilingData.totalQuerySeq;
    totalKeySeq_ = tilingData.totalKeySeq;
    totalValueSeq_ = tilingData.totalValueSeq;
    batch_ = tilingData.batch;
    headNum_ = tilingData.headNum;
    headDim_ = tilingData.headDim;
    usedCore_ = tilingData.usedCore;
    splitHeadNum_ = tilingData.splitHeadNum;
    avgHeads_ = tilingData.avgHeads;
    tailHeads_ = tilingData.tailHeads;
    normDim_ = tilingData.normDim;
    ropeDim_ = tilingData.ropeDim;
    ropeActualSeq_ = tilingData.ropeActualSeq;
    eps_ = tilingData.eps;
    scale_ = tilingData.scale;
    alignedNormDim_ = CeilAlign(normDim_, static_cast<uint32_t>(ONE_BLK_SIZE / sizeof(DTYPE_QUERY)));
    alignedNormNum_ = CeilAlign(avgHeads_, static_cast<uint32_t>(ONE_BLK_SIZE / sizeof(float)));
    alignedRopeDim_ = CeilAlign(ropeDim_, static_cast<uint32_t>(ONE_BLK_SIZE / sizeof(DTYPE_QUERY)));
    uint32_t minUbAvgHeadsNum = Max(avgHeads_, MIN_UB_AVGHEADS_NUM);

    // Initialize buffer pools
    uint32_t bufSize = NUM_THREE * minUbAvgHeadsNum * alignedNormDim_ * sizeof(float) +
                       MIN_SHARE_BUFFER; //  grad_output, tmepND buf, shareCast buf, extra shareCast buf

    uint32_t poolSize =
        NUM_TWO * DOUBLE_BUFFER * minUbAvgHeadsNum * alignedNormDim_ * sizeof(DTYPE_QUERY); // grad_output, gradIn Queue
    if constexpr (normType != NormType::NONE || addedNormType != NormType::NONE) {
        poolSize += (DOUBLE_BUFFER * minUbAvgHeadsNum * alignedNormDim_ * sizeof(DTYPE_QUERY) +
                     minUbAvgHeadsNum * alignedNormDim_ * sizeof(float));                  // input Queue & buf
        poolSize += (NUM_TWO * DOUBLE_BUFFER + NUM_TWO) * alignedNormNum_ * sizeof(float); // mean, rstd Queue & buf
        poolSize += minUbAvgHeadsNum * alignedNormDim_ * sizeof(float);                    // share buf

        if constexpr (normType == NormType::LAYER_NORM_AFFINE || normType == NormType::LAYER_NORM_AFFINE_ACROSS_HEADS ||
                      addedNormType == NormType::LAYER_NORM_AFFINE ||
                      addedNormType == NormType::LAYER_NORM_AFFINE_ACROSS_HEADS) {
            poolSize += SINGLE_BUFFER * alignedNormDim_ * sizeof(DTYPE_QUERY) +
                        minUbAvgHeadsNum * alignedNormDim_ * sizeof(float) // weight Queue & buf
                        + NUM_TWO * DOUBLE_BUFFER * alignedNormDim_ * sizeof(float) +
                        NUM_TWO * alignedNormDim_ * sizeof(float); // gradWeight, gradBias Queue & buf
        }
    }
    if constexpr (ropeType != RopeType::NONE) {
        poolSize += SINGLE_BUFFER * alignedRopeDim_ * NUM_TWO * sizeof(DTYPE_ROPE_SIN); // sin & cos
        poolSize +=
            alignedRopeDim_ * sizeof(int32_t) + alignedRopeDim_ * NUM_TWO * sizeof(float); // mask, sin, cos fp32
    }

    poolSize = Max(poolSize, static_cast<uint32_t>(NUM_THREE * TOTAL_NORM_NUM * alignedNormDim_ * sizeof(float)));

    pipe_->InitBuffer(buf_, bufSize);
    pipe_->InitBufPool(queryPool_, poolSize);
    pipe_->InitBufPool(keyPool_, poolSize, queryPool_);
    pipe_->InitBufPool(valuePool_, poolSize, queryPool_);
    pipe_->InitBufPool(encoderQueryPool_, poolSize, queryPool_);
    pipe_->InitBufPool(encoderKeyPool_, poolSize, queryPool_);
    pipe_->InitBufPool(encoderValuePool_, poolSize, queryPool_);
    pipe_->InitBufPool(postPool_, poolSize, queryPool_);

    uint32_t maxAffineSeq = 0;
    if constexpr (normType == NormType::LAYER_NORM_AFFINE || normType == NormType::LAYER_NORM_AFFINE_ACROSS_HEADS) {
        gradNormQueryWeightGM_.SetGlobalBuffer((__gm__ DTYPE_QUERY *)gradNormQueryWeight_, normDim_);
        gradNormQueryBiasGM_.SetGlobalBuffer((__gm__ DTYPE_QUERY *)gradNormQueryBias_, normDim_);
        gradNormKeyWeightGM_.SetGlobalBuffer((__gm__ DTYPE_QUERY *)gradNormKeyWeight_, normDim_);
        gradNormKeyBiasGM_.SetGlobalBuffer((__gm__ DTYPE_QUERY *)gradNormKeyBias_, normDim_);
        maxAffineSeq = Max(querySeq_, keySeq_);
    }
    if constexpr (addedNormType == NormType::LAYER_NORM_AFFINE ||
                  addedNormType == NormType::LAYER_NORM_AFFINE_ACROSS_HEADS) {
        gradNormAddedQueryWeightGM_.SetGlobalBuffer((__gm__ DTYPE_QUERY *)gradNormAddedQueryWeight_, normDim_);
        gradNormAddedQueryBiasGM_.SetGlobalBuffer((__gm__ DTYPE_QUERY *)gradNormAddedQueryBias_, normDim_);
        gradNormAddedKeyWeightGM_.SetGlobalBuffer((__gm__ DTYPE_QUERY *)gradNormAddedKeyWeight_, normDim_);
        gradNormAddedKeyBiasGM_.SetGlobalBuffer((__gm__ DTYPE_QUERY *)gradNormAddedKeyBias_, normDim_);
        maxAffineSeq = Max(maxAffineSeq, Max(encoderQuerySeq_, encoderValueSeq_));
    }

    if constexpr (normType == NormType::LAYER_NORM_AFFINE || normType == NormType::LAYER_NORM_AFFINE_ACROSS_HEADS ||
                  addedNormType == NormType::LAYER_NORM_AFFINE ||
                  addedNormType == NormType::LAYER_NORM_AFFINE_ACROSS_HEADS) {
        while (avgHeadsLow_ <= minUbAvgHeadsNum / 2) {
            avgHeadsLow_ <<= 1;
        }

        uint64_t headDimNums =
            static_cast<uint64_t>(batch_) * ((maxAffineSeq + GetBlockNum() - 1) / GetBlockNum()) * splitHeadNum_;
        uint64_t current = 1;
        while (current < headDimNums && avgHeadsLow_ != 1) {
            current *= avgHeadsLow_;
            affineBlockNums_++;
        }
        affineBlockNums_ = affineBlockNums_ == 0 ? 1 : affineBlockNums_;

        workspaceGM_.SetGlobalBuffer((__gm__ float *)workspace_,
                                     TOTAL_NORM_NUM * usedCore_ * alignedNormDim_ +
                                         affineBlockNums_ * usedCore_ * avgHeadsLow_ * NUM_TWO * alignedNormDim_);
        workspaceAffineGM_ = workspaceGM_[TOTAL_NORM_NUM * usedCore_ * alignedNormDim_];
    }
}

template <NormType normType, NormType addedNormType, RopeType ropeType, ConcatOrder concatOrder>
__aicore__ inline void NormRopeConcatGrad<normType, addedNormType, ropeType, concatOrder>::GetSeqRange(
    uint32_t &startSeq, uint32_t &endSeq, uint32_t totalSeq, uint32_t usedCore)
{
    uint32_t eachCoreSeq = totalSeq / usedCore;
    uint32_t tailCoreSeq = totalSeq % usedCore;
    if (blockIdx_ < Min(usedCore, totalSeq)) {
        if (blockIdx_ < tailCoreSeq) {
            startSeq = blockIdx_ * (eachCoreSeq + 1);
            endSeq = startSeq + eachCoreSeq + 1;
        } else {
            startSeq = blockIdx_ * eachCoreSeq + tailCoreSeq;
            endSeq = startSeq + eachCoreSeq;
        }
    } else {
        startSeq = totalSeq;
        endSeq = startSeq;
    }
}

template <NormType normType, NormType addedNormType, RopeType ropeType, ConcatOrder concatOrder>
__aicore__ inline void NormRopeConcatGrad<normType, addedNormType, ropeType, concatOrder>::Prepare()
{
    if constexpr (normType == NormType::NONE && addedNormType == NormType::NONE) {
        return;
    }

    if (blockIdx_ == 0) {
        // gm grad weith&bias zero init
        if constexpr (normType == NormType::LAYER_NORM_AFFINE || normType == NormType::LAYER_NORM_AFFINE_ACROSS_HEADS) {
            InitGlobalMemory(gradNormQueryWeightGM_, this->normDim_, static_cast<DTYPE_QUERY>(0.0f));
            InitGlobalMemory(gradNormQueryBiasGM_, this->normDim_, static_cast<DTYPE_QUERY>(0.0f));
            InitGlobalMemory(gradNormKeyWeightGM_, this->normDim_, static_cast<DTYPE_QUERY>(0.0f));
            InitGlobalMemory(gradNormKeyBiasGM_, this->normDim_, static_cast<DTYPE_QUERY>(0.0f));
        }

        if constexpr (addedNormType == NormType::LAYER_NORM_AFFINE ||
                      addedNormType == NormType::LAYER_NORM_AFFINE_ACROSS_HEADS) {
            InitGlobalMemory(gradNormAddedQueryWeightGM_, this->normDim_, static_cast<DTYPE_QUERY>(0.0f));
            InitGlobalMemory(gradNormAddedQueryBiasGM_, this->normDim_, static_cast<DTYPE_QUERY>(0.0f));
            InitGlobalMemory(gradNormAddedKeyWeightGM_, this->normDim_, static_cast<DTYPE_QUERY>(0.0f));
            InitGlobalMemory(gradNormAddedKeyBiasGM_, this->normDim_, static_cast<DTYPE_QUERY>(0.0f));
        }

        if constexpr (normType == NormType::LAYER_NORM_AFFINE || normType == NormType::LAYER_NORM_AFFINE_ACROSS_HEADS ||
                      addedNormType == NormType::LAYER_NORM_AFFINE ||
                      addedNormType == NormType::LAYER_NORM_AFFINE_ACROSS_HEADS) {
            InitGlobalMemory(workspaceGM_,
                             TOTAL_NORM_NUM * usedCore_ * alignedNormDim_ +
                                 affineBlockNums_ * usedCore_ * avgHeadsLow_ * NUM_TWO * alignedNormDim_,
                             static_cast<float>(0.0f));
        }
    }
    SyncAll();
}

template <NormType normType, NormType addedNormType, RopeType ropeType, ConcatOrder concatOrder>
__aicore__ inline void NormRopeConcatGrad<normType, addedNormType, ropeType, concatOrder>::Process()
{
    Prepare();
    ProcessQuery();
    ProcessKey();
    ProcessValue();
    PostProcess();
}

template <NormType normType, NormType addedNormType, RopeType ropeType, ConcatOrder concatOrder>
__aicore__ inline void NormRopeConcatGrad<normType, addedNormType, ropeType, concatOrder>::ProcessQuery()
{
    uint32_t outputQuerySeq = 0;
    uint32_t outputEncoderSeq = querySeq_;
    if constexpr (concatOrder == ConcatOrder::AFTER_ENCODER) {
        outputQuerySeq = encoderQuerySeq_;
        outputEncoderSeq = 0;
    }
    ProcessQK(queryPool_, gradQueryOutput_, query_, normQueryWeight_, normQueryBias_, normQueryMean_, normQueryRstd_,
              gradQuery_, gradNormQueryWeight_, gradNormQueryBias_, querySeq_, totalQuerySeq_, outputQuerySeq);
    ProcessEncoderQK(encoderQueryPool_, gradQueryOutput_, encoderQuery_, normAddedQueryWeight_, normAddedQueryBias_,
                     normAddedQueryMean_, normAddedQueryRstd_, gradEncoderQuery_, gradNormAddedQueryWeight_,
                     gradNormAddedQueryBias_, encoderQuerySeq_, totalQuerySeq_, outputEncoderSeq);
}

template <NormType normType, NormType addedNormType, RopeType ropeType, ConcatOrder concatOrder>
__aicore__ inline void NormRopeConcatGrad<normType, addedNormType, ropeType, concatOrder>::ProcessKey()
{
    uint32_t outputKeySeq = 0;
    uint32_t outputEncoderSeq = keySeq_;
    if constexpr (concatOrder == ConcatOrder::AFTER_ENCODER) {
        outputKeySeq = encoderKeySeq_;
        outputEncoderSeq = 0;
    }
    ProcessQK(keyPool_, gradKeyOutput_, key_, normKeyWeight_, normKeyBias_, normKeyMean_, normKeyRstd_, gradKey_,
              gradNormKeyWeight_, gradNormKeyBias_, keySeq_, totalKeySeq_, outputKeySeq);
    ProcessEncoderQK(encoderKeyPool_, gradKeyOutput_, encoderKey_, normAddedKeyWeight_, normAddedKeyBias_,
                     normAddedKeyMean_, normAddedKeyRstd_, gradEncoderKey_, gradNormAddedKeyWeight_,
                     gradNormAddedKeyBias_, encoderKeySeq_, totalKeySeq_, outputEncoderSeq);
}

template <NormType normType, NormType addedNormType, RopeType ropeType, ConcatOrder concatOrder>
__aicore__ inline void NormRopeConcatGrad<normType, addedNormType, ropeType, concatOrder>::ProcessValue()
{
    uint32_t outputValueSeq = 0;
    uint32_t outputEncoderSeq = valueSeq_;
    if constexpr (concatOrder == ConcatOrder::AFTER_ENCODER) {
        outputValueSeq = encoderValueSeq_;
        outputEncoderSeq = 0;
    }
    ProcessV(valuePool_, gradValueOutput_, gradValue_, valueSeq_, totalValueSeq_, outputValueSeq);
    ProcessV(encoderValuePool_, gradValueOutput_, gradEncoderValue_, encoderValueSeq_, totalValueSeq_,
             outputEncoderSeq);
}

template <NormType normType, NormType addedNormType, RopeType ropeType, ConcatOrder concatOrder>
__aicore__ inline void NormRopeConcatGrad<normType, addedNormType, ropeType, concatOrder>::ProcessQK(
    TBufPool<TPosition::VECCALC, BUF_ID_SIZE> &bufPool, GM_ADDR xGradOut, GM_ADDR x, GM_ADDR weight, GM_ADDR bias,
    GM_ADDR mean, GM_ADDR rstd, GM_ADDR xGrad, GM_ADDR weightGrad, GM_ADDR biasGrad, uint32_t xSeq, uint32_t totalSeq,
    uint32_t outputSeq)
{
    uint32_t startSeq, endSeq;
    GetSeqRange(startSeq, endSeq, xSeq, usedCore_);
    if (startSeq < endSeq) {
        RopeOperationBackward<ropeType> ropeOp(bufPool, xGradOut, ropeSin_, ropeCos_, nullptr, ropeActualSeq_, totalSeq,
                                               startSeq, ropeDim_, avgHeads_, alignedRopeDim_);
        NormOperationBackward<normType> normOp(bufPool, x, weight, nullptr, mean, rstd, xGrad, weightGrad, biasGrad,
                                               eps_, scale_, normDim_, avgHeads_, alignedNormDim_, alignedNormNum_,
                                               affineBlockNums_, avgHeadsLow_);
        ropeOp.Prepare(startSeq);
        normOp.Prepare();
        for (uint32_t s = startSeq; s < endSeq; ++s) {
            ropeOp.PreProcess(outputSeq + s);
            for (uint32_t b = 0; b < batch_; ++b) {
                uint64_t outGradOffset = static_cast<uint64_t>(b) * headNum_ * totalSeq * headDim_ +
                                         static_cast<uint64_t>(outputSeq + s) * headDim_;
                uint64_t inOffset = static_cast<uint64_t>(b * xSeq + s) * headNum_ * headDim_;
                uint64_t normOffset = static_cast<uint64_t>(b * xSeq + s) * headNum_;
                for (uint32_t n = 0; n < splitHeadNum_ - 1; ++n) {
                    LocalTensor<float> buf = buf_.Get<float>();
                    ropeOp.Process(buf, outGradOffset, avgHeads_);
                    normOp.Process(buf, workspaceAffineGM_, inOffset, normOffset, avgHeads_);
                    outGradOffset += static_cast<uint64_t>(avgHeads_) * totalSeq * headDim_;
                    inOffset += static_cast<uint64_t>(avgHeads_) * headDim_;
                    normOffset += avgHeads_;
                }
                // tail
                LocalTensor<float> buf = buf_.Get<float>();
                ropeOp.Process(buf, outGradOffset, tailHeads_);
                normOp.Process(buf, workspaceAffineGM_, inOffset, normOffset, tailHeads_);
                outGradOffset += static_cast<uint64_t>(tailHeads_) * totalSeq * headDim_;
                inOffset += static_cast<uint64_t>(tailHeads_) * headDim_;
                normOffset += tailHeads_;
            }
        }
        LocalTensor<float> buf = buf_.Get<float>();
        normOp.PostProcess(buf, workspaceAffineGM_, workspaceGM_, workspaceId_);
        PipeBarrier<PIPE_ALL>();
        bufPool.Reset();
    }
    workspaceId_ += 1;
}

template <NormType normType, NormType addedNormType, RopeType ropeType, ConcatOrder concatOrder>
__aicore__ inline void NormRopeConcatGrad<normType, addedNormType, ropeType, concatOrder>::ProcessEncoderQK(
    TBufPool<TPosition::VECCALC, BUF_ID_SIZE> &bufPool, GM_ADDR xGradOut, GM_ADDR x, GM_ADDR weight, GM_ADDR bias,
    GM_ADDR mean, GM_ADDR rstd, GM_ADDR xGrad, GM_ADDR weightGrad, GM_ADDR biasGrad, uint32_t xSeq, uint32_t totalSeq,
    uint32_t outputSeq)
{
    uint32_t startSeq, endSeq;
    GetSeqRange(startSeq, endSeq, xSeq, usedCore_);
    if (startSeq < endSeq) {
        RopeOperationBackward<ropeType> ropeOp(bufPool, xGradOut, ropeSin_, ropeCos_, nullptr, ropeActualSeq_, totalSeq,
                                               startSeq, ropeDim_, avgHeads_, alignedRopeDim_);
        NormOperationBackward<addedNormType> normOp(bufPool, x, weight, nullptr, mean, rstd, xGrad, weightGrad,
                                                    biasGrad, eps_, scale_, normDim_, avgHeads_, alignedNormDim_,
                                                    alignedNormNum_, affineBlockNums_, avgHeadsLow_);
        ropeOp.Prepare(startSeq);
        normOp.Prepare();
        for (uint32_t s = startSeq; s < endSeq; ++s) {
            ropeOp.PreProcess(outputSeq + s);
            for (uint32_t b = 0; b < batch_; ++b) {
                uint64_t outGradOffset = static_cast<uint64_t>(b) * headNum_ * totalSeq * headDim_ +
                                         static_cast<uint64_t>(outputSeq + s) * headDim_;
                uint64_t inOffset = static_cast<uint64_t>(b * xSeq + s) * headNum_ * headDim_;
                uint64_t normOffset = static_cast<uint64_t>(b * xSeq + s) * headNum_;
                for (uint32_t n = 0; n < splitHeadNum_ - 1; ++n) {
                    LocalTensor<float> buf = buf_.Get<float>();
                    ropeOp.Process(buf, outGradOffset, avgHeads_);
                    normOp.Process(buf, workspaceAffineGM_, inOffset, normOffset, avgHeads_);
                    outGradOffset += static_cast<uint64_t>(avgHeads_) * totalSeq * headDim_;
                    inOffset += static_cast<uint64_t>(avgHeads_) * headDim_;
                    normOffset += avgHeads_;
                }
                // tail
                LocalTensor<float> buf = buf_.Get<float>();
                ropeOp.Process(buf, outGradOffset, tailHeads_);
                normOp.Process(buf, workspaceAffineGM_, inOffset, normOffset, tailHeads_);
                outGradOffset += static_cast<uint64_t>(tailHeads_) * totalSeq * headDim_;
                inOffset += static_cast<uint64_t>(tailHeads_) * headDim_;
                normOffset += tailHeads_;
            }
        }
        LocalTensor<float> buf = buf_.Get<float>();
        normOp.PostProcess(buf, workspaceAffineGM_, workspaceGM_, workspaceId_);
        PipeBarrier<PIPE_ALL>();
        bufPool.Reset();
    }
    workspaceId_ += 1;
}

template <NormType normType, NormType addedNormType, RopeType ropeType, ConcatOrder concatOrder>
__aicore__ inline void NormRopeConcatGrad<normType, addedNormType, ropeType, concatOrder>::ProcessV(
    TBufPool<TPosition::VECCALC, BUF_ID_SIZE> &bufPool, GM_ADDR xGradOut, GM_ADDR xGrad, uint32_t xSeq,
    uint32_t totalSeq, uint32_t outputSeq)
{
    uint32_t startSeq, endSeq;
    GetSeqRange(startSeq, endSeq, xSeq, usedCore_);
    if (startSeq < endSeq) {
        RopeOperationBackward<RopeType::NONE> ropeOp(bufPool, xGradOut, ropeSin_, ropeCos_, nullptr, ropeActualSeq_,
                                                     totalSeq, startSeq, ropeDim_, avgHeads_, alignedRopeDim_);

        NormOperationBackward<NormType::NONE> normOp(bufPool, nullptr, nullptr, nullptr, nullptr, nullptr, xGrad,
                                                     nullptr, nullptr, eps_, scale_, normDim_, avgHeads_,
                                                     alignedNormDim_, alignedNormNum_, affineBlockNums_, avgHeadsLow_);
        ropeOp.Prepare(startSeq);
        normOp.Prepare();
        for (uint32_t s = startSeq; s < endSeq; ++s) {
            ropeOp.PreProcess(s);
            for (uint32_t b = 0; b < batch_; ++b) {
                uint64_t outGradOffset = static_cast<uint64_t>(b) * headNum_ * totalSeq * headDim_ +
                                         static_cast<uint64_t>(outputSeq + s) * headDim_;
                uint64_t inOffset = static_cast<uint64_t>(b * xSeq + s) * headNum_ * headDim_;
                uint64_t normOffset = static_cast<uint64_t>(b * xSeq + s) * headNum_;
                for (uint32_t n = 0; n < splitHeadNum_ - 1; ++n) {
                    LocalTensor<float> buf = buf_.Get<float>();
                    ropeOp.Process(buf, outGradOffset, avgHeads_);
                    normOp.Process(buf, workspaceAffineGM_, inOffset, normOffset, avgHeads_);
                    outGradOffset += static_cast<uint64_t>(avgHeads_) * totalSeq * headDim_;
                    inOffset += static_cast<uint64_t>(avgHeads_) * headDim_;
                    normOffset += avgHeads_;
                }
                // tail
                LocalTensor<float> buf = buf_.Get<float>();
                ropeOp.Process(buf, outGradOffset, tailHeads_);
                normOp.Process(buf, workspaceAffineGM_, inOffset, normOffset, tailHeads_);
                outGradOffset += static_cast<uint64_t>(tailHeads_) * totalSeq * headDim_;
                inOffset += static_cast<uint64_t>(tailHeads_) * headDim_;
                normOffset += tailHeads_;
            }
        }
        PipeBarrier<PIPE_ALL>();
        bufPool.Reset();
    }
}

template <NormType normType, NormType addedNormType, RopeType ropeType, ConcatOrder concatOrder>
__aicore__ inline void NormRopeConcatGrad<normType, addedNormType, ropeType, concatOrder>::PostProcess()
{
    if constexpr (normType != NormType::LAYER_NORM_AFFINE && normType != NormType::LAYER_NORM_AFFINE_ACROSS_HEADS &&
                  addedNormType != NormType::LAYER_NORM_AFFINE &&
                  addedNormType != NormType::LAYER_NORM_AFFINE_ACROSS_HEADS) {
        return;
    }
    SyncAll();

    postPool_.InitBuffer(workspaceGetQue_, SINGLE_BUFFER, TOTAL_NORM_NUM * alignedNormDim_ * sizeof(float));
    postPool_.InitBuffer(workspacePushQue_, SINGLE_BUFFER, TOTAL_NORM_NUM * alignedNormDim_ * sizeof(DTYPE_QUERY));
    postPool_.InitBuffer(workspaceBuf_, TOTAL_NORM_NUM * alignedNormDim_ * sizeof(float));
    workspaceShareBuf_ = workspaceBuf_.Get<float>(); //[headdim]

    if (blockIdx_ != 0) {
        return;
    }

    Duplicate(workspaceShareBuf_, 0.0f, TOTAL_NORM_NUM * alignedNormDim_);
    PipeBarrier<PIPE_V>();
    workspaceParams_.blockLen = TOTAL_NORM_NUM * alignedNormDim_ * sizeof(float);
    for (int i = 0; i < usedCore_; i++) {
        LocalTensor<float> workspaceGetBuf = workspaceGetQue_.AllocTensor<float>();
        DataCopyPad(workspaceGetBuf, workspaceGM_[i * TOTAL_NORM_NUM * alignedNormDim_], workspaceParams_,
                    padWorkspaceParams_);
        workspaceGetQue_.EnQue(workspaceGetBuf);

        LocalTensor<float> workspaceOne2 = workspaceGetQue_.DeQue<float>();
        Add(workspaceShareBuf_, workspaceShareBuf_, workspaceOne2, TOTAL_NORM_NUM * alignedNormDim_);
        PipeBarrier<PIPE_V>();
        workspaceGetQue_.FreeTensor(workspaceOne2);
    }

    // copy to weight & bias gm
    LocalTensor<DTYPE_QUERY> workspacePushBuf = workspacePushQue_.AllocTensor<DTYPE_QUERY>();
    if (sizeof(DTYPE_QUERY) == sizeof(float)) {
        Adds(workspacePushBuf.ReinterpretCast<float>(), workspaceShareBuf_, 0.0f, TOTAL_NORM_NUM * alignedNormDim_);
    } else {
        Cast(workspacePushBuf, workspaceShareBuf_, RoundMode::CAST_ROUND, TOTAL_NORM_NUM * alignedNormDim_);
    }
    PipeBarrier<PIPE_V>();
    workspacePushQue_.EnQue(workspacePushBuf);

    LocalTensor<DTYPE_QUERY> workspacePushBuf2 = workspacePushQue_.DeQue<DTYPE_QUERY>();
    workspaceParams_.blockLen = normDim_ * sizeof(DTYPE_QUERY);
    if constexpr (normType == NormType::LAYER_NORM_AFFINE || normType == NormType::LAYER_NORM_AFFINE_ACROSS_HEADS) {
        DataCopyPad(gradNormQueryWeightGM_, workspacePushBuf2, workspaceParams_);
        DataCopyPad(gradNormQueryBiasGM_, workspacePushBuf2[alignedNormDim_], workspaceParams_);
        DataCopyPad(gradNormKeyWeightGM_, workspacePushBuf2[4 * alignedNormDim_], workspaceParams_);
        DataCopyPad(gradNormKeyBiasGM_, workspacePushBuf2[5 * alignedNormDim_], workspaceParams_);
    }

    if constexpr (addedNormType == NormType::LAYER_NORM_AFFINE ||
                  addedNormType == NormType::LAYER_NORM_AFFINE_ACROSS_HEADS) {
        DataCopyPad(gradNormAddedQueryWeightGM_, workspacePushBuf2[2 * alignedNormDim_], workspaceParams_);
        DataCopyPad(gradNormAddedQueryBiasGM_, workspacePushBuf2[3 * alignedNormDim_], workspaceParams_);
        DataCopyPad(gradNormAddedKeyWeightGM_, workspacePushBuf2[6 * alignedNormDim_], workspaceParams_);
        DataCopyPad(gradNormAddedKeyBiasGM_, workspacePushBuf2[7 * alignedNormDim_], workspaceParams_);
    }
    workspacePushQue_.FreeTensor(workspacePushBuf2);
}
} // namespace nrcg
#endif // NORM_ROPE_CONCAT_GRAD_H
