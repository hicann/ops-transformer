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
 * \file add_rms_norm_merge_n.h
 * \brief
 */
#ifndef MC2_ADD_RMS_NORM_MERGE_N_H
#define MC2_ADD_RMS_NORM_MERGE_N_H
#include "mararn_rms_norm_base.h"
#include "adv_api/pad/broadcast.h"
#include "adv_api/quantization/ascend_dequant.h"

using namespace AscendC;

constexpr uint32_t DOUBLE_BUFFER_QUEUE = 2;
template <typename T>
class KernelAddRmsNormMergeN {
public:
    __aicore__ inline KernelAddRmsNormMergeN() {}
    __aicore__ inline void Init(GM_ADDR gammaGM, Mc2Tiling::AddRMSNormTilingData &tiling, TPipe *pipe,
                                uint32_t numBlocks)
    {
        ASSERT(numBlocks != 0 && "Block dim can not be zero!");
        this->numBlocks_ = numBlocks;
        this->numRow_ = tiling.num_row;
        this->numCol_ = tiling.num_col;
        uint32_t numPerBlock = ONE_BLK_SIZE / sizeof(T);
        this->numColAlign_ = AlignUp(numCol_, numPerBlock);
        this->blockFactor_ = tiling.block_factor;
        this->rowFactor_ = tiling.row_factor;
        this->ubFactor_ = tiling.ub_factor;
        this->epsilon_ = tiling.epsilon;
        this->avgFactor_ = (numCol_ != 0) ? (float)1.0 / numCol_ : 0;

        if (GetBlockIdx() < numBlocks_ - 1) {
            this->rowWork_ = blockFactor_;
        } else {
            this->rowWork_ = numRow_ - (numBlocks_ - 1) * blockFactor_;
        }
        this->gmBlockOffset_ = GetBlockIdx() * blockFactor_ * numCol_;
        this->gmBlockSize_ = rowWork_ * numCol_;

        // get start index for current core, core parallel
        gamma_.SetGlobalBuffer((__gm__ T *)gammaGM, numCol_);

        // pipe alloc memory to queue, the unit is Bytes
        pipe->InitBuffer(inQueueX_, DOUBLE_BUFFER_QUEUE, ubFactor_ * sizeof(T));
        pipe->InitBuffer(inQueueGamma_, 1, ubFactor_ * sizeof(T));
        pipe->InitBuffer(outQueueY_, DOUBLE_BUFFER_QUEUE, ubFactor_ * sizeof(T));
        pipe->InitBuffer(outQueueRstd_, 1, rowFactor_ * sizeof(float));
        pipe->InitBuffer(xFp32Buf_, ubFactor_ * sizeof(float));
        pipe->InitBuffer(sqxBuf_, ubFactor_ * sizeof(float));
        pipe->InitBuffer(tmpBuf_, rowFactor_ * NUM_PER_REP_FP32 * sizeof(float));
    }

    __aicore__ inline void ComputeProcess(GM_ADDR normOutGM, GM_ADDR residualGM, GM_ADDR yGM,
                                          Mc2Tiling::AddRMSNormTilingData &tilingData, uint32_t addRmsNormCount,
                                          uint32_t rcvCnt)
    {
        uint64_t cOffset = CalcShapeOffset(sizeof(T), tilingData.num_row, tilingData.num_col); // 偏移*size
        for (; addRmsNormCount <= rcvCnt; ++addRmsNormCount) {
            normOut_.SetGlobalBuffer((__gm__ T *)normOutGM + gmBlockOffset_, gmBlockSize_);
            residual_.SetGlobalBuffer((__gm__ T *)residualGM + gmBlockOffset_, gmBlockSize_);
            y_.SetGlobalBuffer((__gm__ T *)yGM + gmBlockOffset_, gmBlockSize_);
            Process();
            normOutGM += cOffset;
            residualGM += cOffset;
            yGM += cOffset;
        }
    }

    __aicore__ inline void Process()
    {
        CopyInGamma();
        LocalTensor<T> gammaLocal = inQueueGamma_.DeQue<T>();
        BroadCastGamma(gammaLocal);
        uint32_t i_o_max = CeilDiv(rowWork_, rowFactor_);
        uint32_t row_tail = rowWork_ - (i_o_max - 1) * rowFactor_;

        for (uint32_t i_o = 0; i_o < i_o_max - 1; i_o++) {
            SubProcess(i_o, rowFactor_, gammaLocal);
        }
        SubProcess(i_o_max - 1, row_tail, gammaLocal);
        inQueueGamma_.FreeTensor(gammaLocal);
    }

    __aicore__ inline void SubProcess(uint32_t i_o, uint32_t calc_row_num, LocalTensor<T> &gammaLocal)
    {
        uint32_t gm_bias = i_o * rowFactor_ * numCol_;
        uint32_t elementNum = calc_row_num * numColAlign_;
        CopyInX(gm_bias, calc_row_num);
        LocalTensor<T> xLocal = ComputeX(elementNum);
        CopyOutX(gm_bias, calc_row_num);
        LocalTensor<float> rstdLocal = outQueueRstd_.AllocTensor<float>();
        ComputeRstd(xLocal, rstdLocal, calc_row_num);
        outQueueRstd_.EnQue<float>(rstdLocal);
        CopyOutRstd(i_o, calc_row_num);
        ComputeY(xLocal, gammaLocal, rstdLocal, calc_row_num);
        CopyOutY(gm_bias, calc_row_num);
    }

private:
    __aicore__ inline void CopyInX(uint32_t gm_bias, uint32_t calc_row_num)
    {
        LocalTensor<T> x1Local = inQueueX_.AllocTensor<T>();
        if (isNumColAlign_) {
            DataCopyCustom<T>(x1Local, normOut_[gm_bias], calc_row_num * numCol_);
        } else {
            DataCopyCustom<T>(x1Local, normOut_[gm_bias], calc_row_num, numCol_);
        }
        inQueueX_.EnQue(x1Local);
        LocalTensor<T> x2Local = inQueueX_.AllocTensor<T>();
        if (isNumColAlign_) {
            DataCopyCustom<T>(x2Local, residual_[gm_bias], calc_row_num * numCol_);
        } else {
            DataCopyCustom<T>(x2Local, residual_[gm_bias], calc_row_num, numCol_);
        }
        inQueueX_.EnQue(x2Local);
    }

    __aicore__ inline LocalTensor<T> ComputeX(uint32_t elementNum)
    {
        LocalTensor<T> x1Local = inQueueX_.DeQue<T>();
        LocalTensor<T> x2Local = inQueueX_.DeQue<T>();
        LocalTensor<T> xLocal = outQueueY_.AllocTensor<T>();
        if constexpr (!IsSameType<T, bfloat16_t>::value) {
            Add(xLocal, x1Local, x2Local, elementNum);
        } else {
            LocalTensor<float> x1Fp32 = xFp32Buf_.Get<float>();
            LocalTensor<float> x2Fp32 = sqxBuf_.Get<float>();
            Cast(x1Fp32, x1Local, RoundMode::CAST_NONE, elementNum);
            Cast(x2Fp32, x2Local, RoundMode::CAST_NONE, elementNum);
            PipeBarrier<PIPE_V>();
            Add(x1Fp32, x1Fp32, x2Fp32, elementNum);
            PipeBarrier<PIPE_V>();
            Cast(xLocal, x1Fp32, RoundMode::CAST_RINT, elementNum);
        }
        inQueueX_.FreeTensor(x1Local);
        inQueueX_.FreeTensor(x2Local);
        outQueueY_.EnQue(xLocal);
        PipeBarrier<PIPE_V>();
        return xLocal;
    }

    __aicore__ inline void CopyOutX(uint32_t gm_bias, uint32_t calc_row_num)
    {
        // CopyOut x1 + x2
        auto xOut = outQueueY_.DeQue<T>();
        if (isNumColAlign_) {
            DataCopyCustom<T>(y_[gm_bias], xOut, calc_row_num * numCol_);
        } else {
            DataCopyCustom<T>(y_[gm_bias], xOut, calc_row_num, numCol_);
        }
        outQueueY_.FreeTensor(xOut);
    }

    __aicore__ inline void CopyInGamma()
    {
        LocalTensor<T> gammaLocal = inQueueGamma_.AllocTensor<T>();
        DataCopyCustom<T>(gammaLocal, gamma_, numCol_);
        inQueueGamma_.EnQue(gammaLocal);
    }

    __aicore__ inline void BroadCastGamma(LocalTensor<T> &gammaLocal)
    {
        const uint32_t srcShape[2] = {1, numColAlign_};
        const uint32_t dstShape[2] = {rowFactor_, numColAlign_};
        LocalTensor<uint8_t> tmpLocal = tmpBuf_.Get<uint8_t>();
        if constexpr (IsSameType<T, bfloat16_t>::value) {
            LocalTensor<half> interpreLocal = gammaLocal.template ReinterpretCast<half>();
            BroadCast<half, DIM_NUM, 0>(interpreLocal, interpreLocal, dstShape, srcShape, tmpLocal);
        } else {
            BroadCast<T, DIM_NUM, 0>(gammaLocal, gammaLocal, dstShape, srcShape, tmpLocal);
        }
    }

    __aicore__ inline void ComputeRstd(LocalTensor<T> xLocal, LocalTensor<float> rstdLocal, uint32_t calc_row_num)
    {
        uint32_t elementNum = calc_row_num * numColAlign_;
        LocalTensor<float> sqx = sqxBuf_.Get<float>();
        LocalTensor<float> tmpLocal = tmpBuf_.Get<float>();
        LocalTensor<float> x_fp32 = xFp32Buf_.Get<float>();
        Cast(x_fp32, xLocal, RoundMode::CAST_NONE, elementNum);
        PipeBarrier<PIPE_V>();
        Mul(sqx, x_fp32, x_fp32, elementNum);
        PipeBarrier<PIPE_V>();

        Muls(sqx, sqx, avgFactor_, elementNum);
        PipeBarrier<PIPE_V>();

        ReduceSumMultiN(rstdLocal, sqx, tmpLocal, calc_row_num, numCol_, numColAlign_);
        PipeBarrier<PIPE_V>();
        Adds(rstdLocal, rstdLocal, epsilon_, calc_row_num);
        PipeBarrier<PIPE_V>();

        Sqrt(rstdLocal, rstdLocal, calc_row_num);
        Duplicate(tmpLocal, (float)1.0, calc_row_num);
        PipeBarrier<PIPE_V>();

        Div(rstdLocal, tmpLocal, rstdLocal, calc_row_num);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ComputeY(LocalTensor<T> xLocal, LocalTensor<T> gammaLocal, LocalTensor<float> rstdLocal,
                                    uint32_t calc_row_num)
    {
        uint32_t elementNum = calc_row_num * numColAlign_;
        LocalTensor<float> sqx = sqxBuf_.Get<float>();
        auto sharedTmpLocal = tmpBuf_.Get<uint8_t>();
        const uint32_t srcShape[2] = {calc_row_num, 1};
        const uint32_t dstShape[2] = {calc_row_num, numColAlign_};
        BroadCast<float, DIM_NUM, 1>(sqx, rstdLocal, dstShape, srcShape, sharedTmpLocal);
        PipeBarrier<PIPE_V>();

        LocalTensor<T> yLocal = outQueueY_.AllocTensor<T>();
        LocalTensor<float> x_fp32 = xFp32Buf_.Get<float>();
        Mul(x_fp32, x_fp32, sqx, elementNum);
        PipeBarrier<PIPE_V>();
        if constexpr (IsSameType<T, half>::value) {
            Cast(yLocal, x_fp32, RoundMode::CAST_NONE, elementNum);
        } else {
            Cast(yLocal, x_fp32, RoundMode::CAST_RINT, elementNum);
        }
        PipeBarrier<PIPE_V>();

        if constexpr (!IsSameType<T, bfloat16_t>::value) {
            Mul(yLocal, yLocal, gammaLocal, elementNum);
        } else {
            LocalTensor<float> x_fp32 = xFp32Buf_.Get<float>();
            Cast(x_fp32, yLocal, RoundMode::CAST_NONE, elementNum);
            Cast(sqx, gammaLocal, RoundMode::CAST_NONE, elementNum);
            PipeBarrier<PIPE_V>();
            Mul(x_fp32, x_fp32, sqx, elementNum);
            PipeBarrier<PIPE_V>();
            Cast(yLocal, x_fp32, RoundMode::CAST_RINT, elementNum);
        }
        PipeBarrier<PIPE_V>();
        outQueueY_.EnQue<T>(yLocal);
    }

    __aicore__ inline void CopyOutY(uint32_t progress, uint32_t calc_row_num)
    {
        LocalTensor<T> yLocal = outQueueY_.DeQue<T>();
        if (isNumColAlign_) {
            DataCopyCustom<T>(normOut_[progress], yLocal, calc_row_num * numCol_);
        } else {
            DataCopyCustom<T>(normOut_[progress], yLocal, calc_row_num, numCol_);
        }
        outQueueY_.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOutRstd(uint32_t outer_progress, uint32_t num)
    {
        LocalTensor<float> rstdLocal = outQueueRstd_.DeQue<float>();
        outQueueRstd_.FreeTensor(rstdLocal);
    }

private:
    // create queues for input, in this case depth is equal to buffer num
    TQue<QuePosition::VECIN, 1> inQueueGamma_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER_QUEUE> inQueueX_;
    TQue<QuePosition::VECOUT, 1> outQueueRstd_;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER_QUEUE> outQueueY_;

    TBuf<TPosition::VECCALC> xFp32Buf_;
    TBuf<TPosition::VECCALC> sqxBuf_;
    TBuf<TPosition::VECCALC> tmpBuf_;
    GlobalTensor<T> normOut_;
    GlobalTensor<T> residual_;
    GlobalTensor<T> gamma_;
    GlobalTensor<T> y_;

    uint32_t numRow_;
    uint32_t numCol_;
    uint32_t numColAlign_;
    uint32_t blockFactor_; // number of calculations rows on each core
    uint32_t rowFactor_;
    uint32_t ubFactor_;
    float epsilon_;
    float avgFactor_;

    uint32_t rowWork_ = 1;
    bool isNumColAlign_;
    uint32_t numBlocks_;
    uint64_t gmBlockOffset_;
    uint64_t gmBlockSize_;
};
#endif // MC2_ADD_RMS_NORM_MERGE_N_H
