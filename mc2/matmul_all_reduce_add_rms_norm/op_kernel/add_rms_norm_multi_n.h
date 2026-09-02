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
 * \file add_rms_norm_multi_n.h
 * \brief
 */
#ifndef ADD_RMS_NORM_MULTI_N_H
#define ADD_RMS_NORM_MULTI_N_H
#include "mararn_rms_norm_base.h"
using namespace AscendC;

template <typename T>
class KernelAddRmsNormMultiN {
public:
    __aicore__ inline KernelAddRmsNormMultiN() {}
    __aicore__ inline void Init(GM_ADDR gamma, Mc2Tiling::AddRMSNormTilingData &tilingData, TPipe *Ppipe,
                                uint32_t numBlocks)
    {
        ASSERT(numBlocks != 0 && "Block dim can not be zero!");
        this->numBlocks = numBlocks;
        this->numRow = tilingData.num_row;
        this->numCol = tilingData.num_col;
        uint32_t numPerBlock = ONE_BLK_SIZE / sizeof(T);
        this->numColAlign = AlignUp(tilingData.num_col, numPerBlock);
        this->blockFactor = tilingData.block_factor;
        this->rowFactor = tilingData.row_factor;
        this->ubFactor = tilingData.ub_factor;
        this->epsilon = tilingData.epsilon;
        this->avgFactor = (float)1.0 / tilingData.num_col;

        if (GetBlockIdx() < (numBlocks - 1)) {
            this->rowWork = blockFactor;
        } else if (GetBlockIdx() == (numBlocks - 1)) {
            this->rowWork = numRow - (numBlocks - 1) * blockFactor;
        } else {
        }

        // get start index for current core, core parallel
        gammaGm.SetGlobalBuffer((__gm__ T *)gamma, numCol);

        // pipe alloc memory to queue, the unit is Bytes
        Ppipe->InitBuffer(inQueueX, DOUBLE_BUFFER_NUM, ubFactor * sizeof(T));
        Ppipe->InitBuffer(inQueueGamma, SINGLE_BUFFER_NUM, numColAlign * sizeof(T));
        Ppipe->InitBuffer(outQueueY, DOUBLE_BUFFER_NUM, ubFactor * sizeof(T));
        Ppipe->InitBuffer(rstdBuf, rowFactor * NUM_PER_BLK_FP32 * sizeof(float));
        Ppipe->InitBuffer(xFp32Buf, ubFactor * sizeof(float));

        Ppipe->InitBuffer(sqxBuf, ubFactor * sizeof(float));
        Ppipe->InitBuffer(reduceFp32Buf, NUM_PER_REP_FP32 * sizeof(float));
        Ppipe->InitBuffer(offsetBuf, rowFactor * NUM_PER_BLK_FP32 * sizeof(uint32_t));
    }

    __aicore__ inline void InitGlobalATensor(GM_ADDR normOut, GM_ADDR residual, GM_ADDR y)
    {
        if (GetBlockIdx() < numBlocks - 1) {
            this->rowWork = blockFactor;
        } else if (GetBlockIdx() == numBlocks - 1) {
            this->rowWork = numRow - (numBlocks - 1) * blockFactor;
        }
        normOutGm.SetGlobalBuffer((__gm__ T *)normOut + GetBlockIdx() * blockFactor * numCol, rowWork * numCol);
        residualGm.SetGlobalBuffer((__gm__ T *)residual + GetBlockIdx() * blockFactor * numCol, rowWork * numCol);
        yGm.SetGlobalBuffer((__gm__ T *)y + GetBlockIdx() * blockFactor * numCol, rowWork * numCol);
    }

    __aicore__ inline void ComputeProcess(GM_ADDR normOut, GM_ADDR residual, GM_ADDR y,
                                          Mc2Tiling::AddRMSNormTilingData &tilingData, uint32_t addRmsNormCount,
                                          uint32_t rcvCnt)
    {
        auto x1Addr = normOut;
        auto x2Addr = residual;
        auto yAddr = y;
        uint64_t cOffset = CalcShapeOffset(sizeof(T), tilingData.num_row, tilingData.num_col); // 偏移*size
        while (addRmsNormCount <= rcvCnt) {
            InitGlobalATensor(x1Addr, x2Addr, yAddr);
            Process();
            x1Addr += cOffset;
            yAddr += cOffset;
            x2Addr += cOffset;
            addRmsNormCount += 1;
        }
    }

    __aicore__ inline void Process()
    {
        CopyInGamma();
        LocalTensor<T> gammaLocal = inQueueGamma.DeQue<T>();

        LocalTensor<uint32_t> offsetLocal = offsetBuf.Get<uint32_t>();
        for (uint32_t i = 0; i < rowFactor; i++) {
            Duplicate(offsetLocal[i * NUM_PER_BLK_FP32], i * ONE_BLK_SIZE, NUM_PER_BLK_FP32);
        }
        uint32_t i_o_max = CeilDiv(rowWork, rowFactor);
        uint32_t row_tail = rowWork - (i_o_max - 1) * rowFactor;
        for (uint32_t i_o = 0; i_o < i_o_max - 1; i_o++) {
            SubProcess(i_o, rowFactor, gammaLocal);
        }
        SubProcess(i_o_max - 1, row_tail, gammaLocal);
        inQueueGamma.FreeTensor(gammaLocal);
    }

    __aicore__ inline void SubProcess(uint32_t i_o, uint32_t calc_row_num, LocalTensor<T> &gammaLocal)
    {
        uint32_t gm_bias = i_o * rowFactor * numCol;
        CopyInX(gm_bias, calc_row_num);

        LocalTensor<T> xLocal = ComputeX(calc_row_num);
        CopyOutX(gm_bias, calc_row_num);

        LocalTensor<float> rstdLocal = rstdBuf.Get<float>();
        ComputeRstd(xLocal, rstdLocal, calc_row_num);

        ComputeY(xLocal, gammaLocal, rstdLocal, calc_row_num);
        CopyOutY(gm_bias, calc_row_num);
    }

private:
    __aicore__ inline void CopyInX(uint32_t gm_bias, uint32_t calc_row_num)
    {
        LocalTensor<T> x1Local = inQueueX.AllocTensor<T>();
        DataCopyCustom<T>(x1Local, normOutGm[gm_bias], calc_row_num * numCol);
        inQueueX.EnQue(x1Local);

        LocalTensor<T> x2Local = inQueueX.AllocTensor<T>();
        DataCopyCustom<T>(x2Local, residualGm[gm_bias], calc_row_num * numCol);
        inQueueX.EnQue(x2Local);
    }

    __aicore__ inline LocalTensor<T> ComputeX(uint32_t calc_row_num)
    {
        uint32_t calc_num = calc_row_num * numColAlign;
        LocalTensor<T> x1Local = inQueueX.DeQue<T>();
        LocalTensor<T> x2Local = inQueueX.DeQue<T>();
        LocalTensor<T> xLocal = outQueueY.AllocTensor<T>();

        Add(xLocal, x1Local, x2Local, calc_num);
        inQueueX.FreeTensor(x1Local);
        inQueueX.FreeTensor(x2Local);
        outQueueY.EnQue(xLocal);
        PipeBarrier<PIPE_V>();
        return xLocal;
    }

    __aicore__ inline void CopyOutX(uint32_t gm_bias, uint32_t calc_row_num)
    {
        // CopyOut x1 + x2
        auto x_out = outQueueY.DeQue<T>();
        DataCopyCustom<T>(yGm[gm_bias], x_out, calc_row_num * numCol);
        outQueueY.FreeTensor(x_out);
    }

    __aicore__ inline void CopyInGamma()
    {
        LocalTensor<T> gammaLocal = inQueueGamma.AllocTensor<T>();
        DataCopyCustom<T>(gammaLocal, gammaGm, numCol);
        inQueueGamma.EnQue(gammaLocal);
    }

    __aicore__ inline void ComputeRstd(LocalTensor<T> xLocal, LocalTensor<float> rstdLocal, uint32_t calc_row_num)
    {
        LocalTensor<float> x_fp32 = xFp32Buf.Get<float>();
        LocalTensor<float> sqx = sqxBuf.Get<float>();
        LocalTensor<float> reduce_buf_local = reduceFp32Buf.Get<float>();
        Cast(x_fp32, xLocal, RoundMode::CAST_NONE, calc_row_num * numColAlign);
        PipeBarrier<PIPE_V>();

        Mul(sqx, x_fp32, x_fp32, calc_row_num * numColAlign);
        PipeBarrier<PIPE_V>();

        Muls(sqx, sqx, avgFactor, calc_row_num * numColAlign);
        PipeBarrier<PIPE_V>();

        for (uint32_t i_i = 0; i_i < calc_row_num; i_i++) {
            ReduceSumCustom(rstdLocal[i_i * NUM_PER_BLK_FP32], sqx[i_i * numColAlign], reduce_buf_local, numCol);
        }

        Adds(rstdLocal, rstdLocal, epsilon, calc_row_num * NUM_PER_BLK_FP32);
        PipeBarrier<PIPE_V>();

        Sqrt(rstdLocal, rstdLocal, calc_row_num * NUM_PER_BLK_FP32);
        Duplicate(reduce_buf_local, (float)1.0, NUM_PER_BLK_FP32);
        PipeBarrier<PIPE_V>();

        int32_t repeatTimes = calc_row_num * NUM_PER_BLK_FP32 / NUM_PER_REP_FP32;
        int32_t tailCount = calc_row_num * NUM_PER_BLK_FP32 % NUM_PER_REP_FP32;
        int32_t bodyCount = repeatTimes * NUM_PER_REP_FP32;

        if (likely(repeatTimes > 0)) {
            Div(rstdLocal, reduce_buf_local, rstdLocal, NUM_PER_REP_FP32, repeatTimes,
                {1, 0, 1, DEFAULT_REPEAT_STRIDE, 0, DEFAULT_REPEAT_STRIDE});
        }
        if (unlikely(tailCount != 0)) {
            Div(rstdLocal[bodyCount], reduce_buf_local, rstdLocal[bodyCount], tailCount, 1,
                {1, 0, 1, DEFAULT_REPEAT_STRIDE, 0, DEFAULT_REPEAT_STRIDE});
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ComputeY(LocalTensor<T> xLocal, LocalTensor<T> gammaLocal, LocalTensor<float> rstdLocal,
                                    uint32_t calc_row_num)
    {
        LocalTensor<float> x_fp32 = xFp32Buf.Get<float>();
        LocalTensor<uint32_t> offsetLocal = offsetBuf.Get<uint32_t>();
        Gather(rstdLocal, rstdLocal, offsetLocal, 0U, calc_row_num * NUM_PER_BLK_FP32);
        PipeBarrier<PIPE_V>();
        int32_t repeatTimes = numCol / NUM_PER_REP_FP32;
        int32_t tailCount = numCol % NUM_PER_REP_FP32;
        int32_t bodyCount = repeatTimes * NUM_PER_REP_FP32;
        for (uint32_t i_i = 0; i_i < calc_row_num; i_i++) {
            if (likely(repeatTimes > 0)) {
                Mul(x_fp32[i_i * numColAlign], x_fp32[i_i * numColAlign], rstdLocal[i_i * NUM_PER_BLK_FP32],
                    NUM_PER_REP_FP32, repeatTimes, {1, 1, 0, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, 0});
            }
            if (unlikely(tailCount != 0)) {
                Mul(x_fp32[i_i * numColAlign + bodyCount], x_fp32[i_i * numColAlign + bodyCount],
                    rstdLocal[i_i * NUM_PER_BLK_FP32], tailCount, 1,
                    {1, 1, 0, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, 0});
            }
        }
        PipeBarrier<PIPE_V>();
        LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();
        Cast(yLocal, x_fp32, RoundMode::CAST_NONE, calc_row_num * numColAlign);
        PipeBarrier<PIPE_V>();

        for (uint32_t i_i = 0; i_i < calc_row_num; i_i++) {
            Mul(yLocal[i_i * numColAlign], gammaLocal, yLocal[i_i * numColAlign], numCol);
        }
        PipeBarrier<PIPE_V>();
        outQueueY.EnQue<T>(yLocal);
    }

    __aicore__ inline void CopyOutY(uint32_t progress, uint32_t calc_row_num)
    {
        LocalTensor<T> yLocal = outQueueY.DeQue<T>();
        DataCopyCustom<T>(normOutGm[progress], yLocal, calc_row_num * numCol);
        outQueueY.FreeTensor(yLocal);
    }

private:
    // create queues for input, in this case depth is equal to buffer num
    TQue<QuePosition::VECIN, SINGLE_BUFFER_NUM> inQueueGamma;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER_NUM> inQueueX;
    TBuf<TPosition::VECCALC> rstdBuf;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER_NUM> outQueueY;

    TBuf<TPosition::VECCALC> xFp32Buf;
    TBuf<TPosition::VECCALC> sqxBuf;
    TBuf<TPosition::VECCALC> reduceFp32Buf;
    TBuf<TPosition::VECCALC> offsetBuf;
    GlobalTensor<T> normOutGm;
    GlobalTensor<T> residualGm;
    GlobalTensor<T> gammaGm;
    GlobalTensor<float> rstdGm;
    GlobalTensor<T> yGm;

    uint32_t numRow;
    uint32_t numCol;
    uint32_t blockFactor; // number of calculations rows on each core
    uint32_t rowFactor;
    uint32_t ubFactor;
    float epsilon;
    float avgFactor;
    uint32_t numColAlign;
    uint32_t numBlocks;
    uint32_t rowWork = 1;
};
#endif // ADD_RMS_NORM_MULTI_N_H
