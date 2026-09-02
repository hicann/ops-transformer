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
 * \file add_rms_norm_split_d.h
 * \brief
 */
#ifndef ADD_RMS_NORM_SPLIT_D_H
#define ADD_RMS_NORM_SPLIT_D_H
#include "mararn_rms_norm_base.h"
using namespace AscendC;

template <typename T>
class KernelAddRmsNormSplitD {
public:
    __aicore__ inline KernelAddRmsNormSplitD() {}
    __aicore__ inline void Init(GM_ADDR gamma, Mc2Tiling::AddRMSNormTilingData &tilingData, TPipe *Ppipe,
                                uint32_t numBlocks)
    {
        ASSERT(numBlocks != 0 && "Block dim can not be zero!");
        this->numBlocks = numBlocks;
        this->numRow = tilingData.num_row;
        this->numCol = tilingData.num_col;
        this->blockFactor = tilingData.block_factor;
        this->rowFactor = tilingData.row_factor;
        this->ubFactor = tilingData.ub_factor;
        this->epsilon = tilingData.epsilon;
        this->avgFactor = (float)1.0 / tilingData.num_col;

        if (GetBlockIdx() < numBlocks - 1) {
            this->rowWork = blockFactor;
        } else if (GetBlockIdx() == numBlocks - 1) {
            this->rowWork = numRow - (numBlocks - 1) * blockFactor;
        } else {
        }
        // get start index for current core, core parallel
        gammaGm.SetGlobalBuffer((__gm__ T *)gamma, numCol);

        // Ppipe alloc memory to queue, the unit is Bytes.
        // We need 2 buffers here for both x1 and x2.
        Ppipe->InitBuffer(inQueueX, SINGLE_BUFFER_NUM, 2 * ubFactor * sizeof(T));
        Ppipe->InitBuffer(inQueueGamma, SINGLE_BUFFER_NUM, ubFactor * sizeof(T));
        Ppipe->InitBuffer(outQueueY, SINGLE_BUFFER_NUM, ubFactor * sizeof(T));
        Ppipe->InitBuffer(rstdBuf, rowFactor * sizeof(float));

        Ppipe->InitBuffer(xFp32Buf, ubFactor * sizeof(float));
        Ppipe->InitBuffer(sqxBuf, ubFactor * sizeof(float));
        Ppipe->InitBuffer(sumBuf, rowFactor * NUM_PER_BLK_FP32 * sizeof(float));
        Ppipe->InitBuffer(reduceFp32Buf, NUM_PER_REP_FP32 * sizeof(float));
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
        uint32_t i_o_max = CeilDiv(rowWork, rowFactor);
        uint32_t row_tail = rowWork - (i_o_max - 1) * rowFactor;
        uint32_t j_max = CeilDiv(numCol, ubFactor);
        uint32_t col_tail = numCol - (j_max - 1) * ubFactor;
        for (uint32_t i_o = 0; i_o < i_o_max - 1; i_o++) {
            SubProcess(i_o, rowFactor, j_max, col_tail);
        }
        SubProcess(i_o_max - 1, row_tail, j_max, col_tail);
    }

    __aicore__ inline void SubProcess(uint32_t i_o, uint32_t calc_row_num, uint32_t j_max, uint32_t col_tail)
    {
        LocalTensor<float> sumLocal = sumBuf.Get<float>();
        LocalTensor<float> rstdLocal = rstdBuf.Get<float>();
        Duplicate(rstdLocal, (float)0.0, calc_row_num);
        PipeBarrier<PIPE_V>();
        for (uint32_t j = 0; j < j_max - 1; j++) {
            ComputeFormer(i_o, calc_row_num, j, rstdLocal, sumLocal, ubFactor);
        }
        // do tail
        ComputeFormer(i_o, calc_row_num, j_max - 1, rstdLocal, sumLocal, col_tail);
        ComputeRstd(rstdLocal, calc_row_num);

        for (uint32_t j = 0; j < j_max - 1; j++) {
            ComputeLatter(i_o, calc_row_num, j, rstdLocal, ubFactor);
        }
        ComputeLatter(i_o, calc_row_num, j_max - 1, rstdLocal, col_tail);
    }

private:
    __aicore__ inline void CopyInAndAdd(uint32_t i_idx, uint32_t j_idx, uint32_t num)
    {
        LocalTensor<T> x1x2_in = inQueueX.AllocTensor<T>();
        LocalTensor<T> x1_in = x1x2_in[0];
        LocalTensor<T> x2_in = x1x2_in[ubFactor];
        DataCopyCustom<T>(x1_in, normOutGm[i_idx * numCol + j_idx * ubFactor], num);
        DataCopyCustom<T>(x2_in, residualGm[i_idx * numCol + j_idx * ubFactor], num);
        inQueueX.EnQue(x1x2_in);
        LocalTensor<T> x1x2Local = inQueueX.DeQue<T>();

        auto x1Local = x1x2Local[0];
        auto x2Local = x1x2Local[ubFactor];

        LocalTensor<T> xLocal = outQueueY.AllocTensor<T>();

        if constexpr (IsSameType<T, half>::value) {
            LocalTensor<float> x1_fp32 = xFp32Buf.Get<float>();

            Add(xLocal, x1Local, x2Local, num);
            PipeBarrier<PIPE_V>();
            Cast(x1_fp32, xLocal, RoundMode::CAST_NONE, num);
            PipeBarrier<PIPE_V>();
            // x1+x2 saved in x1_fp32
        } else if constexpr (IsSameType<T, bfloat16_t>::value) {
            LocalTensor<float> x1_fp32 = xFp32Buf.Get<float>();
            LocalTensor<float> x2_fp32 = x1x2Local.template ReinterpretCast<float>();

            Cast(x1_fp32, x1Local, RoundMode::CAST_NONE, num);
            PipeBarrier<PIPE_V>();
            Cast(x2_fp32, x2Local, RoundMode::CAST_NONE, num);
            PipeBarrier<PIPE_V>();

            Add(x1_fp32, x1_fp32, x2_fp32, num);
            PipeBarrier<PIPE_V>();
            Cast(xLocal, x1_fp32, RoundMode::CAST_RINT, num);
            PipeBarrier<PIPE_V>();
            // x1+x2 saved in x1_fp32
        }
        inQueueX.FreeTensor(x1x2Local);

        // copy out to x_out
        outQueueY.EnQue(xLocal);
        auto x_out = outQueueY.DeQue<T>();
        DataCopyCustom<T>(yGm[i_idx * numCol + j_idx * ubFactor], x_out, num);
        outQueueY.FreeTensor(x_out);
    }

    __aicore__ inline void ComputeFormer(uint32_t i_o_idx, uint32_t calc_row_num, uint32_t j_idx,
                                         LocalTensor<float> &rstdLocal, LocalTensor<float> &sumLocal, uint32_t num)
    {
        for (uint32_t i_i = 0; i_i < calc_row_num; i_i++) {
            CopyInAndAdd(i_o_idx * rowFactor + i_i, j_idx, num);
            ComputeSum(i_i, sumLocal, num);
        }
        BlockReduceSumFP32(sumLocal, sumLocal, calc_row_num * NUM_PER_BLK_FP32);
        Add(rstdLocal, rstdLocal, sumLocal, calc_row_num);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ComputeSum(uint32_t i_i_idx, LocalTensor<float> &sumLocal, uint32_t num)
    {
        LocalTensor<float> sqx = sqxBuf.Get<float>();
        LocalTensor<float> reduce_buf_local = reduceFp32Buf.Get<float>();
        LocalTensor<float> x_fp32 = xFp32Buf.Get<float>();
        PipeBarrier<PIPE_V>();
        Mul(sqx, x_fp32, x_fp32, num);

        PipeBarrier<PIPE_V>();
        Muls(sqx, sqx, avgFactor, num);
        PipeBarrier<PIPE_V>();
        // 8 means 8 fp32 pre block
        ReduceSumFP32ToBlock(sumLocal[i_i_idx * 8], sqx, reduce_buf_local, num);
    }

    __aicore__ inline void ComputeRstd(LocalTensor<float> rstdLocal, uint32_t num)
    {
        LocalTensor<float> reduce_buf_local = reduceFp32Buf.Get<float>();
        Adds(rstdLocal, rstdLocal, epsilon, num);
        PipeBarrier<PIPE_V>();
        Sqrt(rstdLocal, rstdLocal, num);
        Duplicate(reduce_buf_local, (float)1.0, num);
        PipeBarrier<PIPE_V>();
        Div(rstdLocal, reduce_buf_local, rstdLocal, num);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ComputeLatter(uint32_t i_o_idx, uint32_t calc_row_num, uint32_t j_idx,
                                         LocalTensor<float> &rstdLocal, uint32_t num)
    {
        CopyInGamma(j_idx, num);
        LocalTensor<T> gammaLocal = inQueueGamma.DeQue<T>();
        for (uint32_t i_i = 0; i_i < calc_row_num; i_i++) {
            CopyInX(i_o_idx * rowFactor + i_i, j_idx, num);
            ComputeY(i_i, gammaLocal, rstdLocal, num);
            CopyOutY(i_o_idx * rowFactor + i_i, j_idx, num);
        }
        inQueueGamma.FreeTensor(gammaLocal);
    }

    __aicore__ inline void CopyInGamma(uint32_t j_idx, uint32_t num)
    {
        LocalTensor<T> gammaLocal = inQueueGamma.AllocTensor<T>();
        DataCopyCustom<T>(gammaLocal, gammaGm[j_idx * ubFactor], num);
        inQueueGamma.EnQue(gammaLocal);
    }

    __aicore__ inline void CopyInX(uint32_t i_idx, uint32_t j_idx, uint32_t num)
    {
        LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        DataCopyCustom<T>(xLocal, yGm[i_idx * numCol + j_idx * ubFactor], num);
        inQueueX.EnQue<T>(xLocal);
        if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {
            LocalTensor<float> x_fp32 = xFp32Buf.Get<float>();
            LocalTensor<T> xLocal = inQueueX.DeQue<T>();
            Cast(x_fp32, xLocal, RoundMode::CAST_NONE, num);
            PipeBarrier<PIPE_V>();
            inQueueX.FreeTensor(xLocal);
        }
    }

    __aicore__ inline void ComputeY(uint32_t i_i_idx, LocalTensor<half> &gammaLocal, LocalTensor<float> &rstdLocal,
                                    uint32_t num)
    {
        LocalTensor<float> x_fp32 = xFp32Buf.Get<float>();
        LocalTensor<float> sqx = sqxBuf.Get<float>();
        event_t event_v_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(event_v_s);
        WaitFlag<HardEvent::V_S>(event_v_s);
        float rstd_value = rstdLocal.GetValue(i_i_idx);
        event_t event_s_v = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(event_s_v);
        WaitFlag<HardEvent::S_V>(event_s_v);
        PipeBarrier<PIPE_V>();
        Muls(x_fp32, x_fp32, rstd_value, num);
        PipeBarrier<PIPE_V>();
        LocalTensor<half> yLocal = outQueueY.AllocTensor<half>();
        Cast(yLocal, x_fp32, RoundMode::CAST_NONE, num);
        PipeBarrier<PIPE_V>();
        Mul(yLocal, gammaLocal, yLocal, num);
        PipeBarrier<PIPE_V>();
        outQueueY.EnQue<half>(yLocal);
    }

    __aicore__ inline void ComputeY(uint32_t i_i_idx, LocalTensor<bfloat16_t> &gammaLocal,
                                    LocalTensor<float> &rstdLocal, uint32_t num)
    {
        LocalTensor<float> x_fp32 = xFp32Buf.Get<float>();
        LocalTensor<float> sqx = sqxBuf.Get<float>();
        event_t event_v_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(event_v_s);
        WaitFlag<HardEvent::V_S>(event_v_s);
        float rstd_value = rstdLocal.GetValue(i_i_idx);
        event_t event_s_v = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(event_s_v);
        WaitFlag<HardEvent::S_V>(event_s_v);
        PipeBarrier<PIPE_V>();
        Muls(x_fp32, x_fp32, rstd_value, num);
        PipeBarrier<PIPE_V>();
        LocalTensor<bfloat16_t> yLocal = outQueueY.AllocTensor<bfloat16_t>();
        Cast(yLocal, x_fp32, RoundMode::CAST_RINT, num);
        PipeBarrier<PIPE_V>();
        Cast(x_fp32, yLocal, RoundMode::CAST_NONE, num);
        PipeBarrier<PIPE_V>();
        Cast(sqx, gammaLocal, RoundMode::CAST_NONE, num);
        PipeBarrier<PIPE_V>();
        Mul(x_fp32, x_fp32, sqx, num);
        PipeBarrier<PIPE_V>();
        Cast(yLocal, x_fp32, RoundMode::CAST_RINT, num);
        PipeBarrier<PIPE_V>();
        outQueueY.EnQue<bfloat16_t>(yLocal);
    }

    __aicore__ inline void CopyOutY(uint32_t i_idx, uint32_t j_idx, uint32_t num)
    {
        LocalTensor<T> yLocal = outQueueY.DeQue<T>();
        DataCopyCustom<T>(normOutGm[i_idx * numCol + j_idx * ubFactor], yLocal, num);
        outQueueY.FreeTensor(yLocal);
    }

private:
    // create queues for input, in this case depth is equal to buffer num
    TQue<QuePosition::VECIN, SINGLE_BUFFER_NUM> inQueueX, inQueueGamma;
    // create queues for output, in this case depth is equal to buffer num
    TQue<QuePosition::VECOUT, SINGLE_BUFFER_NUM> outQueueY;
    TBuf<TPosition::VECCALC> rstdBuf;
    TBuf<TPosition::VECCALC> xFp32Buf;
    TBuf<TPosition::VECCALC> sqxBuf;
    TBuf<TPosition::VECCALC> sumBuf;
    TBuf<TPosition::VECCALC> reduceFp32Buf;

    GlobalTensor<T> normOutGm;
    GlobalTensor<T> residualGm;
    GlobalTensor<T> gammaGm;
    GlobalTensor<T> yGm;

    uint32_t numRow;
    uint32_t numCol;
    uint32_t blockFactor; // number of calculations rows on each core
    uint32_t rowFactor;
    uint32_t ubFactor;
    float epsilon;
    float avgFactor;
    uint32_t numBlocks;
    uint32_t rowWork = 1;

    int tempbufNum;
};
#endif
