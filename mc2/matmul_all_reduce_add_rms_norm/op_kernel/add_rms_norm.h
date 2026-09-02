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
 * \file add_rms_norm.h
 * \brief
 */

#ifndef ADD_RMS_NORM_H
#define ADD_RMS_NORM_H
#include "mararn_rms_norm_base.h"
using namespace AscendC;

template <typename T>
class KernelAddRmsNorm {
public:
    __aicore__ inline KernelAddRmsNorm() {}
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
        }

        // get start index for current core, core parallel
        gammaGm.SetGlobalBuffer((__gm__ T *)gamma, numCol);

        // Ppipe alloc memory to queue, the unit is Bytes
        Ppipe->InitBuffer(inQueueX, SINGLE_BUFFER_NUM, ubFactor * sizeof(T));
        Ppipe->InitBuffer(inQueueGamma, SINGLE_BUFFER_NUM, ubFactor * sizeof(T));
        Ppipe->InitBuffer(outQueueY, SINGLE_BUFFER_NUM, ubFactor * sizeof(T));

        if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {
            Ppipe->InitBuffer(xFp32Buf, ubFactor * sizeof(float));
        }
        Ppipe->InitBuffer(sqxBuf, ubFactor * sizeof(float));
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
        CopyInGamma();
        LocalTensor<T> gammaLocal = inQueueGamma.DeQue<T>();
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
        for (uint32_t i_i = 0; i_i < calc_row_num; i_i++) {
            uint32_t gm_bias = (i_o * rowFactor + i_i) * numCol;
            CopyIn(gm_bias);
            Compute(gammaLocal);
            CopyOutY(gm_bias);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t gm_bias)
    {
        LocalTensor<T> x1Local_in = inQueueX.AllocTensor<T>();
        LocalTensor<T> x2Local = sqxBuf.Get<T>();
        LocalTensor<T> xLocal = outQueueY.AllocTensor<T>();

        if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {
            x2Local = x2Local[ubFactor];
        }

        DataCopyCustom<T>(x1Local_in, normOutGm[gm_bias], numCol);
        DataCopyCustom<T>(x2Local, residualGm[gm_bias], numCol);
        inQueueX.EnQue(x1Local_in);
        auto x1Local = inQueueX.DeQue<T>();

        if constexpr (IsSameType<T, half>::value) {
            LocalTensor<float> x1_fp32 = xFp32Buf.Get<float>();
            Add(xLocal, x1Local, x2Local, numCol);
            PipeBarrier<PIPE_V>();
            Cast(x1_fp32, xLocal, RoundMode::CAST_NONE, numCol);
            PipeBarrier<PIPE_V>();
        } else if constexpr (IsSameType<T, bfloat16_t>::value) {
            LocalTensor<float> x1_fp32 = xFp32Buf.Get<float>();
            LocalTensor<float> x2_fp32 = sqxBuf.Get<float>();
            Cast(x1_fp32, x1Local, RoundMode::CAST_NONE, numCol);
            Cast(x2_fp32, x2Local, RoundMode::CAST_NONE, numCol);
            PipeBarrier<PIPE_V>();
            Add(x1_fp32, x1_fp32, x2_fp32, numCol);
            PipeBarrier<PIPE_V>();
            Cast(xLocal, x1_fp32, RoundMode::CAST_RINT, numCol);
            PipeBarrier<PIPE_V>();
        }
        inQueueX.FreeTensor(x1Local);

        // CopyOut x1 + x2
        outQueueY.EnQue(xLocal);
        auto x_out = outQueueY.DeQue<T>();
        DataCopyCustom<T>(yGm[gm_bias], x_out, numCol);
        outQueueY.FreeTensor(x_out);
    }

    __aicore__ inline void CopyInGamma()
    {
        LocalTensor<T> gammaLocal = inQueueGamma.AllocTensor<T>();
        DataCopyCustom<T>(gammaLocal, gammaGm, numCol);
        inQueueGamma.EnQue(gammaLocal);
    }

    __aicore__ inline void Compute(LocalTensor<bfloat16_t> gammaLocal)
    {
        LocalTensor<float> x_fp32 = xFp32Buf.Get<float>();
        LocalTensor<float> sqx = sqxBuf.Get<float>();
        LocalTensor<float> reduce_buf_local = reduceFp32Buf.Get<float>();

        Mul(sqx, x_fp32, x_fp32, numCol);
        PipeBarrier<PIPE_V>();

        Muls(sqx, sqx, avgFactor, numCol);
        PipeBarrier<PIPE_V>();

        ReduceSumCustom(sqx, sqx, reduce_buf_local, numCol);
        PipeBarrier<PIPE_V>();

        Adds(sqx, sqx, epsilon, 1);
        PipeBarrier<PIPE_V>();

        Sqrt(sqx, sqx, 1);
        Duplicate(reduce_buf_local, (float)1.0, 1);
        PipeBarrier<PIPE_V>();

        Div(sqx, reduce_buf_local, sqx, 1);
        PipeBarrier<PIPE_V>();

        event_t event_v_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(event_v_s);
        WaitFlag<HardEvent::V_S>(event_v_s);
        float rstd_value = sqx.GetValue(0);
        event_t event_s_v = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(event_s_v);
        WaitFlag<HardEvent::S_V>(event_s_v);
        PipeBarrier<PIPE_V>();
        Muls(x_fp32, x_fp32, rstd_value, numCol);
        PipeBarrier<PIPE_V>();

        LocalTensor<bfloat16_t> yLocal = outQueueY.AllocTensor<bfloat16_t>();
        Cast(yLocal, x_fp32, RoundMode::CAST_RINT, numCol);
        PipeBarrier<PIPE_V>();

        Cast(x_fp32, yLocal, RoundMode::CAST_NONE, numCol);
        PipeBarrier<PIPE_V>();

        Cast(sqx, gammaLocal, RoundMode::CAST_NONE, numCol); // gamma_fp32 reuse sqx
        PipeBarrier<PIPE_V>();
        Mul(x_fp32, x_fp32, sqx, numCol);
        PipeBarrier<PIPE_V>();

        Cast(yLocal, x_fp32, RoundMode::CAST_RINT, numCol);
        PipeBarrier<PIPE_V>();

        event_t event_v_mte = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(event_v_mte);
        WaitFlag<HardEvent::V_MTE2>(event_v_mte);

        outQueueY.EnQue<bfloat16_t>(yLocal);
    }

    __aicore__ inline void Compute(LocalTensor<half> gammaLocal)
    {
        LocalTensor<float> x_fp32 = xFp32Buf.Get<float>();
        LocalTensor<float> sqx = sqxBuf.Get<float>();
        LocalTensor<float> reduce_buf_local = reduceFp32Buf.Get<float>();

        Mul(sqx, x_fp32, x_fp32, numCol);
        PipeBarrier<PIPE_V>();

        Muls(sqx, sqx, avgFactor, numCol);
        PipeBarrier<PIPE_V>();

        ReduceSumCustom(sqx, sqx, reduce_buf_local, numCol);
        PipeBarrier<PIPE_V>();

        Adds(sqx, sqx, epsilon, 1);
        PipeBarrier<PIPE_V>();

        Sqrt(sqx, sqx, 1);
        Duplicate(reduce_buf_local, (float)1.0, 1);
        PipeBarrier<PIPE_V>();
        Div(sqx, reduce_buf_local, sqx, 1);
        PipeBarrier<PIPE_V>();

        event_t event_v_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(event_v_s);
        WaitFlag<HardEvent::V_S>(event_v_s);
        float rstd_value = sqx.GetValue(0);
        event_t event_s_v = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(event_s_v);
        WaitFlag<HardEvent::S_V>(event_s_v);
        PipeBarrier<PIPE_V>();

        Muls(x_fp32, x_fp32, rstd_value, numCol);
        PipeBarrier<PIPE_V>();
        LocalTensor<half> yLocal = outQueueY.AllocTensor<half>();
        Cast(yLocal, x_fp32, RoundMode::CAST_NONE, numCol);

        event_t event_v_mte = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(event_v_mte);
        WaitFlag<HardEvent::V_MTE2>(event_v_mte);

        PipeBarrier<PIPE_V>();
        Mul(yLocal, gammaLocal, yLocal, numCol);
        PipeBarrier<PIPE_V>();
        outQueueY.EnQue<half>(yLocal);
    }

    __aicore__ inline void CopyOutY(uint32_t progress)
    {
        LocalTensor<T> yLocal = outQueueY.DeQue<T>();
        DataCopyCustom<T>(normOutGm[progress], yLocal, numCol);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe *Ppipe = nullptr;
    // create queues for input, in this case depth is equal to buffer num
    TQue<QuePosition::VECIN, SINGLE_BUFFER_NUM> inQueueX, inQueueGamma;
    // create queues for output, in this case depth is equal to buffer num
    TQue<QuePosition::VECOUT, SINGLE_BUFFER_NUM> outQueueY;

    TBuf<TPosition::VECCALC> xFp32Buf;
    TBuf<TPosition::VECCALC> sqxBuf;
    TBuf<TPosition::VECCALC> reduceFp32Buf;
    GlobalTensor<T> normOutGm;
    GlobalTensor<T> residualGm;
    GlobalTensor<T> gammaGm;
    GlobalTensor<T> yGm;

    uint32_t numRow;
    uint32_t numCol;
    uint32_t blockFactor; // number of calculations rows on each core
    uint32_t ubFactor;
    uint32_t rowFactor;
    float epsilon;
    float avgFactor;
    uint32_t numBlocks;

    uint32_t rowWork = 1;
    uint32_t cptCount = 1;
};
#endif // ADD_RMS_NORM_H
