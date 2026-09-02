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
 * \file add_rms_norm_single_n.h
 * \brief
 */
#ifndef MC2_ADD_RMS_NORM_SINGLE_N_H
#define MC2_ADD_RMS_NORM_SINGLE_N_H
#include "mararn_rms_norm_base.h"

using namespace AscendC;

constexpr uint64_t SINGLE_N_NUM_BLOCK = 192;
constexpr uint64_t SINGLE_N_BLOCK_SIZE = 1024;
constexpr uint64_t SINGLE_N_BUFFER_SIZE = (SINGLE_N_NUM_BLOCK - 1) * SINGLE_N_BLOCK_SIZE; // 195584 bytes

template <typename T>
class KernelAddRmsNormSingleN {
public:
    __aicore__ inline KernelAddRmsNormSingleN() {}
    __aicore__ inline void Init(GM_ADDR gammaGM, Mc2Tiling::AddRMSNormTilingData &tiling, TPipe *pipe,
                                uint32_t numBlocks)
    {
        ASSERT(numBlocks != 0 && "Block dim can not be zero!");
        this->numCol_ = tiling.num_col;
        this->ubFactor_ = tiling.ub_factor;
        this->epsilon_ = tiling.epsilon;
        this->avgFactor_ = (numCol_ != 0) ? (float)1.0 / numCol_ : 0;
        // get start index for current core, core parallel
        gamma_.SetGlobalBuffer((__gm__ T *)gammaGM, numCol_);
        pipe->InitBuffer(unitBuf_, SINGLE_N_BUFFER_SIZE);
    }

    __aicore__ inline void Process()
    {
        if constexpr (IsSameType<T, half>::value) {
            ProcessFp16();
        } else {
            ProcessBf16();
        }
    }

    __aicore__ inline void ComputeProcess(GM_ADDR normOutGM, GM_ADDR residualGM, GM_ADDR yGM,
                                          Mc2Tiling::AddRMSNormTilingData &tilingData, uint32_t addRmsNormCount,
                                          uint32_t rcvCnt)
    {
        uint64_t cOffset = CalcShapeOffset(sizeof(T), tilingData.num_row, tilingData.num_col); // 偏移*size
        for (; addRmsNormCount <= rcvCnt; ++addRmsNormCount) {
            normOut_.SetGlobalBuffer((__gm__ T *)normOutGM + GetBlockIdx() * numCol_, numCol_);
            residual_.SetGlobalBuffer((__gm__ T *)residualGM + GetBlockIdx() * numCol_, numCol_);
            y_.SetGlobalBuffer((__gm__ T *)yGM + GetBlockIdx() * numCol_, numCol_);
            Process();
            normOutGM += cOffset;
            residualGM += cOffset;
            yGM += cOffset;
        }
    }

private:
    __aicore__ inline void ProcessFp16()
    {
        LocalTensor<float> ubLocal = unitBuf_.Get<float>();
        LocalTensor<T> xLocal = ubLocal.template ReinterpretCast<T>();
        LocalTensor<T> x1Local = xLocal[0];
        LocalTensor<T> x2Local = xLocal[ubFactor_];
        LocalTensor<float> xFp32Local = ubLocal[ubFactor_];
        LocalTensor<float> sqxLocal = ubLocal[ubFactor_ * 2];
        LocalTensor<float> tmpLocal = ubLocal[ubFactor_ * 3];

        DataCopyCustom<T>(x1Local, normOut_, numCol_);
        event_t eventMTE2V1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventMTE2V1);
        DataCopyCustom<T>(x2Local, residual_, numCol_);
        event_t eventMTE2V2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventMTE2V2);
        WaitFlag<HardEvent::MTE2_V>(eventMTE2V1);
        WaitFlag<HardEvent::MTE2_V>(eventMTE2V2);
        Add(x1Local, x1Local, x2Local, numCol_);
        PipeBarrier<PIPE_V>();

        // copy gamma
        event_t eventVMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(eventVMTE2);
        WaitFlag<HardEvent::V_MTE2>(eventVMTE2);

        DataCopyCustom<T>(x2Local, gamma_, numCol_); // gammaLocal use x2Local
        SetFlag<HardEvent::MTE2_V>(eventMTE2V2);

        // copy x out
        event_t eventVMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventVMTE3);
        WaitFlag<HardEvent::V_MTE3>(eventVMTE3);
        DataCopyCustom<T>(y_, x1Local, numCol_);
        event_t eventMTE3V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(eventMTE3V);

        Cast(xFp32Local, x1Local, RoundMode::CAST_NONE, numCol_);
        PipeBarrier<PIPE_V>();
        Mul(sqxLocal, xFp32Local, xFp32Local, numCol_);
        PipeBarrier<PIPE_V>();
        Muls(sqxLocal, sqxLocal, avgFactor_, numCol_);
        PipeBarrier<PIPE_V>();
        ReduceSumCustom(sqxLocal, sqxLocal, tmpLocal, numCol_);
        PipeBarrier<PIPE_V>();
        Adds(sqxLocal, sqxLocal, epsilon_, 1);
        PipeBarrier<PIPE_V>();
        Sqrt(sqxLocal, sqxLocal, 1);
        Duplicate(tmpLocal, (float)1.0, 1);
        PipeBarrier<PIPE_V>();
        Div(sqxLocal, tmpLocal, sqxLocal, 1);
        PipeBarrier<PIPE_V>();

        event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventVS);
        WaitFlag<HardEvent::V_S>(eventVS);
        float rstdValue = sqxLocal.GetValue(0);
        event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventSV);
        WaitFlag<HardEvent::S_V>(eventSV);

        Muls(xFp32Local, xFp32Local, rstdValue, numCol_);
        PipeBarrier<PIPE_V>();
        WaitFlag<HardEvent::MTE3_V>(eventMTE3V);
        Cast(x1Local, xFp32Local, RoundMode::CAST_NONE, numCol_);
        PipeBarrier<PIPE_V>();
        WaitFlag<HardEvent::MTE2_V>(eventMTE2V2);
        Mul(x1Local, x1Local, x2Local, numCol_);
        SetFlag<HardEvent::V_MTE3>(eventVMTE3);
        WaitFlag<HardEvent::V_MTE3>(eventVMTE3);
        DataCopyCustom<T>(normOut_, x1Local, numCol_);
    }

    __aicore__ inline void ProcessBf16()
    {
        LocalTensor<float> ubLocal = unitBuf_.Get<float>();
        LocalTensor<T> xLocal = ubLocal.template ReinterpretCast<T>();
        LocalTensor<T> x1Local = xLocal[0];
        LocalTensor<T> x2Local = xLocal[ubFactor_];
        LocalTensor<float> xFp32Local = ubLocal[ubFactor_];
        LocalTensor<float> sqxLocal = ubLocal[ubFactor_ * 2];
        LocalTensor<float> tmpLocal = ubLocal[ubFactor_ * 3];

        DataCopyCustom<T>(x1Local, normOut_, numCol_);
        event_t eventMTE2V1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventMTE2V1);
        DataCopyCustom<T>(x2Local, residual_, numCol_);
        event_t eventMTE2V2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventMTE2V2);
        WaitFlag<HardEvent::MTE2_V>(eventMTE2V1);
        Cast(xFp32Local, x1Local, RoundMode::CAST_NONE, numCol_);
        WaitFlag<HardEvent::MTE2_V>(eventMTE2V2);
        Cast(sqxLocal, x2Local, RoundMode::CAST_NONE, numCol_);
        PipeBarrier<PIPE_V>();
        Add(xFp32Local, xFp32Local, sqxLocal, numCol_);
        PipeBarrier<PIPE_V>();
        Cast(x1Local, xFp32Local, RoundMode::CAST_RINT, numCol_);
        PipeBarrier<PIPE_V>();
        // copy gamma
        event_t eventVMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(eventVMTE2);
        WaitFlag<HardEvent::V_MTE2>(eventVMTE2);

        DataCopyCustom<T>(x2Local, gamma_, numCol_); // gammaLocal use x2Local
        SetFlag<HardEvent::MTE2_V>(eventMTE2V2);

        // copy x out
        event_t eventVMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventVMTE3);
        WaitFlag<HardEvent::V_MTE3>(eventVMTE3);
        DataCopyCustom<T>(y_, x1Local, numCol_);
        event_t eventMTE3V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(eventMTE3V);

        Cast(xFp32Local, x1Local, RoundMode::CAST_NONE, numCol_);
        PipeBarrier<PIPE_V>();
        Mul(sqxLocal, xFp32Local, xFp32Local, numCol_);
        PipeBarrier<PIPE_V>();
        Muls(sqxLocal, sqxLocal, avgFactor_, numCol_);
        PipeBarrier<PIPE_V>();
        ReduceSumCustom(sqxLocal, sqxLocal, tmpLocal, numCol_);
        PipeBarrier<PIPE_V>();
        Adds(sqxLocal, sqxLocal, epsilon_, 1);
        PipeBarrier<PIPE_V>();
        Sqrt(sqxLocal, sqxLocal, 1);
        Duplicate(tmpLocal, (float)1.0, 1);
        PipeBarrier<PIPE_V>();
        Div(sqxLocal, tmpLocal, sqxLocal, 1);
        PipeBarrier<PIPE_V>();

        event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventVS);
        WaitFlag<HardEvent::V_S>(eventVS);
        float rstdValue = sqxLocal.GetValue(0);
        event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventSV);
        WaitFlag<HardEvent::S_V>(eventSV);
        Muls(xFp32Local, xFp32Local, rstdValue, numCol_);
        PipeBarrier<PIPE_V>();
        WaitFlag<HardEvent::MTE3_V>(eventMTE3V);

        Cast(x1Local, xFp32Local, RoundMode::CAST_RINT, numCol_);
        PipeBarrier<PIPE_V>();
        Cast(xFp32Local, x1Local, RoundMode::CAST_NONE, numCol_);
        PipeBarrier<PIPE_V>();
        WaitFlag<HardEvent::MTE2_V>(eventMTE2V2);

        Cast(sqxLocal, x2Local, RoundMode::CAST_NONE, numCol_);
        PipeBarrier<PIPE_V>();
        Mul(xFp32Local, xFp32Local, sqxLocal, numCol_);
        PipeBarrier<PIPE_V>();
        Cast(x1Local, xFp32Local, RoundMode::CAST_RINT, numCol_);
        SetFlag<HardEvent::V_MTE3>(eventVMTE3);
        WaitFlag<HardEvent::V_MTE3>(eventVMTE3);
        DataCopyCustom<T>(normOut_, x1Local, numCol_);
    }

private:
    TBuf<TPosition::VECCALC> unitBuf_;
    GlobalTensor<T> normOut_;
    GlobalTensor<T> residual_;
    GlobalTensor<T> gamma_;
    GlobalTensor<T> y_;

    uint32_t numCol_;
    uint32_t ubFactor_;
    float epsilon_;
    float avgFactor_;
};
#endif // MC2_ADD_RMS_NORM_SINGLE_N_H
