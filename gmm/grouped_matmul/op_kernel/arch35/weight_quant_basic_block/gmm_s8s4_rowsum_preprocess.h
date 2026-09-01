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
 * \file gmm_s8s4_rowsum_preprocess.h
 * \brief Compute the ordinary FP32 row sum of an int8 GMM activation.
 */
#pragma once

#if ASC_DEVKIT_MAJOR >= 9
#include <kernel_basic_intf.h>
#else
#include <kernel_operator.h>
#include <kernel_operator_intf.h>
#endif

namespace GROUPED_MATMUL::S8S4V5 {

class S8S4RowSumPreprocess {
public:
    struct Params {
        GM_ADDR xGmAddr{nullptr};
        GM_ADDR rowSumGmAddr{nullptr};
        uint64_t m{0};
        uint64_t k{0};
        bool enabled{false};
    };

    __aicore__ inline S8S4RowSumPreprocess() {}
    __aicore__ inline ~S8S4RowSumPreprocess() {}

    __aicore__ inline void Init(const Params &params, AscendC::TPipe *pipe)
    {
        m_ = params.m;
        k_ = params.k;
        enabled_ = params.enabled;
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(params.xGmAddr));
        rowSumGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(params.rowSumGmAddr));
        if ASCEND_IS_AIV {
            pipe->InitBuffer(xBuffer_, k_ * sizeof(int8_t));
            pipe->InitBuffer(halfBuffer_, k_ * sizeof(half));
            pipe->InitBuffer(floatBuffer_, k_ * sizeof(float));
            pipe->InitBuffer(reduceWorkBuffer_, REDUCE_WORK_BYTES);
            pipe->InitBuffer(reduceOutBuffer_, DATA_BLOCK_BYTES);
        }
    }

    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIC {
            return;
        }
        if (!enabled_ || m_ == 0 || k_ == 0) {
            return;
        }
        const uint64_t taskIdx = AscendC::GetBlockIdx();
        const uint64_t taskCount = static_cast<uint64_t>(AscendC::GetBlockNum()) * AscendC::GetTaskRation();
        if (taskCount == 0) {
            return;
        }
        for (uint64_t row = taskIdx; row < m_; row += taskCount) {
            ProcessRow(row);
        }
    }

    __aicore__ inline void operator()(const Params &params, AscendC::TPipe *pipe)
    {
        Init(params, pipe);
        Process();
    }

private:
    static constexpr uint32_t DATA_BLOCK_BYTES = 32;
    // Same scratch size used by the existing S8S4 MSD row-sum preprocess.
    static constexpr uint32_t REDUCE_WORK_BYTES = 256 * sizeof(float);

    __aicore__ inline void ProcessRow(uint64_t row)
    {
        auto xLocal = xBuffer_.Get<int8_t>();
        auto halfLocal = halfBuffer_.Get<half>();
        auto floatLocal = floatBuffer_.Get<float>();
        auto reduceWorkLocal = reduceWorkBuffer_.Get<float>();
        auto reduceOutLocal = reduceOutBuffer_.Get<float>();
        const uint32_t elementCount = static_cast<uint32_t>(k_);
        const AscendC::DataCopyPadExtParams<int8_t> padParams{false, 0, 0, 0};
        const AscendC::DataCopyExtParams copyInParams{1, static_cast<uint32_t>(elementCount * sizeof(int8_t)), 0, 0, 0};
        AscendC::DataCopyPad(xLocal, xGm_[row * k_], copyInParams, padParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::Cast(halfLocal, xLocal, AscendC::RoundMode::CAST_NONE, elementCount);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(floatLocal, halfLocal, AscendC::RoundMode::CAST_NONE, elementCount);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ReduceSum(reduceOutLocal, floatLocal, reduceWorkLocal, elementCount);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        const AscendC::DataCopyExtParams copyOutParams{1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
        AscendC::DataCopyPad(rowSumGm_[row], reduceOutLocal, copyOutParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
    }

    AscendC::GlobalTensor<int8_t> xGm_;
    AscendC::GlobalTensor<float> rowSumGm_;
    AscendC::TBuf<AscendC::TPosition::VECIN> xBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> halfBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> floatBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduceWorkBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECOUT> reduceOutBuffer_;
    uint64_t m_{0};
    uint64_t k_{0};
    bool enabled_{false};
};

} // namespace GROUPED_MATMUL::S8S4V5
