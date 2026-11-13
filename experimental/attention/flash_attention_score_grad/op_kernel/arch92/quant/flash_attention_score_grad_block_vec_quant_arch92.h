/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef FLASH_ATTENTION_SCORE_GRAD_BLOCK_VEC_QUANT_ARCH92_H_
#define FLASH_ATTENTION_SCORE_GRAD_BLOCK_VEC_QUANT_ARCH92_H_
#include "flash_attention_score_grad_vec_context_quant_arch92.h"
#include "vector_api/vf_anti_quant_compute_p_ds_arch92.h"
namespace FagBaseApi {
namespace quant {
TEMPLATES_DEF
class FAGBlockVecQuant : public FAGBlockVecQuantContext<TEMPLATE_ARGS> {
    using Base = FAGBlockVecQuantContext<TEMPLATE_ARGS>;
    // Stats use one cache and event 0; P/dS output retains its two independent slots.
    static constexpr uint32_t STATS_EVENT_ID = 0;
    LocalTensor<float> stats;
    int64_t cachedTask = -1;
    bool statsPending = false;

public:
    __aicore__ inline void InitStatsBuffer(const LocalTensor<float>& common)
    {
        stats = common;

        this->permTensor = common[1536].template ReinterpretCast<uint16_t>();
        for (uint16_t i = 0; i < 32; ++i) {
            for (uint16_t j = 0; j < 4; ++j)
                this->permTensor.SetValue(i * 4 + j, j * 32 + i);
        }
    }
    __aicore__ inline void AllocEventID()
    {
        Base::AllocEventID();
        for (uint32_t i = 0; i < FagBaseApi::quant::SP_SLOTS; ++i) {
            SetFlag<HardEvent::MTE3_V>(i);
        }
    }
    __aicore__ inline void FreeEventID()
    {
        Base::FreeEventID();
        for (uint32_t i = 0; i < FagBaseApi::quant::SP_SLOTS; ++i) {
            WaitFlag<HardEvent::MTE3_V>(i);
        }
    }
    __aicore__ inline void PrepareStats(int64_t task, int64_t offset, uint32_t rows)
    {
        if (cachedTask == task)
            return;
        WaitFlag<HardEvent::V_MTE2>(STATS_EVENT_ID);
        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = rows * sizeof(float);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        copyParams.rsv = 0;
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
        DataCopyPad(stats, this->softmaxMaxGm[offset], copyParams, padParams);
        DataCopyPad(stats[512], this->softmaxSumGm[offset], copyParams, padParams);
        DataCopyPad(stats[1024], this->sfmgWorkspaceGm[offset], copyParams, padParams);
        SetFlag<HardEvent::MTE2_V>(STATS_EVENT_ID);
        cachedTask = task;
        statsPending = true;
    }
    __aicore__ inline void FinishStats()
    {
        SetFlag<HardEvent::V_MTE2>(STATS_EVENT_ID);
    }
    // Shared arithmetic/statistics path; transport and ownership are handled by callers.
    __aicore__ inline void ComputeHalf(const LocalTensor<float>& sp, const LocalTensor<float>& dp,
                                       const LocalTensor<INPUT_TYPE>& pOut, const LocalTensor<INPUT_TYPE>& dsOut,
                                       uint32_t half, FagConstInfo& info, FagRunInfo& run)
    {
        uint32_t rows = run.quantRunInfo.innerS1RealSize[run.quantRunInfo.s1Idx];
        uint16_t n = rows > half * 64 ? Min<uint32_t>(64, rows - half * 64) : 0;
        uint16_t realM = run.quantRunInfo.innerS2RealSize[run.quantRunInfo.s2Idx];
        if (n != 0) {
            LocalTensor<float> max, sum, delta;
            {
                if (statsPending) {
                    WaitFlag<HardEvent::MTE2_V>(STATS_EVENT_ID);
                    NormalizeQuantStats(stats, stats[512], run.commonRunInfo.s1RealSize, info.pScaleD);
                    statsPending = false;
                }
                uint32_t off = run.quantRunInfo.s1Idx * 128 + half * 64;
                max = stats[off];
                sum = stats[512 + off];
                delta = stats[1024 + off];
            }
            ComputePDSQuant(sp, dp, pOut, dsOut, this->permTensor, max, sum, delta, realM, n,
                            run.quantRunInfo.qkDScale * info.scaleValue, run.quantRunInfo.vdyDScale, info.pScaleD,
                            info.pScaleD * info.dsScale);
        }
    }
    // Two complete FP32 inputs share neither address nor lifetime with their
    // quantized outputs. Consume BOTH query halves before returning input FREE.
    __aicore__ inline void ProcessFullOutput(const LocalTensor<INPUT_TYPE>& pl1, const LocalTensor<INPUT_TYPE>& dsl1,
                                             const LocalTensor<float>& sp, const LocalTensor<float>& dp,
                                             const LocalTensor<INPUT_TYPE>& pOut, const LocalTensor<INPUT_TYPE>& dsOut,
                                             uint32_t slot, FagConstInfo& info, FagRunInfo& run)
    {
        WaitFlag<HardEvent::MTE3_V>(slot);
        for (uint32_t half = 0; half < 2; ++half) {
            ComputeHalf(sp[half * 64], dp[half * 64], pOut[half * 8192], dsOut[half * 8192], half, info, run);
        }
        CrossCoreSetFlag<4, PIPE_V>(slot);
        // Do not hold the input slot while waiting for the old L1 consumers.
        if ((run.quantRunInfo.s1Idx & 1) == 0) {
            CrossCoreWaitFlag<4, PIPE_MTE3>(SYNC_UB2L1_P_FLAG);
            CrossCoreWaitFlag<4, PIPE_MTE3>(SYNC_UB2L1_DS_FLAG);
        }
        SetFlag<HardEvent::V_MTE3>(slot);
        WaitFlag<HardEvent::V_MTE3>(slot);
        uint16_t rows = run.quantRunInfo.innerS2RealSize[run.quantRunInfo.s2Idx];
        uint16_t cols = run.quantRunInfo.innerS1RealSize[run.quantRunInfo.s1Idx];
        uint16_t m = Ceil<uint16_t>(rows, 8) * 8;
        // Each NZ plane contains 128 rows x 32 bytes. Copy all valid query
        // planes in one command per tensor; preserve the 4096-byte L1 pitch.
        DataCopyParams copyParams;
        copyParams.blockCount = Ceil<uint16_t>(cols, 32);
        copyParams.blockLen = m;
        copyParams.srcStride = 128 - m;
        copyParams.dstStride = 128 - m;
        auto pDst = pl1.template ReinterpretCast<int8_t>();
        auto dsDst = dsl1.template ReinterpretCast<int8_t>();
        auto pSrc = pOut.template ReinterpretCast<int8_t>();
        auto dsSrc = dsOut.template ReinterpretCast<int8_t>();
        DataCopy(pDst, pSrc, copyParams);
        DataCopy(dsDst, dsSrc, copyParams);
        SetFlag<HardEvent::MTE3_V>(slot);
    }
};

TEMPLATES_DEF
class FAGBlockVecQuantDummy : public FAGBlockVecQuantContextDummy<TEMPLATE_ARGS> {
public:
    __aicore__ inline void InitStatsBuffer(const LocalTensor<float>&) {}
    __aicore__ inline void PrepareStats(int64_t, int64_t, uint32_t) {}
    __aicore__ inline void FinishStats() {}

    __aicore__ inline void ProcessFullOutput(const LocalTensor<INPUT_TYPE>&, const LocalTensor<INPUT_TYPE>&,
                                             const LocalTensor<float>&, const LocalTensor<float>&,
                                             const LocalTensor<INPUT_TYPE>&, const LocalTensor<INPUT_TYPE>&, uint32_t,
                                             FagConstInfo&, FagRunInfo&)
    {}
};
} // namespace quant
} // namespace FagBaseApi
#endif
