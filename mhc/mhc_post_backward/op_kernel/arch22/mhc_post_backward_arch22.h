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
 * \file mhc_post_backward_arch22.h
 * \brief
 */

#ifndef MHC_POST_BACKWARD_ARCH22_H
#define MHC_POST_BACKWARD_ARCH22_H

#include "kernel_operator.h"
#include "mhc_post_backward_tiling_data_arch22.h"
using namespace AscendC;

constexpr float ZERO = 0;

template <typename T>
class KernelMhcPostBackward {
public:
    __aicore__ inline KernelMhcPostBackward() {}

    __aicore__ inline void Init(GM_ADDR grad_y, GM_ADDR x, GM_ADDR h_res, GM_ADDR h_out, GM_ADDR h_post, GM_ADDR grad_x,
                                GM_ADDR grad_h_res, GM_ADDR grad_h_out, GM_ADDR grad_h_post,
                                const MhcPostBackwardTilingDataArch22 &tilingData, TPipe *pipe);
    __aicore__ inline void Process();

protected:
    static constexpr uint64_t BUFFER_NUM = 1;

    __aicore__ inline void VecMatmulMknk(LocalTensor<float> &A, LocalTensor<float> &B, LocalTensor<float> &C,
                                         LocalTensor<float> &broadcastBuffer, LocalTensor<float> &reduceBuffer,
                                         uint32_t m, uint32_t k, uint32_t n, uint32_t alignN);
    __aicore__ inline void VecMatmulMkkn(LocalTensor<float> &A, LocalTensor<float> &B, LocalTensor<float> &C,
                                         LocalTensor<float> &broadcastBuffer, uint32_t m, uint32_t k, uint32_t n);

    TBuf<TPosition::VECCALC> dFPostResCastBuf, FOutCastBuf, HLPostBuf, xLCastBuf, HLResBuf;
    TBuf<TPosition::VECCALC> dHLPostBuf, dFOutCastBuf, dHLResBuf, dxLCastBuf;
    TBuf<TPosition::VECCALC> dFPostResBuf, FOutBuf, xLBuf, dFOutBuf, dxLBuf;
    TBuf<TPosition::VECCALC> dHLResTmpBuf1, dHLResTmpBuf2, dHLResTmpBuf3, dHLResTmpBuf4;
    TBuf<TPosition::VECCALC> dxLTmpBuf2;

    GlobalTensor<T> dFPostResGm, FOutGm, xLGm;
    GlobalTensor<float> HLResGm, HLPostGm;

    GlobalTensor<T> dxLGm, dFOutGm;
    GlobalTensor<float> dHResGm, dHPostGm;

    LocalTensor<float> dFPostResUb, FOutUb, HLPostUb, xLUb, HLResUb;
    LocalTensor<T> dFPostResCastUb, FOutCastUb, xLCastUb, dFOutCastUb, dxLCastUb;
    LocalTensor<float> dHLPostUb, dFOutUb, dHLResUb, dxLUb;

    LocalTensor<float> dHLResTmp1, dHLResTmp2, dHLResTmp3;

    LocalTensor<float> dxLTmp2;

    uint64_t coreUsed = 0;
    uint64_t singleCoreBS = 0;
    uint64_t tailBS = 0;
    uint64_t frontCore = 0;
    uint64_t tailCore = 0;

    uint64_t blockChannel = 0;
    uint64_t channel = 0;
    uint64_t n = 0;
    uint64_t alignN = 0;
    uint64_t loopC = 0;
    uint64_t tailC = 0;

    uint64_t dFPostResSize = 0;
    uint64_t xSize = 0;
    uint64_t hResSize = 0;
    uint64_t hOutSize = 0;
    uint64_t hPostSize = 0;

    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPadExtParams<float> padParamsFloat{false, 0, 0, 0};
};

template <typename T>
__aicore__ inline void KernelMhcPostBackward<T>::Init(GM_ADDR grad_y, GM_ADDR x, GM_ADDR h_res, GM_ADDR h_out,
                                                      GM_ADDR h_post, GM_ADDR grad_x, GM_ADDR grad_h_res,
                                                      GM_ADDR grad_h_out, GM_ADDR grad_h_post,
                                                      const MhcPostBackwardTilingDataArch22 &tilingData, TPipe *pipe)
{
    this->coreUsed = tilingData.coreUsed;
    this->singleCoreBS = tilingData.singleCoreBS;
    this->tailBS = tilingData.tailBS;
    this->frontCore = tilingData.frontCore;
    this->tailCore = tilingData.tailCore;
    this->coreUsed = tilingData.coreUsed;

    this->blockChannel = tilingData.blockChannel;
    this->channel = tilingData.channel;
    this->n = tilingData.n;
    this->alignN = tilingData.alignN;
    this->loopC = tilingData.loopC;
    this->tailC = tilingData.tailC;

    this->dFPostResSize = tilingData.dFPostResSize;
    this->xSize = tilingData.xSize;
    this->hResSize = tilingData.hResSize;
    this->hOutSize = tilingData.hOutSize;
    this->hPostSize = tilingData.hPostSize;

    this->dFPostResGm.SetGlobalBuffer((__gm__ T *)grad_y, this->dFPostResSize);
    this->FOutGm.SetGlobalBuffer((__gm__ T *)h_out, this->hOutSize);
    this->xLGm.SetGlobalBuffer((__gm__ T *)x, this->xSize);
    this->HLResGm.SetGlobalBuffer((__gm__ float *)h_res, this->hResSize);
    this->HLPostGm.SetGlobalBuffer((__gm__ float *)h_post, this->hPostSize);

    this->dxLGm.SetGlobalBuffer((__gm__ T *)grad_x, this->xSize);
    this->dFOutGm.SetGlobalBuffer((__gm__ T *)grad_h_out, this->hOutSize);
    this->dHResGm.SetGlobalBuffer((__gm__ float *)grad_h_res, this->hResSize);
    this->dHPostGm.SetGlobalBuffer((__gm__ float *)grad_h_post, this->hPostSize);

    pipe->InitBuffer(this->dFPostResCastBuf, this->n * this->blockChannel * sizeof(T));
    pipe->InitBuffer(this->FOutCastBuf, this->blockChannel * sizeof(T));
    pipe->InitBuffer(this->HLPostBuf, this->n * sizeof(float));
    pipe->InitBuffer(this->xLCastBuf, this->n * this->blockChannel * sizeof(T));
    pipe->InitBuffer(this->HLResBuf, this->n * this->n * sizeof(float));

    pipe->InitBuffer(this->dHLPostBuf, this->alignN * sizeof(float));
    pipe->InitBuffer(this->dFOutCastBuf, this->blockChannel * sizeof(T));
    pipe->InitBuffer(this->dHLResBuf, this->n * this->alignN * sizeof(float));
    pipe->InitBuffer(this->dxLCastBuf, this->n * this->blockChannel * sizeof(T));

    pipe->InitBuffer(this->dFPostResBuf, this->n * this->blockChannel * sizeof(float)); // [n, tileC]
    pipe->InitBuffer(this->FOutBuf, this->blockChannel * sizeof(float));                // [1, tileC]
    pipe->InitBuffer(this->xLBuf, this->n * this->blockChannel * sizeof(float));        // [n, tileC]
    pipe->InitBuffer(this->dFOutBuf, this->blockChannel * sizeof(float));               // [1, tileC]
    pipe->InitBuffer(this->dxLBuf, this->n * this->blockChannel * sizeof(float));       // [n, tileC]

    pipe->InitBuffer(this->dHLResTmpBuf1, this->blockChannel * this->n * sizeof(float)); // [tilC, n] [k, n]
    pipe->InitBuffer(this->dHLResTmpBuf2, this->blockChannel * this->n * this->n * sizeof(float));
    pipe->InitBuffer(this->dHLResTmpBuf3, this->n * this->alignN * sizeof(float)); // [n, n] [m, n]
    pipe->InitBuffer(this->dHLResTmpBuf4, this->n * this->n * sizeof(uint8_t));    // [n, n] [m, n]

    pipe->InitBuffer(this->dxLTmpBuf2, this->n * this->n * sizeof(float));
}

template <typename T>
__aicore__ inline void KernelMhcPostBackward<T>::Process()
{
    uint32_t coreId = GetBlockIdx();

    if (coreId >= this->coreUsed) {
        return;
    }

    uint64_t startIdx = coreId > this->frontCore ?
                            (coreId - this->frontCore) * this->tailBS + this->frontCore * this->singleCoreBS :
                            coreId * this->singleCoreBS;

    uint64_t endIdx = startIdx + ((coreId < this->frontCore) ? this->singleCoreBS : this->tailBS);

    this->dFPostResCastUb = this->dFPostResCastBuf.template Get<T>();
    this->FOutCastUb = this->FOutCastBuf.template Get<T>();
    this->HLPostUb = this->HLPostBuf.template Get<float>();
    this->xLCastUb = this->xLCastBuf.template Get<T>();
    this->HLResUb = this->HLResBuf.template Get<float>();

    this->dHLPostUb = this->dHLPostBuf.template Get<float>();
    this->dFOutCastUb = this->dFOutCastBuf.template Get<T>();
    this->dHLResUb = this->dHLResBuf.template Get<float>();
    this->dxLCastUb = this->dxLCastBuf.template Get<T>();

    this->dFPostResUb = this->dFPostResBuf.template Get<float>();
    this->FOutUb = this->FOutBuf.template Get<float>();
    this->xLUb = this->xLBuf.template Get<float>();
    this->dFOutUb = this->dFOutBuf.template Get<float>();
    this->dxLUb = this->dxLBuf.template Get<float>();

    this->dHLResTmp1 = this->dHLResTmpBuf1.template Get<float>();
    this->dHLResTmp2 = this->dHLResTmpBuf2.template Get<float>();
    this->dHLResTmp3 = this->dHLResTmpBuf3.template Get<float>();

    this->dxLTmp2 = this->dxLTmpBuf2.template Get<float>();

    for (uint64_t i = startIdx; i < endIdx; i++) {
        // Hlpost 和Hlres 驻留
        SetFlag<HardEvent::V_MTE2>(0);
        WaitFlag<HardEvent::V_MTE2>(0);

        DataCopyExtParams copyParamsHLPostUb{1, static_cast<uint32_t>(this->n * sizeof(float)), 0, 0, 0};
        DataCopyPad(this->HLPostUb, this->HLPostGm[i * this->n], copyParamsHLPostUb, this->padParamsFloat);
        DataCopyExtParams copyParamsHLResUb{1, static_cast<uint32_t>(this->n * this->n * sizeof(float)), 0, 0, 0};
        DataCopyPad(this->HLResUb, this->HLResGm[i * this->n * this->n], copyParamsHLResUb, this->padParamsFloat);

        // 清0
        SetFlag<HardEvent::MTE3_V>(0);
        WaitFlag<HardEvent::MTE3_V>(0);

        Duplicate(this->dHLPostUb, float(0.0), this->alignN);
        Duplicate(this->dHLResUb, float(0.0), this->n * this->alignN);

        for (int j = 0; j < this->loopC; j++) {
            uint32_t channelStride = this->channel - this->blockChannel;
            SetFlag<HardEvent::V_MTE2>(0);
            WaitFlag<HardEvent::V_MTE2>(0);

            SetFlag<HardEvent::MTE3_V>(1);
            WaitFlag<HardEvent::MTE3_V>(1);

            DataCopyExtParams copyParamsdFPostResUb{static_cast<uint16_t>(this->n),
                                                    static_cast<uint32_t>(this->blockChannel * sizeof(T)),
                                                    static_cast<uint32_t>(channelStride * sizeof(T)), 0, 0};
            DataCopyPad(this->dFPostResCastUb, this->dFPostResGm[i * this->n * this->channel + j * this->blockChannel],
                        copyParamsdFPostResUb, this->padParams);
            DataCopyExtParams copyParamsFOutUb{1, static_cast<uint32_t>(this->blockChannel * sizeof(T)), 0, 0, 0};
            DataCopyPad(this->FOutCastUb, this->FOutGm[i * this->channel + j * this->blockChannel], copyParamsFOutUb,
                        this->padParams);

            DataCopyExtParams copyParamsXLUb{static_cast<uint16_t>(this->n),
                                             static_cast<uint32_t>(this->blockChannel * sizeof(T)),
                                             static_cast<uint32_t>(channelStride * sizeof(T)), 0, 0};
            DataCopyPad(this->xLCastUb, this->xLGm[i * this->n * this->channel + j * this->blockChannel],
                        copyParamsXLUb, this->padParams);

            SetFlag<HardEvent::MTE2_V>(0);
            WaitFlag<HardEvent::MTE2_V>(0);

            SetFlag<HardEvent::MTE3_V>(0);
            WaitFlag<HardEvent::MTE3_V>(0);

            // dHLPost : Fout @ dFPostRes.T [1, blockChannel] @ [n, blockChannel]^T
            // 搬运量：[1, n, blockChannel] + [1, 1, blockChannel]
            // 输出：[1, n] 驻留在ub做累加
            Cast(this->dFPostResUb, this->dFPostResCastUb, RoundMode::CAST_NONE,
                 this->n * this->blockChannel);                                                   // bf16--> fp32
            Cast(this->FOutUb, this->FOutCastUb, RoundMode::CAST_NONE, this->blockChannel);       // bf16--> fp32
            Cast(this->xLUb, this->xLCastUb, RoundMode::CAST_NONE, this->n * this->blockChannel); // bf16--> fp32

            VecMatmulMknk(this->FOutUb, this->dFPostResUb, this->dHLPostUb, this->dHLResTmp1, this->dHLResTmp3, 1,
                          this->blockChannel, this->n, this->alignN);

            // dHres: dF@x^T //x@dF^T
            // [n, C]@[C, n] = [n, n] 驻留在ub做累加
            VecMatmulMknk(this->xLUb, this->dFPostResUb, this->dHLResUb, this->dHLResTmp1, this->dHLResTmp3, this->n,
                          this->blockChannel, this->n, this->alignN);

            // dFout: H_post@dF
            // [1, n]@[n, C] = [1, C] 直接搬出
            Duplicate(this->dFOutUb, float(0.0), this->blockChannel);
            VecMatmulMkkn(this->HLPostUb, this->dFPostResUb, this->dFOutUb, this->dHLResTmp2, 1, this->n,
                          this->blockChannel);

            // dx_l: H_res^T@dF
            // [n, n]@[n, C] = [n, C] 直接搬出,
            Duplicate(this->dxLUb, float(0.0), this->n * this->blockChannel);
            VecMatmulMkkn(this->HLResUb, this->dFPostResUb, this->dxLUb, this->dHLResTmp2, this->n, this->n,
                          this->blockChannel);

            DataCopyExtParams copyParamsdxLGm{static_cast<uint16_t>(this->n),
                                              static_cast<uint32_t>(this->blockChannel * sizeof(T)), 0,
                                              static_cast<uint32_t>(channelStride * sizeof(T)), 0};
            DataCopyExtParams copyParamsdFOutGm{1, static_cast<uint32_t>(this->blockChannel * sizeof(T)), 0, 0, 0};
            Cast(this->dFOutCastUb, this->dFOutUb, RoundMode::CAST_ROUND, this->blockChannel);       // float--> bf16
            Cast(this->dxLCastUb, this->dxLUb, RoundMode::CAST_ROUND, this->n * this->blockChannel); // float--> bf16

            SetFlag<HardEvent::V_MTE3>(0);
            WaitFlag<HardEvent::V_MTE3>(0);
            DataCopyPad(this->dxLGm[i * this->n * this->channel + j * this->blockChannel], this->dxLCastUb,
                        copyParamsdxLGm);
            DataCopyPad(this->dFOutGm[i * this->channel + j * this->blockChannel], this->dFOutCastUb,
                        copyParamsdFOutGm);
        }

        if (this->tailC != 0) {
            // 尾部C的处理
            uint32_t channelStride = this->channel - this->tailC;

            SetFlag<HardEvent::V_MTE2>(0);
            WaitFlag<HardEvent::V_MTE2>(0);

            SetFlag<HardEvent::MTE3_V>(1);
            WaitFlag<HardEvent::MTE3_V>(1);

            DataCopyExtParams copyParamsdFPostResUb{static_cast<uint16_t>(this->n),
                                                    static_cast<uint32_t>(this->tailC * sizeof(T)),
                                                    static_cast<uint32_t>(channelStride * sizeof(T)), 0, 0};
            DataCopyPad(this->dFPostResCastUb,
                        this->dFPostResGm[i * this->n * this->channel + this->loopC * this->blockChannel],
                        copyParamsdFPostResUb, this->padParams);

            DataCopyExtParams copyParamsFOutUb{1, static_cast<uint32_t>(this->tailC * sizeof(T)), 0, 0, 0};
            DataCopyPad(this->FOutCastUb, this->FOutGm[i * this->channel + this->loopC * this->blockChannel],
                        copyParamsFOutUb, this->padParams);
            DataCopyExtParams copyParamsXLUb{static_cast<uint16_t>(this->n),
                                             static_cast<uint32_t>(this->tailC * sizeof(T)),
                                             static_cast<uint32_t>(channelStride * sizeof(T)), 0, 0};
            DataCopyPad(this->xLCastUb, this->xLGm[i * this->n * this->channel + this->loopC * this->blockChannel],
                        copyParamsXLUb, this->padParams);
            SetFlag<HardEvent::MTE2_V>(0);
            WaitFlag<HardEvent::MTE2_V>(0);

            SetFlag<HardEvent::MTE3_V>(0);
            WaitFlag<HardEvent::MTE3_V>(0);

            // dHLPost : Fout @ dFPostRes.T [1, tailC] @ [n, tailC]^T
            // 搬运量：[1, n, tailC] + [1, 1, tailC]
            // 输出：[1, n] 驻留在ub做累加
            Cast(this->dFPostResUb, this->dFPostResCastUb, RoundMode::CAST_NONE, this->n * this->tailC); // bf16--> fp32
            Cast(this->FOutUb, this->FOutCastUb, RoundMode::CAST_NONE, this->tailC);                     // bf16--> fp32
            Cast(this->xLUb, this->xLCastUb, RoundMode::CAST_NONE, this->n * this->tailC);               // bf16--> fp32

            VecMatmulMknk(this->FOutUb, this->dFPostResUb, this->dHLPostUb, this->dHLResTmp1, this->dHLResTmp3, 1,
                          this->tailC, this->n, this->alignN);

            // dHres: dF@x^T
            // [n, tailC]@[tailC, n] = [n, n] 驻留在ub做累加
            VecMatmulMknk(this->xLUb, this->dFPostResUb, this->dHLResUb, this->dHLResTmp1, this->dHLResTmp3, this->n,
                          this->tailC, this->n, this->alignN);

            // dFout: H_post@dF
            // [1, n]@[n, tailC] = [1, tailC] 直接搬出
            Duplicate(this->dFOutUb, float(0.0), this->tailC);
            VecMatmulMkkn(this->HLPostUb, this->dFPostResUb, this->dFOutUb, this->dHLResTmp2, 1, this->n, this->tailC);
            // dx_l: H_res^T@dF
            // [n, n]@[n, tailC] = [n, tailC] 直接搬出
            Duplicate(this->dxLUb, float(0.0), this->n * this->tailC);
            VecMatmulMkkn(this->HLResUb, this->dFPostResUb, this->dxLUb, this->dHLResTmp2, this->n, this->n,
                          this->tailC);

            DataCopyExtParams copyParamsdxLGm{static_cast<uint16_t>(this->n),
                                              static_cast<uint32_t>(this->tailC * sizeof(T)), 0,
                                              static_cast<uint32_t>(channelStride * sizeof(T)), 0};
            DataCopyExtParams copyParamsdFOutGm{1, static_cast<uint32_t>(this->tailC * sizeof(T)), 0, 0, 0};
            Cast(this->dFOutCastUb, this->dFOutUb, RoundMode::CAST_ROUND, this->tailC);       // float--> bf16
            Cast(this->dxLCastUb, this->dxLUb, RoundMode::CAST_ROUND, this->n * this->tailC); // float--> bf16

            SetFlag<HardEvent::V_MTE3>(0);
            WaitFlag<HardEvent::V_MTE3>(0);
            DataCopyPad(this->dxLGm[i * this->n * this->channel + this->loopC * this->blockChannel], this->dxLCastUb,
                        copyParamsdxLGm);
            DataCopyPad(this->dFOutGm[i * this->channel + this->loopC * this->blockChannel], this->dFOutCastUb,
                        copyParamsdFOutGm);
        }

        DataCopyExtParams copyParamsdHLResGm{static_cast<uint16_t>(this->n),
                                             static_cast<uint32_t>(this->n * sizeof(float)), 0, 0, 0};
        DataCopyExtParams copyParamsdHLPostGm{1, static_cast<uint32_t>(this->n * sizeof(float)), 0, 0, 0};

        SetFlag<HardEvent::V_MTE3>(2);
        WaitFlag<HardEvent::V_MTE3>(2);
        DataCopyPad(this->dHPostGm[i * this->n], this->dHLPostUb, copyParamsdHLPostGm);
        DataCopyPad(this->dHResGm[i * this->n * this->n], this->dHLResUb, copyParamsdHLResGm);
    }
}

template <typename T>
__aicore__ inline void KernelMhcPostBackward<T>::VecMatmulMknk(LocalTensor<float> &A, LocalTensor<float> &B,
                                                               LocalTensor<float> &C,
                                                               LocalTensor<float> &broadcastBuffer,
                                                               LocalTensor<float> &reduceBuffer, uint32_t m, uint32_t k,
                                                               uint32_t n, uint32_t alignN)
{
    // Compute matrix multiplication using vector instructions.
    uint32_t reduceShape[] = {n, k};
    uint32_t broadcastSrcShape[] = {1, k};
    constexpr bool isReuse = true;

    auto tempBuffer = this->dHLResTmpBuf4.template Get<uint8_t>();
    for (int32_t i = 0; i < m; i++) {
        // 逐元素相乘
        PipeBarrier<PIPE_V>();
        Mul(broadcastBuffer[0], A[i * k], B[0], k);
        Mul(broadcastBuffer[k], A[i * k], B[k], k);
        Mul(broadcastBuffer[2 * k], A[i * k], B[2 * k], k);
        Mul(broadcastBuffer[3 * k], A[i * k], B[3 * k], k);
        // 沿第0维求和
        PipeBarrier<PIPE_V>();
        ReduceSum<float, AscendC::Pattern::Reduce::AR, isReuse>(reduceBuffer[i * alignN], broadcastBuffer, tempBuffer,
                                                                reduceShape, true);
    }

    // 累加到输出 C
    PipeBarrier<PIPE_V>();
    Add(C, C, reduceBuffer, m * alignN);
}

template <typename T>
__aicore__ inline void KernelMhcPostBackward<T>::VecMatmulMkkn(LocalTensor<float> &A, LocalTensor<float> &B,
                                                               LocalTensor<float> &C,
                                                               LocalTensor<float> &broadcastBuffer, uint32_t m,
                                                               uint32_t k, uint32_t n)
{
    uint32_t broadcastSrcShape[] = {m * k, 1};
    uint32_t broadcastDstShape[] = {m * k, n};
    Broadcast<float, 2, 1>(broadcastBuffer, A, broadcastDstShape, broadcastSrcShape);
    PipeBarrier<PIPE_V>();
    for (int32_t i = 0; i < m; i++) {
        for (int32_t j = 0; j < k; j++) {
            MulAddDst(C[i * n], broadcastBuffer[(i * m + j) * n], B[j * n], n);
        }
    }
}
#endif
