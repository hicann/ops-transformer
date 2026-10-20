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
 * \file service_rms_norm_arch92.h
 * \brief
 */

#ifndef SERVICE_RMS_NORM_arch92_H
#define SERVICE_RMS_NORM_arch92_H

#include "mla_prolog_comm_arch92.h"
#include "mla_prolog_vector_comm_arch92.h"

#include "vf/vf_comm_arch92.h"
#include "vf/vf_rms_norm_arch92.h"
#include "vf/vf_dynamic_quant_arch92.h"

namespace MlaProlog {

/**
 * @brief RmsNormNormal rmsNorm流程
 * @param outputLocal 输出tensor，[B, S1, H]
 * @param inputGm 输入tensor，[B, S1, H]，dtype支持bf16，fp32，int32
 * @param gammaLocal 系数gamma [H]，dtype只支持bf16
 * @param dequantScaleWDqLocal 权重w反量化系数
 * @param dequantScaleXLocal x反量化系数
 * @param shareTmpUb 临时buffer 所需空间[2 * cnt * sizeof(float) + ALIGN_BLOCK_SIZE]
          ALIGN_BLOCK_SIZE = 32Bytes, cnt = row * col
 * @param rmsNormParams rms所需系数，包括
          reciprocal rmsnorm系数reciprocal
          epsilon rmsnorm系数epsilon
          row 处理的行数；预留参数，当前仅支持单个batch的处理，row为1，对应S1
          col 列数，对应H
 */
template <typename T, typename GammaType, typename C, typename O, typename U, bool knAccumNoDequant = false>
__aicore__ inline void RmsNormNormal(const LocalTensor<O>& outputLocal, const GlobalTensor<T>& inputGm,
                                     const LocalTensor<GammaType>& gammaLocal,
                                     const LocalTensor<float>& dequantScaleWDqLocal,
                                     const LocalTensor<float>& dequantScaleXLocal,
                                     const LocalTensor<uint8_t>& shareTmpUb, RmsNormParam& rmsNormParams,
                                     const SplitKnAccumParams& splitKnParams = DefaultSplitKnAccumParams())
{
    int64_t cnt = rmsNormParams.row * rmsNormParams.col;
    LocalTensor<C> xFp32Local = shareTmpUb.ReinterpretCast<C>();

    // load input [1, col]
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(rmsNormParams.col * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

    const bool knAccum = splitKnParams.NeedAccumulate();
    if (knAccum) {
        if constexpr (std::is_same<T, C>::value) {
            AccumulateKnPartials<C>(xFp32Local, inputGm, 0, static_cast<uint32_t>(rmsNormParams.row),
                                    static_cast<uint32_t>(rmsNormParams.col), splitKnParams.kGroups,
                                    splitKnParams.partialStride);
            if constexpr (knAccumNoDequant) {
                LocalTensor<bfloat16_t> bf16Stage = xFp32Local[cnt].template ReinterpretCast<bfloat16_t>();
                // fp32→bf16 非饱和舍入，对齐非切KN路径 fixpipe F322BF16 语义(保留+-inf/NaN)；
                // 普通 Cast(CAST_RINT) 为饱和语义(+-inf→+-MAX, NaN→0)，会改变溢出数值
                DataSyncBarrier<MemDsbT::UB>();
                CastF32ToB16NoSat(bf16Stage, xFp32Local, static_cast<uint32_t>(cnt));
                Cast(xFp32Local, bf16Stage, RoundMode::CAST_NONE, cnt); // bf16→fp32(回读，恢复fp32计算)
            }
        }
    }

    if constexpr (std::is_same<T, float>::value && std::is_same<U, FP8E8M0>::value) {
        if (!knAccum) {
            DataCopyPad(xFp32Local, inputGm, copyParams, padParams);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        }
    } else if constexpr (std::is_same<T, float>::value && std::is_same<U, float>::value) {
        if (!knAccum) {
            DataCopyPad(xFp32Local, inputGm, copyParams, padParams);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        }
        if constexpr (!knAccumNoDequant) {
            Rectangle rectangleParams{
                static_cast<uint32_t>(rmsNormParams.row), static_cast<uint32_t>(rmsNormParams.col),
                static_cast<uint32_t>(rmsNormParams.col) // columnStride
            };
            Dequant(xFp32Local, xFp32Local, dequantScaleWDqLocal, dequantScaleXLocal, rectangleParams);
        }
    } else if constexpr (std::is_same<T, int32_t>::value) {
        LocalTensor<T> xInt32Local = shareTmpUb.ReinterpretCast<T>();
        if (knAccum) {
            AccumulateKnPartials<T>(xInt32Local, inputGm, 0, static_cast<uint32_t>(rmsNormParams.row),
                                    static_cast<uint32_t>(rmsNormParams.col), splitKnParams.kGroups,
                                    splitKnParams.partialStride);
        } else {
            DataCopyPad(xInt32Local, inputGm, copyParams, padParams);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
        }
        Rectangle rectangleParams{
            static_cast<uint32_t>(rmsNormParams.row), static_cast<uint32_t>(rmsNormParams.col),
            static_cast<uint32_t>(rmsNormParams.col) // columnStride
        };
        Dequant(xFp32Local, xInt32Local, dequantScaleWDqLocal, dequantScaleXLocal, rectangleParams);
    } else {
        LocalTensor<T> inputLocal = xFp32Local[rmsNormParams.col].template ReinterpretCast<T>();
        DataCopyPad(inputLocal, inputGm, copyParams, padParams);

        SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
        // Cast input to fp32 [1, col]
        Cast(xFp32Local, inputLocal, RoundMode::CAST_NONE, cnt);
    }
    SetFlag<HardEvent::MTE3_V>(EVENT_ID1);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);

    if constexpr (std::is_same<C, O>::value) {
        RmsNormVF<C, GammaType, C, C>(outputLocal, xFp32Local, gammaLocal, rmsNormParams);
    } else {
        RmsNormVF<C, GammaType, C, C>(xFp32Local, xFp32Local, gammaLocal, rmsNormParams);
        // Cast xFp32 to outputLocal
        Cast(outputLocal, xFp32Local, RoundMode::CAST_RINT, cnt);
    }
}

/**
 * @brief RmsNormDynamicQuant rmsNorm + dynamicQuant融合流程
          目前C均为float
 * @param outputLocal 输出tensor，[B, S1, H], 量化完后为int8
 * @param outputScales dynamicQuant计算系数输出
 * @param inputGm 输入tensor，[B, S1, H]，dtype支持bf16，fp32，int32
 * @param gammaLocal 输入tensor，[H, ], dtype只支持bf16
 * @param smoothLocal dynamicQuant平滑参数
 * @param dequantScaleWDqLocal 权重w反量化系数
 * @param dequantScaleXLocal x反量化系数
 * @param shareTmpUb 临时buffer 所需空间[(4 * cnt + 8) * sizeof(float)] cnt = row * col
 * @param rmsNormParams rms所需系数，包括
          reciprocal rmsnorm系数reciprocal
          epsilon rmsnorm系数epsilon
          row 处理的行数；预留参数，当前仅支持单个batch的处理，row为1，对应S1
          col 列数，对应H
 * @param enableSmoothScalesCq 表示是否有smoothGm的需求
 * @param splitKnParams ★迁移(Phase B): 切KN 部分和累加参数(默认 {0,0} → NeedAccumulate()==false,
 *        非切KN 行为零变化)。rmsNormCqOffset 指向 ps[0][m](KGroup-0 层内布局与原版相同),
 *        累加靠 partialStride = stepBatchSize×headSizeCq 跨 K 组取同一行。
 */
template <typename T, typename GammaType, typename SmoothType, typename C, typename O, typename U>
__aicore__ inline void RmsNormDynamicQuant(const LocalTensor<O>& outputLocal, const LocalTensor<U>& outputScales,
                                           const GlobalTensor<T>& inputGm, const LocalTensor<GammaType>& gammaLocal,
                                           const LocalTensor<SmoothType>& smoothLocal,
                                           const LocalTensor<float>& dequantScaleWDqLocal,
                                           const LocalTensor<float>& dequantScaleXLocal,
                                           const LocalTensor<uint8_t>& shareTmpUb, RmsNormParam& rmsNormParams,
                                           bool enableSmoothScalesCq = true,
                                           const SplitKnAccumParams& splitKnParams = DefaultSplitKnAccumParams())
{
    int64_t cnt = rmsNormParams.row * rmsNormParams.col;
    LocalTensor<C> xFp32Local = shareTmpUb.ReinterpretCast<C>();
    RmsNormNormal<T, GammaType, C, C, U>(xFp32Local, inputGm, gammaLocal, dequantScaleWDqLocal, dequantScaleXLocal,
                                         shareTmpUb[rmsNormParams.col * sizeof(C)], rmsNormParams, splitKnParams);
    if constexpr (std::is_same<O, fp8_e4m3fn_t>::value && std::is_same<U, fp8_e8m0_t>::value) {
        LocalTensor<bfloat16_t> xBf16Local = xFp32Local[cnt].template ReinterpretCast<bfloat16_t>();
        Cast(xBf16Local, xFp32Local, RoundMode::CAST_ROUND, cnt);
        LocalTensor<uint16_t> outputScalesLocal = outputScales.template ReinterpretCast<uint16_t>();
        LocalTensor<int8_t> outLocal = outputLocal.template ReinterpretCast<int8_t>();
        LocalTensor<uint8_t> tmpLocal = xBf16Local[cnt].template ReinterpretCast<uint8_t>();
        DynamicQuantPerBlockMxfp8Vf<bfloat16_t, fp8_e4m3fn_t>(outLocal, outputScalesLocal, xBf16Local, tmpLocal,
                                                              rmsNormParams.row, rmsNormParams.col);
        LocalTensor<uint8_t> scale = outputScalesLocal.template ReinterpretCast<uint8_t>();
    } else {
        if (enableSmoothScalesCq) {
            Mul(xFp32Local, xFp32Local, smoothLocal, cnt);
        }
        DynamicQuantPerTokenVf(outputLocal, outputScales, xFp32Local, rmsNormParams.row, rmsNormParams.col);
    }
}

template <typename C>
__aicore__ inline void RmsNormKnAccumIssue(const LocalTensor<uint8_t>& slotUb, const GlobalTensor<C>& inputGm,
                                           uint32_t row, uint32_t col, const SplitKnAccumParams& splitKnParams,
                                           uint32_t evtId)
{
    LocalTensor<C> bUb = slotUb[col * sizeof(C)].template ReinterpretCast<C>();
    AccumulateKnPartialsIssue<C>(bUb, inputGm, 0, row, col, splitKnParams.kGroups, splitKnParams.partialStride, evtId);
}

template <typename GammaType, typename SmoothType, typename C, typename O, typename U>
__aicore__ inline void RmsNormKnDqCompute(
    const LocalTensor<O>& outputLocal, const LocalTensor<U>& outputScales, const LocalTensor<GammaType>& gammaLocal,
    const LocalTensor<SmoothType>& smoothLocal, const LocalTensor<float>& dequantScaleWDqLocal,
    const LocalTensor<float>& dequantScaleXLocal, const LocalTensor<uint8_t>& slotUb, RmsNormParam& rmsNormParams,
    bool enableSmoothScalesCq, const SplitKnAccumParams& splitKnParams, uint32_t evtId)
{
    int64_t cnt = rmsNormParams.row * rmsNormParams.col;
    LocalTensor<C> aUb = slotUb.template ReinterpretCast<C>();                                // A 区
    LocalTensor<C> bUb = slotUb[rmsNormParams.col * sizeof(C)].template ReinterpretCast<C>(); // B 区
    AccumulateKnPartialsReduce<C>(bUb, static_cast<uint32_t>(rmsNormParams.row),
                                  static_cast<uint32_t>(rmsNormParams.col), splitKnParams.kGroups, evtId);
    Rectangle rectangleParams{static_cast<uint32_t>(rmsNormParams.row), static_cast<uint32_t>(rmsNormParams.col),
                              static_cast<uint32_t>(rmsNormParams.col)};
    Dequant(bUb, bUb, dequantScaleWDqLocal, dequantScaleXLocal, rectangleParams);
    SetFlag<HardEvent::MTE3_V>(EVENT_ID1);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);
    RmsNormVF<C, GammaType, C, C>(aUb, bUb, gammaLocal, rmsNormParams);
    if (enableSmoothScalesCq) {
        Mul(aUb, aUb, smoothLocal, cnt);
    }
    DynamicQuantPerTokenVf(outputLocal, outputScales, aUb, rmsNormParams.row, rmsNormParams.col);
}

template <typename GammaType, typename SmoothType, typename O, typename U>
__aicore__ inline void RmsNormKnDqComputeInt32(
    const LocalTensor<O>& outputLocal, const LocalTensor<U>& outputScales, const LocalTensor<GammaType>& gammaLocal,
    const LocalTensor<SmoothType>& smoothLocal, const LocalTensor<float>& dequantScaleWDqLocal,
    const LocalTensor<float>& dequantScaleXLocal, const LocalTensor<uint8_t>& slotUb, RmsNormParam& rmsNormParams,
    bool enableSmoothScalesCq, const SplitKnAccumParams& splitKnParams, uint32_t evtId)
{
    int64_t cnt = rmsNormParams.row * rmsNormParams.col;
    LocalTensor<float> aUb = slotUb.template ReinterpretCast<float>(); // A 区
    LocalTensor<int32_t> bUb =
        slotUb[rmsNormParams.col * sizeof(int32_t)].template ReinterpretCast<int32_t>(); // B 区(int32 部分和)
    AccumulateKnPartialsReduce<int32_t>(bUb, static_cast<uint32_t>(rmsNormParams.row),
                                        static_cast<uint32_t>(rmsNormParams.col), splitKnParams.kGroups, evtId);
    Rectangle rectangleParams{static_cast<uint32_t>(rmsNormParams.row), static_cast<uint32_t>(rmsNormParams.col),
                              static_cast<uint32_t>(rmsNormParams.col)};
    Dequant(aUb, bUb, dequantScaleWDqLocal, dequantScaleXLocal, rectangleParams);
    SetFlag<HardEvent::MTE3_V>(EVENT_ID1);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);
    RmsNormVF<float, GammaType, float, float>(aUb, aUb, gammaLocal, rmsNormParams);
    if (enableSmoothScalesCq) {
        Mul(aUb, aUb, smoothLocal, cnt);
    }
    DynamicQuantPerTokenVf(outputLocal, outputScales, aUb, rmsNormParams.row, rmsNormParams.col);
}

template <typename GammaType, typename O, typename U>
__aicore__ inline void RmsNormKnMxCompute(const LocalTensor<O>& outputLocal, const LocalTensor<U>& outputScales,
                                          const LocalTensor<GammaType>& gammaLocal, const LocalTensor<uint8_t>& slotUb,
                                          RmsNormParam& rmsNormParams, const SplitKnAccumParams& splitKnParams,
                                          uint32_t evtId)
{
    int64_t cnt = rmsNormParams.row * rmsNormParams.col;
    LocalTensor<float> aUb = slotUb.template ReinterpretCast<float>(); // A 区
    LocalTensor<float> bUb =
        slotUb[rmsNormParams.col * sizeof(float)].template ReinterpretCast<float>(); // B 区(float 部分和)
    AccumulateKnPartialsReduce<float>(bUb, static_cast<uint32_t>(rmsNormParams.row),
                                      static_cast<uint32_t>(rmsNormParams.col), splitKnParams.kGroups, evtId);
    SetFlag<HardEvent::MTE3_V>(EVENT_ID1);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);
    RmsNormVF<float, GammaType, float, float>(aUb, bUb, gammaLocal, rmsNormParams);
    LocalTensor<bfloat16_t> xBf16Local = bUb.template ReinterpretCast<bfloat16_t>();
    Cast(xBf16Local, aUb, RoundMode::CAST_ROUND, cnt);
    LocalTensor<uint16_t> outputScalesLocal = outputScales.template ReinterpretCast<uint16_t>();
    LocalTensor<int8_t> outLocal = outputLocal.template ReinterpretCast<int8_t>();
    LocalTensor<uint8_t> tmpLocal = xBf16Local[cnt].template ReinterpretCast<uint8_t>();
    DynamicQuantPerBlockMxfp8Vf<bfloat16_t, fp8_e4m3fn_t>(outLocal, outputScalesLocal, xBf16Local, tmpLocal,
                                                          rmsNormParams.row, rmsNormParams.col);
}

template <typename GammaType, typename C, typename O>
__aicore__ inline void RmsNormKnNqCompute(const LocalTensor<O>& outputLocal, const LocalTensor<GammaType>& gammaLocal,
                                          const LocalTensor<uint8_t>& slotUb, RmsNormParam& rmsNormParams,
                                          const SplitKnAccumParams& splitKnParams, uint32_t evtId)
{
    int64_t cnt = rmsNormParams.row * rmsNormParams.col;
    LocalTensor<C> aUb = slotUb.template ReinterpretCast<C>();                                // A 区
    LocalTensor<C> bUb = slotUb[rmsNormParams.col * sizeof(C)].template ReinterpretCast<C>(); // B 区
    AccumulateKnPartialsReduce<C>(bUb, static_cast<uint32_t>(rmsNormParams.row),
                                  static_cast<uint32_t>(rmsNormParams.col), splitKnParams.kGroups, evtId);
    {
        LocalTensor<bfloat16_t> bf16Stage = bUb[cnt].template ReinterpretCast<bfloat16_t>();
        // fp32→bf16 非饱和舍入，对齐非切KN路径 fixpipe F322BF16 语义(保留+-inf/NaN)；
        // 普通 Cast(CAST_RINT) 为饱和语义(+-inf→+-MAX, NaN→0)，会改变溢出数值
        DataSyncBarrier<MemDsbT::UB>();
        CastF32ToB16NoSat(bf16Stage, bUb, static_cast<uint32_t>(cnt));
        Cast(bUb, bf16Stage, RoundMode::CAST_NONE, cnt); // bf16→fp32(回读，恢复fp32计算)
    }
    SetFlag<HardEvent::MTE3_V>(EVENT_ID1);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);
    RmsNormVF<C, GammaType, C, C>(aUb, bUb, gammaLocal, rmsNormParams);
    Cast(outputLocal, aUb, RoundMode::CAST_RINT, cnt); // fp32 → bf16(与 RmsNormNormal 的 O≠C 尾段同款)
}
} // namespace MlaProlog

#endif
