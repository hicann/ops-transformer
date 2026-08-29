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
 * \file apply_rotary_pos_emb_grad_common.h
 * \brief
 */

#ifndef APPLY_ROTARY_POS_EMB_GRAD_COMMON_H
#define APPLY_ROTARY_POS_EMB_GRAD_COMMON_H

#include "kernel_operator.h"
#include "op_kernel/load_store_utils.h"
#include "op_kernel/math_util.h"

// 复用原有 apply_rotary_pos_emb 常量定义
#if __has_include("../../apply_rotary_pos_emb/arch35/apply_rotary_pos_emb_common.h")
#include "../../apply_rotary_pos_emb/arch35/apply_rotary_pos_emb_common.h"
#else
#include "../../apply_rotary_pos_emb/op_kernel/arch35/apply_rotary_pos_emb_common.h"
#endif

using namespace AscendC;

namespace ApplyRotaryPosEmbGrad {

constexpr uint32_t VL_FLOAT32_SIZE = GetVRegSize() / sizeof(float);
constexpr uint32_t BLOCK_TYPE_SIZE = GetUbBlockSize();
constexpr uint32_t HALF_COEF = 2;
constexpr uint32_t DOUBLE_BUFFER = 2;

/*
    grad_out_1 = cos_1 * grad_1 + sin_2 * grad_2
    grad_out_2 = cos_2 * grad_2 - sin_1 * grad_1
*/
template <typename T>
__aicore__ inline void HalfGradAlignVF(const LocalTensor<T> &sinTensor, const LocalTensor<T> &cosTensor,
                                       const LocalTensor<T> &inTensor, const LocalTensor<T> &outTensor, uint32_t dLen,
                                       uint32_t dAlign, uint16_t currSNum, uint16_t currDNum)
{
    __ubuf__ T *sinUb = (__ubuf__ T *)sinTensor.GetPhyAddr();
    __ubuf__ T *cosUb = (__ubuf__ T *)cosTensor.GetPhyAddr();
    __ubuf__ T *inUb = (__ubuf__ T *)inTensor.GetPhyAddr();
    __ubuf__ T *outUb = (__ubuf__ T *)outTensor.GetPhyAddr();
    uint32_t halfD = dLen / HALF_COEF;
    uint32_t halfDAlign = Ops::Base::CeilAlign(halfD, static_cast<uint32_t>(BLOCK_TYPE_SIZE / sizeof(T)));
    uint16_t repeatTimes = Ops::Base::CeilDiv(halfD, VL_FLOAT32_SIZE);
    __ubuf__ T *currInUb, *currOutUb, *currSinUb, *currCosUb;

    __VEC_SCOPE__
    {
        Reg::RegTensor<float> vregIn, vregHalfIn, vregSin, vregHalfSin, vregCos, vregHalfCos, vregOut, vregHalfOut;
        Reg::MaskReg preg;
        for (uint16_t sIdx = 0; sIdx < currSNum; sIdx++) {
            currSinUb = sinUb + sIdx * dAlign;
            currCosUb = cosUb + sIdx * dAlign;
            for (uint16_t row = 0; row < currDNum; row++) {
                currInUb = inUb + (sIdx * currDNum + row) * dAlign;
                currOutUb = outUb + (sIdx * currDNum + row) * dAlign;
                uint32_t updateCnt = halfD;
                for (uint16_t i = 0; i < repeatTimes; i++) {
                    preg = Reg::UpdateMask<float>(updateCnt);
                    int32_t offset = i * VL_FLOAT32_SIZE;
                    int32_t halfOffset = offset + halfDAlign;
                    ops::LoadTwoTensorForDtypeT<T>(currInUb, currInUb, vregIn, vregHalfIn, preg, preg, offset,
                                                   halfOffset);
                    ops::LoadTwoTensorForDtypeT<T>(currSinUb, currSinUb, vregSin, vregHalfSin, preg, preg, offset,
                                                   halfOffset);
                    ops::LoadTwoTensorForDtypeT<T>(currCosUb, currCosUb, vregCos, vregHalfCos, preg, preg, offset,
                                                   halfOffset);

                    Mul(vregOut, vregCos, vregIn, preg);
                    Mul(vregHalfSin, vregHalfSin, vregHalfIn, preg);
                    Add(vregOut, vregOut, vregHalfSin, preg);
                    Mul(vregSin, vregSin, vregIn, preg);
                    Mul(vregHalfCos, vregHalfCos, vregHalfIn, preg);
                    Sub(vregHalfOut, vregHalfCos, vregSin, preg);

                    ops::StoreOneTensorForDtypeT<T>(currOutUb, vregOut, preg, offset);
                    ops::StoreOneTensorForDtypeT<T>(currOutUb, vregHalfOut, preg, halfOffset);
                }
            }
        }
    }
}

// ============================================================================
// HalfGradCosSinPartialVF — 合并 dx + grad_cos/grad_sin 部分积 (dCosFlag=1 公共实现)
//   dx[0:d/2]   = grad[0:d/2]*cos[0:d/2] + grad[d/2:d]*sin[d/2:d]
//   dx[d/2:d]   = grad[d/2:d]*cos[d/2:d] - grad[0:d/2]*sin[0:d/2]
//   isAccumulate=false (SET):  cosPartial = grad*x, sinPartial = grad*rot(x)
//   isAccumulate=true  (ADD):  cosPartial += grad*x, sinPartial += grad*rot(x)
//   (rotate(x) = cat(-x2, x1))
// ============================================================================
template <typename T>
__aicore__ inline void HalfGradCosSinPartialVF(const LocalTensor<T> &sinTensor, const LocalTensor<T> &cosTensor,
                                               const LocalTensor<T> &gradIn, const LocalTensor<T> &gradOut,
                                               const LocalTensor<T> &xIn, const LocalTensor<float> &cosPartial,
                                               const LocalTensor<float> &sinPartial, uint32_t dLen, uint32_t dAlign,
                                               uint32_t partialDAlign, int64_t ubFactorN, uint16_t currSNum,
                                               uint16_t currDNum, bool isAccumulate)
{
    __ubuf__ T *sinUb = (__ubuf__ T *)sinTensor.GetPhyAddr();
    __ubuf__ T *cosUb = (__ubuf__ T *)cosTensor.GetPhyAddr();
    __ubuf__ T *gradUb = (__ubuf__ T *)gradIn.GetPhyAddr();
    __ubuf__ T *gradOutUb = (__ubuf__ T *)gradOut.GetPhyAddr();
    __ubuf__ T *xUb = (__ubuf__ T *)xIn.GetPhyAddr();
    __ubuf__ float *cosPartialUb = (__ubuf__ float *)cosPartial.GetPhyAddr();
    __ubuf__ float *sinPartialUb = (__ubuf__ float *)sinPartial.GetPhyAddr();
    uint32_t halfD = dLen / HALF_COEF;
    uint32_t halfDAlign = dAlign / HALF_COEF;
    uint32_t halfPartialDAlign = partialDAlign / HALF_COEF;
    uint16_t repeatTimes = Ops::Base::CeilDiv(halfD, VL_FLOAT32_SIZE);
    __ubuf__ T *currGradUb, *currGradOutUb, *currSinUb, *currCosUb, *currXUb;
    __ubuf__ float *currCosPartialUb, *currSinPartialUb;

    __VEC_SCOPE__
    {
        Reg::RegTensor<float> vregGrad1, vregGrad2, vregCos1, vregCos2, vregSin1, vregSin2, vregX1, vregX2;
        Reg::RegTensor<float> vregTmp1, vregTmp2, vregTmp3, vregTmp4, vregTmp5, vregTmp6, vregTmp7, vregTmp8;
        Reg::MaskReg preg;
        for (uint16_t sIdx = 0; sIdx < currSNum; sIdx++) {
            currSinUb = sinUb + sIdx * dAlign;
            currCosUb = cosUb + sIdx * dAlign;
            for (uint16_t row = 0; row < currDNum; row++) {
                currGradUb = gradUb + (sIdx * currDNum + row) * dAlign;
                currGradOutUb = gradOutUb + (sIdx * currDNum + row) * dAlign;
                currXUb = xUb + (sIdx * currDNum + row) * dAlign;
                currCosPartialUb = cosPartialUb + (sIdx * ubFactorN + row) * partialDAlign;
                currSinPartialUb = sinPartialUb + (sIdx * ubFactorN + row) * partialDAlign;
                uint32_t updateCnt = halfD;
                for (uint16_t i = 0; i < repeatTimes; i++) {
                    preg = Reg::UpdateMask<float>(updateCnt);
                    int32_t offset = i * VL_FLOAT32_SIZE;
                    int32_t halfOffset = offset + halfDAlign;

                    ops::LoadTwoTensorForDtypeT<T>(currGradUb, currGradUb, vregGrad1, vregGrad2, preg, preg, offset,
                                                   halfOffset);
                    ops::LoadTwoTensorForDtypeT<T>(currCosUb, currCosUb, vregCos1, vregCos2, preg, preg, offset,
                                                   halfOffset);
                    ops::LoadTwoTensorForDtypeT<T>(currSinUb, currSinUb, vregSin1, vregSin2, preg, preg, offset,
                                                   halfOffset);
                    ops::LoadTwoTensorForDtypeT<T>(currXUb, currXUb, vregX1, vregX2, preg, preg, offset, halfOffset);

                    // dx[0:d/2] = grad[0:d/2]*cos[0:d/2] + grad[d/2:d]*sin[d/2:d]
                    Mul(vregTmp1, vregCos1, vregGrad1, preg);
                    Mul(vregTmp2, vregSin2, vregGrad2, preg);
                    Add(vregTmp1, vregTmp1, vregTmp2, preg);
                    // dx[d/2:d] = grad[d/2:d]*cos[d/2:d] - grad[0:d/2]*sin[0:d/2]
                    Mul(vregTmp3, vregCos2, vregGrad2, preg);
                    Mul(vregTmp4, vregSin1, vregGrad1, preg);
                    Sub(vregTmp3, vregTmp3, vregTmp4, preg);

                    // grad_cos = grad*x (前/后半)
                    Mul(vregTmp5, vregGrad1, vregX1, preg);
                    Mul(vregTmp6, vregGrad2, vregX2, preg);
                    // grad_sin = -(grad1*x2) + grad2*x1
                    Mul(vregTmp7, vregGrad1, vregX2, preg);
                    Muls(vregTmp7, vregTmp7, -1.0f, preg);
                    Mul(vregTmp8, vregGrad2, vregX1, preg);

                    if (!isAccumulate) {
                        ops::StoreOneTensorForDtypeT<T>(currGradOutUb, vregTmp1, preg, offset);
                        ops::StoreOneTensorForDtypeT<T>(currGradOutUb, vregTmp3, preg, halfOffset);
                        ops::StoreOneTensorForDtypeT<float>(currCosPartialUb, vregTmp5, preg, offset);
                        ops::StoreOneTensorForDtypeT<float>(currCosPartialUb, vregTmp6, preg,
                                                            halfPartialDAlign + offset);
                        ops::StoreOneTensorForDtypeT<float>(currSinPartialUb, vregTmp7, preg, offset);
                        ops::StoreOneTensorForDtypeT<float>(currSinPartialUb, vregTmp8, preg,
                                                            halfPartialDAlign + offset);
                    } else {
                        ops::StoreOneTensorForDtypeT<T>(currGradOutUb, vregTmp1, preg, offset);
                        ops::StoreOneTensorForDtypeT<T>(currGradOutUb, vregTmp3, preg, halfOffset);

                        ops::LoadTwoTensorForDtypeT<float>(currCosPartialUb, currCosPartialUb, vregCos1, vregCos2, preg,
                                                           preg, offset, halfPartialDAlign + offset);
                        Add(vregCos1, vregCos1, vregTmp5, preg);
                        Add(vregCos2, vregCos2, vregTmp6, preg);
                        ops::StoreOneTensorForDtypeT<float>(currCosPartialUb, vregCos1, preg, offset);
                        ops::StoreOneTensorForDtypeT<float>(currCosPartialUb, vregCos2, preg,
                                                            halfPartialDAlign + offset);

                        ops::LoadTwoTensorForDtypeT<float>(currSinPartialUb, currSinPartialUb, vregSin1, vregSin2, preg,
                                                           preg, offset, halfPartialDAlign + offset);
                        Add(vregSin1, vregSin1, vregTmp7, preg);
                        Add(vregSin2, vregSin2, vregTmp8, preg);
                        ops::StoreOneTensorForDtypeT<float>(currSinPartialUb, vregSin1, preg, offset);
                        ops::StoreOneTensorForDtypeT<float>(currSinPartialUb, vregSin2, preg,
                                                            halfPartialDAlign + offset);
                    }
                }
            }
        }
    }
}

// ============================================================================
// ProcessGradTile — 单 tile 单路径公共流程: CopyIn grad(+x) -> Compute -> CopyOut dx
//   kDcosFlag=true : CopyIn x(query/key) + HalfGradCosSinPartialVF 合并计算 dx 与部分积
//   kDcosFlag=false: 仅 HalfGradAlignVF 计算 dx
// 搬运统一按 D/2 拆分：每个半行独立 32B padding 对齐 (cos/sin/grad/x/dx 同布局)
// ============================================================================
template <typename T, bool kDcosFlag, int32_t bufferNum>
__aicore__ inline void ProcessGradTile(const LocalTensor<T> &sinTensor, const LocalTensor<T> &cosTensor,
                                       GlobalTensor<T> &gradXGm, GlobalTensor<T> &gradXOutGm, GlobalTensor<T> &xGm,
                                       TQue<QuePosition::VECIN, bufferNum> &gradXInQue,
                                       TQue<QuePosition::VECOUT, bufferNum> &gradXOutQue,
                                       TQue<QuePosition::VECIN, bufferNum> &xInQue, int64_t gradGmOffset,
                                       int64_t currXN, uint32_t currBS, LocalTensor<float> &cosPartial,
                                       LocalTensor<float> &sinPartial, bool isAccumulate, uint32_t dLen,
                                       uint32_t dAlign, uint32_t partialDAlign, int64_t ubFactorN)
{
    uint32_t dHalfByteSize = static_cast<uint32_t>(dLen / HALF_COEF * sizeof(T));

    // CopyIn: gradX，一次 DataCopyPad 按 D/2 拆分搬运。
    LocalTensor<T> gradXIn = gradXInQue.template AllocTensor<T>();
    DataCopyExtParams copyParams = {static_cast<uint16_t>(currBS * currXN * HALF_COEF), dHalfByteSize, 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(gradXIn, gradXGm[gradGmOffset], copyParams, padParams);
    gradXInQue.EnQue(gradXIn);
    gradXIn = gradXInQue.template DeQue<T>();
    LocalTensor<T> gradXOut = gradXOutQue.template AllocTensor<T>();

    // CopyIn: x (query/key, 仅 dCosFlag) + Compute
    LocalTensor<T> xIn;
    if constexpr (kDcosFlag) {
        xIn = xInQue.template AllocTensor<T>();
        DataCopyPad(xIn, xGm[gradGmOffset], copyParams, padParams);
        xInQue.EnQue(xIn);
        xIn = xInQue.template DeQue<T>();
        HalfGradCosSinPartialVF<T>(sinTensor, cosTensor, gradXIn, gradXOut, xIn, cosPartial, sinPartial, dLen, dAlign,
                                   partialDAlign, ubFactorN, currBS, currXN, isAccumulate);
    } else {
        // HalfGradAlignVF: currSNum=currBS(cos 行数), currDNum=currXN(每 cos 行下的 n 行)
        HalfGradAlignVF<T>(sinTensor, cosTensor, gradXIn, gradXOut, dLen, dAlign, currBS, currXN);
    }

    // CopyOut: dx (currBS 行 × currXN 列)
    gradXOutQue.EnQue(gradXOut);
    gradXOut = gradXOutQue.template DeQue<T>();
    DataCopyPad(gradXOutGm[gradGmOffset], gradXOut, copyParams);
    gradXOutQue.FreeTensor(gradXOut);

    gradXInQue.FreeTensor(gradXIn);
    if constexpr (kDcosFlag) {
        xInQue.FreeTensor(xIn);
    }
}

// ============================================================================
// BatchHalfGradContiguousVF — A 模板 half gradient
// UB 中每行按 dAlign = CeilAlign(D/2, BLOCK/sizeof(T)) * 2 存放（半行分别 32B 对齐落位），
// 后半段偏移为 dAlign/2，保证 D 非 16 对齐（如 D=100）时向量 Load/Store 地址仍然对齐
// ============================================================================
template <typename T, bool IsBBoardcast>
__aicore__ inline void BatchHalfGradContiguousVF(__ubuf__ T *in, __ubuf__ T *cos, __ubuf__ T *sin, __ubuf__ T *out,
                                                 uint16_t sLength, uint16_t bLength, uint16_t nLength, int64_t d,
                                                 int64_t dAlign, int64_t ubFactorS, int64_t ubFactorN)
{
    uint32_t dHalfSize = d / HALF_COEF;
    uint16_t dLoopCount = (dHalfSize + VL_FLOAT32_SIZE - 1) / VL_FLOAT32_SIZE;
    uint32_t dHalfOffset = dAlign / HALF_COEF;

    int32_t bStepUb = ubFactorN * ubFactorS * dAlign;
    int32_t nStepUb = ubFactorS * dAlign;

    __VEC_SCOPE__
    {
        Reg::RegTensor<float> inPart1Reg, inPart2Reg, cosPart1Reg, cosPart2Reg, sinPart1Reg, sinPart2Reg;
        Reg::MaskReg pregLoop;
        __ubuf__ T *currInUb, *currOutUb, *currSinUb, *currCosUb;
        for (uint16_t bIdx = 0; bIdx < bLength; bIdx++) {
            for (uint16_t nIdx = 0; nIdx < nLength; nIdx++) {
                for (uint16_t sIdx = 0; sIdx < sLength; sIdx++) {
                    uint32_t count = dHalfSize;
                    currInUb = in + bIdx * bStepUb + nIdx * nStepUb + sIdx * dAlign;
                    currOutUb = out + bIdx * bStepUb + nIdx * nStepUb + sIdx * dAlign;
                    if constexpr (IsBBoardcast) {
                        currCosUb = cos + sIdx * dAlign;
                        currSinUb = sin + sIdx * dAlign;
                    } else {
                        currCosUb = cos + bIdx * nStepUb + sIdx * dAlign;
                        currSinUb = sin + bIdx * nStepUb + sIdx * dAlign;
                    }
                    for (uint16_t i = 0; i < dLoopCount; i++) {
                        pregLoop = Reg::UpdateMask<float>(count);
                        ops::LoadOneTensorForDtypeT<T>(currInUb, inPart1Reg, pregLoop, i * VL_FLOAT32_SIZE);
                        ops::LoadOneTensorForDtypeT<T>(currInUb, inPart2Reg, pregLoop,
                                                       i * VL_FLOAT32_SIZE + dHalfOffset);
                        ops::LoadOneTensorForDtypeT<T>(currCosUb, cosPart1Reg, pregLoop, i * VL_FLOAT32_SIZE);
                        ops::LoadOneTensorForDtypeT<T>(currCosUb, cosPart2Reg, pregLoop,
                                                       i * VL_FLOAT32_SIZE + dHalfOffset);
                        ops::LoadOneTensorForDtypeT<T>(currSinUb, sinPart1Reg, pregLoop, i * VL_FLOAT32_SIZE);
                        ops::LoadOneTensorForDtypeT<T>(currSinUb, sinPart2Reg, pregLoop,
                                                       i * VL_FLOAT32_SIZE + dHalfOffset);
                        Mul(cosPart1Reg, inPart1Reg, cosPart1Reg, pregLoop);
                        Mul(sinPart2Reg, sinPart2Reg, inPart2Reg, pregLoop);
                        Add(cosPart1Reg, cosPart1Reg, sinPart2Reg, pregLoop);
                        Mul(cosPart2Reg, inPart2Reg, cosPart2Reg, pregLoop);
                        Mul(sinPart1Reg, inPart1Reg, sinPart1Reg, pregLoop);
                        Sub(cosPart2Reg, cosPart2Reg, sinPart1Reg, pregLoop);
                        ops::StoreOneTensorForDtypeT<T>(currOutUb, cosPart1Reg, pregLoop, i * VL_FLOAT32_SIZE);
                        ops::StoreOneTensorForDtypeT<T>(currOutUb, cosPart2Reg, pregLoop,
                                                        i * VL_FLOAT32_SIZE + dHalfOffset);
                    }
                }
            }
        }
    }
}

} // namespace ApplyRotaryPosEmbGrad

#endif // APPLY_ROTARY_POS_EMB_GRAD_COMMON_H
