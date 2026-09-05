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
 * \file block_attention_residuals_grad_regbase_common.h
 * \brief arch35 RegBase VF helpers for block_attention_residuals_grad
 */
#ifndef BLOCK_ATTENTION_RESIDUALS_GRAD_REGBASE_COMMON_H
#define BLOCK_ATTENTION_RESIDUALS_GRAD_REGBASE_COMMON_H

#include "kernel_operator.h"

namespace NsBlockAttentionResidualsGrad {
namespace RegBase {

using namespace AscendC;
using namespace AscendC::MicroAPI;

constexpr uint32_t GRAD_ELEM_PER_BLK_FP32 = 8;
constexpr uint32_t GRAD_SCALAR_LOCAL_ELEMS = GRAD_ELEM_PER_BLK_FP32;

constexpr uint32_t kVlFp32 = AscendC::VECTOR_REG_WIDTH / static_cast<uint32_t>(sizeof(float));

__aicore__ inline uint32_t CeilDivU32(uint32_t a, uint32_t b)
{
    return (a + b - 1U) / b;
}

__aicore__ inline uint32_t RoundUpFp32(uint32_t num)
{
    return CeilDivU32(num, GRAD_ELEM_PER_BLK_FP32) * GRAD_ELEM_PER_BLK_FP32;
}

/*!
 * Half-interval binary-tree fold parameters for ReduceMulSumTree, mirroring the
 * arch22 reduction so the A5 accumulation order matches the passing A2/A3 path.
 */
constexpr uint32_t REDUCE_FOLD_LANE_NUM = 64;
constexpr int32_t REDUCE_FOLD_INTERVAL = 2;
constexpr int32_t REDUCE_FOLD_SHIFT_1 = 1;
constexpr int32_t REDUCE_FOLD_SHIFT_2 = 2;
constexpr int32_t REDUCE_FOLD_SHIFT_4 = 4;
constexpr int32_t REDUCE_FOLD_SHIFT_8 = 8;
constexpr int32_t REDUCE_FOLD_SHIFT_16 = 16;

__aicore__ inline int32_t ArgFindPowerTwo(int32_t n)
{
    n |= n >> REDUCE_FOLD_SHIFT_1;
    n |= n >> REDUCE_FOLD_SHIFT_2;
    n |= n >> REDUCE_FOLD_SHIFT_4;
    n |= n >> REDUCE_FOLD_SHIFT_8;
    n |= n >> REDUCE_FOLD_SHIFT_16;
    return (n + 1) >> 1;
}

/*!
 * Kahan compensated summation over at most REDUCE_FOLD_LANE_NUM lanes,
 * matching the final step of the arch22 ReduceSumHalfInterval helper.
 */
__aicore__ inline float ReduceSumKahan(const LocalTensor<float> &srcLocal, int32_t count)
{
    float sum = 0.0f;
    float comp = 0.0f;
    for (int32_t i = 0; i < count; i++) {
        const float y = srcLocal.GetValue(i) - comp;
        const float t = sum + y;
        comp = (t - sum) - y;
        sum = t;
    }
    return sum;
}

/*! buf[0, len) += buf[srcOffset, srcOffset + len), register-level vector fold. */
__aicore__ inline void AddFoldHalf(const LocalTensor<float> &buf, uint32_t len, uint32_t srcOffset)
{
    __local_mem__ float *bufAddr = (__local_mem__ float *)buf.GetPhyAddr();
    const uint16_t repeatTimes = static_cast<uint16_t>((len + kVlFp32 - 1U) / kVlFp32);
    uint32_t sreg = len;
    __VEC_SCOPE__
    {
        RegTensor<float> x, y, sum;
        MaskReg mask;
        for (uint16_t i = 0; i < repeatTimes; ++i) {
            mask = UpdateMask<float>(sreg);
            DataCopy<float, LoadDist::DIST_NORM>(x, bufAddr + i * kVlFp32);
            DataCopy<float, LoadDist::DIST_NORM>(y, bufAddr + srcOffset + i * kVlFp32);
            Add(sum, x, y, mask);
            DataCopy(bufAddr + i * kVlFp32, sum, mask);
        }
    }
}

/*!
 * Binary-tree reduction: dst[0] = sum(productBuf[0, elemCount)).
 * The caller materializes src0 * src1 into productBuf first, same as arch22.
 * The products are folded with the half-interval tree used by arch22 (tail into
 * head, halve until 64 lanes), then combined with a Kahan summation.
 */
__aicore__ inline void ReduceMulSumTree(const LocalTensor<float> &dstScalar, const LocalTensor<float> &productBuf,
                                        uint32_t elemCount, event_t eventSV, event_t eventVS)
{
    if (elemCount == 0U) {
        return;
    }
    int32_t reduceCount = static_cast<int32_t>(elemCount);
    if (elemCount > REDUCE_FOLD_LANE_NUM) {
        int32_t bodyCount = ArgFindPowerTwo(reduceCount);
        const int32_t tailCount = reduceCount - bodyCount;
        if (tailCount > 0) {
            AddFoldHalf(productBuf, static_cast<uint32_t>(tailCount), static_cast<uint32_t>(bodyCount));
        }
        while (bodyCount > static_cast<int32_t>(REDUCE_FOLD_LANE_NUM)) {
            bodyCount = bodyCount / REDUCE_FOLD_INTERVAL;
            AddFoldHalf(productBuf, static_cast<uint32_t>(bodyCount), static_cast<uint32_t>(bodyCount));
        }
        reduceCount = static_cast<int32_t>(REDUCE_FOLD_LANE_NUM);
    }

    SetFlag<HardEvent::V_S>(eventVS);
    WaitFlag<HardEvent::V_S>(eventVS);
    dstScalar.SetValue(0, ReduceSumKahan(productBuf, reduceCount));
    SetFlag<HardEvent::S_V>(eventSV);
    WaitFlag<HardEvent::S_V>(eventSV);
}

/*!
 * Kahan-compensated accumulation of a newly loaded src into a running sum.
 * sumTensorList[outPos] holds the running sum and the other slot the
 * compensation; roles swap after each update (same scheme as arch22).
 */
__aicore__ inline void KahanSumUpdate(LocalTensor<float> &inputTensor, LocalTensor<float> sumTensorList[2], int32_t len,
                                      int32_t &outPos)
{
    LocalTensor<float> sumTensor = sumTensorList[outPos];
    LocalTensor<float> eTensor = sumTensorList[1 - outPos];
    AscendC::Sub(inputTensor, inputTensor, eTensor, len); // y = x - e
    AscendC::Add(eTensor, inputTensor, sumTensor, len);   // t = y + s
    AscendC::Sub(sumTensor, eTensor, sumTensor, len);     // e = t - s
    AscendC::Sub(sumTensor, sumTensor, inputTensor, len); // e = (t - s) - y
    outPos = 1 - outPos;
}

/*! sum += src * scalarSrc[0]; plain accumulation to keep the A5 UB footprint small. */
__aicore__ inline void FusedMulAddAccumulate(const LocalTensor<float> &sum, const LocalTensor<float> &src,
                                             const LocalTensor<float> &scalarSrc, uint32_t elemCount)
{
    __local_mem__ float *sumAddr = (__local_mem__ float *)sum.GetPhyAddr();
    __local_mem__ float *srcAddr = (__local_mem__ float *)src.GetPhyAddr();
    __local_mem__ float *scAddr = (__local_mem__ float *)scalarSrc.GetPhyAddr();
    uint32_t sreg = elemCount;
    const uint16_t repeatTimes = static_cast<uint16_t>((elemCount + kVlFp32 - 1U) / kVlFp32);
    __VEC_SCOPE__
    {
        RegTensor<float> scReg, x0, s0, acc;
        MaskReg mask;
        DataCopy<float, LoadDist::DIST_BRC_B32>(scReg, scAddr);
        for (uint16_t i = 0; i < repeatTimes; ++i) {
            mask = UpdateMask<float>(sreg);
            DataCopy<float, LoadDist::DIST_NORM>(x0, srcAddr + i * kVlFp32);
            DataCopy<float, LoadDist::DIST_NORM>(s0, sumAddr + i * kVlFp32);
            Mul(acc, x0, scReg, mask);
            Add(acc, acc, s0, mask);
            DataCopy(sumAddr + i * kVlFp32, acc, mask);
        }
    }
}

/*! Copy one float between compact meta slots without requiring 32B alignment. */
__aicore__ inline void CopyScalarToDense(const LocalTensor<float> &dst, const LocalTensor<float> &src)
{
    // DIST_BRC_B32 / DIST_FIRST_ELEMENT_B32 按 32B 块搬运，而 gradProb/probs/
    // gradScore/invRms 都是 4B 步进的紧凑标量槽，非 32B 对齐时会被硬件取整到
    // 错误的 32B 块。这里改为标量单元 SetValue/GetValue，与 arch22 和 split_h 一致。
    LocalTensor<float> d = dst;
    LocalTensor<float> s = src;
    d.SetValue(0, s.GetValue(0));
}

/*!
 * gradV = gradHiddenState * prob + scoreWeight * sc + vRow * varScale
 * scalarBase[0] = prob, scalarBase[24] = sc, scalarBase[32] = varScale
 */
__aicore__ inline void FusedGradVF(const LocalTensor<float> &dst, const LocalTensor<float> &gradHiddenState,
                                   const LocalTensor<float> &vRow, const LocalTensor<float> &scoreWeight,
                                   const LocalTensor<float> &scalarBase, uint32_t hiddenSize)
{
    __local_mem__ float *dstAddr = (__local_mem__ float *)dst.GetPhyAddr();
    __local_mem__ float *gradHiddenStateAddr = (__local_mem__ float *)gradHiddenState.GetPhyAddr();
    __local_mem__ float *vAddr = (__local_mem__ float *)vRow.GetPhyAddr();
    __local_mem__ float *swAddr = (__local_mem__ float *)scoreWeight.GetPhyAddr();
    __local_mem__ float *scalarAddr = (__local_mem__ float *)scalarBase.GetPhyAddr();

    // The dual-half path is only enabled when both halves are whole vectors
    // (H is a multiple of 2*kVlFp32). Every other shape, including odd H and
    // partial vector tails, stays on the single masked path below.
    if ((hiddenSize & (2U * kVlFp32 - 1U)) == 0U) {
        const uint32_t halfCount = hiddenSize >> 1;
        const uint16_t repeatTimes = static_cast<uint16_t>(halfCount / kVlFp32);
        __local_mem__ float *dstAddr2 = dstAddr + halfCount;
        __local_mem__ float *gradHiddenStateAddr2 = gradHiddenStateAddr + halfCount;
        __local_mem__ float *vAddr2 = vAddr + halfCount;
        __local_mem__ float *swAddr2 = swAddr + halfCount;
        __VEC_SCOPE__
        {
            RegTensor<float> probReg, scReg, varReg;
            RegTensor<float> gradHiddenState0, gradHiddenState1, v0, v1, sw0, sw1, acc0, acc1;
            MaskReg mask = CreateMask<float, MaskPattern::ALL>();
            DataCopy<float, LoadDist::DIST_BRC_B32>(probReg, scalarAddr);
            DataCopy<float, LoadDist::DIST_BRC_B32>(scReg, scalarAddr + 3U * GRAD_ELEM_PER_BLK_FP32);
            DataCopy<float, LoadDist::DIST_BRC_B32>(varReg, scalarAddr + 4U * GRAD_ELEM_PER_BLK_FP32);
            for (uint16_t i = 0; i < repeatTimes; ++i) {
                DataCopy<float, LoadDist::DIST_NORM>(gradHiddenState0, gradHiddenStateAddr + i * kVlFp32);
                DataCopy<float, LoadDist::DIST_NORM>(gradHiddenState1, gradHiddenStateAddr2 + i * kVlFp32);
                DataCopy<float, LoadDist::DIST_NORM>(v0, vAddr + i * kVlFp32);
                DataCopy<float, LoadDist::DIST_NORM>(v1, vAddr2 + i * kVlFp32);
                DataCopy<float, LoadDist::DIST_NORM>(sw0, swAddr + i * kVlFp32);
                DataCopy<float, LoadDist::DIST_NORM>(sw1, swAddr2 + i * kVlFp32);
                Mul(acc0, gradHiddenState0, probReg, mask);
                MulAddDst(acc0, sw0, scReg, mask);
                MulAddDst(acc0, v0, varReg, mask);
                Mul(acc1, gradHiddenState1, probReg, mask);
                MulAddDst(acc1, sw1, scReg, mask);
                MulAddDst(acc1, v1, varReg, mask);
                DataCopy(dstAddr + i * kVlFp32, acc0, mask);
                DataCopy(dstAddr2 + i * kVlFp32, acc1, mask);
            }
        }
    } else {
        uint32_t sreg = hiddenSize;
        const uint16_t repeatTimes = static_cast<uint16_t>((hiddenSize + kVlFp32 - 1U) / kVlFp32);
        __VEC_SCOPE__
        {
            RegTensor<float> probReg, scReg, varReg;
            RegTensor<float> gradHiddenState0, v0, sw0, acc0;
            MaskReg mask;
            MaskReg maskAll = CreateMask<float, MaskPattern::ALL>();
            DataCopy<float, LoadDist::DIST_BRC_B32>(probReg, scalarAddr);
            DataCopy<float, LoadDist::DIST_BRC_B32>(scReg, scalarAddr + 3U * GRAD_ELEM_PER_BLK_FP32);
            DataCopy<float, LoadDist::DIST_BRC_B32>(varReg, scalarAddr + 4U * GRAD_ELEM_PER_BLK_FP32);
            for (uint16_t i = 0; i < repeatTimes; ++i) {
                mask = UpdateMask<float>(sreg);
                DataCopy<float, LoadDist::DIST_NORM>(gradHiddenState0, gradHiddenStateAddr + i * kVlFp32);
                DataCopy<float, LoadDist::DIST_NORM>(v0, vAddr + i * kVlFp32);
                DataCopy<float, LoadDist::DIST_NORM>(sw0, swAddr + i * kVlFp32);
                Mul(acc0, gradHiddenState0, probReg, mask);
                MulAddDst(acc0, sw0, scReg, mask);
                MulAddDst(acc0, v0, varReg, mask);
                DataCopy(dstAddr + i * kVlFp32, acc0, mask);
            }
        }
    }
}

/*! gradScore = probs*gradProb - probs*scalar, matching _softmax_backward_data. */
__aicore__ inline void SoftmaxBackwardVf(const LocalTensor<float> &gradScore, const LocalTensor<float> &probs,
                                         const LocalTensor<float> &gradProb, const LocalTensor<float> &scalar,
                                         uint32_t blockCount)
{
    __local_mem__ float *gsAddr = (__local_mem__ float *)gradScore.GetPhyAddr();
    __local_mem__ float *pAddr = (__local_mem__ float *)probs.GetPhyAddr();
    __local_mem__ float *gpAddr = (__local_mem__ float *)gradProb.GetPhyAddr();
    __local_mem__ float *scalarAddr = (__local_mem__ float *)scalar.GetPhyAddr();
    uint32_t sreg = blockCount;
    const uint16_t repeatTimes = static_cast<uint16_t>((blockCount + kVlFp32 - 1U) / kVlFp32);
    __VEC_SCOPE__
    {
        RegTensor<float> scalarReg, pReg, gReg, tmp, tmp2;
        MaskReg mask;
        DataCopy<float, LoadDist::DIST_BRC_B32>(scalarReg, scalarAddr);
        for (uint16_t i = 0; i < repeatTimes; ++i) {
            mask = UpdateMask<float>(sreg);
            DataCopy<float, LoadDist::DIST_NORM>(pReg, pAddr + i * kVlFp32);
            DataCopy<float, LoadDist::DIST_NORM>(gReg, gpAddr + i * kVlFp32);
            Mul(tmp, pReg, gReg, mask);
            Mul(tmp2, pReg, scalarReg, mask);
            Sub(tmp, tmp, tmp2, mask);
            DataCopy(gsAddr + i * kVlFp32, tmp, mask);
        }
    }
}

} // namespace RegBase
} // namespace NsBlockAttentionResidualsGrad

#endif // BLOCK_ATTENTION_RESIDUALS_GRAD_REGBASE_COMMON_H
