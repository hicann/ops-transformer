/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file engram_fused_weight_kernel.h
 * \brief EngramFusedWeight 持久核通用路径 (key 0)
 *
 * 块数多于核数时持久核跨块/跨 UB 循环; 头核与尾核的循环次数 / 尾块元素数由 tiling 下发
 * (blockFactor / tailBlockFactor / perCoreTailElem / tailCoreTailElem), 块内按 VL 两次展开,
 * 尾块以 mask 收尾。
 */

#ifndef ENGRAM_FUSED_WEIGHT_H
#define ENGRAM_FUSED_WEIGHT_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

namespace EngramFusedWeight {
using namespace AscendC;
using AscendC::Reg::CastTrait;
using AscendC::Reg::CreateMask;
using AscendC::Reg::LoadAlign;
using AscendC::Reg::LoadDist;
using AscendC::Reg::MaskMergeMode;
using AscendC::Reg::MaskReg;
using AscendC::Reg::RegLayout;
using AscendC::Reg::RegTensor;
using AscendC::Reg::SatMode;
using AscendC::Reg::StoreAlign;
using AscendC::Reg::UpdateMask;

constexpr CastTrait castTraitB162B32Even = {
    RegLayout::ZERO,
    SatMode::UNKNOWN,
    MaskMergeMode::ZEROING,
    RoundMode::UNKNOWN,
};

template <typename T0, typename T1>
class EngramFusedWeightRegbase {
public:
    __aicore__ inline EngramFusedWeightRegbase(const EngramFusedWeightTilingData* tilingData, TPipe* pipe)
    {
        pipe_ = pipe;
        tl_ = tilingData;
    }

    __aicore__ inline void Init(GM_ADDR weightHidden, GM_ADDR weightEmbed, GM_ADDR weightFused)
    {
        if (GetBlockIdx() >= tl_->usedCoreNums) {
            return;
        }

        // Init GM Tensor
        x1Gm.SetGlobalBuffer((__gm__ T0*)weightHidden);
        x2Gm.SetGlobalBuffer((__gm__ T0*)weightEmbed);
        yGm.SetGlobalBuffer((__gm__ T1*)weightFused);

        // Init Pipe
        int64_t xBufSize = tl_->blockSize * sizeof(T0);
        int64_t yBufSize = tl_->blockSize * sizeof(T1);
        pipe_->InitBuffer(inQueueX1, NUM_TWO, xBufSize);
        pipe_->InitBuffer(inQueueX2, NUM_TWO, xBufSize);
        pipe_->InitBuffer(outQueueY, NUM_TWO, yBufSize);
    }

    __aicore__ inline void Process()
    {
        if (GetBlockIdx() >= tl_->usedCoreNums) {
            return;
        }

        int64_t offset = GetBlockIdx() * tl_->perCoreElem;
        int64_t loopNum = GetBlockIdx() == tl_->usedCoreNums - 1 ? tl_->tailBlockFactor : tl_->blockFactor;
        int64_t tailLen = GetBlockIdx() == tl_->usedCoreNums - 1 ? tl_->tailCoreTailElem : tl_->perCoreTailElem;
        VL = AscendC::VECTOR_REG_WIDTH / sizeof(float);

        for (int64_t i = 0; i < loopNum; ++i) {
            int64_t numLen = (i == loopNum - 1) ? tailLen : tl_->blockSize;
            CopyIn(offset + i * tl_->blockSize, numLen);
            Compute(numLen);
            CopyOut(offset + i * tl_->blockSize, numLen);
        }
    }

private:
    __aicore__ inline void CopyIn(int64_t offset, int64_t numLen)
    {
        LocalTensor<T0> x1Local = inQueueX1.AllocTensor<T0>();
        LocalTensor<T0> x2Local = inQueueX2.AllocTensor<T0>();

        DataCopyExtParams copyParams{
            static_cast<uint16_t>(1),                   // blockCount
            static_cast<uint32_t>(numLen * sizeof(T0)), // blockLen
            static_cast<uint32_t>(0),                   // srcStride
            static_cast<uint32_t>(0),                   // dstStride
            0                                           // rsv
        };
        DataCopyPadExtParams<T0> padParams = {false, 0, 0, 0};
        DataCopyPad<T0>(x1Local, x1Gm[offset], copyParams, padParams);
        DataCopyPad<T0>(x2Local, x2Gm[offset], copyParams, padParams);
        inQueueX1.EnQue(x1Local);
        inQueueX2.EnQue(x2Local);
    }

    __aicore__ inline void Compute(int64_t numLen)
    {
        LocalTensor<T0> x1Local = inQueueX1.DeQue<T0>();
        LocalTensor<T0> x2Local = inQueueX2.DeQue<T0>();
        LocalTensor<T1> yLocal = outQueueY.AllocTensor<T1>();

        __ubuf__ T0* x1Addr = (__ubuf__ T0*)x1Local.GetPhyAddr();
        __ubuf__ T0* x2Addr = (__ubuf__ T0*)x2Local.GetPhyAddr();
        __ubuf__ T1* yAddr = (__ubuf__ T1*)yLocal.GetPhyAddr();

        uint16_t loop = static_cast<uint16_t>(numLen + VL - 1) / VL;
        uint16_t halfLoop = loop / NUM_TWO;
        uint16_t tailLoop = loop - halfLoop * NUM_TWO;
        uint32_t remainder = halfLoop * NUM_TWO;
        uint32_t sreg0 = static_cast<uint32_t>(numLen);
        uint32_t stride = NUM_TWO * VL;

        __ubuf__ T0* x1AddrTwo = x1Addr + VL;
        __ubuf__ T0* x2AddrTwo = x2Addr + VL;
        __ubuf__ T1* yAddrTwo = yAddr + VL;
        __ubuf__ T0* x1AddrRemain = x1Addr + remainder * VL;
        __ubuf__ T0* x2AddrRemain = x2Addr + remainder * VL;
        __ubuf__ T1* yAddrRemain = yAddr + remainder * VL;

        __VEC_SCOPE__
        {
            RegTensor<T0> x1Bf16One;
            RegTensor<T0> x1Bf16Two;
            RegTensor<T0> x2Bf16One;
            RegTensor<T0> x2Bf16Two;
            RegTensor<float> x1F32One;
            RegTensor<float> x1F32Two;
            RegTensor<float> x2F32One;
            RegTensor<float> x2F32Two;
            RegTensor<float> yF32One;
            RegTensor<float> yF32Two;
            MaskReg pregOne;
            MaskReg pregTwo;

            for (uint16_t i = 0; i < halfLoop; ++i) {
                pregOne = UpdateMask<float>(sreg0);
                pregTwo = UpdateMask<float>(sreg0);
                LoadAlign<T0, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(x1Bf16One, x1Addr + i * stride);
                LoadAlign<T0, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(x2Bf16One, x2Addr + i * stride);
                LoadAlign<T0, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(x1Bf16Two, x1AddrTwo + i * stride);
                LoadAlign<T0, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(x2Bf16Two, x2AddrTwo + i * stride);
                Cast<float, T0, castTraitB162B32Even>(x1F32One, x1Bf16One, pregOne);
                Cast<float, T0, castTraitB162B32Even>(x1F32Two, x1Bf16Two, pregTwo);
                Cast<float, T0, castTraitB162B32Even>(x2F32One, x2Bf16One, pregOne);
                Cast<float, T0, castTraitB162B32Even>(x2F32Two, x2Bf16Two, pregTwo);
                Mul(yF32One, x1F32One, x2F32One, pregOne);
                Mul(yF32Two, x1F32Two, x2F32Two, pregTwo);
                StoreAlign(yAddr + i * stride, yF32One, pregOne);
                StoreAlign(yAddrTwo + i * stride, yF32Two, pregTwo);
            }
            for (uint16_t i = 0; i < tailLoop; ++i) {
                pregOne = UpdateMask<float>(sreg0);
                LoadAlign<T0, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(x1Bf16One, x1AddrRemain);
                LoadAlign<T0, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(x2Bf16One, x2AddrRemain);
                Cast<float, T0, castTraitB162B32Even>(x1F32One, x1Bf16One, pregOne);
                Cast<float, T0, castTraitB162B32Even>(x2F32One, x2Bf16One, pregOne);
                Mul(yF32One, x1F32One, x2F32One, pregOne);
                StoreAlign(yAddrRemain, yF32One, pregOne);
            }
        }

        inQueueX1.FreeTensor(x1Local);
        inQueueX2.FreeTensor(x2Local);
        outQueueY.EnQue(yLocal);
    }

    __aicore__ inline void CopyOut(int64_t offset, int64_t numLen)
    {
        LocalTensor<T1> yLocal = outQueueY.DeQue<T1>();
        DataCopyExtParams copyParams{
            static_cast<uint16_t>(1),                   // blockCount
            static_cast<uint32_t>(numLen * sizeof(T1)), // blockLen
            static_cast<uint32_t>(0),                   // srcStride
            static_cast<uint32_t>(0),                   // dstStride
            0                                           // rsv
        };
        DataCopyPad<T1>(yGm[offset], yLocal, copyParams);
        outQueueY.FreeTensor(yLocal);
    }

    uint32_t VL = 0;
    constexpr static uint32_t NUM_TWO = 2;

    TQue<QuePosition::VECIN, NUM_TWO> inQueueX1;
    TQue<QuePosition::VECIN, NUM_TWO> inQueueX2;
    TQue<QuePosition::VECOUT, NUM_TWO> outQueueY;
    GlobalTensor<T0> x1Gm;
    GlobalTensor<T0> x2Gm;
    GlobalTensor<T1> yGm;

    /* ascendc variable */
    TPipe* pipe_ = nullptr;

    const EngramFusedWeightTilingData* tl_ = nullptr;
};
} // namespace EngramFusedWeight

#endif // ENGRAM_FUSED_WEIGHT_H
