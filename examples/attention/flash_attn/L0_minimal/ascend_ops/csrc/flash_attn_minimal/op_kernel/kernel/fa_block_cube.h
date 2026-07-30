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
 * \file fa_block_cube.h
 * \brief Cube-side compute — static tensor, template block size.
 */
#ifndef FA_BLOCK_CUBE_H
#define FA_BLOCK_CUBE_H

#ifdef __DAV_C310_CUBE__

#include "../memcopy/offset_calculator.h"
#include "../matmul/matmul.h"
#include "../memcopy/memory_copy.h"
#include "../fa_kernel_public.h"
#include "kernel_operator_list_tensor_intf.h"
using namespace AscendC;
using namespace AscendC::Impl::Detail;
using namespace fa_base_matmul;

static constexpr uint32_t UB_BMM1_ELEMS = 64 * 128;
static constexpr uint32_t UB_BMM2_ELEMS = 64 * 128;
static constexpr uint32_t L0_AB_ELEMS   = 128 * 128;
static constexpr uint32_t L0C_ELEMS     = 128 * 128;

static constexpr uint32_t FLAG_BMM1_READY  = 0;
static constexpr uint32_t FLAG_L1P_READY   = 1;
static constexpr uint32_t FLAG_BMM2_READY  = 2;

using IN_T  = bfloat16_t;
using T     = float;
using OUT_T = bfloat16_t;

static constexpr uint32_t E_L1Q    = 0;
static constexpr uint32_t E_L1K    = 1;
static constexpr uint32_t E_L1V    = 2;
#define             E_L0AB(s)    (3 + (s))
static constexpr uint32_t E_L0C    = 5;

static constexpr FixpipeConfig COMMON_FP_CFG = {CO2Layout::ROW_MAJOR, true};

template <uint32_t N_BASE, uint32_t D_SIZE>
__aicore__ inline void FixpipeMm1(const LocalTensor<float> &dst, const LocalTensor<float> &l0C,
    uint32_t mActual, uint32_t nActual)
{
    uint32_t nSize = (nActual + 7) >> 3 << 3;
    uint32_t mSize = (mActual + 1) >> 1 << 1;
    FixpipeParamsC310<CO2Layout::ROW_MAJOR> p;
    p.nSize = nSize;
    p.mSize = mSize;
    p.srcStride = ((mSize + 15) / 16) * 16;
    p.dstStride = nSize;
    p.dualDstCtl = 1;
    p.params.ndNum = 1;
    p.params.srcNdStride = 0;
    p.params.dstNdStride = 0;
    Fixpipe<float, float, COMMON_FP_CFG>(dst, l0C, p);
}

template <uint32_t M_BASE, uint32_t D_SIZE>
__aicore__ inline void FixpipeMm2(const LocalTensor<float> &dst, const LocalTensor<float> &l0C,
    uint32_t mActual, uint32_t nActual)
{
    uint32_t nSize = (nActual + 7) >> 3 << 3;
    uint32_t mSize = (mActual + 1) >> 1 << 1;
    FixpipeParamsC310<CO2Layout::ROW_MAJOR> p;
    p.nSize = nSize;
    p.mSize = mSize;
    p.srcStride = ((mSize + 15) / 16) * 16;
    p.dstStride = ((D_SIZE + 15) >> 4 << 4);
    p.dualDstCtl = 1;
    p.params.ndNum = 1;
    p.params.srcNdStride = 0;
    p.params.dstNdStride = 0;
    Fixpipe<float, float, COMMON_FP_CFG>(dst, l0C, p);
}

template <uint32_t M_BASE, uint32_t N_BASE, uint32_t D_SIZE>
__aicore__ inline void ComputeMm1(
    LocalTensor<IN_T> &l1Q, LocalTensor<IN_T> &l1K,
    LocalTensor<IN_T> &l0A, LocalTensor<IN_T> &l0B,
    LocalTensor<float> &l0C,
    LocalTensor<float> &ubBmm1,
    FaGmTensor<IN_T, GmFormat::BSND> &qGm,
    FaGmTensor<IN_T, GmFormat::BSND> &kGm,
    uint32_t bIdx, uint32_t n1Head, uint32_t n2Head,
    uint32_t s1BlockBase, uint32_t s2BlockBase,
    uint32_t mActual, uint32_t nActual,
    bool isFirstS2Loop, bool isLastS2Loop)
{
    CrossCoreWaitFlag<4, PIPE_FIX>(FLAG_BMM1_READY);
    CrossCoreWaitFlag<4, PIPE_FIX>(FLAG_BMM1_READY + 16);

    if (isFirstS2Loop) {
        Mutex::Lock<PIPE_MTE2>(E_L1Q);
        FaL1Tensor<IN_T> qL1{.tensor = l1Q, .rowCount = (((uint32_t)M_BASE + 15) >> 4 << 4)};
        CopyMatrixGmToL1<IN_T, GmFormat::BSND>(qL1, qGm, bIdx, n1Head, s1BlockBase, mActual);
        Mutex::Unlock<PIPE_MTE2>(E_L1Q);
        Mutex::Lock<PIPE_MTE1>(E_L1Q);
    }

    Mutex::Lock<PIPE_MTE2>(E_L1K);
    {
        FaL1Tensor<IN_T> kL1{.tensor = l1K, .rowCount = (((uint32_t)N_BASE + 15) >> 4 << 4)};
        CopyMatrixGmToL1<IN_T, GmFormat::BSND>(kL1, kGm, bIdx, n2Head, s2BlockBase, nActual);
    }
    Mutex::Unlock<PIPE_MTE2>(E_L1K);

    Mutex::Lock<PIPE_MTE1>(E_L1K);
    Mutex::Lock<PIPE_M>(E_L0C);
    {
        MMParam param = MakeMMParam(mActual, nActual, (uint32_t)D_SIZE, false, true);
        MatmulFull<IN_T, IN_T, float, static_cast<uint32_t>(128), static_cast<uint32_t>(128), static_cast<uint32_t>(128), ABLayout::MK, ABLayout::KN>(
            l1Q, l1K, l0A, l0B, l0C, param, E_L0AB(0));
    }
    Mutex::Unlock<PIPE_MTE1>(E_L1K);
    Mutex::Unlock<PIPE_M>(E_L0C);

    if (isLastS2Loop) {
        Mutex::Unlock<PIPE_MTE1>(E_L1Q);
    }

    Mutex::Lock<PIPE_FIX>(E_L0C);
    FixpipeMm1<N_BASE, D_SIZE>(ubBmm1, l0C, mActual, nActual);
    Mutex::Unlock<PIPE_FIX>(E_L0C);

    CrossCoreSetFlag<4, PIPE_FIX>(FLAG_BMM1_READY);
    CrossCoreSetFlag<4, PIPE_FIX>(FLAG_BMM1_READY + 16);
}

template <uint32_t M_BASE, uint32_t N_BASE, uint32_t D_SIZE>
__aicore__ inline void ComputeMm2(
    LocalTensor<IN_T> &l1P, LocalTensor<IN_T> &l1V,
    LocalTensor<IN_T> &l0A, LocalTensor<IN_T> &l0B,
    LocalTensor<float> &l0C, LocalTensor<float> &ubBmm2,
    FaGmTensor<IN_T, GmFormat::BSND> &vGm,
    uint32_t bIdx, uint32_t n2Head, uint32_t s2BlockBase,
    uint32_t mActual, uint32_t nActual)
{
    CrossCoreWaitFlag<4, PIPE_MTE1>(FLAG_L1P_READY);
    CrossCoreWaitFlag<4, PIPE_MTE1>(FLAG_L1P_READY + 16);
    CrossCoreWaitFlag<4, PIPE_FIX>(FLAG_BMM2_READY);
    CrossCoreWaitFlag<4, PIPE_FIX>(FLAG_BMM2_READY + 16);

    Mutex::Lock<PIPE_MTE2>(E_L1V);
    {
        FaL1Tensor<IN_T> vL1{.tensor = l1V, .rowCount = AlignUp((uint32_t)N_BASE, 16U)};
        CopyMatrixGmToL1<IN_T, GmFormat::BSND>(vL1, vGm, bIdx, n2Head, s2BlockBase, nActual);
    }
    Mutex::Unlock<PIPE_MTE2>(E_L1V);

    Mutex::Lock<PIPE_MTE1>(E_L1V);
    Mutex::Lock<PIPE_M>(E_L0C);
    {
        MMParam param = MakeMMParam(mActual, (uint32_t)D_SIZE, nActual, false, false);
        param.realM = mActual;
        MatmulFull<IN_T, IN_T, float, 128, 128, 128, ABLayout::MK, ABLayout::KN>(
            l1P, l1V, l0A, l0B, l0C, param, E_L0AB(0));
    }
    Mutex::Unlock<PIPE_MTE1>(E_L1V);
    Mutex::Unlock<PIPE_M>(E_L0C);

    Mutex::Lock<PIPE_FIX>(E_L0C);
    FixpipeMm2<M_BASE, D_SIZE>(ubBmm2, l0C, mActual, nActual);
    Mutex::Unlock<PIPE_FIX>(E_L0C);

    CrossCoreSetFlag<4, PIPE_FIX>(FLAG_BMM2_READY);
    CrossCoreSetFlag<4, PIPE_FIX>(FLAG_BMM2_READY + 16);
}

template <uint32_t M_BASE = 128, uint32_t N_BASE = 128, uint32_t D_SIZE = 128>
__aicore__ inline void CubeFunc(__gm__ uint8_t *query, __gm__ uint8_t *key,
                                __gm__ uint8_t *value, __gm__ uint8_t *workspace,
                                __gm__ uint8_t *tiling, uint32_t blockDim,
                                uint32_t B, uint32_t N1, uint32_t N2, uint32_t S1, uint32_t S2)
{
    static constexpr uint32_t L1_P_ELEMS  = M_BASE * N_BASE;
    static constexpr uint32_t L1_Q_ELEMS  = M_BASE * D_SIZE;
    static constexpr uint32_t L1_KV_ELEMS = N_BASE * D_SIZE;

    uint32_t bn1Total = B * N1;
    uint32_t bn1PerCore = (bn1Total + blockDim - 1) / blockDim;
    uint32_t aicIdx = GetBlockIdx();
    uint32_t bn1Start = aicIdx * bn1PerCore;
    uint32_t bn1End = bn1Start + bn1PerCore;
    if (bn1End > bn1Total) {
        bn1End = bn1Total;
    }

    uint32_t numS1Blocks = (S1 + M_BASE - 1) / M_BASE;
    uint32_t numS2Blocks = (S2 + N_BASE - 1) / N_BASE;

    FaGmTensor<IN_T, GmFormat::BSND> qGm((__gm__ IN_T *)query, B, N1, S1, (uint32_t)D_SIZE);
    FaGmTensor<IN_T, GmFormat::BSND> kGm((__gm__ IN_T *)key,   B, N2, S2, (uint32_t)D_SIZE);
    FaGmTensor<IN_T, GmFormat::BSND> vGm((__gm__ IN_T *)value, B, N2, S2, (uint32_t)D_SIZE);

    uint32_t l1Addr = 0;
    LocalTensor<IN_T> l1P = LocalTensor<IN_T>(TPosition::A1, l1Addr, L1_P_ELEMS);
    l1Addr += L1_P_ELEMS * sizeof(IN_T);
    LocalTensor<IN_T> l1Q = LocalTensor<IN_T>(TPosition::A1, l1Addr, L1_Q_ELEMS);
    l1Addr += L1_Q_ELEMS * sizeof(IN_T);
    LocalTensor<IN_T> l1K = LocalTensor<IN_T>(TPosition::A1, l1Addr, L1_KV_ELEMS);
    l1Addr += L1_KV_ELEMS * sizeof(IN_T);
    LocalTensor<IN_T> l1V = LocalTensor<IN_T>(TPosition::A1, l1Addr, L1_KV_ELEMS);

    LocalTensor<IN_T> l0A = LocalTensor<IN_T>(TPosition::A2, 0, L0_AB_ELEMS);
    LocalTensor<IN_T> l0B = LocalTensor<IN_T>(TPosition::B2, 0, L0_AB_ELEMS);

    LocalTensor<float> l0C = LocalTensor<float>(TPosition::CO1, 0, L0C_ELEMS);

    uint32_t ubAddr = 0;
    LocalTensor<float> ubBmm1 = LocalTensor<float>(TPosition::VECIN, ubAddr, UB_BMM1_ELEMS);
    ubAddr += UB_BMM1_ELEMS * sizeof(float);
    LocalTensor<float> ubBmm2 = LocalTensor<float>(TPosition::VECIN, ubAddr, UB_BMM2_ELEMS);

    for (uint32_t bn1 = bn1Start; bn1 < bn1End; bn1++) {
        uint32_t bIdx   = bn1 / N1;
        uint32_t n1Head = bn1 % N1;
        uint32_t n2Head = n1Head * N2 / N1;

        for (uint32_t s1Block = 0; s1Block < numS1Blocks; s1Block++) {
            uint32_t s1BlockBase = s1Block * M_BASE;
            uint32_t mActual = s1BlockBase + M_BASE > S1 ? S1 - s1BlockBase : (uint32_t)M_BASE;
            bool isQFirst = true;

            for (uint32_t s2Block = 0; s2Block < numS2Blocks; s2Block++) {
                uint32_t s2BlockBase = s2Block * N_BASE;
                uint32_t nActual = s2BlockBase + N_BASE > S2 ? S2 - s2BlockBase : (uint32_t)N_BASE;

                ComputeMm1<M_BASE, N_BASE, D_SIZE>(l1Q, l1K, l0A, l0B,
                    l0C, ubBmm1,
                    qGm, kGm, bIdx, n1Head, n2Head,
                    s1BlockBase, s2BlockBase, mActual, nActual,
                    isQFirst, (s2Block + 1) == numS2Blocks);
                isQFirst = false;

                ComputeMm2<M_BASE, N_BASE, D_SIZE>(l1P, l1V, l0A, l0B,
                    l0C, ubBmm2,
                    vGm, bIdx, n2Head, s2BlockBase, mActual, nActual);
            }
        }
    }

    CrossCoreWaitFlag<4, PIPE_FIX>(FLAG_BMM1_READY);
    CrossCoreWaitFlag<4, PIPE_FIX>(FLAG_BMM1_READY + 16);
    CrossCoreWaitFlag<4, PIPE_FIX>(FLAG_BMM2_READY);
    CrossCoreWaitFlag<4, PIPE_FIX>(FLAG_BMM2_READY + 16);
}
#endif // __DAV_C310_CUBE__
#endif // FA_BLOCK_CUBE_H
