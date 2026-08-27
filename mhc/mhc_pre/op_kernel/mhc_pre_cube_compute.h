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
 * \file mhc_pre_cube_compute.h
 * \brief
 */
#ifndef MHC_PRE_CUBE_COMPUTE_H
#define MHC_PRE_CUBE_COMPUTE_H

#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#include "mhc_pre_base.h"
#include "lib/matmul_intf.h"
#include <cstdint>

using AscendC::GlobalTensor;
using AscendC::TPosition;
using namespace AscendC;

namespace MhcPre {

constexpr MatmulConfig MHC_PRE_GRAD_MM1_CFG = GetMDLConfig(true, false, 0, false, false, false, true);

#define MHC_PRE_CUBE_COMPUTE_TEMPLATE_PARAM template <typename T, bool isFac, bool hasResi>
#define MHC_PRE_CUBE_COMPUTE_TEMPLATE_CLASS MhcPreCubeCompute<T, isFac, hasResi>

// 不切K使用的矩阵乘实现
MHC_PRE_CUBE_COMPUTE_TEMPLATE_PARAM
class MhcPreCubeCompute {
public:
    __aicore__ inline MhcPreCubeCompute(){};
    __aicore__ inline void Init(const GlobalTensor<float> &xGm, const GlobalTensor<float> &phiGm,
                                const GlobalTensor<float> &workspaceGlobalAB, int64_t bs, int64_t n, int64_t c,
                                int64_t vecCoreNum);
    __aicore__ inline void ProcessMatmulXPhi(const int32_t taskOffset, const int32_t mm1M);

public:
    GlobalTensor<float> xGm_;
    GlobalTensor<float> phiGm_;
    GlobalTensor<float> workspaceGlobalAB_;
    using AType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>;
    using BType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T, true>;
    using CType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>;

    matmul::MatmulImpl<AType, BType, CType, CType, MHC_PRE_GRAD_MM1_CFG> mm1_;

    int64_t n_, c_, curBs_;
    int64_t mm1K_, mm1M_, mm1N_;
    int64_t ping4Cub = 1;
    uint64_t vecCoreNum_ = 0;
    uint64_t blockIdx_ = 0;
};

MHC_PRE_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void MHC_PRE_CUBE_COMPUTE_TEMPLATE_CLASS::Init(const GlobalTensor<float> &xGm,
                                                                 const GlobalTensor<float> &phiGm,
                                                                 const GlobalTensor<float> &workspaceGlobalAB,
                                                                 int64_t bs, int64_t n, int64_t c, int64_t vecCoreNum)
{
    blockIdx_ = GetBlockIdx();
    xGm_ = xGm;
    phiGm_ = phiGm;
    workspaceGlobalAB_ = workspaceGlobalAB;
    vecCoreNum_ = vecCoreNum;
    n_ = n;
    c_ = c;
    curBs_ = bs;

    if constexpr (hasResi) {
        if constexpr (isFac) {
            mm1N_ = Factorial(n_) + 2 * n_;
        } else {
            mm1N_ = n_ * n_ + 2 * n_;
        }
    } else {
        mm1N_ = 2 * n_;
    }

    mm1K_ = n_ * c_;
}

MHC_PRE_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void MHC_PRE_CUBE_COMPUTE_TEMPLATE_CLASS::ProcessMatmulXPhi(const int32_t taskOffset,
                                                                              const int32_t mm1M)
{
    if (mm1M <= 0)
        return;
    uint64_t xxoffset = blockIdx_ * curBs_ * 2 * mm1K_ + ping4Cub * vecCoreNum_ * curBs_ * mm1K_;
    mm1_.SetTensorA(xGm_[blockIdx_ * curBs_ * 2 * mm1K_ + ping4Cub * vecCoreNum_ * curBs_ * mm1K_]);
    mm1_.SetTensorB(phiGm_, true);
    // mm1_.SetHF32(true, 1);
    mm1_.SetOrgShape(mm1M, mm1N_, mm1K_);
    mm1_.SetSingleShape(mm1M, mm1N_, mm1K_);
    mm1_.template IterateAll<false>(workspaceGlobalAB_[taskOffset * mm1N_]);
    mm1_.End();
    ping4Cub = 1 - ping4Cub;
}
} // namespace MhcPre

#endif // MHC_PRE_CUBE_COMPUTE_H
