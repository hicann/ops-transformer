/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HC_GRAD_NN_FUSION_ADAPTER_H
#define HC_GRAD_NN_FUSION_ADAPTER_H
#include "nn_base/tgrg_mat_mul_base_kernel.h"
namespace TextGatedResidualGrad {
class HcGradNnStrideBlock : public MatmulBaseBlock {
public:
    uint64_t lda = 0, ldb = 0, ldc = 0;
    template <class A, class B, class C, class Bias>
    __aicore__ inline void CalcGMOffset(uint64_t, uint64_t)
    {
        const uint64_t m =
            params_.index / params_.nCntUse * matmulTilingData_->matmulTiling.singleCoreM + params_.mTileAddrOffset;
        const uint64_t n =
            params_.index % params_.nCntUse * matmulTilingData_->matmulTiling.singleCoreN + params_.nTileAddrOffset;
        offset_.offsetA = params_.isTransposeA ? m : m * lda;
        offset_.offsetB = n;
        offset_.offsetC = m * ldc + n;
    }
};
template <typename T, bool TRANS_A, typename C = float>
class HcGradNnKernel
    : public MatmulBaseKernel<matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T, TRANS_A>,
                              matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T>,
                              matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, C>,
                              matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>, HcGradNnStrideBlock> {
public:
    __aicore__ inline void SetStrides(uint32_t m, uint32_t k, uint32_t lda, uint32_t ldb, uint32_t ldc)
    {
        this->block_.lda = lda;
        this->block_.ldb = ldb;
        this->block_.ldc = ldc;
        // SetOrgShape describes physical storage, SetSingleShape in NN Process
        // describes logical tails. In particular packed rows have stride R+16.
        this->mm_.SetOrgShape(TRANS_A ? lda : m, ldb, TRANS_A ? k : lda, k, ldc);
    }
};
} // namespace TextGatedResidualGrad
#endif
