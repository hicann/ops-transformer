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
 * \file text_gated_residual_grad_kernel_arch35.h
 * \brief Ascend 950 specialization of the shared TextGatedResidualGrad kernel.
 */
#ifndef TEXT_GATED_RESIDUAL_GRAD_KERNEL_ARCH35_H
#define TEXT_GATED_RESIDUAL_GRAD_KERNEL_ARCH35_H

#include "text_gated_residual_grad_gemm_arch35.h"
#include "text_gated_residual_grad_kernel_common.h"

namespace TextGatedResidualGrad {

// Ascend 950: CATLASS ping-pong GEMM with TLA; C1b reads dup directly as ColumnMajor.
template <typename D_T, int HAS_GAMMA>
using KernelTextGatedResidualGradArch35 = KernelTextGatedResidualGrad<D_T, HAS_GAMMA, HcGradGemm<D_T>, false, false>;

} // namespace TextGatedResidualGrad

#endif // TEXT_GATED_RESIDUAL_GRAD_KERNEL_ARCH35_H
