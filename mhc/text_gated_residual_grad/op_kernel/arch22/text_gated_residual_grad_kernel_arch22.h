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
 * \file text_gated_residual_grad_kernel_arch22.h
 * \brief Atlas A2/A3 specialization of the shared TextGatedResidualGrad kernel.
 */
#ifndef TEXT_GATED_RESIDUAL_GRAD_KERNEL_ARCH22_H
#define TEXT_GATED_RESIDUAL_GRAD_KERNEL_ARCH22_H

#include "../arch35/text_gated_residual_grad_kernel_common.h"
#include "text_gated_residual_grad_gemm_arch22.h"

namespace TextGatedResidualGrad {

// Atlas A2/A3: CATLASS AtlasA2 BlockMmad; G56 and C1b read Vector results directly as ColumnMajor,
// as on A5. Init/Process/RunCube use the shared pipeline.
template <typename D_T, int HAS_GAMMA>
using KernelTextGatedResidualGradArch22 =
    KernelTextGatedResidualGrad<D_T, HAS_GAMMA, HcGradGemmArch22<D_T>, HC_GRAD_USE_DUP_T, false>;

} // namespace TextGatedResidualGrad

#endif // TEXT_GATED_RESIDUAL_GRAD_KERNEL_ARCH22_H
