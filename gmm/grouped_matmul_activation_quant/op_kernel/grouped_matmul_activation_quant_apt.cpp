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
 * \file grouped_matmul_activation_quant_apt.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "arch35/grouped_matmul_activation_quant_mxquant.h"
#include "arch35/grouped_matmul_activation_quant_tiling_key.h"

namespace {
constexpr uint32_t FLOAT_OVERFLOW_MODE_CTRL = 60U;
} // namespace

using namespace AscendC;
using namespace matmul;

template <int8_t QUANT_B_TRANS, int8_t QUANT_A_TRANS>
__global__ __aicore__ void grouped_matmul_activation_quant(GM_ADDR x, GM_ADDR groupList, GM_ADDR weight,
                                                           GM_ADDR weightScale, GM_ADDR bias, GM_ADDR xScale, GM_ADDR y,
                                                           GM_ADDR yScale, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_NONE_TILING;
    (void)bias;
    TPipe tPipe;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    int64_t oriOverflowMode = AscendC::GetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>();
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
    if (QUANT_B_TRANS == GMM_ACTIVATION_QUANT_NO_TRANS && QUANT_A_TRANS == GMM_ACTIVATION_QUANT_NO_TRANS) {
        GroupedMatmulActivationQuant::GmmActivationMxQuant<AscendC::Te::NDExtLayoutPtn, AscendC::Te::NZLayoutPtn>(
            x, weight, weightScale, xScale, groupList, y, yScale, workspace, tiling);
    } else if (QUANT_B_TRANS == GMM_ACTIVATION_QUANT_TRANS && QUANT_A_TRANS == GMM_ACTIVATION_QUANT_NO_TRANS) {
        GroupedMatmulActivationQuant::GmmActivationMxQuant<AscendC::Te::NDExtLayoutPtn, AscendC::Te::ZNLayoutPtn>(
            x, weight, weightScale, xScale, groupList, y, yScale, workspace, tiling);
    }
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(oriOverflowMode);
}
