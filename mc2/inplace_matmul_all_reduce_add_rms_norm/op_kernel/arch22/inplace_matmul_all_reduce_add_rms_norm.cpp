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
 * \file inplace_matmul_all_reduce_add_rms_norm.cpp
 * \brief
 */

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "lib/matmul_intf.h"

#if defined(__CCE_KT_TEST__)
#define DTYPE_RESIDUAL half
#endif
using DTYPE_Y = DTYPE_RESIDUAL;

#include "../../../matmul_all_reduce/op_kernel/matmul_all_reduce_common.h"
#if defined(MC2_QUANT)
#include "../../../matmul_all_reduce_add_rms_norm/op_kernel/matmul_all_reduce_add_rms_norm_tiling_data.h"
#include "../../../matmul_all_reduce_add_rms_norm/op_kernel/mm_allreduce_add_rms_norm_quant.h"
#elif defined(MC2_WEIGHT_QUANT)
#include "../../../matmul_all_reduce_add_rms_norm/op_kernel/matmul_all_reduce_add_rms_norm_tiling_data.h"
#include "../../../matmul_all_reduce_add_rms_norm/op_kernel/mm_allreduce_add_rms_norm_weight_quant.h"
#else
#include "../../../matmul_all_reduce_add_rms_norm/op_kernel/matmul_all_reduce_add_rms_norm_tiling_data.h"
#include "../../../matmul_all_reduce_add_rms_norm/op_kernel/mm_allreduce_add_rms_norm_910_general.h"
#endif

namespace MatmulAllReduceAddRmsNormImpl {
}

using namespace AscendC;
using namespace MatmulAllReduceAddRmsNormImpl;

extern "C" __global__ __aicore__ void
inplace_matmul_all_reduce_add_rms_norm(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR residualGM, GM_ADDR gammaGM,
                                       GM_ADDR antiquantScaleGM, GM_ADDR antiquantOffsetGM, GM_ADDR dequantGM,
                                       GM_ADDR yGM, GM_ADDR normOutGM, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
#ifdef __CCE_KT_TEST__
    REGISTER_TILING_DEFAULT(Mc2Tiling::MatmulAllReduceAddRmsNormTilingData);
#endif
    if (workspaceGM == nullptr) {
        return;
    }

    GM_ADDR userWS = GetUserWorkspace(workspaceGM);
    if (userWS == nullptr) {
        return;
    }

    /** 1、支持类型
     *    通用910 通用310 A16W4 A16W8 A8W8
     * 2、是否支持L2Cache
     * 3、B矩阵是否做ND2NZ
     * 4、Bias是否做bf162fp16
     */
    TPipe tPipe;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    REGISTER_TILING_DEFAULT(Mc2Tiling::MatmulAllReduceAddRmsNormTilingData);
    // 8 and 4104 belongs to MC2_QUANT_FP16 or MC2_QUANT_BF16
    REGISTER_TILING_FOR_TILINGKEY("(TILING_KEY_VAR == 8) || (TILING_KEY_VAR == 4104)",
                                  Mc2Tiling::QuantMatmulAllReduceAddRmsNormTilingData);
    // More than 10000 belongs to MC2_WEIGHT_QUANT
    REGISTER_TILING_FOR_TILINGKEY("(TILING_KEY_VAR > 10000)",
                                  Mc2Tiling::WeightQuantMatmulAllReduceAddRmsNormTilingData);
#if defined(MC2_QUANT_FP16)
    if (TILING_KEY_IS(8UL)) {
        INVOKE_MC2_ARN_QUANT_910_OP_IMPL(BmmDequant, Mc2CoreType::ON_CUBE_AND_VECTOR, REG_NO_MM_OBJ, int32_t, uint64_t,
                                         DTYPE_Y, false, false);
    } else if (TILING_KEY_IS(4104UL)) {
        INVOKE_MC2_ARN_QUANT_910_OP_IMPL(BmmDequant, Mc2CoreType::ON_CUBE_AND_VECTOR, REG_NO_MM_OBJ, int32_t, uint64_t,
                                         DTYPE_Y, false, true);
    }
#elif defined(MC2_QUANT_BF16)
    if (TILING_KEY_IS(8UL)) {
        INVOKE_MC2_ARN_QUANT_910_OP_IMPL(BmmDequantBf16, Mc2CoreType::ON_VECTOR, REG_MM_OBJ_FOR_ARN, DTYPE_Y, DTYPE_Y,
                                         false, false, true);
    } else if (TILING_KEY_IS(4104UL)) {
        INVOKE_MC2_ARN_QUANT_910_OP_IMPL(BmmDequantBf16, Mc2CoreType::ON_VECTOR, REG_MM_OBJ_FOR_ARN, DTYPE_Y, DTYPE_Y,
                                         false, true, true);
    }
#elif defined(MC2_WEIGHT_QUANT)
    if (TILING_KEY_IS(2293772UL)) {
        INVOKE_MC2_ARN_WEIGHT_QUANT_910_OP_IMPL(false, Mc2QuantType::PER_TENSOR, false);
    } else if (TILING_KEY_IS(3342348UL)) {
        INVOKE_MC2_ARN_WEIGHT_QUANT_910_OP_IMPL(false, Mc2QuantType::PER_TENSOR, true);
    } else if (TILING_KEY_IS(69402636UL)) {
        INVOKE_MC2_ARN_WEIGHT_QUANT_910_OP_IMPL(true, Mc2QuantType::PER_TENSOR, false);
    } else if (TILING_KEY_IS(70451212UL)) {
        INVOKE_MC2_ARN_WEIGHT_QUANT_910_OP_IMPL(true, Mc2QuantType::PER_TENSOR, true);
    } else if (TILING_KEY_IS(4390924UL)) {
        INVOKE_MC2_ARN_WEIGHT_QUANT_910_OP_IMPL(false, Mc2QuantType::PER_CHANNEL, false);
    } else if (TILING_KEY_IS(5439500UL)) {
        INVOKE_MC2_ARN_WEIGHT_QUANT_910_OP_IMPL(false, Mc2QuantType::PER_CHANNEL, true);
    } else if (TILING_KEY_IS(71499788UL)) {
        INVOKE_MC2_ARN_WEIGHT_QUANT_910_OP_IMPL(true, Mc2QuantType::PER_CHANNEL, false);
    } else if (TILING_KEY_IS(72548364UL)) {
        INVOKE_MC2_ARN_WEIGHT_QUANT_910_OP_IMPL(true, Mc2QuantType::PER_CHANNEL, true);
    } else if (TILING_KEY_IS(6488076UL)) {
        INVOKE_MC2_ARN_WEIGHT_QUANT_910_OP_IMPL(false, Mc2QuantType::PER_GROUP, false);
    } else if (TILING_KEY_IS(73596940UL)) {
        INVOKE_MC2_ARN_WEIGHT_QUANT_910_OP_IMPL(true, Mc2QuantType::PER_GROUP, false);
    } else if (TILING_KEY_IS(7536652UL)) {
        INVOKE_MC2_ARN_WEIGHT_QUANT_910_OP_IMPL(false, Mc2QuantType::PER_GROUP, true);
    } else if (TILING_KEY_IS(74645516UL)) {
        INVOKE_MC2_ARN_WEIGHT_QUANT_910_OP_IMPL(true, Mc2QuantType::PER_GROUP, true);
    }
#else
    // 910非量化
    if (TILING_KEY_IS(260UL) || TILING_KEY_IS(256UL)) {
        INVOKE_MC2_ARN_910_OP_IMPL(Mc2MatmulBaseKernel);
    } else if (TILING_KEY_IS(0UL)) {
        INVOKE_MC2_ARN_910_OP_IMPL(Mc2MatmulBaseUnAlignedKernel);
    }
#endif
}
