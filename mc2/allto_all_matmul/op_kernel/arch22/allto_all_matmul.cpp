/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file allto_all_matmul.cpp
 * \brief
 */
#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "lib/matmul_intf.h"
#include "allto_all_matmul_tiling_key.h"
#include "allto_all_matmul.h"
#include "../allto_all_matmul_kernel_base.h" // A3非量化与A5共用流水
#include "../allto_all_matmul_pipeline.h"

using namespace AscendC;
using namespace Mc2Kernel;

#if ((ORIG_DTYPE_X1 == ORIG_DTYPE_X2) && (ORIG_DTYPE_X1 == DT_FLOAT16 || ORIG_DTYPE_X1 == DT_BF16))
#define QUANT_TYPE MC2_NON_QUANT
#elif ((ORIG_DTYPE_X1 == DT_FLOAT16 || ORIG_DTYPE_X1 == DT_BF16) &&                                                    \
       (ORIG_DTYPE_X2 == DT_INT8 || ORIG_DTYPE_X2 == DT_INT4))
#define QUANT_TYPE MC2_DYNAMIC_QUANT
#else
#define QUANT_TYPE MC2_STATIC_QUANT
#endif

#ifndef ALLTO_ALL_MATMUL_A3_FP_IMPL
#define ALLTO_ALL_MATMUL_A3_FP_IMPL(tilingData, pipe)                                                                  \
    do {                                                                                                               \
        DEFINE_MC2_HCCL_FOR_COMMUNICATION(false, HcclServerType::HCCL_SERVER_TYPE_AICPU,                               \
                                          MC2KernelTemplate::MC2AlltoAllContext, AlltoAllMatmulTilingDataA3,           \
                                          MC2KernelTemplate::MC2AlltoAllPrimitives, 0, 1, CommunicationType);          \
        CommunicationType commImplName(&tilingData);                                                                   \
        DEFINE_MC2_TRANSPOSE_FOR_MATH_COMPUTATION(DTYPE_X1, TransposeType);                                            \
        TransposeType transposeImplName(&pipe);                                                                        \
        DEFINE_MC2_MATMUL_CONTEXT_FOR_MATMUL_COMPUTATION_A3_FP(ComputationContextType);                                \
        DEFINE_MC2_MATMUL_FOR_MATMUL_COMPUTATION_A3_FP(false, ALLTO_ALL_MM_TRANSPOSE_X2, ComputationType);             \
        ComputationType matmulImplName(&pipe);                                                                         \
        using SchedulerContextType = AlltoAllMmPipelineContext<ComputationContextType>;                                \
        using SchedulerType =                                                                                          \
            AlltoAllMatmulPipeLine<CommunicationType, TransposeType, ComputationType, SchedulerContextType>;           \
        SchedulerType SchedulerImpl(&commImplName, &transposeImplName, &matmulImplName);                               \
        AlltoAllMatmulKernelBase<SchedulerType, SchedulerContextType, AlltoAllMatmulTilingDataA3> op(&SchedulerImpl);  \
        op.Init(x1, x2, bias, y, all2all_out, workspaceGM, &tilingData, &pipe);                                        \
        op.Process();                                                                                                  \
    } while (0)
#endif

template <bool ALLTO_ALL_MM_HAS_BIAS, bool ALLTO_ALL_MM_TRANSPOSE_X2, int ALLTO_ALL_MM_QUANT_TYPE,
          int ALLTO_ALL_MM_QUANT_BIAS_DTYPE, int ALLTO_ALL_MM_SOC_VERSION>
__global__ __aicore__ void allto_all_matmul(GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR x1_scale, GM_ADDR x2_scale,
                                            GM_ADDR comm_scale, GM_ADDR x1_offset, GM_ADDR x2_offset, GM_ADDR y,
                                            GM_ADDR all2all_out, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2); // A2、A3的kernel task都采用1:2模式

#if (QUANT_TYPE == MC2_NON_QUANT) // int4类型tensor不为cast的src参数，需要隔离
    if constexpr (ALLTO_ALL_MM_SOC_VERSION == SOC_ASCEND910B) {
        // A2结构体注册由tilingkey的编译宏ASCENDC_TPL_TILING_STRUCT_SEL实现
        GET_TILING_DATA_WITH_STRUCT(AlltoAllMatmulTilingData, tilingData, tilingGM);

        if constexpr (ALLTO_ALL_MM_QUANT_TYPE == TILINGKEY_TPL_NOQUANT) { // 有bias无quant
            AlltoAllMatmul<DTYPE_X1, DTYPE_X2, DTYPE_BIAS, DTYPE_X1_SCALE, DTYPE_X2_SCALE, DTYPE_Y, DTYPE_ALL2ALL_OUT,
                           ALLTO_ALL_MM_HAS_BIAS, ALLTO_ALL_MM_TRANSPOSE_X2, QUANT_TYPE>
                op;
            op.Init(x1, x2, bias, nullptr, nullptr, y, all2all_out, workspaceGM, tilingGM);
            op.Process();
        }
    } else if constexpr (ALLTO_ALL_MM_SOC_VERSION == SOC_ASCEND910_93) {
        TPipe pipe;
        // A3结构体注册由tilingkey的编译宏ASCENDC_TPL_TILING_STRUCT_SEL实现
        GET_TILING_DATA_WITH_STRUCT(AlltoAllMatmulTilingDataA3, tilingData, tilingGM);

        if constexpr (ALLTO_ALL_MM_QUANT_BIAS_DTYPE == TILINGKEY_TPL_FP16) {
            using DtypeBias = DTYPE_X1;
            ALLTO_ALL_MATMUL_A3_FP_IMPL(tilingData, pipe);
        } else if constexpr (ALLTO_ALL_MM_QUANT_BIAS_DTYPE == TILINGKEY_TPL_FP32) {
            using DtypeBias = float;
            ALLTO_ALL_MATMUL_A3_FP_IMPL(tilingData, pipe);
        }
    }
#elif (QUANT_TYPE == MC2_DYNAMIC_QUANT)
    GET_TILING_DATA_WITH_STRUCT(AlltoAllMatmulTilingData, tilingData, tilingGM);

    if constexpr ((ALLTO_ALL_MM_QUANT_TYPE == TILINGKEY_TPL_A16W8 || ALLTO_ALL_MM_QUANT_TYPE == TILINGKEY_TPL_A16W4) &&
                  ALLTO_ALL_MM_QUANT_BIAS_DTYPE == TILINGKEY_TPL_FP16) { // x1量化，bias是fp16，x1Scale与x1类型一致
        AlltoAllMatmul<DTYPE_X1, DTYPE_X2, float16_t, DTYPE_X1, DTYPE_X2_SCALE, DTYPE_Y, DTYPE_ALL2ALL_OUT,
                       ALLTO_ALL_MM_HAS_BIAS, ALLTO_ALL_MM_TRANSPOSE_X2, QUANT_TYPE>
            op;
        op.Init(x1, x2, bias, x1_scale, x2_scale, y, all2all_out, workspaceGM, tilingGM);
        op.Process();
    } else if constexpr ((ALLTO_ALL_MM_QUANT_TYPE == TILINGKEY_TPL_A16W8 ||
                          ALLTO_ALL_MM_QUANT_TYPE == TILINGKEY_TPL_A16W4) &&
                         ALLTO_ALL_MM_QUANT_BIAS_DTYPE ==
                             TILINGKEY_TPL_BF16) { // x1量化，int8，bias是bf16，x1Scale与x1类型一致
        AlltoAllMatmul<DTYPE_X1, DTYPE_X2, bfloat16_t, DTYPE_X1, DTYPE_X2_SCALE, DTYPE_Y, DTYPE_ALL2ALL_OUT,
                       ALLTO_ALL_MM_HAS_BIAS, ALLTO_ALL_MM_TRANSPOSE_X2, QUANT_TYPE>
            op;
        op.Init(x1, x2, bias, x1_scale, x2_scale, y, all2all_out, workspaceGM, tilingGM);
        op.Process();
    } else if constexpr ((ALLTO_ALL_MM_QUANT_TYPE == TILINGKEY_TPL_A16W8 ||
                          ALLTO_ALL_MM_QUANT_TYPE == TILINGKEY_TPL_A16W4) &&
                         ALLTO_ALL_MM_QUANT_BIAS_DTYPE ==
                             TILINGKEY_TPL_FP32) { // x1量化，int8，bias是fp32，x1Scale与x1类型一致
        AlltoAllMatmul<DTYPE_X1, DTYPE_X2, float32_t, DTYPE_X1, DTYPE_X2_SCALE, DTYPE_Y, DTYPE_ALL2ALL_OUT,
                       ALLTO_ALL_MM_HAS_BIAS, ALLTO_ALL_MM_TRANSPOSE_X2, QUANT_TYPE>
            op;
        op.Init(x1, x2, bias, x1_scale, x2_scale, y, all2all_out, workspaceGM, tilingGM);
        op.Process();
    }

#elif (QUANT_TYPE == MC2_STATIC_QUANT)
    GET_TILING_DATA_WITH_STRUCT(AlltoAllMatmulTilingData, tilingData, tilingGM);

    if constexpr (ALLTO_ALL_MM_QUANT_TYPE == TILINGKEY_TPL_A4W4 &&
                  ALLTO_ALL_MM_QUANT_BIAS_DTYPE == TILINGKEY_TPL_FP16) { // x1不量化，int4，bias是fp16
        AlltoAllMatmul<int4b_t, int4b_t, float16_t, float, DTYPE_X2_SCALE, DTYPE_Y, DTYPE_ALL2ALL_OUT,
                       ALLTO_ALL_MM_HAS_BIAS, ALLTO_ALL_MM_TRANSPOSE_X2, QUANT_TYPE>
            op;
        op.Init(x1, x2, bias, x1_scale, x2_scale, y, all2all_out, workspaceGM, tilingGM);
        op.Process();
    } else if constexpr (ALLTO_ALL_MM_QUANT_TYPE == TILINGKEY_TPL_A4W4 &&
                         ALLTO_ALL_MM_QUANT_BIAS_DTYPE == TILINGKEY_TPL_BF16) { // x1不量化，int4，bias是bf16
        AlltoAllMatmul<int4b_t, int4b_t, bfloat16_t, float, DTYPE_X2_SCALE, DTYPE_Y, DTYPE_ALL2ALL_OUT,
                       ALLTO_ALL_MM_HAS_BIAS, ALLTO_ALL_MM_TRANSPOSE_X2, QUANT_TYPE>
            op;
        op.Init(x1, x2, bias, x1_scale, x2_scale, y, all2all_out, workspaceGM, tilingGM);
        op.Process();
    } else if constexpr (ALLTO_ALL_MM_QUANT_TYPE == TILINGKEY_TPL_A4W4 &&
                         ALLTO_ALL_MM_QUANT_BIAS_DTYPE == TILINGKEY_TPL_FP32) { // x1不量化，int4，bias是fp32
        AlltoAllMatmul<int4b_t, int4b_t, float32_t, float, DTYPE_X2_SCALE, DTYPE_Y, DTYPE_ALL2ALL_OUT,
                       ALLTO_ALL_MM_HAS_BIAS, ALLTO_ALL_MM_TRANSPOSE_X2, QUANT_TYPE>
            op;
        op.Init(x1, x2, bias, x1_scale, x2_scale, y, all2all_out, workspaceGM, tilingGM);
        op.Process();
    }
#endif
}
