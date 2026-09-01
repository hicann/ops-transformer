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
 * \file allto_all_matmul.cpp
 * \brief
 */
#include <cstring>
#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include <lib/matmul_intf.h>
#include "common.h"
#include "../allto_all_matmul_kernel_base.h"
#include "allto_all_matmul_add_bias_pipeline.h"
#include "allto_all_matmul_tiling_key.h"
#include "allto_all_matmul_tiling_data.h"
#include "allto_all_kc_quant_matmul_arch35.h"
#include "allto_all_kc_quant_matmul_pipeline.h"
#include "allto_all_mx_quant_matmul_arch35.h"
#include "allto_all_mx_quant_matmul_pipeline.h"
#include "../../../common/op_kernel/apace/kernel/fusions/all_to_all_quant_matmul/all_to_all_matmul_tiling_data.h"
#include "../../../common/op_kernel/apace/kernel/fusions/all_to_all_quant_matmul/all_to_all_mx_quant_matmul_hcomm_impl.h"

using AscendC::HcclServerType;
using AscendC::TPipe;
using MC2KernelTemplate::MC2AlltoAllContext;
using MC2KernelTemplate::MC2AlltoAllPrimitives;

#ifndef ALLTO_ALL_MATMUL_APT_FP_IMPL
#define ALLTO_ALL_MATMUL_APT_FP_IMPL(tilingData, pipe, hcclServerType) \
    do { \
        DEFINE_MC2_HCCL_FOR_COMMUNICATION(false, hcclServerType, MC2AlltoAllContext, AlltoAllMatmulTilingData, \
                                          MC2AlltoAllPrimitives, 0, 1, CommunicationType); \
        CommunicationType commImplName(&tilingData); \
        DEFINE_MC2_TRANSPOSE_FOR_MATH_COMPUTATION(DTYPE_X1, TransposeType); \
        TransposeType transposeImplName(&pipe); \
        DEFINE_MC2_MATMUL_CONTEXT_FOR_MATMUL_COMPUTATION_FP(ComputationContextType); \
        DEFINE_MC2_MATMUL_FOR_MATMUL_COMPUTATION_FP(false, X2TRANSPOSE, ComputationType); \
        ComputationType matmulImplName(&pipe); \
        using SchedulerContextType = Mc2Kernel::AlltoAllMmPipelineContext<ComputationContextType>; \
        using SchedulerType = Mc2Kernel::AlltoAllMatmulPipeLine<CommunicationType, TransposeType, ComputationType, \
                                                                SchedulerContextType>; \
        SchedulerType SchedulerImpl(&commImplName, &transposeImplName, &matmulImplName); \
        Mc2Kernel::AlltoAllMatmulKernelBase<SchedulerType, SchedulerContextType, AlltoAllMatmulTilingData> op( \
            &SchedulerImpl); \
        op.Init(x1, x2, bias, y, all2all_out, workspaceGM, &tilingData, &pipe); \
        op.Process(); \
    } while (0)
#endif

// FP 非量化 + 独立 add_bias (ADDBIASSPLITMODE != 0): add_bias pipeline, 按 SplitMode 实例化
#ifndef ALLTO_ALL_MATMUL_APT_FP_SEP_BIAS_IMPL
#define ALLTO_ALL_MATMUL_APT_FP_SEP_BIAS_IMPL(tilingData, pipe, hcclServerType, SplitModeEnum) \
    do { \
        DEFINE_MC2_HCCL_FOR_COMMUNICATION(false, hcclServerType, MC2AlltoAllContext, AlltoAllMatmulTilingData, \
                                          MC2AlltoAllPrimitives, 0, 1, CommunicationType); \
        CommunicationType commImplName(&tilingData); \
        DEFINE_MC2_TRANSPOSE_FOR_MATH_COMPUTATION(DTYPE_X1, TransposeType); \
        TransposeType transposeImplName(&pipe); \
        DEFINE_MC2_ADD_BIAS_FOR_MATH_COMPUTATION(DTYPE_Y, DtypeBias, SplitModeEnum, AddBiasType); \
        AddBiasType addBiasImplName(&pipe); \
        DEFINE_MC2_MATMUL_CONTEXT_FOR_MATMUL_COMPUTATION_FP(ComputationContextType); \
        DEFINE_MC2_MATMUL_FOR_MATMUL_COMPUTATION_FP(false, X2TRANSPOSE, ComputationType); \
        ComputationType matmulImplName(&pipe); \
        using SchedulerContextType = Mc2Kernel::AlltoAllMmAddBiasPipelineContext<ComputationContextType>; \
        using SchedulerType = \
            Mc2Kernel::AlltoAllMatmulAddBiasPipeLine<CommunicationType, TransposeType, ComputationType, AddBiasType, \
                                                     SchedulerContextType>; \
        SchedulerType SchedulerImpl(&commImplName, &transposeImplName, &matmulImplName, &addBiasImplName); \
        Mc2Kernel::AlltoAllMatmulKernelBase<SchedulerType, SchedulerContextType, AlltoAllMatmulTilingData, true> op( \
            &SchedulerImpl); \
        op.Init(x1, x2, bias, y, all2all_out, workspaceGM, &tilingData, &pipe); \
        op.Process(); \
    } while (0)
#endif

#ifndef ALLTO_ALL_KC_QUANT_MATMUL_IMPL
#define ALLTO_ALL_KC_QUANT_MATMUL_IMPL(tilingData, pipe, hcclServerType, MMDataTypeX1, isSmallK) \
    do { \
        DEFINE_MC2_HCCL_FOR_COMMUNICATION(false, hcclServerType, MC2AlltoAllContext, AlltoAllQuantMatmulTilingData, \
                                          MC2AlltoAllPrimitives, 0, 1, CommunicationType); \
        CommunicationType commImplName(&tilingData); \
        DEFINE_MC2_FP8_DYNAMIC_QUANT_PERTOKEN(DTYPE_X1, MMDataTypeX1, TransAndDynamicQuantType, isSmallK); \
        TransAndDynamicQuantType dynamicQuantImplName(&pipe); \
        DEFINE_MC2_MATMUL_CONTEXT_FOR_MATMUL_COMPUTATION_QUANT(ComputationContextType); \
        DEFINE_MC2_MATMUL_FOR_MATMUL_COMPUTATION_QUANT(ComputationType, MMDataTypeX1, DTYPE_X2); \
        ComputationType matmulImplName(&pipe); \
        using SchedulerContextType = Mc2Kernel::AlltoAllKCQmmPipelineContext<ComputationContextType>; \
        using SchedulerType = Mc2Kernel::AlltoAllKCQuantMatmulPipeLine<CommunicationType, TransAndDynamicQuantType, \
                                                                       ComputationType, SchedulerContextType>; \
        SchedulerType SchedulerImpl(&commImplName, &dynamicQuantImplName, &matmulImplName); \
        Mc2Kernel::AlltoAllKcQuantMatmulArch35<SchedulerType, SchedulerContextType, AlltoAllQuantMatmulTilingData> op( \
            &SchedulerImpl); \
        op.Init(x1, x2, bias, y, all2all_out, x1_scale, x2_scale, x2_offset, workspaceGM, &tilingData, &pipe); \
        op.Process(); \
    } while (0)
#endif

#ifndef ALLTO_ALL_MX_QUANT_MATMUL_IMPL
#define ALLTO_ALL_MX_QUANT_MATMUL_IMPL(tilingData, pipe, hcclServerType, hcclDataType, commDataTypeX1, isMxFp4) \
    do { \
        DEFINE_MC2_HCCL_FOR_COMMUNICATION(false, hcclServerType, MC2AlltoAllContext, AlltoAllQuantMatmulTilingData, \
                                          MC2AlltoAllPrimitives, 0, 1, CommunicationType); \
        CommunicationType commImplName(&tilingData); \
        DEFINE_MC2_TRANSPOSE_FOR_MATH_COMPUTATION(commDataTypeX1, TransposeType); \
        TransposeType transposeImplName(&pipe); \
        DEFINE_MC2_TRANSPOSE_FOR_MATH_COMPUTATION(AscendC::fp8_e8m0_t, ScaleTransposeType); \
        ScaleTransposeType scaleTransposeImplName(&pipe); \
        DEFINE_MC2_MATMUL_CONTEXT_FOR_MATMUL_COMPUTATION_MX_QUANT(ComputationContextType); \
        DEFINE_MC2_MATMUL_FOR_MATMUL_COMPUTATION_MX_QUANT(ComputationType); \
        ComputationType matmulImplName(&pipe); \
        using SchedulerContextType = Mc2Kernel::AlltoAllMXQmmPipelineContext<ComputationContextType>; \
        using SchedulerType = \
            Mc2Kernel::AlltoAllMXQuantMatmulPipeLine<CommunicationType, TransposeType, ScaleTransposeType, \
                                                     ComputationType, SchedulerContextType>; \
        SchedulerType SchedulerImpl(&commImplName, &transposeImplName, &scaleTransposeImplName, &matmulImplName); \
        Mc2Kernel::AlltoAllMxQuantMatmulArch35<SchedulerType, SchedulerContextType, AlltoAllQuantMatmulTilingData, \
                                               isMxFp4> \
            op(&SchedulerImpl, hcclDataType); \
        op.Init(x1, x2, bias, y, all2all_out, x1_scale, x2_scale, workspaceGM, &tilingData, &pipe); \
        op.Process(); \
    } while (0)
#endif

template <uint32_t QUANTMODE, bool X2TRANSPOSE, uint32_t DTYPEBIAS, bool ISSMALLK, uint32_t COMMTYPE,
          bool USING_APACE_IMPL, uint32_t ADDBIASSPLITMODE>
__global__ __aicore__ void allto_all_matmul(GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR x1_scale, GM_ADDR x2_scale,
                                            GM_ADDR comm_scale, GM_ADDR x1_offset, GM_ADDR x2_offset, GM_ADDR y,
                                            GM_ADDR all2all_out, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    TPipe pipe;
    constexpr HcclServerType hcclServerType = COMMTYPE == ALL2ALL_COMM_TYPE_CCU ?
                                                  HcclServerType::HCCL_SERVER_TYPE_CCU :
                                                  HcclServerType::HCCL_SERVER_TYPE_AICPU;

#if ((ORIG_DTYPE_X1 == ORIG_DTYPE_X2) && ((ORIG_DTYPE_X1 == DT_FLOAT16) || (ORIG_DTYPE_X1 == DT_BF16)))
    REGISTER_TILING_DEFAULT(AlltoAllMatmulTilingData);
    GET_TILING_DATA_WITH_STRUCT(AlltoAllMatmulTilingData, tilingData, tilingGM);

    using DtypeBias = typename std::conditional<DTYPEBIAS == DTYPE_BIAS_FP32, float, DTYPE_X1>::type;
    if constexpr (ADDBIASSPLITMODE == ADDBIAS_OFF) {
        ALLTO_ALL_MATMUL_APT_FP_IMPL(tilingData, pipe, hcclServerType);
    } else {
        ALLTO_ALL_MATMUL_APT_FP_SEP_BIAS_IMPL(tilingData, pipe, hcclServerType, ADDBIASSPLITMODE);
    }

#else
    REGISTER_TILING_DEFAULT(AlltoAllQuantMatmulTilingData); // 按当前最大的结构体申请内存
#if (((ORIG_DTYPE_X1 == DT_FLOAT8_E4M3FN) || (ORIG_DTYPE_X1 == DT_FLOAT8_E5M2)) && \
     ((ORIG_DTYPE_X2 == DT_FLOAT8_E4M3FN) || (ORIG_DTYPE_X2 == DT_FLOAT8_E5M2)))
    if constexpr (USING_APACE_IMPL) {
        GET_TILING_DATA_WITH_STRUCT(Apace::hcommAllToAllMatmulTilingData, tilingData, tilingGM);
        Apace::AllToAllMxQuantMatmulHcommImpl<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_X1,
                                              Apace::hcommAllToAllMatmulTilingData, hcclServerType, false>
            op(&tilingData);
        op.Init(x1, x2, bias, y, all2all_out, x1_scale, x2_scale, workspaceGM);
        op.Run();
    } else {
        GET_TILING_DATA_WITH_STRUCT(AlltoAllQuantMatmulTilingData, tilingData, tilingGM);
        AscendC::HcclDataType hcclDataType = AscendC::HCCL_DATA_TYPE_FP8E4M3;
        if constexpr (AscendC::IsSameType<DTYPE_X1, float8_e5m2_t>::value) {
            hcclDataType = AscendC::HCCL_DATA_TYPE_FP8E5M2;
        }
        ALLTO_ALL_MX_QUANT_MATMUL_IMPL(tilingData, pipe, hcclServerType, hcclDataType, DTYPE_X1, false);
    }
#elif ((ORIG_DTYPE_X1 == DT_FLOAT4_E2M1) && (ORIG_DTYPE_X2 == DT_FLOAT4_E2M1))
    if constexpr (USING_APACE_IMPL) {
        GET_TILING_DATA_WITH_STRUCT(Apace::hcommAllToAllMatmulTilingData, tilingData, tilingGM);
        Apace::AllToAllMxQuantMatmulHcommImpl<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_X1,
                                              Apace::hcommAllToAllMatmulTilingData, hcclServerType, true>
            op(&tilingData);
        op.Init(x1, x2, bias, y, all2all_out, x1_scale, x2_scale, workspaceGM);
        op.Run();
    } else {
        GET_TILING_DATA_WITH_STRUCT(AlltoAllQuantMatmulTilingData, tilingData, tilingGM);
        ALLTO_ALL_MX_QUANT_MATMUL_IMPL(tilingData, pipe, hcclServerType, AscendC::HCCL_DATA_TYPE_UINT8, uint8_t, true);
    }
#else
    GET_TILING_DATA_WITH_STRUCT(AlltoAllQuantMatmulTilingData, tilingData, tilingGM);
    if constexpr (QUANTMODE == KC_QUANT_FP8E5M2_MODE) {
        ALLTO_ALL_KC_QUANT_MATMUL_IMPL(tilingData, pipe, hcclServerType, float8_e5m2_t, ISSMALLK);
    } else if constexpr (QUANTMODE == KC_QUANT_FP8E4M3_MODE) {
        ALLTO_ALL_KC_QUANT_MATMUL_IMPL(tilingData, pipe, hcclServerType, float8_e4m3_t, ISSMALLK);
    }
#endif
#endif
}
