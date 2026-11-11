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
 * \file grouped_matmul_swiglu_quant_v2_apt.cpp
 * \brief
 */

#include "kernel_tiling/kernel_tiling.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "lib/matmul_intf.h"

#if ORIG_DTYPE_X_SCALE == DT_FLOAT8_E8M0

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 9201 || __NPU_ARCH__ == 9202) && ORIG_DTYPE_X == DT_FLOAT8_E4M3FN && \
    (ORIG_DTYPE_WEIGHT == DT_FLOAT4_E2M1 || ORIG_DTYPE_WEIGHT == DT_HIFLOAT4)
#define GMM_SWIGLU_QUANT_WEIGHT_QUANT_ASW
#include "arch35/weight_quant_asw/grouped_matmul_swiglu_quant_v2_weight_quant_asw_tiling_data.h"
#include "arch35/weight_quant_asw/grouped_matmul_swiglu_quant_v2_weight_quant_asw_tiling_key.h"
#include "arch35/weight_quant_asw/grouped_matmul_swiglu_quant_v2_weight_quant_asw_kernel.h"
using GMMSQWeightQuantAswTilingData = GMMSQArch35Tiling::GMMSQWeightQuantAswTilingData;
#elif ORIG_DTYPE_X == DT_FLOAT8_E4M3FN && ORIG_DTYPE_WEIGHT == DT_FLOAT4_E2M1
// 3510 MxA8W4 uses the resplit controller; 9201/9202 uses the ASW branch above.
#define GMM_SWIGLU_QUANT_WEIGHT_QUANT
#include "arch35/weight_quant_basic_block/grouped_matmul_swiglu_quant_v2_weight_quant_tiling_data.h"
#include "arch35/weight_quant_basic_block/gmmsq_weight_quant_resplit_controller.h"
#include "arch35/weight_quant_basic_block/grouped_matmul_swiglu_quant_v2_weight_quant_tiling_key.h"
#else
// 全量化Mx量化
#include "arch35/grouped_matmul_swiglu_quant_v2_tiling_data.h"
#include "arch35/grouped_matmul_swiglu_quant_v2_mxquant.h"
#if !defined(__NPU_ARCH__) || (__NPU_ARCH__ != 9201 && __NPU_ARCH__ != 9202)
#define GMM_SWIGLU_QUANT_TENSOR_API_ENABLED 1
#include "arch35/grouped_matmul_tensor_api_swiglu_quant_v2_mxfp8_kernel.h"
#else
#define GMM_SWIGLU_QUANT_TENSOR_API_ENABLED 0
#endif
#include "arch35/grouped_matmul_swiglu_quant_v2_mxfp4_weight_nz.h"
#include "arch35/grouped_matmul_swiglu_quant_v2_tiling_key.h"
#endif

#elif ORIG_DTYPE_X_SCALE == DT_FLOAT
#include "arch35/grouped_matmul_swiglu_quant_v2_tiling_data.h"
#include "arch35/grouped_matmul_swiglu_quant_v2_tensor_api_tiling_data.h"
#include "arch35/grouped_matmul_swiglu_quant_v2_pertoken_quant.h"
#include "arch35/grouped_matmul_swiglu_quant_v2_tiling_key.h"
#elif ORIG_DTYPE_X_SCALE == DT_HIFLOAT4_SCALE
#include "arch35/grouped_matmul_swiglu_quant_v2_hif4quant.h"
#include "arch35/grouped_matmul_swiglu_quant_v2_tiling_key.h"
#endif

#define FLOAT_OVERFLOW_MODE_CTRL 60

using namespace AscendC;
using namespace matmul;

namespace {
#if !defined(GMM_SWIGLU_QUANT_WEIGHT_QUANT_ASW) && !defined(GMM_SWIGLU_QUANT_WEIGHT_QUANT)
#if defined(FORMAT_WEIGHT) && FORMAT_WEIGHT == FORMAT_FRACTAL_NZ
constexpr CubeFormat weightFormat = CubeFormat::NZ;
#else
constexpr CubeFormat weightFormat = CubeFormat::ND;
#endif
#if ORIG_DTYPE_X_SCALE == DT_FLOAT8_E8M0
constexpr bool isMxFp4Input =
    (AscendC::IsSameType<DTYPE_X, fp4x2_e2m1_t>::value || AscendC::IsSameType<DTYPE_X, fp4x2_e1m2_t>::value) &&
    (AscendC::IsSameType<DTYPE_WEIGHT, fp4x2_e2m1_t>::value || AscendC::IsSameType<DTYPE_WEIGHT, fp4x2_e1m2_t>::value);
#if GMM_SWIGLU_QUANT_TENSOR_API_ENABLED
constexpr bool isMxFp8Input =
    (AscendC::IsSameType<DTYPE_X, fp8_e4m3fn_t>::value || AscendC::IsSameType<DTYPE_X, fp8_e5m2_t>::value) &&
    (AscendC::IsSameType<DTYPE_WEIGHT, fp8_e4m3fn_t>::value || AscendC::IsSameType<DTYPE_WEIGHT, fp8_e5m2_t>::value);
#endif
#endif
#endif
} // namespace

#if defined(GMM_SWIGLU_QUANT_WEIGHT_QUANT_ASW)
template <int8_t QUANT_B_TRANS, int8_t QUANT_A_TRANS>
#elif defined(GMM_SWIGLU_QUANT_WEIGHT_QUANT)
template <int8_t QUANT_B_TRANS, int8_t QUANT_A_TRANS, bool IS_SINGLE_MULTI_SINGLE>
#else
template <int8_t QUANT_B_TRANS, int8_t QUANT_A_TRANS, int8_t KERNEL_TYPE>
#endif
__global__ __aicore__ void grouped_matmul_swiglu_quant_v2(GM_ADDR x, GM_ADDR xScale, GM_ADDR groupList, GM_ADDR weight,
                                                          GM_ADDR weightScale, GM_ADDR weightAssistanceMatrix,
                                                          GM_ADDR bias, GM_ADDR smoothScale, GM_ADDR y, GM_ADDR yScale,
                                                          GM_ADDR workspace, GM_ADDR tiling)
{
#if defined(GMM_SWIGLU_QUANT_WEIGHT_QUANT_ASW)
    AscendC::InitSocState();
    int64_t oriOverflowMode = AscendC::GetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>();
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
    if (QUANT_B_TRANS == GMM_SWIGLU_QUANT_TRANS && QUANT_A_TRANS == GMM_SWIGLU_QUANT_NO_TRANS) {
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
        REGISTER_TILING_DEFAULT(GMMSQWeightQuantAswTilingData);
        GET_TILING_DATA_MEMBER(GMMSQWeightQuantAswTilingData, gmmWeightQuantAswParam, gmmWeightQuantAswParam_, tiling);
        GET_TILING_DATA_MEMBER(GMMSQWeightQuantAswTilingData, mmTilingData, mmTilingData_, tiling);
        GMMSQWeightQuantAsw::GMMSQWeightQuantASWKernel<DTYPE_X, DTYPE_WEIGHT, DTYPE_X_SCALE, DTYPE_WEIGHT_SCALE,
                                                       DTYPE_Y, DTYPE_Y_SCALE>
            op;
        op.Init(x, weight, xScale, weightScale, groupList, y, yScale, &gmmWeightQuantAswParam_, &mmTilingData_);
        op.Process();
    }
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(oriOverflowMode);
#elif defined(GMM_SWIGLU_QUANT_WEIGHT_QUANT)
    // MxA8W4场景
    AscendC::InitSocState();
    int64_t oriOverflowMode = AscendC::GetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>();
    // enable overflow mode to avoid nan/inf value
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    REGISTER_TILING_DEFAULT(GMMSQArch35Tiling::GMMSQWeightQuantTilingData);
    GET_TILING_DATA_WITH_STRUCT(GMMSQArch35Tiling::GMMSQWeightQuantTilingData, tilingDataIn, tiling);

    GMMSQWeightQuant::GMMSQWeightQuantResplitController<
        DTYPE_X, DTYPE_WEIGHT, DTYPE_WEIGHT_SCALE, DTYPE_X_SCALE, DTYPE_Y, DTYPE_Y_SCALE, GMMSQWeightQuant::MXA8W4_NZNK,
        GMMSQWeightQuant::VEC_ANTIQUANT_CONFIG_DYNAMIC, IS_SINGLE_MULTI_SINGLE>
        controller;

    controller.Init(x, weight, weightScale, groupList, xScale, y, yScale, &tilingDataIn);
    controller.Process();
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(oriOverflowMode);
#else
    TPipe tPipe;
#if ORIG_DTYPE_X_SCALE == DT_HIFLOAT4_SCALE
    REGISTER_TILING_DEFAULT(GMMSwigluQuantBasicApiTilingDataParams);
#else
    REGISTER_TILING_DEFAULT(GMMSwigluQuantV2TilingDataParams);
#endif
#if ORIG_DTYPE_X_SCALE == DT_FLOAT8_E8M0 && GMM_SWIGLU_QUANT_TENSOR_API_ENABLED
    REGISTER_TILING_FOR_TILINGKEY("(TILING_KEY_VAR == 16) || (TILING_KEY_VAR == 17)",
                                  GroupedMatmulSwigluQuantV2TensorApi::GMMSwigluQuantV2TensorApiTilingData);
#endif
    GM_ADDR userWorkspace = GetUserWorkspace(workspace);
    int64_t oriOverflowMode = AscendC::GetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>();
    // enable overflow mode to avoid nan/inf value
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
#if ORIG_DTYPE_X_SCALE == DT_FLOAT8_E8M0
#if GMM_SWIGLU_QUANT_V3_WEIGHT_NZ_ENABLED && GMM_SWIGLU_QUANT_TENSOR_API_ENABLED
    // The backend remains a compile-time choice; V2/V3 epilogue attributes share its tiling payload.
    if constexpr (KERNEL_TYPE == GMM_SWIGLU_QUANT_TENSOR_LEVEL_KERNEL_TYPE) {
        if constexpr (QUANT_B_TRANS == GMM_SWIGLU_QUANT_NO_TRANS) {
            GmmTensorApiSwigluQuantMxFp8Kernel<asc::te::nd_ext_layout_ptn, asc::te::nz_layout_ptn, true>(
                x, weight, weightScale, xScale, weightAssistanceMatrix, smoothScale, groupList, y, yScale, workspace,
                tiling);
        } else {
            GmmTensorApiSwigluQuantMxFp8Kernel<asc::te::nd_ext_layout_ptn, asc::te::zn_layout_ptn, true>(
                x, weight, weightScale, xScale, weightAssistanceMatrix, smoothScale, groupList, y, yScale, workspace,
                tiling);
        }
    } else
#endif
        if constexpr (weightFormat == CubeFormat::NZ && isMxFp4Input) {
        if (QUANT_B_TRANS == GMM_SWIGLU_QUANT_NO_TRANS && QUANT_A_TRANS == GMM_SWIGLU_QUANT_NO_TRANS) {
            GmmSwigluMxFp4WeightNz<Cgmct::Gemm::layout::RowMajor, Cgmct::Gemm::layout::Nz>(
                x, weight, weightScale, xScale, weightAssistanceMatrix, smoothScale, groupList, y, yScale, workspace,
                tiling);
        } else if (QUANT_B_TRANS == GMM_SWIGLU_QUANT_TRANS && QUANT_A_TRANS == GMM_SWIGLU_QUANT_NO_TRANS) {
            GmmSwigluMxFp4WeightNz<Cgmct::Gemm::layout::RowMajor, Cgmct::Gemm::layout::Zn>(
                x, weight, weightScale, xScale, weightAssistanceMatrix, smoothScale, groupList, y, yScale, workspace,
                tiling);
        }
    } else if constexpr (weightFormat == CubeFormat::NZ) {
        if (QUANT_B_TRANS == GMM_SWIGLU_QUANT_NO_TRANS && QUANT_A_TRANS == GMM_SWIGLU_QUANT_NO_TRANS) {
            GmmSwigluAswt<Cgmct::Gemm::layout::RowMajor, Cgmct::Gemm::layout::Nz,
                          GMM_SWIGLU_QUANT_V3_WEIGHT_NZ_ENABLED>(x, weight, weightScale, xScale, weightAssistanceMatrix,
                                                                 smoothScale, groupList, y, yScale, workspace, tiling);
        } else if (QUANT_B_TRANS == GMM_SWIGLU_QUANT_TRANS && QUANT_A_TRANS == GMM_SWIGLU_QUANT_NO_TRANS) {
            GmmSwigluAswt<Cgmct::Gemm::layout::RowMajor, Cgmct::Gemm::layout::Zn,
                          GMM_SWIGLU_QUANT_V3_WEIGHT_NZ_ENABLED>(x, weight, weightScale, xScale, weightAssistanceMatrix,
                                                                 smoothScale, groupList, y, yScale, workspace, tiling);
        }
    } else {
        if (QUANT_B_TRANS == GMM_SWIGLU_QUANT_NO_TRANS && QUANT_A_TRANS == GMM_SWIGLU_QUANT_NO_TRANS) {
#if GMM_SWIGLU_QUANT_TENSOR_API_ENABLED
            if constexpr (KERNEL_TYPE == GMM_SWIGLU_QUANT_TENSOR_LEVEL_KERNEL_TYPE && isMxFp8Input) {
                GmmTensorApiSwigluQuantMxFp8Kernel<asc::te::nd_ext_layout_ptn, asc::te::nd_ext_layout_ptn>(
                    x, weight, weightScale, xScale, weightAssistanceMatrix, smoothScale, groupList, y, yScale,
                    workspace, tiling);
            } else
#endif
            {
                GmmSwigluAswt<Cgmct::Gemm::layout::RowMajor, Cgmct::Gemm::layout::RowMajor>(
                    x, weight, weightScale, xScale, weightAssistanceMatrix, smoothScale, groupList, y, yScale,
                    workspace, tiling);
            }
        } else if (QUANT_B_TRANS == GMM_SWIGLU_QUANT_TRANS && QUANT_A_TRANS == GMM_SWIGLU_QUANT_NO_TRANS) {
#if GMM_SWIGLU_QUANT_TENSOR_API_ENABLED
            if constexpr (KERNEL_TYPE == GMM_SWIGLU_QUANT_TENSOR_LEVEL_KERNEL_TYPE && isMxFp8Input) {
                GmmTensorApiSwigluQuantMxFp8Kernel<asc::te::nd_ext_layout_ptn, asc::te::dn_ext_layout_ptn>(
                    x, weight, weightScale, xScale, weightAssistanceMatrix, smoothScale, groupList, y, yScale,
                    workspace, tiling);
            } else
#endif
            {
                GmmSwigluAswt<Cgmct::Gemm::layout::RowMajor, Cgmct::Gemm::layout::ColumnMajor>(
                    x, weight, weightScale, xScale, weightAssistanceMatrix, smoothScale, groupList, y, yScale,
                    workspace, tiling);
            }
        }
    }
#elif ORIG_DTYPE_X_SCALE == DT_FLOAT
    if constexpr (weightFormat == CubeFormat::NZ) {
        if (QUANT_B_TRANS == GMM_SWIGLU_QUANT_NO_TRANS && QUANT_A_TRANS == GMM_SWIGLU_QUANT_NO_TRANS) {
            GmmSwigluAswtPertoken<Cgmct::Gemm::layout::RowMajor, Cgmct::Gemm::layout::Nz>(
                x, weight, weightScale, xScale, weightAssistanceMatrix, smoothScale, groupList, y, yScale, workspace,
                tiling, &tPipe);
        } else if (QUANT_B_TRANS == GMM_SWIGLU_QUANT_TRANS && QUANT_A_TRANS == GMM_SWIGLU_QUANT_NO_TRANS) {
            GmmSwigluAswtPertoken<Cgmct::Gemm::layout::RowMajor, Cgmct::Gemm::layout::Zn>(
                x, weight, weightScale, xScale, weightAssistanceMatrix, smoothScale, groupList, y, yScale, workspace,
                tiling, &tPipe);
        }
    } else {
        if (QUANT_B_TRANS == GMM_SWIGLU_QUANT_NO_TRANS && QUANT_A_TRANS == GMM_SWIGLU_QUANT_NO_TRANS) {
            GmmSwigluAswtPertoken<Cgmct::Gemm::layout::RowMajor, Cgmct::Gemm::layout::RowMajor>(
                x, weight, weightScale, xScale, weightAssistanceMatrix, smoothScale, groupList, y, yScale, workspace,
                tiling, &tPipe);
        } else if (QUANT_B_TRANS == GMM_SWIGLU_QUANT_TRANS && QUANT_A_TRANS == GMM_SWIGLU_QUANT_NO_TRANS) {
            GmmSwigluAswtPertoken<Cgmct::Gemm::layout::RowMajor, Cgmct::Gemm::layout::ColumnMajor>(
                x, weight, weightScale, xScale, weightAssistanceMatrix, smoothScale, groupList, y, yScale, workspace,
                tiling, &tPipe);
        }
    }
#elif ORIG_DTYPE_X_SCALE == DT_HIFLOAT4_SCALE
    if (QUANT_B_TRANS == GMM_SWIGLU_QUANT_TRANS && QUANT_A_TRANS == GMM_SWIGLU_QUANT_NO_TRANS &&
        KERNEL_TYPE == GMM_SWIGLU_HIF4_QUANT_KERNEL_TYPE) {
        GmmSwigluHiF4<Cgmct::Gemm::layout::RowMajor, Cgmct::Gemm::layout::ColumnMajor>(
            x, weight, weightScale, xScale, weightAssistanceMatrix, smoothScale, groupList, y, yScale, workspace,
            tiling);
    }
#endif
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(oriOverflowMode);
#endif
}
