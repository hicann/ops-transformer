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

#if (ORIG_DTYPE_X == DT_FLOAT8_E4M3FN && ORIG_DTYPE_WEIGHT == DT_FLOAT4_E2M1)
// 伪量化MxA8W4场景识别宏：x=FP8_E4M3FN, w=FP4_E2M1
#define GMM_SWIGLU_QUANT_WEIGHT_QUANT
#include "arch35/weight_quant_basic_block/grouped_matmul_swiglu_quant_v2_weight_quant_tiling_data.h"
#include "arch35/weight_quant_basic_block/gmmsq_weight_quant_resplit_controller.h"
#include "arch35/weight_quant_basic_block/grouped_matmul_swiglu_quant_v2_weight_quant_tiling_key.h"
#else
// 全量化Mx量化
#include "arch35/grouped_matmul_swiglu_quant_v2_mxquant.h"
#include "arch35/grouped_matmul_tensor_api_swiglu_quant_v2_mxfp8_kernel.h"
#include "arch35/grouped_matmul_swiglu_quant_v2_mxfp4_weight_nz.h"
#include "arch35/grouped_matmul_swiglu_quant_v2_tiling_key.h"
#endif

#elif ORIG_DTYPE_X_SCALE == DT_FLOAT
#include "arch35/grouped_matmul_swiglu_quant_v2_pertoken_quant.h"
#include "arch35/grouped_matmul_swiglu_quant_v2_tiling_key.h"
#endif

#define FLOAT_OVERFLOW_MODE_CTRL 60

using namespace AscendC;
using namespace matmul;

namespace {
#if defined(FORMAT_WEIGHT) && FORMAT_WEIGHT == FORMAT_FRACTAL_NZ
constexpr CubeFormat weightFormat = CubeFormat::NZ;
#else
constexpr CubeFormat weightFormat = CubeFormat::ND;
#endif
#if ORIG_DTYPE_X_SCALE == DT_FLOAT8_E8M0
constexpr bool isMxFp4Input =
    (AscendC::IsSameType<DTYPE_X, fp4x2_e2m1_t>::value || AscendC::IsSameType<DTYPE_X, fp4x2_e1m2_t>::value) &&
    (AscendC::IsSameType<DTYPE_WEIGHT, fp4x2_e2m1_t>::value || AscendC::IsSameType<DTYPE_WEIGHT, fp4x2_e1m2_t>::value);
#endif
} // namespace

#if defined(GMM_SWIGLU_QUANT_WEIGHT_QUANT)
template <int8_t QUANT_B_TRANS, int8_t QUANT_A_TRANS, bool IS_SINGLE_MULTI_SINGLE>
#else
template <int8_t QUANT_B_TRANS, int8_t QUANT_A_TRANS, int8_t KERNEL_TYPE>
#endif
__global__ __aicore__ void grouped_matmul_swiglu_quant_v2(GM_ADDR x, GM_ADDR xScale, GM_ADDR groupList, GM_ADDR weight,
                                                          GM_ADDR weightScale, GM_ADDR weightAssistanceMatrix,
                                                          GM_ADDR bias, GM_ADDR smoothScale, GM_ADDR y, GM_ADDR yScale,
                                                          GM_ADDR workspace, GM_ADDR tiling)
{
#if defined(GMM_SWIGLU_QUANT_WEIGHT_QUANT)
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
    GM_ADDR userWorkspace = GetUserWorkspace(workspace);
    int64_t oriOverflowMode = AscendC::GetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>();
    // enable overflow mode to avoid nan/inf value
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
#if ORIG_DTYPE_X_SCALE == DT_FLOAT8_E8M0
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
            GmmSwigluAswt<Cgmct::Gemm::layout::RowMajor, Cgmct::Gemm::layout::Nz>(
                x, weight, weightScale, xScale, weightAssistanceMatrix, smoothScale, groupList, y, yScale, workspace,
                tiling);
        } else if (QUANT_B_TRANS == GMM_SWIGLU_QUANT_TRANS && QUANT_A_TRANS == GMM_SWIGLU_QUANT_NO_TRANS) {
            GmmSwigluAswt<Cgmct::Gemm::layout::RowMajor, Cgmct::Gemm::layout::Zn>(
                x, weight, weightScale, xScale, weightAssistanceMatrix, smoothScale, groupList, y, yScale, workspace,
                tiling);
        }
    } else {
        if (QUANT_B_TRANS == GMM_SWIGLU_QUANT_NO_TRANS && QUANT_A_TRANS == GMM_SWIGLU_QUANT_NO_TRANS) {
            if constexpr (KERNEL_TYPE == GMM_SWIGLU_QUANT_ORIGINAL_KERNEL_TYPE) {
                GmmSwigluAswt<Cgmct::Gemm::layout::RowMajor, Cgmct::Gemm::layout::RowMajor>(
                    x, weight, weightScale, xScale, weightAssistanceMatrix, smoothScale, groupList, y, yScale,
                    workspace, tiling);
            } else {
                GmmTensorApiSwigluQuantMxFp8Kernel<AscendC::Te::NDExtLayoutPtn, AscendC::Te::NDExtLayoutPtn>(
                    x, weight, weightScale, xScale, weightAssistanceMatrix, smoothScale, groupList, y, yScale,
                    workspace, tiling);
            }
        } else if (QUANT_B_TRANS == GMM_SWIGLU_QUANT_TRANS && QUANT_A_TRANS == GMM_SWIGLU_QUANT_NO_TRANS) {
            if constexpr (KERNEL_TYPE == GMM_SWIGLU_QUANT_ORIGINAL_KERNEL_TYPE) {
                GmmSwigluAswt<Cgmct::Gemm::layout::RowMajor, Cgmct::Gemm::layout::ColumnMajor>(
                    x, weight, weightScale, xScale, weightAssistanceMatrix, smoothScale, groupList, y, yScale,
                    workspace, tiling);
            } else {
                GmmTensorApiSwigluQuantMxFp8Kernel<AscendC::Te::NDExtLayoutPtn, AscendC::Te::DNExtLayoutPtn>(
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
#endif
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(oriOverflowMode);
#endif
}
