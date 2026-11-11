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
 * \file grouped_matmul_finalize_routing_apt.cpp
 * \brief
 */

#if (defined(__CCE_AICORE__) && __CCE_AICORE__ == 310) || \
    (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 9201 || __NPU_ARCH__ == 9202))
#include "kernel_utils.h"
#include "kernel_tiling/kernel_tiling.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

// Only the Ascend950 path uses the migrated Tensor API kernels.
// Keep the macro-undefined C310 pass consistent with the existing Ascend950 build.
#if (defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510) || \
    (!defined(__NPU_ARCH__) && defined(__CCE_AICORE__) && __CCE_AICORE__ == 310)
#define GMM_FR_USE_TENSOR_API
#endif

#if defined(ORIG_DTYPE_X) && defined(ORIG_DTYPE_W) && defined(DT_FLOAT8_E4M3FN) && defined(DT_FLOAT4_E2M1)
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310 && \
    (!defined(__NPU_ARCH__) || (__NPU_ARCH__ != 9201 && __NPU_ARCH__ != 9202)) && \
    (ORIG_DTYPE_X == DT_FLOAT8_E4M3FN && (ORIG_DTYPE_W == DT_FLOAT4_E2M1))
#define V310_GMM_FR_ANTI_QUANT
#endif
#endif

#if defined(ORIG_DTYPE_Y) && defined(DT_BF16) && (ORIG_DTYPE_Y == DT_BF16) && \
    (!defined(__NPU_ARCH__) || (__NPU_ARCH__ != 9201 && __NPU_ARCH__ != 9202))
#define GMM_FR_BF16_OUTPUT
#endif

#if defined(ORIG_DTYPE_X) && defined(ORIG_DTYPE_W) && defined(DT_FLOAT4_E2M1) && defined(DT_FLOAT4_E1M2) && \
    defined(DT_FLOAT8_E4M3FN) && defined(DT_FLOAT8_E5M2) && \
    (!defined(__NPU_ARCH__) || (__NPU_ARCH__ != 9201 && __NPU_ARCH__ != 9202))
#if ((ORIG_DTYPE_X == DT_FLOAT4_E2M1 || ORIG_DTYPE_X == DT_FLOAT4_E1M2) && \
     (ORIG_DTYPE_W == DT_FLOAT4_E2M1 || ORIG_DTYPE_W == DT_FLOAT4_E1M2)) || \
    ((ORIG_DTYPE_X == DT_FLOAT8_E4M3FN || ORIG_DTYPE_X == DT_FLOAT8_E5M2) && \
     (ORIG_DTYPE_W == DT_FLOAT8_E4M3FN || ORIG_DTYPE_W == DT_FLOAT8_E5M2))
#define GMM_FR_FULL_MX
#endif
#endif

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 9201 || __NPU_ARCH__ == 9202) && defined(ORIG_DTYPE_X) && \
    defined(ORIG_DTYPE_W) && defined(ORIG_DTYPE_SCALE) && defined(DT_FLOAT8_E4M3FN) && defined(DT_FLOAT4_E2M1) && \
    defined(DT_HIFLOAT4) && defined(DT_FLOAT8_E8M0) && defined(DT_HIFLOAT4_SCALE)
#if (ORIG_DTYPE_X == DT_FLOAT8_E4M3FN) && (ORIG_DTYPE_W == DT_FLOAT4_E2M1 || ORIG_DTYPE_W == DT_HIFLOAT4) && \
    (ORIG_DTYPE_SCALE == DT_FLOAT8_E8M0 || ORIG_DTYPE_SCALE == DT_HIFLOAT4_SCALE)
#define V9201_GMM_FINALIZE_ROUTING_WEIGHT_QUANT
#endif
#if (ORIG_DTYPE_X == DT_HIFLOAT4) && (ORIG_DTYPE_W == DT_HIFLOAT4) && (ORIG_DTYPE_SCALE == DT_HIFLOAT4_SCALE)
#define V9201_GMM_FINALIZE_ROUTING_HIF4
#endif
#endif

#if defined(GMM_FR_BF16_OUTPUT)
#if defined(GMM_FR_FULL_MX)
#include "lib/matmul_intf.h"
#include "arch35/grouped_matmul_finalize_routing_tiling_key.h"
#include "arch35/grouped_matmul_finalize_routing_arch35.h"
#endif
#elif defined(V310_GMM_FR_ANTI_QUANT)
// Weight Quantization scenario (伪量化场景)
#include "arch35/weight_quant_basic_block/gmm_fr_weight_quant_tiling_data.h"
#include "arch35/weight_quant_basic_block/gmm_fr_weight_quant_tiling_key.h"
#if defined(GMM_FR_USE_TENSOR_API)
#include "arch35/grouped_matmul_finalize_routing_a8w4_blaze.h"
#else
#include "arch35/common/basic_block_config.h"
#include "arch35/weight_quant_basic_block/gmm_fr_weight_quant_resplit_controller.h"
#endif
#elif defined(V9201_GMM_FINALIZE_ROUTING_WEIGHT_QUANT) || defined(V9201_GMM_FINALIZE_ROUTING_HIF4)
#include "lib/matmul_intf.h"
#include "arch35/grouped_matmul_finalize_routing_tiling_key.h"
#include "arch35/grouped_matmul_finalize_routing_mx_basic_api.h"
#else
// Full Quantization scenario (全量化场景)
#include "lib/matmul_intf.h"
#include "arch35/grouped_matmul_finalize_routing_tiling_key.h"
#if ORIG_DTYPE_X == DT_FLOAT8_E4M3FN || ORIG_DTYPE_X == DT_FLOAT8_E5M2 || ORIG_DTYPE_X == DT_FLOAT4_E1M2 || \
    ORIG_DTYPE_X == DT_FLOAT4_E2M1
#if !defined(__NPU_ARCH__) || (__NPU_ARCH__ != 9201 && __NPU_ARCH__ != 9202)
#include "arch35/grouped_matmul_finalize_routing_arch35.h"
#else
#include "arch35/grouped_matmul_finalize_routing_mx_legacy.h"
#endif
#endif
#if ORIG_DTYPE_X != DT_FLOAT4_E1M2 && ORIG_DTYPE_X != DT_FLOAT4_E2M1
#if defined(GMM_FR_USE_TENSOR_API)
#include "arch35/grouped_matmul_finalize_routing_pertoken_blaze.h"
#else
#include "arch35/grouped_matmul_finalize_routing_pertoken_dequant.h"
#endif
#endif
#endif

// Values match the existing host/device template-key encoding.
static constexpr uint64_t GMM_FR_NO_TRANS = 0;
static constexpr uint64_t GMM_FR_TRANS = 1;
static constexpr uint64_t GMM_FR_SCALE_MX = 0;
static constexpr uint64_t GMM_FR_SCALE_FP32 = 1;
static constexpr uint64_t GMM_FR_ROW_INDEX_INT64 = 0;
static constexpr uint64_t GMM_FR_ROW_INDEX_INT32 = 1;
static constexpr uint64_t BF16TYPE = 2;
static constexpr uint64_t LOGIT_BF16_TYPE = 1;

#if defined(GMM_FR_FULL_MX)
#if defined(FORMAT_W) && defined(FORMAT_FRACTAL_NZ) && (FORMAT_W == FORMAT_FRACTAL_NZ)
static constexpr bool GMM_FR_WEIGHT_NZ = true;
#else
static constexpr bool GMM_FR_WEIGHT_NZ = false;
#endif

template <int BTRANS, bool WEIGHT_NZ>
struct GmmFrMxLayoutBSelector {
    using type =
        AscendC::Std::conditional_t<BTRANS == GMM_FR_NO_TRANS, AscendC::Te::NZLayoutPtn, AscendC::Te::ZNLayoutPtn>;
};

template <int BTRANS>
struct GmmFrMxLayoutBSelector<BTRANS, false> {
    using type = AscendC::Std::conditional_t<BTRANS == GMM_FR_NO_TRANS, AscendC::Te::NDExtLayoutPtn,
                                             AscendC::Te::DNExtLayoutPtn>;
};

template <int BTRANS, int LOGITTYPE>
__aicore__ inline void RunGmmFrMx(GM_ADDR x, GM_ADDR w, GM_ADDR scale, GM_ADDR bias, GM_ADDR pertokenScale,
                                  GM_ADDR groupList, GM_ADDR shareInput, GM_ADDR logit, GM_ADDR rowIndex,
                                  GM_ADDR offset, GM_ADDR y, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    using LayoutB = typename GmmFrMxLayoutBSelector<BTRANS, GMM_FR_WEIGHT_NZ>::type;
    using LogitType = AscendC::Std::conditional_t<LOGITTYPE == LOGIT_BF16_TYPE, bfloat16_t, float>;
    grouped_matmul_finalize_routing_mx<AscendC::Te::NDExtLayoutPtn, LayoutB, LogitType>(
        x, w, scale, bias, pertokenScale, groupList, shareInput, logit, rowIndex, offset, y, workspaceGM, tilingGM);
}

#endif

#if !defined(GMM_FR_BF16_OUTPUT) && !defined(V310_GMM_FR_ANTI_QUANT) && \
    !defined(V9201_GMM_FINALIZE_ROUTING_WEIGHT_QUANT) && !defined(V9201_GMM_FINALIZE_ROUTING_HIF4) && \
    ORIG_DTYPE_X != DT_FLOAT4_E1M2 && ORIG_DTYPE_X != DT_FLOAT4_E2M1
template <int BTRANS, int SCALETYPE, int ROWINDEXTYPE>
__aicore__ inline void RunGmmFrPertoken(GM_ADDR x, GM_ADDR w, GM_ADDR scale, GM_ADDR bias, GM_ADDR pertokenScale,
                                        GM_ADDR groupList, GM_ADDR shareInput, GM_ADDR logit, GM_ADDR rowIndex,
                                        GM_ADDR offset, GM_ADDR y, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
#if defined(GMM_FR_USE_TENSOR_API)
    using LayoutB =
        AscendC::Std::conditional_t<BTRANS == GMM_FR_NO_TRANS, asc::te::nz_layout_ptn, asc::te::zn_layout_ptn>;
    GroupedMatmulFinalizeRouting::InvokeBlazePertokenKernel<asc::te::nd_ext_layout_ptn, LayoutB, SCALETYPE,
                                                            ROWINDEXTYPE>(
        x, w, scale, bias, pertokenScale, groupList, shareInput, logit, rowIndex, offset, y, workspaceGM, tilingGM);
#else
    using LayoutB =
        AscendC::Std::conditional_t<BTRANS == GMM_FR_NO_TRANS, Cgmct::Gemm::layout::Nz, Cgmct::Gemm::layout::Zn>;
    grouped_matmul_finalize_routing_pertoken_dequant<Cgmct::Gemm::layout::RowMajor, LayoutB, SCALETYPE, ROWINDEXTYPE>(
        x, w, scale, bias, pertokenScale, groupList, shareInput, logit, rowIndex, offset, y, workspaceGM, tilingGM);
#endif
}
#endif

#if defined(V310_GMM_FR_ANTI_QUANT)
template <int ATRANS, int BTRANS, int HASBIAS>
#else
// SCALETYPE 0 is float8e8m0, 1 is fp32, 2 is bf16; ROWINDEXTYPE 0 is int64, 1 is int32;
// LOGITTYPE 0 is fp32, 1 is bf16.
template <int ATRANS, int BTRANS, int SCALETYPE, int ROWINDEXTYPE, int LOGITTYPE>
#endif
__global__ __aicore__ void grouped_matmul_finalize_routing(GM_ADDR x, GM_ADDR w, GM_ADDR scale, GM_ADDR bias,
                                                           GM_ADDR pertoken_scale, GM_ADDR group_list,
                                                           GM_ADDR share_input, GM_ADDR logit, GM_ADDR row_index,
                                                           GM_ADDR offset, GM_ADDR y, GM_ADDR workspaceGM,
                                                           GM_ADDR tilingGM)
{
#if defined(V310_GMM_FR_ANTI_QUANT)
#if !defined(GMM_FR_BF16_OUTPUT)
#ifndef DTYPE_BIAS
#define DTYPE_BIAS bfloat16_t
#endif
    AscendC::InitSocState();
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    REGISTER_TILING_DEFAULT(GMMFinalizeRoutingArch35Tiling::GMMFinalizeRoutingWeightQuantTilingData);
    GET_TILING_DATA_WITH_STRUCT(GMMFinalizeRoutingArch35Tiling::GMMFinalizeRoutingWeightQuantTilingData, tilingData,
                                tilingGM);
#if defined(GMM_FR_USE_TENSOR_API)
    GroupedMatmulFinalizeRouting::InvokeBlazeA8W4Kernel<DTYPE_X, DTYPE_W, DTYPE_SCALE, DTYPE_PERTOKEN_SCALE, DTYPE_BIAS,
                                                        DTYPE_Y, bfloat16_t, float, int64_t, HASBIAS>(
        x, w, scale, bias, group_list, pertoken_scale, logit, row_index, y, share_input, &tilingData);
#else
    const GMMFinalizeRoutingArch35Tiling::GMMFinalizeRoutingWeightQuantTilingData* tiling = &tilingData;

    // Use pre-defined MXA8W4_NZNK config for FP8+FP4 MX (Microscaling) format
    // Config values: aTrans=false, bTrans=true, antiQuantType=MX, hasAntiQuantOffset=false, quantType=NONE,
    // weightFormat=NZ
    GROUPED_MATMUL_FINALIZE_ROUTING::GMMFRWeightQuantResplitController<
        DTYPE_X, DTYPE_W, DTYPE_SCALE, DTYPE_SCALE, DTYPE_PERTOKEN_SCALE, DTYPE_BIAS, DTYPE_Y, bfloat16_t, float,
        int64_t, HASBIAS, WeightQuantBatchMatmulV2::Arch35::MXA8W4_NZNK,
        WeightQuantBatchMatmulV2::Arch35::VEC_ANTIQUANT_CONFIG_DYNAMIC>
        controller;

    controller.Init(x, w, scale, bias, group_list, pertoken_scale, logit, row_index, y, share_input, tiling);
    controller.Process();
#endif
#endif
#elif defined(V9201_GMM_FINALIZE_ROUTING_WEIGHT_QUANT)
    AscendC::TPipe pipe;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
    if constexpr (ATRANS == GMM_FR_NO_TRANS && BTRANS == GMM_FR_TRANS && SCALETYPE == GMM_FR_SCALE_MX &&
                  ROWINDEXTYPE == GMM_FR_ROW_INDEX_INT64) {
        grouped_matmul_finalize_routing_mx_basic_api<Cgmct::Gemm::layout::RowMajor, Cgmct::Gemm::layout::Zn>(
            x, w, scale, bias, pertoken_scale, group_list, share_input, logit, row_index, offset, y, workspaceGM,
            tilingGM);
    }
#elif defined(V9201_GMM_FINALIZE_ROUTING_HIF4)
    AscendC::TPipe pipe;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
    if constexpr (ATRANS == GMM_FR_NO_TRANS && BTRANS == GMM_FR_NO_TRANS && SCALETYPE == GMM_FR_SCALE_MX &&
                  ROWINDEXTYPE == GMM_FR_ROW_INDEX_INT64) {
        grouped_matmul_finalize_routing_mx_basic_api<Cgmct::Gemm::layout::RowMajor, Cgmct::Gemm::layout::RowMajor>(
            x, w, scale, bias, pertoken_scale, group_list, share_input, logit, row_index, offset, y, workspaceGM,
            tilingGM);
    } else if constexpr (ATRANS == GMM_FR_NO_TRANS && BTRANS == GMM_FR_TRANS && SCALETYPE == GMM_FR_SCALE_MX &&
                         ROWINDEXTYPE == GMM_FR_ROW_INDEX_INT64) {
        grouped_matmul_finalize_routing_mx_basic_api<Cgmct::Gemm::layout::RowMajor, Cgmct::Gemm::layout::ColumnMajor>(
            x, w, scale, bias, pertoken_scale, group_list, share_input, logit, row_index, offset, y, workspaceGM,
            tilingGM);
    }
#elif defined(GMM_FR_BF16_OUTPUT)
#if defined(GMM_FR_FULL_MX)
    AscendC::TPipe pipe;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if constexpr (GMM_FR_WEIGHT_NZ && ATRANS == GMM_FR_NO_TRANS && BTRANS == GMM_FR_NO_TRANS &&
                  SCALETYPE == GMM_FR_SCALE_MX && ROWINDEXTYPE == GMM_FR_ROW_INDEX_INT32) {
        RunGmmFrMx<GMM_FR_NO_TRANS, LOGITTYPE>(x, w, scale, bias, pertoken_scale, group_list, share_input, logit,
                                               row_index, offset, y, workspaceGM, tilingGM);
    } else if constexpr (GMM_FR_WEIGHT_NZ && ATRANS == GMM_FR_NO_TRANS && BTRANS == GMM_FR_TRANS &&
                         SCALETYPE == GMM_FR_SCALE_MX && ROWINDEXTYPE == GMM_FR_ROW_INDEX_INT32) {
        RunGmmFrMx<GMM_FR_TRANS, LOGITTYPE>(x, w, scale, bias, pertoken_scale, group_list, share_input, logit,
                                            row_index, offset, y, workspaceGM, tilingGM);
    }
#endif
#else
    AscendC::TPipe pipe;
    // Full Quantization scenario
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 9201 || __NPU_ARCH__ == 9202)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
#else
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
#endif
#if ORIG_DTYPE_X != DT_FLOAT8_E4M3FN && ORIG_DTYPE_X != DT_FLOAT8_E5M2 && ORIG_DTYPE_X != DT_FLOAT4_E1M2 && \
    ORIG_DTYPE_X != DT_FLOAT4_E2M1
    if constexpr (ATRANS == GMM_FR_NO_TRANS && (BTRANS == GMM_FR_NO_TRANS || BTRANS == GMM_FR_TRANS) &&
                  (SCALETYPE == GMM_FR_SCALE_FP32 || SCALETYPE == BF16TYPE) &&
                  (ROWINDEXTYPE == GMM_FR_ROW_INDEX_INT64 || ROWINDEXTYPE == GMM_FR_ROW_INDEX_INT32)) {
        RunGmmFrPertoken<BTRANS, SCALETYPE, ROWINDEXTYPE>(x, w, scale, bias, pertoken_scale, group_list, share_input,
                                                          logit, row_index, offset, y, workspaceGM, tilingGM);
    }
#elif ORIG_DTYPE_X == DT_FLOAT4_E1M2 || ORIG_DTYPE_X == DT_FLOAT4_E2M1
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 9201 || __NPU_ARCH__ == 9202)
    if constexpr (ATRANS == GMM_FR_NO_TRANS && BTRANS == GMM_FR_NO_TRANS && SCALETYPE == GMM_FR_SCALE_MX &&
                  ROWINDEXTYPE == GMM_FR_ROW_INDEX_INT64) {
        grouped_matmul_finalize_routing_mx_legacy<Cgmct::Gemm::layout::RowMajor, Cgmct::Gemm::layout::RowMajor>(
            x, w, scale, bias, pertoken_scale, group_list, share_input, logit, row_index, offset, y, workspaceGM,
            tilingGM);
    } else if constexpr (ATRANS == GMM_FR_NO_TRANS && BTRANS == GMM_FR_TRANS && SCALETYPE == GMM_FR_SCALE_MX &&
                         ROWINDEXTYPE == GMM_FR_ROW_INDEX_INT64) {
        grouped_matmul_finalize_routing_mx_legacy<Cgmct::Gemm::layout::RowMajor, Cgmct::Gemm::layout::ColumnMajor>(
            x, w, scale, bias, pertoken_scale, group_list, share_input, logit, row_index, offset, y, workspaceGM,
            tilingGM);
    }
#else
    if constexpr (ATRANS == GMM_FR_NO_TRANS && BTRANS == GMM_FR_NO_TRANS && SCALETYPE == GMM_FR_SCALE_MX &&
                  ROWINDEXTYPE == GMM_FR_ROW_INDEX_INT64) {
        RunGmmFrMx<GMM_FR_NO_TRANS, LOGITTYPE>(x, w, scale, bias, pertoken_scale, group_list, share_input, logit,
                                               row_index, offset, y, workspaceGM, tilingGM);
    } else if constexpr (ATRANS == GMM_FR_NO_TRANS && BTRANS == GMM_FR_TRANS && SCALETYPE == GMM_FR_SCALE_MX &&
                         ROWINDEXTYPE == GMM_FR_ROW_INDEX_INT64) {
        RunGmmFrMx<GMM_FR_TRANS, LOGITTYPE>(x, w, scale, bias, pertoken_scale, group_list, share_input, logit,
                                            row_index, offset, y, workspaceGM, tilingGM);
    }
#endif
#else
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 9201 || __NPU_ARCH__ == 9202)
    if constexpr (ATRANS == GMM_FR_NO_TRANS && BTRANS == GMM_FR_NO_TRANS && SCALETYPE == GMM_FR_SCALE_MX &&
                  ROWINDEXTYPE == GMM_FR_ROW_INDEX_INT64) {
        grouped_matmul_finalize_routing_mx_legacy<Cgmct::Gemm::layout::RowMajor, Cgmct::Gemm::layout::RowMajor>(
            x, w, scale, bias, pertoken_scale, group_list, share_input, logit, row_index, offset, y, workspaceGM,
            tilingGM);
    } else if constexpr (ATRANS == GMM_FR_NO_TRANS && BTRANS == GMM_FR_TRANS && SCALETYPE == GMM_FR_SCALE_MX &&
                         ROWINDEXTYPE == GMM_FR_ROW_INDEX_INT64) {
        grouped_matmul_finalize_routing_mx_legacy<Cgmct::Gemm::layout::RowMajor, Cgmct::Gemm::layout::ColumnMajor>(
            x, w, scale, bias, pertoken_scale, group_list, share_input, logit, row_index, offset, y, workspaceGM,
            tilingGM);
#else
    if constexpr (ATRANS == GMM_FR_NO_TRANS && BTRANS == GMM_FR_NO_TRANS && SCALETYPE == GMM_FR_SCALE_MX &&
                  ROWINDEXTYPE == GMM_FR_ROW_INDEX_INT64) {
        RunGmmFrMx<GMM_FR_NO_TRANS, LOGITTYPE>(x, w, scale, bias, pertoken_scale, group_list, share_input, logit,
                                               row_index, offset, y, workspaceGM, tilingGM);
    } else if constexpr (ATRANS == GMM_FR_NO_TRANS && BTRANS == GMM_FR_TRANS && SCALETYPE == GMM_FR_SCALE_MX &&
                         ROWINDEXTYPE == GMM_FR_ROW_INDEX_INT64) {
        RunGmmFrMx<GMM_FR_TRANS, LOGITTYPE>(x, w, scale, bias, pertoken_scale, group_list, share_input, logit,
                                            row_index, offset, y, workspaceGM, tilingGM);
#endif
    } else if constexpr (ATRANS == GMM_FR_NO_TRANS && (BTRANS == GMM_FR_NO_TRANS || BTRANS == GMM_FR_TRANS) &&
                         (SCALETYPE == GMM_FR_SCALE_FP32 || SCALETYPE == BF16TYPE) &&
                         (ROWINDEXTYPE == GMM_FR_ROW_INDEX_INT64 || ROWINDEXTYPE == GMM_FR_ROW_INDEX_INT32)) {
        RunGmmFrPertoken<BTRANS, SCALETYPE, ROWINDEXTYPE>(x, w, scale, bias, pertoken_scale, group_list, share_input,
                                                          logit, row_index, offset, y, workspaceGM, tilingGM);
    }
#endif
#endif
}
#endif
