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
 * \file grouped_matmul_apt.cpp
 * \brief
 */

#if ASC_DEVKIT_MAJOR >= 9 && ASC_DEVKIT_MINOR > 0
#define IS_BLAZE true
#else
#define IS_BLAZE false
#endif

#include "grouped_matmul_utils.h"
#include "arch35/grouped_matmul_tiling_data_apt.h"
using GMMWeightQuantTilingData = GroupedMatmulTilingData::GMMWeightQuantTilingData;
using GMMNoQuantTilingData = GroupedMatmulTilingData::GMMNoQuantTilingData;
using GMMQuantTilingData = GroupedMatmulTilingData::GMMQuantTilingData;
using GMMQuantBasicApiTilingData = GroupedMatmulTilingData::GMMQuantBasicApiTilingData;
using GMMS4S4IntQuantTilingData = GroupedMatmulTilingData::GMMS4S4IntQuantTilingData;
using GMMS8S4BasicApiTilingData = GroupedMatmulTilingData::GMMS8S4BasicApiTilingData;
#if defined(V310_GMM_QUANT)
#include "arch35/quant_adaptive_sliding_window_templates/gqmm_tiling_key.h"
#if defined(V310_GMM_QUANT_CUBE) || defined(V310_GMM_QUANT_PERTENSOR_CUBE)
#include "arch35/quant_adaptive_sliding_window_templates/gqmm_cube_on_the_fly.h"
#endif
#if defined(V310_GMM_QUANT_MX) && IS_BLAZE
#include "arch35/quant_adaptive_sliding_window_templates/gqmm_tensor_api_mx_kernel.h"
#elif defined(V310_GMM_QUANT_MX)
#include "arch35/quant_adaptive_sliding_window_templates/gqmm_cgmct_mx_kernel.h"
#endif
#if defined(V310_GMM_QUANT_S4S4) && IS_BLAZE
#include "arch35/quant_adaptive_sliding_window_templates/qgmm_s4s4_kernel.h"
#endif
#if defined(V310_GMM_QUANT_MX) || defined(V310_GMM_QUANT_PERTENSOR_CUBE)
#include "arch35/quant_adaptive_sliding_window_templates/gqmm_init_output.h"
#endif
#if defined(V310_GMM_QUANT_PERTENSOR_CUBE)
#include "arch35/quant_adaptive_sliding_window_templates/gqmm_mix_online_dynamic.h"
#endif
#if defined(V310_GMM_QUANT_PERTILE)
#include "arch35/quant_adaptive_sliding_window_templates/gqmm_cgmct_pertile_kernel.h"
#endif
#elif defined(V310_GMM_ANTI_QUANT)
#include "arch35/weight_quant_basic_block/basic_block_config.h"
#include "arch35/weight_quant_basic_block/grouped_matmul_weight_quant_basic_controller.h"
#include "arch35/weight_quant_basic_block/grouped_matmul_weight_quant_resplit_controller.h"
#include "arch35/weight_quant_basic_block/weight_quant_basic_block.h"
#include "arch35/weight_quant_basic_block/weight_quant_vcv_basic_block.h"
#include "arch35/weight_quant_basic_block/weight_quant_tiling_key.h"
#if IS_BLAZE
#include "arch35/quant_adaptive_sliding_window_templates/gmm_weight_quant_tensor_api_mx_kernel.h"
#endif
#if defined(V310_GMM_S8S4_UINT64_SCALE)
#include "arch35/weight_quant_basic_block/gmm_s8s4_perchannel_tensor_api.h"
#endif
using WeightQuantBatchMatmulV2::Arch35::A16MXF4_NZKN;
using WeightQuantBatchMatmulV2::Arch35::MXA8W4_NZNK;
using WeightQuantBatchMatmulV2::Arch35::QuantType;
using WeightQuantBatchMatmulV2::Arch35::S8S4_NZKN_G;
using WeightQuantBatchMatmulV2::Arch35::WeightQuantMatmulBasicBlock;
using WeightQuantBatchMatmulV2::Arch35::WeightQuantVcvMatmulBasicBlock;
static constexpr VecAntiQuantConfig VEC_ANTIQUANT_CONFIG_0 = {2, 512};
static constexpr VecAntiQuantConfig VEC_ANTIQUANT_CONFIG_1 = {4, 512};
static constexpr VecAntiQuantConfig VEC_ANTIQUANT_CONFIG_2 = {2, 1024};
static constexpr VecAntiQuantConfig VEC_ANTIQUANT_CONFIG_3 = {4, 256};
static constexpr VecAntiQuantConfig VEC_ANTIQUANT_CONFIG_4 = {3, 512};
static constexpr VecAntiQuantConfig VEC_ANTIQUANT_CONFIG_5 = {3, 384};
static constexpr VecAntiQuantConfig VEC_ANTIQUANT_CONFIG_6 = {2, 256}; // A16W4 ND PerGroup
static constexpr VecAntiQuantConfig VEC_ANTIQUANT_CONFIG_DYNAMIC = {4, 0};
#if defined(DT_FLOAT) && defined(ORIG_DTYPE_WEIGHT) && ORIG_DTYPE_WEIGHT == DT_FLOAT
#undef DTYPE_WEIGHT
#define DTYPE_WEIGHT fp4x2_e2m1_t
#endif
#if defined(DT_INT32) && defined(ORIG_DTYPE_WEIGHT) && ORIG_DTYPE_WEIGHT == DT_INT32
#undef DTYPE_WEIGHT
#define DTYPE_WEIGHT AscendC::int4b_t
#undef ORIG_DTYPE_WEIGHT
#define ORIG_DTYPE_WEIGHT DT_INT4
#endif
#else
#include "arch35/non_quant/grouped_matmul_kernel.h"
#include "arch35/non_quant/grouped_matmul_tiling_key.h"
#endif

using namespace AscendC;
using namespace matmul;
using namespace GROUPED_MATMUL;

#ifndef FORMAT_FRACTAL_NZ
#define FORMAT_FRACTAL_NZ
#endif

namespace {
#if defined(FORMAT_WEIGHT) && FORMAT_WEIGHT == FORMAT_FRACTAL_NZ
constexpr CubeFormat wFormat = CubeFormat::NZ;
#else
constexpr CubeFormat wFormat = CubeFormat::ND;
#endif
#if defined(FORMAT_SCALE) && defined(FORMAT_WEIGHT) && defined(ORIG_DTYPE_SCALE) && defined(DT_FLOAT8_E8M0) && \
    FORMAT_SCALE == FORMAT_FRACTAL_NZ && FORMAT_WEIGHT == FORMAT_FRACTAL_NZ && ORIG_DTYPE_SCALE == DT_FLOAT8_E8M0
constexpr bool scaleNz = true;
#else
constexpr bool scaleNz = false;
#endif

} // namespace

template <bool trans = false>
using xType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X, trans>;

template <bool trans = false>
using weightType = MatmulType<AscendC::TPosition::GM, wFormat, DTYPE_X, trans>;

using yType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, MM_DTYPE_Y>;

using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_BIAS>;

#define INVOKE_GMM_WEIGHT_QUANT_BASIC_CONTROLLER_OP_IMPL(templateClass, ...) \
    do { \
        GET_TILING_DATA_MEMBER(GMMWeightQuantTilingData, gmmWeightQuantParam, gmmBaseParams_, tiling); \
        GET_TILING_DATA_MEMBER(GMMWeightQuantTilingData, mmTilingData, mmTilingData_, tiling); \
        GET_TILING_DATA_MEMBER_ADDR(GMMWeightQuantTilingData, gmmArray, gmmArrayAddr_, tiling); \
        templateClass<DTYPE_X, DTYPE_WEIGHT, DTYPE_BIAS, DTYPE_Y, __VA_ARGS__> op; \
        op.Init(x, weight, antiquantScale, antiquantOffset, bias, groupList, y, &gmmBaseParams_, &mmTilingData_, \
                tiling, gmmArrayAddr_, &tPipe); \
        op.Process(); \
    } while (0)

#define INVOKE_GMM_WEIGHT_QUANT_RESPLIT_CONTROLLER_OP_IMPL(templateClass, ...) \
    do { \
        GET_TILING_DATA_MEMBER(GMMWeightQuantTilingData, gmmWeightQuantParam, gmmBaseParams_, tiling); \
        GET_TILING_DATA_MEMBER(GMMWeightQuantTilingData, mmTilingData, mmTilingData_, tiling); \
        templateClass<DTYPE_X, DTYPE_WEIGHT, DTYPE_ANTIQUANT_SCALE, DTYPE_SCALE, float, DTYPE_BIAS, DTYPE_Y, \
                      WeightQuantMatmulBasicBlock, __VA_ARGS__, false> \
            op; \
        op.Init(x, weight, scale, antiquantScale, antiquantOffset, bias, groupList, perTokenScale, y, &gmmBaseParams_, \
                &mmTilingData_, tiling, &tPipe); \
        op.Process(); \
    } while (0)

#define INVOKE_GMM_WEIGHT_QUANT_MXA8W4_CONTROLLER_OP_IMPL(templateClass, ...) \
    do { \
        GET_TILING_DATA_MEMBER(GMMWeightQuantTilingData, gmmWeightQuantParam, gmmBaseParams_, tiling); \
        GET_TILING_DATA_MEMBER(GMMWeightQuantTilingData, mmTilingData, mmTilingData_, tiling); \
        templateClass<DTYPE_X, DTYPE_WEIGHT, DTYPE_ANTIQUANT_SCALE, DTYPE_SCALE, DTYPE_PER_TOKEN_SCALE, DTYPE_BIAS, \
                      DTYPE_Y, WeightQuantMatmulBasicBlock, __VA_ARGS__, IS_SINGLE_MULTI_SINGLE> \
            op; \
        op.Init(x, weight, scale, antiquantScale, antiquantOffset, bias, groupList, perTokenScale, y, &gmmBaseParams_, \
                &mmTilingData_, tiling, &tPipe); \
        op.Process(); \
    } while (0)

#define INVOKE_GMM_WEIGHT_QUANT_VCV_CONTROLLER_OP_IMPL(templateClass, ...) \
    do { \
        GET_TILING_DATA_MEMBER(GMMWeightQuantTilingData, gmmWeightQuantParam, gmmBaseParams_, tiling); \
        GET_TILING_DATA_MEMBER(GMMWeightQuantTilingData, mmTilingData, mmTilingData_, tiling); \
        templateClass<DTYPE_X, DTYPE_WEIGHT, DTYPE_ANTIQUANT_SCALE, DTYPE_SCALE, DTYPE_PER_TOKEN_SCALE, DTYPE_BIAS, \
                      DTYPE_Y, WeightQuantVcvMatmulBasicBlock, __VA_ARGS__, false> \
            op; \
        op.Init(x, weight, scale, antiquantScale, antiquantOffset, bias, groupList, perTokenScale, y, &gmmBaseParams_, \
                &mmTilingData_, tiling, &tPipe); \
        op.Process(); \
    } while (0)

#define GMM_QUANT_IMPL_CLASS(transposeX1, transposeX2, templateClass) \
    do { \
        templateClass<DTYPE_X, DTYPE_WEIGHT, DTYPE_BIAS, DTYPE_SCALE, DTYPE_Y, wFormat, transposeX1, transposeX2> op; \
        GET_TILING_DATA_MEMBER(GMMQuantTilingData, gmmQuantParams, gmmQuantParams_, tiling); \
        GET_TILING_DATA_MEMBER(GMMQuantTilingData, mmTilingData, mmTilingData_, tiling); \
        GET_TILING_DATA_MEMBER_ADDR(GMMQuantTilingData, gmmArray, gmmArrayAddr_, tiling); \
        op.Init(x, weight, bias, scale, groupList, perTokenScale, y, user1, &gmmQuantParams_, &mmTilingData_, \
                gmmArrayAddr_, &tPipe); \
        op.Process(); \
    } while (0)

#define GMM_QUANT_MIX_IMPL_CLASS(transposeX1, transposeX2, templateClass) \
    do { \
        templateClass<DTYPE_X, DTYPE_WEIGHT, DTYPE_BIAS, DTYPE_SCALE, float, DTYPE_Y, wFormat, transposeX1, \
                      transposeX2, DTYPE_L0C_LOCAL> \
            op; \
        GET_TILING_DATA_MEMBER(GMMQuantTilingData, gmmQuantParams, gmmQuantParams_, tiling); \
        GET_TILING_DATA_MEMBER(GMMQuantTilingData, mmTilingData, mmTilingData_, tiling); \
        GET_TILING_DATA_MEMBER_ADDR(GMMQuantTilingData, gmmArray, gmmArrayAddr_, tiling); \
        op.Init(x, weight, bias, scale, groupList, perTokenScale, y, user1, &gmmQuantParams_, &mmTilingData_, \
                gmmArrayAddr_, &tPipe); \
        op.Process(); \
    } while (0)

#define GMM_QUANT_WITH_EMPTY_TENSOR_IMPL_CLASS(transposeX1, transposeX2, templateClass) \
    do { \
        GET_TILING_DATA_MEMBER(GMMQuantTilingData, gmmQuantParams, gmmQuantParams_, tiling); \
        GET_TILING_DATA_MEMBER(GMMQuantTilingData, mmTilingData, mmTilingData_, tiling); \
        GET_TILING_DATA_MEMBER_ADDR(GMMQuantTilingData, gmmArray, gmmArrayAddr_, tiling); \
        if ASCEND_IS_AIC { \
            templateClass<DTYPE_X, DTYPE_WEIGHT, DTYPE_BIAS, DTYPE_SCALE, DTYPE_Y, wFormat, transposeX1, transposeX2> \
                op; \
            op.Init(x, weight, bias, scale, groupList, perTokenScale, y, user1, &gmmQuantParams_, &mmTilingData_, \
                    gmmArrayAddr_, &tPipe); \
            op.Process(); \
        } \
        if ASCEND_IS_AIV { \
            GQmmEmptyTensor<DTYPE_Y>(groupList, y, &gmmQuantParams_, mmTilingData_.M, mmTilingData_.N); \
        } \
    } while (0)

#define GMM_QUANT_BASIC_API_EMPTY_TENSOR_WITHOUT_BIAS_IMPL_CLASS() \
    do { \
        GET_TILING_DATA_MEMBER(GMMQuantBasicApiTilingData, gmmQuantParams, gmmQuantParams_, tiling); \
        GET_TILING_DATA_MEMBER(GMMQuantBasicApiTilingData, mmTilingData, mmTilingData_, tiling); \
        GQmmEmptyTensor<DTYPE_Y>(groupList, y, &gmmQuantParams_, mmTilingData_.m, mmTilingData_.n); \
    } while (0)

#define GMM_QUANT_GB_IMPL_CLASS(xLayout, wLayout, yLayout) \
    do { \
        GET_TILING_DATA_MEMBER(GMMQuantTilingData, gmmQuantParams, gmmQuantParams_, tiling); \
        GET_TILING_DATA_MEMBER(GMMQuantTilingData, mmTilingData, mmTilingData_, tiling); \
        GmmCgmctPerTileKernel<DTYPE_X, DTYPE_WEIGHT, DTYPE_BIAS, DTYPE_SCALE, float, DTYPE_Y, xLayout, wLayout, \
                              yLayout, DTYPE_L0C_LOCAL>(x, weight, bias, scale, groupList, perTokenScale, y, user1, \
                                                        &gmmQuantParams_, &mmTilingData_, &tPipe); \
    } while (0)

#if IS_BLAZE
#define GMM_QUANT_MX_IMPL_CLASS(cgmctXLayout, cgmctWLayout, cgmctYLayout, xLayout, wLayout, yLayout) \
    do { \
        GET_TILING_DATA_MEMBER(GMMQuantBasicApiTilingData, gmmQuantParams, gmmQuantParams_, tiling); \
        GET_TILING_DATA_MEMBER(GMMQuantBasicApiTilingData, mmTilingData, mmTilingData_, tiling); \
        using wLayoutWithScale = \
            AscendC::Std::conditional_t<scaleNz, AscendC::Std::tuple<wLayout, AscendC::Te::NNLayoutPtn>, wLayout>; \
        GmmTensorApiMxKernel<DTYPE_X, DTYPE_WEIGHT, DTYPE_BIAS, DTYPE_SCALE, float, DTYPE_Y, xLayout, \
                             wLayoutWithScale, yLayout, DTYPE_L0C_LOCAL>( \
            x, weight, bias, scale, groupList, perTokenScale, y, user1, &gmmQuantParams_, &mmTilingData_, &tPipe); \
    } while (0)
#else
#define GMM_QUANT_MX_IMPL_CLASS(cgmctXLayout, cgmctWLayout, cgmctYLayout, xLayout, wLayout, yLayout) \
    do { \
        GET_TILING_DATA_MEMBER(GMMQuantBasicApiTilingData, gmmQuantParams, gmmQuantParams_, tiling); \
        GET_TILING_DATA_MEMBER(GMMQuantBasicApiTilingData, mmTilingData, mmTilingData_, tiling); \
        GmmCgmctMxKernel<DTYPE_X, DTYPE_WEIGHT, DTYPE_BIAS, DTYPE_SCALE, float, DTYPE_Y, cgmctXLayout, cgmctWLayout, \
                         cgmctYLayout, DTYPE_L0C_LOCAL>(x, weight, bias, scale, groupList, perTokenScale, y, user1, \
                                                        &gmmQuantParams_, &mmTilingData_, &tPipe); \
    } while (0)
#endif

#if defined(V310_GMM_QUANT_S4S4) && IS_BLAZE
#define GMM_S4S4_IMPL_CLASS(xLayout, wLayout, yLayout) \
    do { \
        GET_TILING_DATA_MEMBER(GMMS4S4IntQuantTilingData, gmmS4S4Params, gmmS4S4Params_, tiling); \
        GET_TILING_DATA_MEMBER(GMMS4S4IntQuantTilingData, mmTilingData, mmTilingData_, tiling); \
        GmmS4S4Kernel<DTYPE_X, DTYPE_WEIGHT, DTYPE_BIAS, DTYPE_SCALE, float, DTYPE_Y, xLayout, wLayout, yLayout, \
                      DTYPE_L0C_LOCAL>(x, weight, bias, scale, groupList, perTokenScale, y, user1, &gmmS4S4Params_, \
                                       &mmTilingData_, &tPipe); \
    } while (0)
#endif

#if defined(V310_GMM_QUANT)
template <int8_t QUANT_B_TRANS, int8_t QUANT_A_TRANS, int8_t KERNEL_TYPE>
#elif defined(V310_GMM_ANTI_QUANT)
template <int8_t W_TYPE, int8_t OFFSET_OR_BIAS_EXIT, int8_t C_QUANT_TYPE, int8_t W_QUANT_TYPE, int8_t WQ_B_TRANS,
          int8_t WQ_A_TRANS, int8_t TEMPLATE_CUSTOM_SC, bool IS_SINGLE_MULTI_SINGLE, int8_t ALGORITHM_SUB_CATEGORY,
          int8_t ALGORITHM_CATEGORY>
#else
template <int8_t NO_QUANT_B_TRANS, int8_t NO_QUANT_A_TRANS>
#endif
__global__ __aicore__ void grouped_matmul(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR scale, GM_ADDR offset,
                                          GM_ADDR antiquantScale, GM_ADDR antiquantOffset, GM_ADDR groupList,
                                          GM_ADDR perTokenScale, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
#if !defined(V310_GMM_ANTI_QUANT) || !defined(ORIG_DTYPE_X) || !defined(DT_FLOAT8_E4M3FN) || \
    ORIG_DTYPE_X != DT_FLOAT8_E4M3FN || !IS_BLAZE
    TPipe tPipe;
#endif
    AscendCUtils::SetOverflow(1);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    GM_ADDR user1 = GetUserWorkspace(workspace);

#ifndef __CCE_KT_TEST__
#if defined(V310_GMM_QUANT) // Quant: A8W8, A4W4
#if defined(V310_GMM_QUANT_S4S4) && IS_BLAZE
    REGISTER_TILING_DEFAULT(GMMS4S4IntQuantTilingData);
#else
    REGISTER_NONE_TILING;
#endif
#if defined(V310_GMM_QUANT_MX) // mx
    if constexpr (wFormat == CubeFormat::ND) {
        if constexpr (QUANT_B_TRANS == GMM_NO_TRANS && QUANT_A_TRANS == GMM_NO_TRANS &&
                      KERNEL_TYPE == GMM_DEQUANT_FIXP) {
            GET_TILING_DATA_WITH_STRUCT(GMMQuantBasicApiTilingData, tilingData, tiling);
            GMM_QUANT_MX_IMPL_CLASS(Cgmct::Gemm::layout::RowMajor, Cgmct::Gemm::layout::RowMajor,
                                    Cgmct::Gemm::layout::RowMajorAlign, AscendC::Te::NDExtLayoutPtn,
                                    AscendC::Te::NDExtLayoutPtn, AscendC::Te::NDExtLayoutPtn);
        } else if constexpr (QUANT_B_TRANS == GMM_TRANS && QUANT_A_TRANS == GMM_NO_TRANS &&
                             KERNEL_TYPE == GMM_DEQUANT_FIXP) {
            GET_TILING_DATA_WITH_STRUCT(GMMQuantBasicApiTilingData, tilingData, tiling);
            GMM_QUANT_MX_IMPL_CLASS(Cgmct::Gemm::layout::RowMajor, Cgmct::Gemm::layout::ColumnMajor,
                                    Cgmct::Gemm::layout::RowMajorAlign, AscendC::Te::NDExtLayoutPtn,
                                    AscendC::Te::DNExtLayoutPtn, AscendC::Te::NDExtLayoutPtn);
        } else if constexpr (QUANT_B_TRANS == GMM_NO_TRANS && QUANT_A_TRANS == GMM_TRANS &&
                             KERNEL_TYPE == GMM_DEQUANT_FIXP) {
            GET_TILING_DATA_WITH_STRUCT(GMMQuantBasicApiTilingData, tilingData, tiling);
            if ASCEND_IS_AIV {
                GMM_QUANT_BASIC_API_EMPTY_TENSOR_WITHOUT_BIAS_IMPL_CLASS();
            }
            if ASCEND_IS_AIC {
                GMM_QUANT_MX_IMPL_CLASS(Cgmct::Gemm::layout::ColumnMajor, Cgmct::Gemm::layout::RowMajor,
                                        Cgmct::Gemm::layout::RowMajorAlign, AscendC::Te::DNExtLayoutPtn,
                                        AscendC::Te::NDExtLayoutPtn, AscendC::Te::NDExtLayoutPtn);
            }
        }
    } else {
        if constexpr (QUANT_B_TRANS == GMM_NO_TRANS && QUANT_A_TRANS == GMM_NO_TRANS &&
                      KERNEL_TYPE == GMM_DEQUANT_FIXP) {
            GET_TILING_DATA_WITH_STRUCT(GMMQuantBasicApiTilingData, tilingData, tiling);
            GMM_QUANT_MX_IMPL_CLASS(Cgmct::Gemm::layout::RowMajor, Cgmct::Gemm::layout::Nz,
                                    Cgmct::Gemm::layout::RowMajorAlign, AscendC::Te::NDExtLayoutPtn,
                                    AscendC::Te::NZLayoutPtn, AscendC::Te::NDExtLayoutPtn);
        } else if constexpr (QUANT_B_TRANS == GMM_TRANS && QUANT_A_TRANS == GMM_NO_TRANS &&
                             KERNEL_TYPE == GMM_DEQUANT_FIXP) {
            GET_TILING_DATA_WITH_STRUCT(GMMQuantBasicApiTilingData, tilingData, tiling);
            GMM_QUANT_MX_IMPL_CLASS(Cgmct::Gemm::layout::RowMajor, Cgmct::Gemm::layout::Zn,
                                    Cgmct::Gemm::layout::RowMajorAlign, AscendC::Te::NDExtLayoutPtn,
                                    AscendC::Te::ZNLayoutPtn, AscendC::Te::NDExtLayoutPtn);
        }
    }
#endif

#if defined(V310_GMM_QUANT_CUBE) || defined(V310_GMM_QUANT_PERTENSOR_CUBE) // scale64/perTensor/double perTensor
    if constexpr (QUANT_B_TRANS == GMM_NO_TRANS && QUANT_A_TRANS == GMM_NO_TRANS && KERNEL_TYPE == GMM_DEQUANT_FIXP) {
        GET_TILING_DATA_WITH_STRUCT(GMMQuantTilingData, tilingData, tiling);
        GMM_QUANT_IMPL_CLASS(false, false, GmmASWKernel);
    } else if constexpr (QUANT_B_TRANS == GMM_TRANS && QUANT_A_TRANS == GMM_NO_TRANS &&
                         KERNEL_TYPE == GMM_DEQUANT_FIXP) {
        GET_TILING_DATA_WITH_STRUCT(GMMQuantTilingData, tilingData, tiling);
        GMM_QUANT_IMPL_CLASS(false, true, GmmASWKernel);
    }
#endif
#if defined(V310_GMM_QUANT_PERTENSOR_CUBE) // perTensor/double perTensor
    if constexpr (QUANT_B_TRANS == GMM_NO_TRANS && QUANT_A_TRANS == GMM_TRANS && KERNEL_TYPE == GMM_DEQUANT_FIXP) {
        GET_TILING_DATA_WITH_STRUCT(GMMQuantTilingData, tilingData, tiling);
        GMM_QUANT_WITH_EMPTY_TENSOR_IMPL_CLASS(true, false, GmmASWKernel);
    }
#endif
#if defined(V310_GMM_QUANT_MIX) // perToken/SPLIT_K/scale bf16/fp32
    if constexpr (QUANT_B_TRANS == GMM_NO_TRANS && QUANT_A_TRANS == GMM_NO_TRANS && KERNEL_TYPE == GMM_DEQUANT_VECTOR) {
        GET_TILING_DATA_WITH_STRUCT(GMMQuantTilingData, tilingData, tiling);
        GMM_QUANT_MIX_IMPL_CLASS(false, false, GQmmMixRegbaseKernel);
    } else if constexpr (QUANT_B_TRANS == GMM_TRANS && QUANT_A_TRANS == GMM_NO_TRANS &&
                         KERNEL_TYPE == GMM_DEQUANT_VECTOR) {
        GET_TILING_DATA_WITH_STRUCT(GMMQuantTilingData, tilingData, tiling);
        GMM_QUANT_MIX_IMPL_CLASS(false, true, GQmmMixRegbaseKernel);
    } else if constexpr (QUANT_B_TRANS == GMM_NO_TRANS && QUANT_A_TRANS == GMM_TRANS &&
                         KERNEL_TYPE == GMM_DEQUANT_VECTOR) {
        GET_TILING_DATA_WITH_STRUCT(GMMQuantTilingData, tilingData, tiling);
        GMM_QUANT_MIX_IMPL_CLASS(true, false, GQmmMixRegbaseKernel);
    }
#endif
#if defined(V310_GMM_QUANT_PERTILE)
    if constexpr (QUANT_B_TRANS == GMM_NO_TRANS && QUANT_A_TRANS == GMM_NO_TRANS &&
                  KERNEL_TYPE == GMM_PERGROUP_PERBLOCK) {
        GET_TILING_DATA_WITH_STRUCT(GMMQuantTilingData, tilingData, tiling);
        GMM_QUANT_GB_IMPL_CLASS(Cgmct::Gemm::layout::RowMajor, Cgmct::Gemm::layout::RowMajor,
                                Cgmct::Gemm::layout::RowMajorAlign);
    } else if constexpr (QUANT_B_TRANS == GMM_TRANS && QUANT_A_TRANS == GMM_NO_TRANS &&
                         KERNEL_TYPE == GMM_PERGROUP_PERBLOCK) {
        GET_TILING_DATA_WITH_STRUCT(GMMQuantTilingData, tilingData, tiling);
        GMM_QUANT_GB_IMPL_CLASS(Cgmct::Gemm::layout::RowMajor, Cgmct::Gemm::layout::ColumnMajor,
                                Cgmct::Gemm::layout::RowMajorAlign);
    } else if constexpr (QUANT_B_TRANS == GMM_NO_TRANS && QUANT_A_TRANS == GMM_TRANS &&
                         KERNEL_TYPE == GMM_PERGROUP_PERBLOCK) {
        GET_TILING_DATA_WITH_STRUCT(GMMQuantTilingData, tilingData, tiling);
        GMM_QUANT_GB_IMPL_CLASS(Cgmct::Gemm::layout::ColumnMajor, Cgmct::Gemm::layout::RowMajor,
                                Cgmct::Gemm::layout::RowMajorAlign);
    }
#endif
#if defined(V310_GMM_QUANT_S4S4) && IS_BLAZE
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2); // S4S4 需 1:2 AIC:AIV
    if constexpr (wFormat == CubeFormat::ND) {
        if constexpr (QUANT_B_TRANS == GMM_NO_TRANS && QUANT_A_TRANS == GMM_NO_TRANS && KERNEL_TYPE == GMM_S4S4_MIX) {
            GET_TILING_DATA_WITH_STRUCT(GMMS4S4IntQuantTilingData, tilingData, tiling);
            GMM_S4S4_IMPL_CLASS(AscendC::Te::NDExtLayoutPtn, AscendC::Te::NDExtLayoutPtn, AscendC::Te::NDExtLayoutPtn);
        }
    } else {
        if constexpr ((QUANT_B_TRANS == GMM_NO_TRANS || QUANT_B_TRANS == GMM_TRANS) && QUANT_A_TRANS == GMM_NO_TRANS &&
                      KERNEL_TYPE == GMM_S4S4_MIX) {
            GET_TILING_DATA_WITH_STRUCT(GMMS4S4IntQuantTilingData, tilingData, tiling);
            GMM_S4S4_IMPL_CLASS(AscendC::Te::NDExtLayoutPtn, AscendC::Te::NZLayoutPtn, AscendC::Te::NDExtLayoutPtn);
        }
    }
#endif
#elif defined(V310_GMM_ANTI_QUANT)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
#if defined(V310_GMM_S8S4_UINT64_SCALE)
    constexpr bool IS_S8S4_V5_COMMON =
        (W_TYPE == WQGMM_ND || W_TYPE == WQGMM_FRACTAL_NZ) &&
        OFFSET_OR_BIAS_EXIT == WQGMM_ANTIQUANT_OFFSET_NOT_EXIST_BIAS_NOT_EXIST && W_QUANT_TYPE == WQGMM_NONE &&
        WQ_B_TRANS == WQGMM_NO_TRANS && WQ_A_TRANS == WQGMM_NO_TRANS &&
        TEMPLATE_CUSTOM_SC == WQGMM_S8S4_FIXED_TEMPLATE && IS_SINGLE_MULTI_SINGLE == WQGMM_NOT_SINGLE_MULTI_SINGLE &&
        ALGORITHM_SUB_CATEGORY == WQGMM_VDEFAULT && ALGORITHM_CATEGORY == WQGMM_MULTI_SCALE_DEQUANT;
    constexpr bool IS_S8S4_V5_PER_GROUP = IS_S8S4_V5_COMMON && C_QUANT_TYPE == WQGMM_PER_GROUP;
    constexpr bool IS_S8S4_V5_PER_CHANNEL =
        IS_S8S4_V5_COMMON && C_QUANT_TYPE == WQGMM_PER_CHANNEL && AscendC::IsSameType<DTYPE_Y, half>::value;
    REGISTER_TILING_DEFAULT(GMMS8S4BasicApiTilingData);
    if constexpr (IS_S8S4_V5_PER_GROUP || IS_S8S4_V5_PER_CHANNEL) {
        GET_TILING_DATA_WITH_STRUCT(GMMS8S4BasicApiTilingData, s8s4TilingData, tiling);
        GROUPED_MATMUL::S8S4V5::InvokeGmmS8S4TensorApi<DTYPE_Y, wFormat>(
            x, weight, scale, offset, groupList, perTokenScale, y, user1, &tPipe, &s8s4TilingData);
    }
#else
    REGISTER_TILING_DEFAULT(GMMWeightQuantTilingData);
    {
#if ORIG_DTYPE_X == DT_INT8
        if constexpr (W_TYPE == WQGMM_FRACTAL_NZ &&
                      OFFSET_OR_BIAS_EXIT == WQGMM_ANTIQUANT_OFFSET_NOT_EXIST_BIAS_NOT_EXIST &&
                      C_QUANT_TYPE == WQGMM_NONE && W_QUANT_TYPE == WQGMM_PER_CHANNEL && WQ_B_TRANS == WQGMM_NO_TRANS &&
                      WQ_A_TRANS == WQGMM_NO_TRANS && TEMPLATE_CUSTOM_SC == WQGMM_MTE2_INNER_SIZE_512_BUF_NUM_DEFAULT &&
                      ALGORITHM_SUB_CATEGORY == WQGMM_N_FIRST_TAIL_RESPLIT &&
                      ALGORITHM_CATEGORY == WQGMM_VECTOR_ANTIQUANT) {
            INVOKE_GMM_WEIGHT_QUANT_VCV_CONTROLLER_OP_IMPL(GMMWeightQuantResplitController, S8S4_NZKN_G,
                                                           VEC_ANTIQUANT_CONFIG_4);
        } else if constexpr (W_TYPE == WQGMM_FRACTAL_NZ &&
                             OFFSET_OR_BIAS_EXIT == WQGMM_ANTIQUANT_OFFSET_NOT_EXIST_BIAS_NOT_EXIST &&
                             C_QUANT_TYPE == WQGMM_NONE && W_QUANT_TYPE == WQGMM_PER_CHANNEL &&
                             WQ_B_TRANS == WQGMM_NO_TRANS && WQ_A_TRANS == WQGMM_NO_TRANS &&
                             TEMPLATE_CUSTOM_SC == WQGMM_MTE2_INNER_SIZE_384_BUF_NUM_3 &&
                             ALGORITHM_SUB_CATEGORY == WQGMM_N_FIRST_TAIL_RESPLIT &&
                             ALGORITHM_CATEGORY == WQGMM_VECTOR_ANTIQUANT) {
            INVOKE_GMM_WEIGHT_QUANT_VCV_CONTROLLER_OP_IMPL(GMMWeightQuantResplitController, S8S4_NZKN_G,
                                                           VEC_ANTIQUANT_CONFIG_5);
        }
#elif ORIG_DTYPE_X == DT_FLOAT8_E4M3FN
        if constexpr (W_TYPE == WQGMM_FRACTAL_NZ &&
                      OFFSET_OR_BIAS_EXIT == WQGMM_ANTIQUANT_OFFSET_NOT_EXIST_BIAS_NOT_EXIST &&
                      C_QUANT_TYPE == WQGMM_NONE && W_QUANT_TYPE == WQGMM_MX && WQ_B_TRANS == WQGMM_TRANS &&
                      WQ_A_TRANS == WQGMM_NO_TRANS && TEMPLATE_CUSTOM_SC == WQGMM_MTE2_INNER_SIZE_DYNAMIC_BUF_NUM_4 &&
                      ALGORITHM_SUB_CATEGORY == WQGMM_N_FIRST_TAIL_RESPLIT &&
                      ALGORITHM_CATEGORY == WQGMM_VECTOR_ANTIQUANT) {
#if IS_BLAZE
            GET_TILING_DATA_MEMBER(GMMWeightQuantTilingData, gmmWeightQuantParam, gmmBaseParams_, tiling);
            GET_TILING_DATA_MEMBER(GMMWeightQuantTilingData, mmTilingData, mmTilingData_, tiling);
            GROUPED_MATMUL::GmmWeightQuantTensorApiMxKernel<DTYPE_X, DTYPE_WEIGHT, DTYPE_ANTIQUANT_SCALE, DTYPE_BIAS,
                                                            DTYPE_Y, IS_SINGLE_MULTI_SINGLE>(
                x, weight, antiquantScale, bias, groupList, perTokenScale, y, &gmmBaseParams_, &mmTilingData_);
#else
            INVOKE_GMM_WEIGHT_QUANT_MXA8W4_CONTROLLER_OP_IMPL(GMMWeightQuantResplitController, MXA8W4_NZNK,
                                                              VEC_ANTIQUANT_CONFIG_DYNAMIC);
#endif
        }
#elif ORIG_DTYPE_ANTIQUANT_SCALE == DT_FLOAT8_E8M0
        if constexpr (W_TYPE == WQGMM_FRACTAL_NZ &&
                      OFFSET_OR_BIAS_EXIT == WQGMM_ANTIQUANT_OFFSET_NOT_EXIST_BIAS_NOT_EXIST &&
                      C_QUANT_TYPE == WQGMM_NONE && W_QUANT_TYPE == WQGMM_MX && WQ_B_TRANS == WQGMM_NO_TRANS &&
                      WQ_A_TRANS == WQGMM_NO_TRANS && TEMPLATE_CUSTOM_SC == WQGMM_MTE2_INNER_SIZE_256_BUF_NUM_4 &&
                      ALGORITHM_SUB_CATEGORY == WQGMM_N_FIRST_TAIL_RESPLIT &&
                      ALGORITHM_CATEGORY == WQGMM_VECTOR_ANTIQUANT) {
            INVOKE_GMM_WEIGHT_QUANT_RESPLIT_CONTROLLER_OP_IMPL(GMMWeightQuantResplitController, A16MXF4_NZKN,
                                                               VEC_ANTIQUANT_CONFIG_3);
        }
#else
        if constexpr (W_TYPE == WQGMM_ND && OFFSET_OR_BIAS_EXIT == WQGMM_ANTIQUANT_OFFSET_NOT_EXIST_BIAS_NOT_EXIST &&
                      C_QUANT_TYPE == WQGMM_NONE && W_QUANT_TYPE == WQGMM_PER_CHANNEL && WQ_B_TRANS == WQGMM_TRANS &&
                      WQ_A_TRANS == WQGMM_NO_TRANS && TEMPLATE_CUSTOM_SC == WQGMM_MTE2_INNER_SIZE_256_BUF_NUM_4 &&
                      ALGORITHM_SUB_CATEGORY == WQGMM_N_FIRST_TAIL_RESPLIT &&
                      ALGORITHM_CATEGORY == WQGMM_VECTOR_ANTIQUANT) {
            static constexpr WqmmConfig wqmmCfg = {false,           true,          QuantType::PER_CHANNEL, false,
                                                   QuantType::NONE, CubeFormat::ND};
            INVOKE_GMM_WEIGHT_QUANT_RESPLIT_CONTROLLER_OP_IMPL(GMMWeightQuantResplitController, wqmmCfg,
                                                               VEC_ANTIQUANT_CONFIG_3);
        } else if constexpr (W_TYPE == WQGMM_ND && OFFSET_OR_BIAS_EXIT == WQGMM_ANTIQUANT_OFFSET_EXIST_BIAS_NOT_EXIST &&
                             C_QUANT_TYPE == WQGMM_NONE && W_QUANT_TYPE == WQGMM_PER_CHANNEL &&
                             WQ_B_TRANS == WQGMM_TRANS && WQ_A_TRANS == WQGMM_NO_TRANS &&
                             TEMPLATE_CUSTOM_SC == WQGMM_MTE2_INNER_SIZE_256_BUF_NUM_4 &&
                             ALGORITHM_SUB_CATEGORY == WQGMM_N_FIRST_TAIL_RESPLIT &&
                             ALGORITHM_CATEGORY == WQGMM_VECTOR_ANTIQUANT) {
            static constexpr WqmmConfig wqmmCfg = {false,           true,          QuantType::PER_CHANNEL, true,
                                                   QuantType::NONE, CubeFormat::ND};
            INVOKE_GMM_WEIGHT_QUANT_RESPLIT_CONTROLLER_OP_IMPL(GMMWeightQuantResplitController, wqmmCfg,
                                                               VEC_ANTIQUANT_CONFIG_3);
        } else if constexpr (W_TYPE == WQGMM_ND &&
                             OFFSET_OR_BIAS_EXIT == WQGMM_ANTIQUANT_OFFSET_NOT_EXIST_BIAS_NOT_EXIST &&
                             C_QUANT_TYPE == WQGMM_NONE && W_QUANT_TYPE == WQGMM_PER_CHANNEL &&
                             WQ_B_TRANS == WQGMM_TRANS && WQ_A_TRANS == WQGMM_NO_TRANS &&
                             TEMPLATE_CUSTOM_SC == WQGMM_MTE2_INNER_SIZE_256_BUF_NUM_4 &&
                             ALGORITHM_SUB_CATEGORY == WQGMM_N_FIRST_BASIC_BLOCK &&
                             ALGORITHM_CATEGORY == WQGMM_VECTOR_ANTIQUANT) {
            static constexpr WqmmConfig wqmmCfg = {false,           true,          QuantType::PER_CHANNEL, false,
                                                   QuantType::NONE, CubeFormat::ND};
            INVOKE_GMM_WEIGHT_QUANT_BASIC_CONTROLLER_OP_IMPL(GMMWeightQuantBasicController, wqmmCfg,
                                                             VEC_ANTIQUANT_CONFIG_3);
        } else if constexpr (W_TYPE == WQGMM_ND && OFFSET_OR_BIAS_EXIT == WQGMM_ANTIQUANT_OFFSET_EXIST_BIAS_NOT_EXIST &&
                             C_QUANT_TYPE == WQGMM_NONE && W_QUANT_TYPE == WQGMM_PER_CHANNEL &&
                             WQ_B_TRANS == WQGMM_TRANS && WQ_A_TRANS == WQGMM_NO_TRANS &&
                             TEMPLATE_CUSTOM_SC == WQGMM_MTE2_INNER_SIZE_256_BUF_NUM_4 &&
                             ALGORITHM_SUB_CATEGORY == WQGMM_N_FIRST_BASIC_BLOCK &&
                             ALGORITHM_CATEGORY == WQGMM_VECTOR_ANTIQUANT) {
            static constexpr WqmmConfig wqmmCfg = {false,           true,          QuantType::PER_CHANNEL, true,
                                                   QuantType::NONE, CubeFormat::ND};
            INVOKE_GMM_WEIGHT_QUANT_BASIC_CONTROLLER_OP_IMPL(GMMWeightQuantBasicController, wqmmCfg,
                                                             VEC_ANTIQUANT_CONFIG_3);
        } else if constexpr (W_TYPE == WQGMM_ND &&
                             OFFSET_OR_BIAS_EXIT == WQGMM_ANTIQUANT_OFFSET_NOT_EXIST_BIAS_NOT_EXIST &&
                             C_QUANT_TYPE == WQGMM_NONE && W_QUANT_TYPE == WQGMM_PER_CHANNEL &&
                             WQ_B_TRANS == WQGMM_NO_TRANS && WQ_A_TRANS == WQGMM_NO_TRANS &&
                             TEMPLATE_CUSTOM_SC == WQGMM_MTE2_INNER_SIZE_256_BUF_NUM_4 &&
                             ALGORITHM_SUB_CATEGORY == WQGMM_N_FIRST_BASIC_BLOCK &&
                             ALGORITHM_CATEGORY == WQGMM_VECTOR_ANTIQUANT) {
            static constexpr WqmmConfig wqmmCfg = {false, false,           QuantType::PER_CHANNEL,
                                                   false, QuantType::NONE, CubeFormat::ND};
            INVOKE_GMM_WEIGHT_QUANT_BASIC_CONTROLLER_OP_IMPL(GMMWeightQuantBasicController, wqmmCfg,
                                                             VEC_ANTIQUANT_CONFIG_3);
        } else if constexpr (W_TYPE == WQGMM_ND && OFFSET_OR_BIAS_EXIT == WQGMM_ANTIQUANT_OFFSET_EXIST_BIAS_NOT_EXIST &&
                             C_QUANT_TYPE == WQGMM_NONE && W_QUANT_TYPE == WQGMM_PER_CHANNEL &&
                             WQ_B_TRANS == WQGMM_NO_TRANS && WQ_A_TRANS == WQGMM_NO_TRANS &&
                             TEMPLATE_CUSTOM_SC == WQGMM_MTE2_INNER_SIZE_256_BUF_NUM_4 &&
                             ALGORITHM_SUB_CATEGORY == WQGMM_N_FIRST_BASIC_BLOCK &&
                             ALGORITHM_CATEGORY == WQGMM_VECTOR_ANTIQUANT) {
            static constexpr WqmmConfig wqmmCfg = {false, false,           QuantType::PER_CHANNEL,
                                                   true,  QuantType::NONE, CubeFormat::ND};
            INVOKE_GMM_WEIGHT_QUANT_BASIC_CONTROLLER_OP_IMPL(GMMWeightQuantBasicController, wqmmCfg,
                                                             VEC_ANTIQUANT_CONFIG_3);
        } else if constexpr (W_TYPE == WQGMM_ND &&
                             (OFFSET_OR_BIAS_EXIT == WQGMM_ANTIQUANT_OFFSET_NOT_EXIST_BIAS_NOT_EXIST ||
                              OFFSET_OR_BIAS_EXIT == WQGMM_ANTIQUANT_OFFSET_EXIST_BIAS_NOT_EXIST) &&
                             C_QUANT_TYPE == WQGMM_NONE && W_QUANT_TYPE == WQGMM_PER_GROUP &&
                             (WQ_B_TRANS == WQGMM_NO_TRANS || WQ_B_TRANS == WQGMM_TRANS) &&
                             WQ_A_TRANS == WQGMM_NO_TRANS &&
                             TEMPLATE_CUSTOM_SC == WQGMM_MTE2_INNER_SIZE_256_BUF_NUM_2 &&
                             ALGORITHM_SUB_CATEGORY == WQGMM_N_FIRST_BASIC_BLOCK &&
                             ALGORITHM_CATEGORY == WQGMM_VECTOR_ANTIQUANT) {
            static constexpr WqmmConfig wqmmCfg = {false,
                                                   WQ_B_TRANS == WQGMM_TRANS,
                                                   QuantType::PER_GROUP,
                                                   OFFSET_OR_BIAS_EXIT == WQGMM_ANTIQUANT_OFFSET_EXIST_BIAS_NOT_EXIST,
                                                   QuantType::NONE,
                                                   CubeFormat::ND};
            INVOKE_GMM_WEIGHT_QUANT_BASIC_CONTROLLER_OP_IMPL(GMMWeightQuantBasicController, wqmmCfg,
                                                             VEC_ANTIQUANT_CONFIG_6);
        }
#endif
    }
#endif
#else
    REGISTER_TILING_DEFAULT(GMMNoQuantTilingData);
    if constexpr (NO_QUANT_B_TRANS == GMM_NO_TRANS && NO_QUANT_A_TRANS == GMM_NO_TRANS) {
        if constexpr (wFormat == CubeFormat::NZ) {
            GroupedMatMulKernel<AscendC::Te::NDExtLayoutPtn, AscendC::Te::NZLayoutPtn>(x, weight, bias, groupList, y,
                                                                                       tiling);
        } else {
            GroupedMatMulKernel<AscendC::Te::NDExtLayoutPtn, AscendC::Te::NDExtLayoutPtn>(x, weight, bias, groupList, y,
                                                                                          tiling);
        }
    } else if constexpr (NO_QUANT_B_TRANS == GMM_NO_TRANS && NO_QUANT_A_TRANS == GMM_TRANS) { // x transposed
        if constexpr (wFormat == CubeFormat::NZ) {
            GroupedMatMulKernel<AscendC::Te::DNExtLayoutPtn, AscendC::Te::NZLayoutPtn>(x, weight, bias, groupList, y,
                                                                                       tiling);
        } else {
            GroupedMatMulKernel<AscendC::Te::DNExtLayoutPtn, AscendC::Te::NDExtLayoutPtn>(x, weight, bias, groupList, y,
                                                                                          tiling);
        }
    } else if constexpr (NO_QUANT_B_TRANS == GMM_TRANS && NO_QUANT_A_TRANS == GMM_NO_TRANS) { // weight transposed
        if constexpr (wFormat == CubeFormat::NZ) {
            GroupedMatMulKernel<AscendC::Te::NDExtLayoutPtn, AscendC::Te::ZNLayoutPtn>(x, weight, bias, groupList, y,
                                                                                       tiling);
        } else {
            GroupedMatMulKernel<AscendC::Te::NDExtLayoutPtn, AscendC::Te::DNExtLayoutPtn>(x, weight, bias, groupList, y,
                                                                                          tiling);
        }
    }
#endif
#endif
}
