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
 * \file matmul_all_reduce_apt.cpp
 * \brief A5
 */

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "lib/matmul_intf.h"
#include "../matmul_all_reduce_common.h"
#include "matmul_all_reduce_empty_tensor_k_general_arch35.h"
#include "matmul_all_reduce_apt_tiling_key.h"

// david非量化
#if ((ORIG_DTYPE_X1 == ORIG_DTYPE_X2) && ((ORIG_DTYPE_X1 == DT_FLOAT16) || (ORIG_DTYPE_X1 == DT_BF16)))
#include "matmul_all_reduce_910_general_arch35.h"
#endif

#if defined(WEIGHT_W4_W8)
#include "matmul_all_reduce_weight_quant_arch35.h"
#endif

#if defined(WEIGHT_F8) || defined(WEIGHT_W4_W8)
#include "matmul_all_reduce_weight_quant_adaptive_split.h"
using Mc2WeightQuantBatchMatmulV2::Arch35::WeightQuantBatchMatmulV2BasicBlockController;
namespace {
static constexpr Mc2WeightQuantBatchMatmulV2::Arch35::VecAntiQuantConfig VEC_ANTIQUANT_CONFIG_0 = {2,
                                                                                                   512}; // b 转置场景
static constexpr Mc2WeightQuantBatchMatmulV2::Arch35::VecAntiQuantConfig VEC_ANTIQUANT_CONFIG_1 = {4, 512};
static constexpr Mc2WeightQuantBatchMatmulV2::Arch35::VecAntiQuantConfig VEC_ANTIQUANT_CONFIG_2 = {2, 1024};
static constexpr Mc2WeightQuantBatchMatmulV2::Arch35::VecAntiQuantConfig VEC_ANTIQUANT_CONFIG_3 = {4,
                                                                                                   256}; // b 非转置场景
static constexpr Mc2WeightQuantBatchMatmulV2::Arch35::VecAntiQuantConfig VEC_ANTIQUANT_CONFIG_4 = {2, 256}; // solomon
} // namespace
#endif

#if ((ORIG_DTYPE_X1 == ORIG_DTYPE_X2) && (ORIG_DTYPE_X1 == DT_INT8)) ||                                                \
    (((ORIG_DTYPE_X1 == ORIG_DTYPE_X2) && (ORIG_DTYPE_X1 == DT_HIFLOAT8)) ||                                           \
     (((ORIG_DTYPE_X1 == DT_FLOAT8_E4M3FN) || (ORIG_DTYPE_X1 == DT_FLOAT8_E5M2)) &&                                    \
      ((ORIG_DTYPE_X2 == DT_FLOAT8_E4M3FN) || (ORIG_DTYPE_X2 == DT_FLOAT8_E5M2)))) ||                                  \
    ((ORIG_DTYPE_X1 == DT_FLOAT4_E2M1) && (ORIG_DTYPE_X2 == DT_FLOAT4_E2M1))
#include "matmul_all_reduce_quant_pertoken.h"
#include "matmul_all_reduce_quant_pertoken_comm_int8_arch35.h"
#include "matmul_all_reduce_quant_arch35.h"
#include "matmul_all_reduce_quant_comm_int8.h"
#include "matmul_all_reduce_quant_perblock.h"
#include "matmul_all_reduce_quant_commfp8_mixed_calc.h"
#endif

namespace MatmulAllReduceImpl {
}

using namespace AscendC;
using namespace MatmulAllReduceImpl;

template <TPL_APT_PARAMS_FP_MM, TPL_APT_A2A_RS_AG, TPL_APT_COMM_MODE>
__global__ __aicore__ void fp_matmul_all_reduce(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR addGM,
                                                GM_ADDR antiquantScaleGM, GM_ADDR antiquantOffsetGM, GM_ADDR dequantGM,
                                                GM_ADDR pertokenGM, GM_ADDR commQuantScale1GM,
                                                GM_ADDR commQuantScale2GM, GM_ADDR cGM, GM_ADDR workspaceGM,
                                                GM_ADDR tilingGM, TPipe &tPipe, GM_ADDR userWS)
{
#if ((ORIG_DTYPE_X1 == ORIG_DTYPE_X2) && ((ORIG_DTYPE_X1 == DT_FLOAT16) || (ORIG_DTYPE_X1 == DT_BF16)))
    // 910非量化
    if constexpr (!MATMUL_WITH_ADD) {
        INVOKE_MC2_910_OP_IMPL(Mc2MatmulV3Advanced::Mc2MatmulAswKernel, Mc2CoreType::ON_CUBE, APT_A2A_RS_AG,
                               APT_COMM_MODE);
    } else if constexpr (MATMUL_WITH_ADD) {
        INVOKE_MC2_910_OP_IMPL(Mc2MatmulV3Advanced::Mc2MatmulAswKernel, Mc2CoreType::ON_CUBE_AND_VECTOR, APT_A2A_RS_AG,
                               APT_COMM_MODE);
    }
#endif
}

template <TPL_APT_PARAMS_COMM, TPL_APT_PARAMS_QUANT_MM, TPL_APT_A2A_RS_AG, TPL_APT_COMM_MODE>
__global__ __aicore__ void quant_matmul_all_reduce(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR addGM,
                                                   GM_ADDR antiquantScaleGM, GM_ADDR antiquantOffsetGM,
                                                   GM_ADDR dequantGM, GM_ADDR pertokenGM, GM_ADDR commQuantScale1GM,
                                                   GM_ADDR commQuantScale2GM, GM_ADDR cGM, GM_ADDR workspaceGM,
                                                   GM_ADDR tilingGM, TPipe &tPipe, GM_ADDR userWS)
{
#if defined(DAVID_QUANT_INT8_OUT_FP16)
#undef DTYPE_BIAS
#define DTYPE_BIAS int32_t
    if constexpr (!SCENARIO_MXFP8 && COMMDTPYE == COMMDTPYE_DEFAULT && KERNEL_TYPE == NO_VEC_EPILOGUE_WITH_MMAPI) {
        INVOKE_MC2_QUANT_910_OP_IMPL(AscendC::MatMulASWKernel, Mc2CoreType::ON_CUBE_AND_VECTOR, APT_A2A_RS_AG,
                                     APT_COMM_MODE, uint64_t, false, TPL_TRANS_B);
    } else if constexpr (COMMDTPYE == COMMDTPYE_INT8 && KERNEL_TYPE == NO_VEC_EPILOGUE_WITH_MMAPI) {
        INVOKE_MC2_QUANT_COMM_INT8_910_OP_IMPL(AscendC::MatMulASWKernel, Mc2CoreType::ON_CUBE_AND_VECTOR, APT_COMM_MODE,
                                               uint64_t, false, TPL_TRANS_B);
    }

    if constexpr (COMMDTPYE == COMMDTPYE_DEFAULT && KERNEL_TYPE == VEC_EPILOGUE_WITH_MMAPI) {
        INVOKE_BATCH_MATMUL_QUANT_PERTOKEN_IMPL(Mc2QuantBatchMatmulV3::Mc2QuantBmmPertokenRegbaseKernel,
                                                Mc2CoreType::ON_CUBE_AND_VECTOR, APT_A2A_RS_AG, APT_COMM_MODE, float,
                                                false, TPL_TRANS_B);
    } else if constexpr (COMMDTPYE == COMMDTPYE_INT8 && KERNEL_TYPE == VEC_EPILOGUE_WITH_MMAPI) {
        INVOKE_BATCH_MATMUL_QUANT_PERTOKEN_COMM_INT8_IMPL(Mc2QuantBatchMatmulV3::Mc2QuantBmmPertokenRegbaseKernel,
                                                          Mc2CoreType::ON_CUBE_AND_VECTOR, APT_COMM_MODE, float, false,
                                                          TPL_TRANS_B);
    }
#elif defined(DAVID_QUANT_INT8_OUT_BF16)
#undef DTYPE_BIAS
#define DTYPE_BIAS int32_t
    if constexpr (!SCENARIO_MXFP8 && COMMDTPYE == COMMDTPYE_DEFAULT && KERNEL_TYPE == NO_VEC_EPILOGUE_WITH_MMAPI) {
        INVOKE_MC2_QUANT_910_OP_IMPL(AscendC::MatMulASWKernel, Mc2CoreType::ON_CUBE_AND_VECTOR, APT_A2A_RS_AG,
                                     APT_COMM_MODE, DTYPE_Y, false, TPL_TRANS_B);
    } else if constexpr (COMMDTPYE == COMMDTPYE_INT8 && KERNEL_TYPE == NO_VEC_EPILOGUE_WITH_MMAPI) {
        INVOKE_MC2_QUANT_COMM_INT8_910_OP_IMPL(AscendC::MatMulASWKernel, Mc2CoreType::ON_CUBE_AND_VECTOR, APT_COMM_MODE,
                                               DTYPE_Y, false, TPL_TRANS_B);
    }

    if constexpr (COMMDTPYE == COMMDTPYE_DEFAULT && KERNEL_TYPE == VEC_EPILOGUE_WITH_MMAPI) {
        INVOKE_BATCH_MATMUL_QUANT_PERTOKEN_IMPL(Mc2QuantBatchMatmulV3::Mc2QuantBmmPertokenRegbaseKernel,
                                                Mc2CoreType::ON_CUBE_AND_VECTOR, APT_A2A_RS_AG, APT_COMM_MODE, DTYPE_Y,
                                                false, TPL_TRANS_B);
    } else if constexpr (COMMDTPYE == COMMDTPYE_INT8 && KERNEL_TYPE == VEC_EPILOGUE_WITH_MMAPI) {
        INVOKE_BATCH_MATMUL_QUANT_PERTOKEN_COMM_INT8_IMPL(Mc2QuantBatchMatmulV3::Mc2QuantBmmPertokenRegbaseKernel,
                                                          Mc2CoreType::ON_CUBE_AND_VECTOR, APT_COMM_MODE, DTYPE_Y,
                                                          false, TPL_TRANS_B);
    }
#elif ((ORIG_DTYPE_X1 == DT_FLOAT4_E2M1) && (ORIG_DTYPE_X2 == DT_FLOAT4_E2M1))
// fp8,hif8和mixfp的场景，bias都是float
#undef DTYPE_BIAS
#define DTYPE_BIAS float
    if constexpr (TPL_TRANS_B && !SCENARIO_MXFP8 && COMMDTPYE == COMMDTPYE_DEFAULT &&
                  KERNEL_TYPE == NO_VEC_EPILOGUE_WITH_MMAPI) {
        INVOKE_MC2_QUANT_MXFP_910_OP_IMPL(AscendC::MatMulASWKernel, Mc2CoreType::ON_CUBE_AND_VECTOR, APT_A2A_RS_AG,
                                          APT_COMM_MODE, false, true);
    }
#elif (((ORIG_DTYPE_X1 == ORIG_DTYPE_X2) && (ORIG_DTYPE_X1 == DT_HIFLOAT8)) ||                                         \
       (((ORIG_DTYPE_X1 == DT_FLOAT8_E4M3FN) || (ORIG_DTYPE_X1 == DT_FLOAT8_E5M2)) &&                                  \
        ((ORIG_DTYPE_X2 == DT_FLOAT8_E4M3FN) || (ORIG_DTYPE_X2 == DT_FLOAT8_E5M2))))
// fp8,hif8和mixfp的场景，bias都是float
#undef DTYPE_BIAS
#define DTYPE_BIAS float
#if (ORIG_DTYPE_X1 != DT_HIFLOAT8)
    if constexpr (TPL_TRANS_B && SCENARIO_MXFP8 && COMMDTPYE == COMMDTPYE_DEFAULT &&
                  KERNEL_TYPE == NO_VEC_EPILOGUE_WITH_MMAPI) {
        INVOKE_MC2_QUANT_MXFP_910_OP_IMPL(AscendC::MatMulASWKernel, Mc2CoreType::ON_CUBE_AND_VECTOR, APT_A2A_RS_AG,
                                          APT_COMM_MODE, false, true);
    } else if constexpr (COMMDTPYE == COMMDTPYE_FP8 && KERNEL_TYPE == VEC_EPILOGUE_WITH_MMAPI) {
        INVOKE_MC2_COMM_FP8_MIXED_CALC_910_OP_IMPL(Mc2QuantBatchMatmulV3::Mc2QuantBmmPertokenRegbaseKernel,
                                                   Mc2CoreType::ON_CUBE_AND_VECTOR, APT_COMM_MODE, false, TPL_TRANS_B);
    }
#endif
    if constexpr (!SCENARIO_MXFP8 && COMMDTPYE == COMMDTPYE_DEFAULT && KERNEL_TYPE == NO_VEC_EPILOGUE_WITH_MMAPI) {
        INVOKE_MC2_QUANT_910_OP_IMPL(AscendC::MatMulASWKernel, Mc2CoreType::ON_CUBE_AND_VECTOR, APT_A2A_RS_AG,
                                     APT_COMM_MODE, uint64_t, false, TPL_TRANS_B);
    } else if constexpr (COMMDTPYE == COMMDTPYE_DEFAULT && KERNEL_TYPE == VEC_EPILOGUE_WITH_MMAPI) {
        INVOKE_BATCH_MATMUL_QUANT_PERTOKEN_IMPL(Mc2QuantBatchMatmulV3::Mc2QuantBmmPertokenRegbaseKernel,
                                                Mc2CoreType::ON_CUBE_AND_VECTOR, APT_A2A_RS_AG, APT_COMM_MODE, float,
                                                false, TPL_TRANS_B);
    } else if constexpr (COMMDTPYE == COMMDTPYE_DEFAULT && KERNEL_TYPE == VEC_EPILOGUE_WITH_CUSTOM_MM) {
        INVOKE_MC2_QUANT_PERBLOCK_910_OP_IMPL(Mc2QuantBatchMatmulV3::MatMulPerBlockASW, Mc2CoreType::ON_CUBE_AND_VECTOR,
                                              APT_A2A_RS_AG, APT_COMM_MODE, false, TPL_TRANS_B);
    }
#endif
}

template <TPL_APT_PARAMS_COMM, TPL_APT_PARAMS_WEIGHT_QUANT_MM, TPL_APT_A2A_RS_AG, TPL_APT_COMM_MODE>
__global__ __aicore__ void
weight_quant_matmul_allreduce(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR addGM, GM_ADDR antiquantScaleGM,
                              GM_ADDR antiquantOffsetGM, GM_ADDR dequantGM, GM_ADDR pertokenGM,
                              GM_ADDR commQuantScale1GM, GM_ADDR commQuantScale2GM, GM_ADDR cGM, GM_ADDR workspaceGM,
                              GM_ADDR tilingGM, TPipe &tPipe, GM_ADDR userWS)
{
#if defined(WEIGHT_W4_W8) || defined(WEIGHT_F8)
#if defined(WEIGHT_W4_W8)
#undef DTYPE_BIAS
#define DTYPE_BIAS DTYPE_X1
    if constexpr (ANTIQUANT_TYPE == TPL_PER_GROUP) {
        INVOKE_MC2_WEIGHT_QUANT_KERNEL(TPL_TRANS_B, Mc2QuantType::PER_GROUP, HAS_ANTIQUANT_OFFSET, false,

                                       APT_A2A_RS_AG, APT_COMM_MODE);
    }
#endif
    if constexpr (TEMPLATE_CUSTOM != TPL_MTE2_INNER_SIZE_1024_BUF_NUM_2 && TPL_TRANS_B &&
                  ANTIQUANT_TYPE == TPL_PER_CHANNEL && !TPL_EXIST_BIAS && !BAIS_IS_FP32) {
        INVOKE_MC2_WEIGHT_QUANT_ADAPTIVE_SPLIT_KERNEL(true, HAS_ANTIQUANT_OFFSET, Mc2QuantType::PER_CHANNEL, DTYPE_BIAS,
                                                      VEC_ANTIQUANT_CONFIG_0, APT_A2A_RS_AG, APT_COMM_MODE);
    } else if constexpr (TEMPLATE_CUSTOM == TPL_MTE2_INNER_SIZE_1024_BUF_NUM_2 && HAS_ANTIQUANT_OFFSET && TPL_TRANS_B &&
                         ANTIQUANT_TYPE == TPL_PER_CHANNEL && !TPL_EXIST_BIAS && !BAIS_IS_FP32) {
        INVOKE_MC2_WEIGHT_QUANT_ADAPTIVE_SPLIT_KERNEL(true, true, Mc2QuantType::PER_CHANNEL, DTYPE_BIAS,
                                                      VEC_ANTIQUANT_CONFIG_2, APT_A2A_RS_AG, APT_COMM_MODE);
    } else if constexpr (TEMPLATE_CUSTOM == TPL_MTE2_INNER_SIZE_1024_BUF_NUM_2 && !HAS_ANTIQUANT_OFFSET &&
                         TPL_TRANS_B && ANTIQUANT_TYPE == TPL_PER_CHANNEL && !TPL_EXIST_BIAS && !BAIS_IS_FP32) {
        INVOKE_MC2_WEIGHT_QUANT_ADAPTIVE_SPLIT_KERNEL(true, false, Mc2QuantType::PER_CHANNEL, DTYPE_BIAS,
                                                      VEC_ANTIQUANT_CONFIG_2, APT_A2A_RS_AG, APT_COMM_MODE);
    } else if constexpr (!TPL_TRANS_B && ANTIQUANT_TYPE == TPL_PER_CHANNEL && !TPL_EXIST_BIAS && !BAIS_IS_FP32) {
        INVOKE_MC2_WEIGHT_QUANT_ADAPTIVE_SPLIT_KERNEL(false, HAS_ANTIQUANT_OFFSET, Mc2QuantType::PER_CHANNEL,
                                                      DTYPE_BIAS, VEC_ANTIQUANT_CONFIG_3, APT_A2A_RS_AG, APT_COMM_MODE);
    } else if constexpr (TPL_TRANS_B && ANTIQUANT_TYPE == TPL_PER_TENSOR && !TPL_EXIST_BIAS && !BAIS_IS_FP32) {
        INVOKE_MC2_WEIGHT_QUANT_ADAPTIVE_SPLIT_KERNEL(true, HAS_ANTIQUANT_OFFSET, Mc2QuantType::PER_TENSOR, DTYPE_BIAS,
                                                      VEC_ANTIQUANT_CONFIG_0, APT_A2A_RS_AG, APT_COMM_MODE);
    } else if constexpr (!TPL_TRANS_B && ANTIQUANT_TYPE == TPL_PER_TENSOR && !TPL_EXIST_BIAS && !BAIS_IS_FP32) {
        INVOKE_MC2_WEIGHT_QUANT_ADAPTIVE_SPLIT_KERNEL(false, HAS_ANTIQUANT_OFFSET, Mc2QuantType::PER_TENSOR, DTYPE_BIAS,
                                                      VEC_ANTIQUANT_CONFIG_3, APT_A2A_RS_AG, APT_COMM_MODE);
    }
#if defined(ORIG_DTYPE_X1) && defined(DT_BF16) && (ORIG_DTYPE_X1 == DT_BF16)
#undef DTYPE_BIAS
#define DTYPE_BIAS float
    if constexpr (TPL_TRANS_B && ANTIQUANT_TYPE == TPL_PER_CHANNEL && TPL_EXIST_BIAS && BAIS_IS_FP32) {
        INVOKE_MC2_WEIGHT_QUANT_ADAPTIVE_SPLIT_KERNEL(true, HAS_ANTIQUANT_OFFSET, Mc2QuantType::PER_CHANNEL, float,
                                                      VEC_ANTIQUANT_CONFIG_0, APT_A2A_RS_AG, APT_COMM_MODE);
    } else if constexpr (!TPL_TRANS_B && ANTIQUANT_TYPE == TPL_PER_CHANNEL && TPL_EXIST_BIAS && BAIS_IS_FP32) {
        INVOKE_MC2_WEIGHT_QUANT_ADAPTIVE_SPLIT_KERNEL(false, HAS_ANTIQUANT_OFFSET, Mc2QuantType::PER_CHANNEL, float,
                                                      VEC_ANTIQUANT_CONFIG_3, APT_A2A_RS_AG, APT_COMM_MODE);
    } else if constexpr (TPL_TRANS_B && ANTIQUANT_TYPE == TPL_PER_TENSOR && TPL_EXIST_BIAS && BAIS_IS_FP32) {
        INVOKE_MC2_WEIGHT_QUANT_ADAPTIVE_SPLIT_KERNEL(true, HAS_ANTIQUANT_OFFSET, Mc2QuantType::PER_TENSOR, float,
                                                      VEC_ANTIQUANT_CONFIG_0, APT_A2A_RS_AG, APT_COMM_MODE);
    } else if constexpr (!TPL_TRANS_B && ANTIQUANT_TYPE == TPL_PER_TENSOR && TPL_EXIST_BIAS && BAIS_IS_FP32) {
        INVOKE_MC2_WEIGHT_QUANT_ADAPTIVE_SPLIT_KERNEL(false, HAS_ANTIQUANT_OFFSET, Mc2QuantType::PER_TENSOR, float,
                                                      VEC_ANTIQUANT_CONFIG_3, APT_A2A_RS_AG, APT_COMM_MODE);
    }
#endif
#endif
}

template <uint8_t APT_MM_TYPE, TPL_APT_PARAMS_COMM, TPL_APT_A2A_RS_AG, TPL_APT_COMM_MODE, TPL_APT_PARAMS_FP_MM,
          TPL_APT_PARAMS_QUANT_MM, TPL_APT_PARAMS_WEIGHT_QUANT_MM>
__global__ __aicore__ void matmul_all_reduce(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR addGM,
                                             GM_ADDR antiquantScaleGM, GM_ADDR antiquantOffsetGM, GM_ADDR dequantGM,
                                             GM_ADDR pertokenGM, GM_ADDR commQuantScale1GM, GM_ADDR commQuantScale2GM,
                                             GM_ADDR cGM, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(Mc2Tiling::QuantMatmulAllReduceTilingDataA5); // 按当前最大的结构体申请内存
    if (workspaceGM == nullptr) {
        return;
    }

    GM_ADDR userWS = GetUserWorkspace(workspaceGM);
    if (userWS == nullptr) {
        return;
    }

    TPipe tPipe;

    if constexpr (APT_MM_TYPE == MMTYPE_FP_NULL_TENSOR || APT_MM_TYPE == MMTYPE_WEIQUANT_NULL_TENSOR) {
        INVOKE_MC2_EMPTY_TENSOR_OP_IMPL(APT_COMM_MODE);
    } else if constexpr (APT_MM_TYPE == MMTYPE_FP_MM) {
        fp_matmul_all_reduce<APT_PARAMS_FP_MM, APT_A2A_RS_AG, APT_COMM_MODE>(
            aGM, bGM, biasGM, addGM, antiquantScaleGM, antiquantOffsetGM, dequantGM, pertokenGM, commQuantScale1GM,
            commQuantScale2GM, cGM, workspaceGM, tilingGM, tPipe, userWS);
    } else if constexpr (APT_MM_TYPE == MMTYPE_QUANT_MM) {
        quant_matmul_all_reduce<APT_PARAMS_COMM, APT_PARAMS_QUANT_MM, APT_A2A_RS_AG, APT_COMM_MODE>(
            aGM, bGM, biasGM, addGM, antiquantScaleGM, antiquantOffsetGM, dequantGM, pertokenGM, commQuantScale1GM,
            commQuantScale2GM, cGM, workspaceGM, tilingGM, tPipe, userWS);
    } else if constexpr (APT_MM_TYPE == MMTYPE_WEIGHT_QUANT_MM) {
        weight_quant_matmul_allreduce<APT_PARAMS_COMM, APT_PARAMS_WEIGHT_QUANT_MM, APT_A2A_RS_AG, APT_COMM_MODE>(
            aGM, bGM, biasGM, addGM, antiquantScaleGM, antiquantOffsetGM, dequantGM, pertokenGM, commQuantScale1GM,
            commQuantScale2GM, cGM, workspaceGM, tilingGM, tPipe, userWS);
    }
}
