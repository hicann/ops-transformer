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
 * \file all_gather_matmul_v2_apt.cpp
 * \brief
 */

#include "lib/matmul_intf.h"
#include "../agm_v2_common.h"
#include "../all_gather_matmul_v2_apt_tiling_key.h"

#if ((ORIG_DTYPE_X1 == ORIG_DTYPE_X2) && ((ORIG_DTYPE_X1 == DT_FLOAT16) || (ORIG_DTYPE_X1 == DT_BF16)))
#include "all_gather_matmul_fp16_bf16.h"
#endif
#if (((ORIG_DTYPE_X1 == ORIG_DTYPE_X2) && (ORIG_DTYPE_X1 == DT_HIFLOAT8)) ||                                           \
     (((ORIG_DTYPE_X1 == DT_FLOAT8_E4M3FN) || (ORIG_DTYPE_X1 == DT_FLOAT8_E5M2)) &&                                    \
      ((ORIG_DTYPE_X2 == DT_FLOAT8_E4M3FN) || (ORIG_DTYPE_X2 == DT_FLOAT8_E5M2))) ||                                   \
     ((ORIG_DTYPE_X1 == DT_FLOAT4_E2M1) && (ORIG_DTYPE_X2 == DT_FLOAT4_E2M1)))
#include "all_gather_quant_bmm.h"
#include "all_gather_quant_bmm_perblock.h"
#endif

using namespace Mc2Tiling;
using namespace AllGatherMatmulImpl;

#define INVOKE_ALLGATHERMM_FP16_BF16_V2_OP_IMPL(templateClass, isTransB, hcclCommType, ...)                            \
    do {                                                                                                               \
        using aType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, A_DTYPE, false>;                              \
        using bType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, B_DTYPE, isTransB>;                           \
        using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, typename BiasType<BIAS_DTYPE>::type>;      \
        using cType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, C_DTYPE>;                                     \
        REGISTER_TILING_DEFAULT(Mc2Tiling::AllGatherMatmulTilingDataV2);                                               \
        auto tiling = (__gm__ Mc2Tiling::AllGatherMatmulTilingDataV2 *)tilingGM;                                       \
        __gm__ void *mc2InitTiling = (__gm__ void *)(&(tiling->mc2InitTiling));                                        \
        __gm__ void *mc2CcTiling = (__gm__ void *)(&(tiling->mc2CcTiling));                                            \
        GET_TILING_DATA(tilingData, tilingGM);                                                                         \
        templateClass<aType, bType, biasType, cType, hcclCommType> op;                                                 \
        op.Init(aGM, bGM, biasGM, cGM, (__gm__ uint8_t *)context, workspaceGM, gatherOut, &tilingData, mc2InitTiling,  \
                mc2CcTiling, &pipe);                                                                                   \
        op.Process();                                                                                                  \
    } while (0)

#define INVOKE_ALL_GATHER_QUANT_BATCHMATMUL_OP_IMPL(templateClass, isMxFp4, hcclCommType, ...)                         \
    do {                                                                                                               \
        REGISTER_TILING_DEFAULT(Mc2Tiling::AllGatherMatmulTilingDataFp8);                                              \
        auto tiling = (__gm__ Mc2Tiling::AllGatherMatmulTilingDataFp8 *)tilingGM;                                      \
        __gm__ void *mc2InitTiling = (__gm__ void *)(&(tiling->mc2InitTiling));                                        \
        __gm__ void *mc2CcTiling = (__gm__ void *)(&(tiling->mc2CcTiling));                                            \
        GET_TILING_DATA(tilingData, tilingGM);                                                                         \
        templateClass<DTYPE_X1, DTYPE_X2, DTYPE_BIAS, uint64_t, DTYPE_Y, __VA_ARGS__, isMxFp4, hcclCommType> op;       \
        op.Init(aGM, bGM, scaleInv1, scaleInv2, biasGM, scale, cGM, gatherOut, workspaceGM, (__gm__ uint8_t *)context, \
                &tilingData, mc2InitTiling, mc2CcTiling, &pipe);                                                       \
        op.Process();                                                                                                  \
    } while (0)

#define INVOKE_ALL_GATHER_QUANT_BATCHMATMUL_MX_OP_IMPL(templateClass, isMxFp4, hcclCommType, ...)                      \
    do {                                                                                                               \
        REGISTER_TILING_DEFAULT(Mc2Tiling::AllGatherMatmulTilingDataFp8);                                              \
        auto tiling = (__gm__ Mc2Tiling::AllGatherMatmulTilingDataFp8 *)tilingGM;                                      \
        __gm__ void *mc2InitTiling = (__gm__ void *)(&(tiling->mc2InitTiling));                                        \
        __gm__ void *mc2CcTiling = (__gm__ void *)(&(tiling->mc2CcTiling));                                            \
        GET_TILING_DATA(tilingData, tilingGM);                                                                         \
        templateClass<DTYPE_X1, DTYPE_X2, DTYPE_BIAS, fp8_e8m0_t, DTYPE_Y, __VA_ARGS__, isMxFp4, hcclCommType> op;     \
        op.Init(aGM, bGM, scaleInv1, scaleInv2, biasGM, scale, cGM, gatherOut, workspaceGM, (__gm__ uint8_t *)context, \
                &tilingData, mc2InitTiling, mc2CcTiling, &pipe);                                                       \
        op.Process();                                                                                                  \
    } while (0)

template <TPL_PARAMS_COMM, TPL_QUANT_BMM_PARAMS_COMM, TPL_HCCL_COMM_MODE>
__global__ __aicore__ void all_gather_matmul_v2(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR scaleInv1,
                                                GM_ADDR scaleInv2, GM_ADDR scale, GM_ADDR cGM, GM_ADDR gatherOut,
                                                GM_ADDR amax, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    TPipe pipe;
    __gm__ HcclCombinOpParam *context = (__gm__ HcclCombinOpParam *)(GetHcclContext<0>());

#if (((ORIG_DTYPE_X1 == ORIG_DTYPE_X2) && (ORIG_DTYPE_X1 == DT_HIFLOAT8)) ||                                           \
     (((ORIG_DTYPE_X1 == DT_FLOAT8_E4M3FN) || (ORIG_DTYPE_X1 == DT_FLOAT8_E5M2)) &&                                    \
      ((ORIG_DTYPE_X2 == DT_FLOAT8_E4M3FN) || (ORIG_DTYPE_X2 == DT_FLOAT8_E5M2))) ||                                   \
     ((ORIG_DTYPE_X1 == DT_FLOAT4_E2M1) && (ORIG_DTYPE_X2 == DT_FLOAT4_E2M1)))
// MX
#if (ORIG_DTYPE_X1 != DT_HIFLOAT8)
    if constexpr (SCALETYPE == SCALE_TYPE_IS_MX && !INPUT_IS_BF16FP16 && QUANTMMMODE == TPL_DEFAULT_MODE) {
        if (((ORIG_DTYPE_X1 == DT_FLOAT8_E4M3FN) || (ORIG_DTYPE_X1 == DT_FLOAT8_E5M2)) &&
            ((ORIG_DTYPE_X2 == DT_FLOAT8_E4M3FN) || (ORIG_DTYPE_X2 == DT_FLOAT8_E5M2))) {
            INVOKE_ALL_GATHER_QUANT_BATCHMATMUL_MX_OP_IMPL(AllGatherQuantBmm, false, HCCL_COMM_MODE, false, TRANS_B);
        } else if ((ORIG_DTYPE_X1 == DT_FLOAT4_E2M1) && (ORIG_DTYPE_X2 == DT_FLOAT4_E2M1)) {
            INVOKE_ALL_GATHER_QUANT_BATCHMATMUL_MX_OP_IMPL(AllGatherQuantBmm, true, HCCL_COMM_MODE, false, TRANS_B);
        }
    }
#endif
    if constexpr (SCALETYPE == SCALE_TYPE_NOT_IS_MX && !INPUT_IS_BF16FP16 && QUANTMMMODE == TPL_DEFAULT_MODE) {
        INVOKE_ALL_GATHER_QUANT_BATCHMATMUL_OP_IMPL(AllGatherQuantBmm, false, HCCL_COMM_MODE, false, TRANS_B);
    } else if constexpr (SCALETYPE == SCALE_TYPE_NOT_IS_MX && !INPUT_IS_BF16FP16 && QUANTMMMODE == TPL_PERBLOCK_MODE) {
        INVOKE_ALL_GATHER_QUANT_BATCHMATMUL_PERBLOCK_OP_IMPL(Mc2QuantBatchMatmulV3::MatMulPerBlockASWNonContiguous,
                                                             Mc2CoreType::ON_CUBE, HCCL_COMM_MODE, false, TRANS_B);
    }
#elif ((ORIG_DTYPE_X1 == ORIG_DTYPE_X2) && ((ORIG_DTYPE_X1 == DT_FLOAT16) || (ORIG_DTYPE_X1 == DT_BF16)))
    if constexpr (SCALETYPE == SCALE_TYPE_NOT_IS_MX && INPUT_IS_BF16FP16 &&
                  OUTPUTDTYPE == OUTPUT_TYPE_IS_FP16_BF16) { // full mesh+ no nd2nz +biasNoNeedCast
        INVOKE_ALLGATHERMM_FP16_BF16_V2_OP_IMPL(AllGatherMatmulFP16BF16, TRANS_B, HCCL_COMM_MODE);
    }
#endif
}
