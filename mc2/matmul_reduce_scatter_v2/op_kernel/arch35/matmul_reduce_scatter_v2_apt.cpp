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
 * \file matmul_reduce_scatter_v2_apt.cpp
 * \brief
 */

#include "../matmul_reduce_scatter_v2_apt_tiling_key.h"
#include "lib/matmul_intf.h"
#include "../common_def.h"

// HiF4 uses the same MatmulWithScaleMx APACE block, with native scale metadata.
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 9201) && defined(DT_HIFLOAT4) && \
    ((ORIG_DTYPE_X1 == ORIG_DTYPE_X2) && (ORIG_DTYPE_X1 == DT_HIFLOAT4))
#define MMRS_APACE_HIF4_DTYPE 1
#else
#define MMRS_APACE_HIF4_DTYPE 0
#endif

// MX APACE 仅在 960 编译，且排除 HiFloat8 等不支持的输入类型。
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 9201) && \
    ((((ORIG_DTYPE_X1 == DT_FLOAT8_E4M3FN) || (ORIG_DTYPE_X1 == DT_FLOAT8_E5M2)) && \
      ((ORIG_DTYPE_X2 == DT_FLOAT8_E4M3FN) || (ORIG_DTYPE_X2 == DT_FLOAT8_E5M2))) || \
     ((ORIG_DTYPE_X1 == ORIG_DTYPE_X2) && (ORIG_DTYPE_X1 == DT_FLOAT4_E2M1)) || MMRS_APACE_HIF4_DTYPE)
#define MMRS_APACE_MX_DTYPE 1
#else
#define MMRS_APACE_MX_DTYPE 0
#endif

// 非量化 APACE 仅在 960 编译。
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 9201) && (ORIG_DTYPE_X1 == ORIG_DTYPE_X2) && \
    ((ORIG_DTYPE_X1 == DT_FLOAT16) || (ORIG_DTYPE_X1 == DT_BF16))
#define MMRS_APACE_FP16_BF16_DTYPE 1
#else
#define MMRS_APACE_FP16_BF16_DTYPE 0
#endif

#if MMRS_APACE_FP16_BF16_DTYPE || MMRS_APACE_MX_DTYPE
#include "../../../common/op_kernel/apace/kernel/fusions/matmul_reduce_scatter_v2/matmul_reduce_scatter_v2_hcomm_impl.h"
#endif

#if ((ORIG_DTYPE_X1 == ORIG_DTYPE_X2) && ((ORIG_DTYPE_X1 == DT_FLOAT16) || (ORIG_DTYPE_X1 == DT_BF16)))
#include "matmul_a2a_vec_reduce_fp16_bf16.h" // 新实现：All2All + Vec Reduce
#include "matmul_reduce_scatter_fp16_bf16.h" // 旧实现：原生 ReduceScatter
using namespace Mc2Tiling;
#endif
#if (((ORIG_DTYPE_X1 == ORIG_DTYPE_X2) && (ORIG_DTYPE_X1 == DT_HIFLOAT8)) || \
     (((ORIG_DTYPE_X1 == DT_FLOAT8_E4M3FN) || (ORIG_DTYPE_X1 == DT_FLOAT8_E5M2)) && \
      ((ORIG_DTYPE_X2 == DT_FLOAT8_E4M3FN) || (ORIG_DTYPE_X2 == DT_FLOAT8_E5M2))) || \
     ((ORIG_DTYPE_X1 == ORIG_DTYPE_X2) && (ORIG_DTYPE_X1 == DT_FLOAT4_E2M1)))
#include "quant_bmm_a2a_vec_reduce_fp8_hif8.h" // 新实现：All2All + Vec Reduce
#include "quant_bmm_reduce_scatter_fp8_hif8.h" // 旧实现：原生 ReduceScatter
using namespace Mc2Tiling;
#endif

using namespace MatmulReduceScatterV2Impl;

#define INVOKE_MMREDUCESCATTER_FP16_BF16_OP_IMPL(templateClass) \
    do { \
        using AType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, A_DTYPE, false>; \
        using CType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, C_DTYPE>; \
        templateClass<AType, BType, BiasType, CType, TPL_COMM_MODE> op; \
        op.Init(aGM, bGM, biasGM, cGM, (GM_ADDR)context, workspaceGM, &tilingData, &pipe); \
        op.Process(); \
    } while (0)

#define INVOKE_QUANT_BATCHMM_REDUCE_SCATTER_OP_IMPL(templateClass, ...) \
    do { \
        if (tilingData.debugMode != static_cast<uint8_t>(MC2_DEBUG_ONLY_AICPU)) { \
            using mmClass = MatMulASWKernel<DTYPE_X1, DTYPE_X2, float, DTYPE_BIAS, DTYPE_Y, CubeFormat::ND, \
                                            CubeFormat::ND, CubeFormat::ND, __VA_ARGS__>; \
            templateClass<DTYPE_X1, DTYPE_X2, DTYPE_Y, float, mmClass, false, __VA_ARGS__, TPL_COMM_MODE> op; \
            op.Init(aGM, bGM, biasGM, x1ScaleGM, x2ScaleGM, cGM, (GM_ADDR)context, workspaceGM, &tilingData, &pipe); \
            op.Process(); \
        } \
    } while (0)

#define INVOKE_QUANT_BATCHMM_PERTENSOR_MXFP8_REDUCE_SCATTER_OP_IMPL(templateClass, ...) \
    do { \
        if (tilingData.debugMode != static_cast<uint8_t>(MC2_DEBUG_ONLY_AICPU)) { \
            using mmClass = MatMulASWKernel<DTYPE_X1, DTYPE_X2, fp8_e8m0_t, DTYPE_BIAS, DTYPE_Y, CubeFormat::ND, \
                                            CubeFormat::ND, CubeFormat::ND, __VA_ARGS__>; \
            templateClass<DTYPE_X1, DTYPE_X2, DTYPE_Y, fp8_e8m0_t, mmClass, false, __VA_ARGS__, TPL_COMM_MODE> op; \
            op.Init(aGM, bGM, biasGM, x1ScaleGM, x2ScaleGM, cGM, (GM_ADDR)context, workspaceGM, &tilingData, &pipe); \
            op.Process(); \
        } \
    } while (0)

#define INVOKE_QUANT_BATCHMM_PERBLOCK_REDUCE_SCATTER_OP_IMPL(templateClass, ...) \
    do { \
        if (tilingData.debugMode != static_cast<uint8_t>(MC2_DEBUG_ONLY_AICPU)) { \
            using mmClass = Mc2QuantBatchMatmulV3::MatMulPerBlockASWNonContiguous< \
                DTYPE_X1, DTYPE_X2, DTYPE_BIAS, DTYPE_Y, CubeFormat::ND, CubeFormat::ND, CubeFormat::ND, __VA_ARGS__>; \
            templateClass<DTYPE_X1, DTYPE_X2, DTYPE_Y, float, mmClass, true, __VA_ARGS__, TPL_COMM_MODE> op; \
            op.Init(aGM, bGM, biasGM, x1ScaleGM, x2ScaleGM, cGM, (GM_ADDR)context, workspaceGM, &tilingData, &pipe); \
            op.Process(); \
        } \
    } while (0)

template <bool TPL_ISPERBLOCK, bool TPL_TRANSA, bool TPL_TRANSB, uint8_t TPL_INPUT, uint8_t TPL_OUTPUTDTYPE,
          uint8_t TPL_SCALETYPE, uint8_t TPL_COMMALG, uint8_t TPL_COMM_MODE, bool TPL_USING_APACE>
__global__ __aicore__ void matmul_reduce_scatter_v2(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR x1ScaleGM,
                                                    GM_ADDR x2ScaleGM, GM_ADDR quantScaleGM, GM_ADDR cGM,
                                                    GM_ADDR amaxOutGM, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    // david算子模板
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 9201)
    // A6(ascend960) AIC:AIV为1:1
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
#else
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
#endif
    AscendC::TPipe pipe;
    auto context = AscendC::GetHcclContext<0>();
#if ((ORIG_DTYPE_X1 == ORIG_DTYPE_X2) && ((ORIG_DTYPE_X1 == DT_FLOAT16) || (ORIG_DTYPE_X1 == DT_BF16)))
    REGISTER_TILING_DEFAULT(Mc2Tiling::MatmulReduceScatterV2TilingData);
#elif MMRS_APACE_HIF4_DTYPE
    // HiFP4 仅走 APACE，使用其实际输出的 TilingData 作为默认结构。
    REGISTER_TILING_DEFAULT(hcommMatmulReduceScatterV2TilingData);
#elif (((ORIG_DTYPE_X1 == ORIG_DTYPE_X2) && (ORIG_DTYPE_X1 == DT_HIFLOAT8)) || \
       (((ORIG_DTYPE_X1 == DT_FLOAT8_E4M3FN) || (ORIG_DTYPE_X1 == DT_FLOAT8_E5M2)) && \
        ((ORIG_DTYPE_X2 == DT_FLOAT8_E4M3FN) || (ORIG_DTYPE_X2 == DT_FLOAT8_E5M2))) || \
       ((ORIG_DTYPE_X1 == ORIG_DTYPE_X2) && (ORIG_DTYPE_X1 == DT_FLOAT4_E2M1)))
    // MXFP4/MXFP8 保留旧路径的默认结构，APACE 再显式解析自己的结构。
    REGISTER_TILING_DEFAULT(Mc2Tiling::QuantBatchMatmulV3ReduceScatterTilingData);
#endif
#if MMRS_APACE_FP16_BF16_DTYPE || MMRS_APACE_MX_DTYPE
    if constexpr (TPL_USING_APACE) {
#if MMRS_APACE_FP16_BF16_DTYPE
        using ApaceTilingData = Mc2Tiling::MatmulReduceScatterV2TilingData;
        GET_TILING_DATA_WITH_STRUCT(Mc2Tiling::MatmulReduceScatterV2TilingData, tilingData, tilingGM);
#else
        using ApaceTilingData = hcommMatmulReduceScatterV2TilingData;
        GET_TILING_DATA_WITH_STRUCT(hcommMatmulReduceScatterV2TilingData, tilingData, tilingGM);
#endif
        if ASCEND_IS_AIC {
            constexpr AscendC::HcclServerType hcclServerType = TPL_COMM_MODE == TPL_CCU_COMM_MODE ?
                                                                   AscendC::HcclServerType::HCCL_SERVER_TYPE_CCU :
                                                                   AscendC::HcclServerType::HCCL_SERVER_TYPE_AICPU;
            Apace::MatmulReduceScatterV2HcommImpl<DTYPE_X1, DTYPE_X2, DTYPE_Y, TPL_TRANSB, ApaceTilingData,
                                                  hcclServerType>
                op(&tilingData);
            op.Init(aGM, bGM, biasGM, x1ScaleGM, x2ScaleGM, cGM, (GM_ADDR)context, workspaceGM);
            op.Run();
        }
    } else
#endif
    {
#if !MMRS_APACE_HIF4_DTYPE
        GET_TILING_DATA(tilingData, tilingGM);
#endif
#if ((ORIG_DTYPE_X1 == ORIG_DTYPE_X2) && ((ORIG_DTYPE_X1 == DT_FLOAT16) || (ORIG_DTYPE_X1 == DT_BF16)))
        // bf16/fp16 场景
        using BiasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, typename BiasType<BIAS_DTYPE>::type>;
        if constexpr (!TPL_ISPERBLOCK && TPL_INPUT == INPUT_TYPE_IS_FP16_BF16 &&
                      TPL_OUTPUTDTYPE == OUTPUT_TYPE_IS_FP8 && TPL_SCALETYPE == TPL_X1_X2_DTYPE_IS_OTHER) {
            using BType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, B_DTYPE, TPL_TRANSB>;
            if constexpr (TPL_COMMALG == TPL_CCU_ALL2ALL_VEC_REDUCE) {
                INVOKE_MMREDUCESCATTER_FP16_BF16_OP_IMPL(MatmulA2AVecReduceFP16BF16);
            } else if constexpr (TPL_COMMALG == TPL_CCU_REDUCESUM_PEER_ONLY) {
                INVOKE_MMREDUCESCATTER_FP16_BF16_OP_IMPL(MatmulReduceScatterFP16BF16PeerOnly);
            } else {
                INVOKE_MMREDUCESCATTER_FP16_BF16_OP_IMPL(MatmulReduceScatterFP16BF16);
            }
        }
#elif (((ORIG_DTYPE_X1 == ORIG_DTYPE_X2) && ((ORIG_DTYPE_X1 == DT_HIFLOAT8))) || \
       (((ORIG_DTYPE_X1 == DT_FLOAT8_E4M3FN) || (ORIG_DTYPE_X1 == DT_FLOAT8_E5M2)) && \
        ((ORIG_DTYPE_X2 == DT_FLOAT8_E4M3FN) || (ORIG_DTYPE_X2 == DT_FLOAT8_E5M2))) || \
       ((ORIG_DTYPE_X1 == ORIG_DTYPE_X2) && (ORIG_DTYPE_X1 == DT_FLOAT4_E2M1)))
#if (ORIG_DTYPE_X1 != DT_HIFLOAT8)
        if constexpr (TPL_SCALETYPE == TPL_X1_X2_DTYPE_IS_FP8E8M0) {
            if constexpr (TPL_COMMALG == TPL_CCU_ALL2ALL_VEC_REDUCE) {
                INVOKE_QUANT_BATCHMM_PERTENSOR_MXFP8_REDUCE_SCATTER_OP_IMPL(QuantBmmA2AVecReduceFP8HiF8, false,
                                                                            TPL_TRANSB);
            } else if constexpr (TPL_COMMALG == TPL_CCU_REDUCESUM_PEER_ONLY) {
                INVOKE_QUANT_BATCHMM_PERTENSOR_MXFP8_REDUCE_SCATTER_OP_IMPL(QuantBMMReduceScatterPeerOnly, false,
                                                                            TPL_TRANSB);
            } else {
                INVOKE_QUANT_BATCHMM_PERTENSOR_MXFP8_REDUCE_SCATTER_OP_IMPL(QuantBMMReduceScatter, false, TPL_TRANSB);
            }
        }
#endif
        if constexpr (!TPL_ISPERBLOCK && TPL_INPUT == INPUT_TYPE_IS_FP8 && TPL_SCALETYPE == TPL_X1_X2_DTYPE_IS_OTHER) {
            if constexpr (TPL_COMMALG == TPL_CCU_ALL2ALL_VEC_REDUCE) {
                INVOKE_QUANT_BATCHMM_REDUCE_SCATTER_OP_IMPL(QuantBmmA2AVecReduceFP8HiF8, TPL_TRANSA, TPL_TRANSB);
            } else if constexpr (TPL_COMMALG == TPL_CCU_REDUCESUM_PEER_ONLY) {
                INVOKE_QUANT_BATCHMM_REDUCE_SCATTER_OP_IMPL(QuantBMMReduceScatterPeerOnly, TPL_TRANSA, TPL_TRANSB);
            } else {
                INVOKE_QUANT_BATCHMM_REDUCE_SCATTER_OP_IMPL(QuantBMMReduceScatter, TPL_TRANSA, TPL_TRANSB);
            }
        } else if constexpr (TPL_ISPERBLOCK && TPL_INPUT == INPUT_TYPE_IS_FP8 &&
                             TPL_SCALETYPE == TPL_X1_X2_DTYPE_IS_OTHER) {
            if constexpr (TPL_COMMALG == TPL_CCU_ALL2ALL_VEC_REDUCE) {
                INVOKE_QUANT_BATCHMM_PERBLOCK_REDUCE_SCATTER_OP_IMPL(QuantBmmA2AVecReduceFP8HiF8, TPL_TRANSA,
                                                                     TPL_TRANSB);
            } else {
                INVOKE_QUANT_BATCHMM_PERBLOCK_REDUCE_SCATTER_OP_IMPL(QuantBMMReduceScatter, TPL_TRANSA, TPL_TRANSB);
            }
        }
#endif
    }
}

#undef MMRS_APACE_MX_DTYPE
#undef MMRS_APACE_HIF4_DTYPE
#undef MMRS_APACE_FP16_BF16_DTYPE
