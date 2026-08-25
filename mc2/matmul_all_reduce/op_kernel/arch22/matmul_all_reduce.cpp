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
 * \file matmul_all_reduce.cpp
 * \brief arch22/A2/A3/910b/910_93
 */

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "lib/matmul_intf.h"
#include "../matmul_all_reduce_common.h"
#include "../matmul_all_reduce_tiling_key.h"

#ifdef MC2_QUANT
#include "../quant_matmul_all_reduce_tiling_data.h"
#include "matmul_all_reduce_quant.h"
#include "matmul_all_reduce_quant_pertoken_comm_int8_arch22.h"
#ifdef MC2_QUANT_FP16
#include "matmul_all_reduce_quant_fp16_comm_int8.h"
#else
#include "matmul_all_reduce_quant_bf16_comm_int8.h"
#endif // MC2_QUANT_FP16
#else  // 全量化&伪量化
#include "matmul_all_reduce_empty_tensor_k_general.h"
#ifdef MC2_WEIGHT_QUANT
#include "matmul_all_reduce_weight_quant.h"
#include "../weight_quant_matmul_all_reduce_tiling_data.h"
#else // 非量化
#include "matmul_all_reduce_910_general.h"
#include "../unquant_matmul_all_reduce_tiling_data.h"
#endif // MC2_WEIGHT_QUANT
#endif // MC2_QUANT

#if ((ORIG_DTYPE_X1 == DT_INT8) && (ORIG_DTYPE_Y == DT_BF16))
#define INVOKE_QUANT_BATCH_MATMUL_DEQUANT_BF16_IMPL_COMM_INT8(templateClass, opTile, opTail, ...)                      \
    do {                                                                                                               \
        GET_TILING_DATA_MEMBER(Mc2Tiling::QuantMatmulAllReduceTilingData, msg, msg, tilingGM);                         \
        if (msg.debugMode != static_cast<uint8_t>(DebugMode::MC2_DEBUG_ONLY_AICPU)) {                                  \
            GET_TILING_DATA_WITH_STRUCT(Mc2Tiling::QuantMatmulAllReduceTilingData, tilingData, tilingGM);              \
            templateClass<DTYPE_X1, DTYPE_X2, FORMAT_X1, FORMAT_X2, DTYPE_Y, DTYPE_Y, int8_t, __VA_ARGS__> matmul;     \
            const Mc2Tiling::QuantMatmulAllReduceTilingData *QuantMatmulAllReduceTiling = &tilingData;                 \
            const Mc2QuantBatchMatmulV3TilingData *qBmmV3TilingData = &(QuantMatmulAllReduceTiling->tilematmulTiling); \
            const AscendC::tiling::TCubeTiling *mmTilingTile = &(qBmmV3TilingData->matmulTiling);                      \
            const Mc2QuantBatchMatmulV3TilingData *qBmmV3TilingDataTail =                                              \
                &(QuantMatmulAllReduceTiling->tailmatmulTiling);                                                       \
            const AscendC::tiling::TCubeTiling *mmTilingTail = &(qBmmV3TilingDataTail->matmulTiling);                  \
            REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), opTile.mm, mmTilingTile, opTail.mm, mmTilingTail);         \
            matmul.Init(aGM, bGM, biasGM, addGM, dequantGM, commQuantScale1GM, commQuantScale2GM, cGM, userWS,         \
                        &tilingData, &tPipe);                                                                          \
            matmul.Process(opTile, opTail);                                                                            \
            tPipe.Destroy();                                                                                           \
        }                                                                                                              \
    } while (0)
#endif // (ORIG_DTYPE_X1 == DT_INT8) && (ORIG_DTYPE_Y == DT_BF16)

#if ((ORIG_DTYPE_X1 == DT_INT8) && (ORIG_DTYPE_Y == DT_FLOAT16))
#define INVOKE_QUANT_BMM_DEQUANT_FP16_IMPL_COMM_INT8(templateClass, ...)                                               \
    do {                                                                                                               \
        GET_TILING_DATA_WITH_STRUCT(Mc2Tiling::QuantMatmulAllReduceTilingData, tilingData, tilingGM);                  \
        templateClass<DTYPE_X1, DTYPE_X2, int32_t, DTYPE_Y, int8_t, __VA_ARGS__> op;                                   \
        op.Init(aGM, bGM, dequantGM, biasGM, addGM, cGM, workspaceGM, &tilingData, &tPipe);                            \
        op.InitScale(commQuantScale1GM, commQuantScale2GM);                                                            \
        op.Process();                                                                                                  \
    } while (0)
#endif

#if ((ORIG_DTYPE_X1 == DT_INT8) && (ORIG_DTYPE_X2 == DT_INT8))
#define INVOKE_QUANT_BATCH_MATMUL_DEQUANT_PERTOKEN_COMM_INT8_IMPL(templateClass, scaleType, opTile, opTail, ...)       \
    do {                                                                                                               \
        GET_TILING_DATA_MEMBER(Mc2Tiling::QuantMatmulAllReduceTilingData, msg, msg, tilingGM);                         \
        if (msg.debugMode != static_cast<uint8_t>(DebugMode::MC2_DEBUG_ONLY_AICPU)) {                                  \
            GET_TILING_DATA_WITH_STRUCT(Mc2Tiling::QuantMatmulAllReduceTilingData, tilingData, tilingGM);              \
            templateClass<DTYPE_X1, DTYPE_X2, FORMAT_X1, FORMAT_X2, scaleType, DTYPE_Y, int8_t, __VA_ARGS__> matmul;   \
            const Mc2Tiling::QuantMatmulAllReduceTilingData *QuantMatmulAllReduceTiling = &tilingData;                 \
            const Mc2QuantBatchMatmulV3TilingData *qBmmV3TilingData = &(QuantMatmulAllReduceTiling->tilematmulTiling); \
            const AscendC::tiling::TCubeTiling *mmTilingTile = &(qBmmV3TilingData->matmulTiling);                      \
            const Mc2QuantBatchMatmulV3TilingData *qBmmV3TilingDataTail =                                              \
                &(QuantMatmulAllReduceTiling->tailmatmulTiling);                                                       \
            const AscendC::tiling::TCubeTiling *mmTilingTail = &(qBmmV3TilingDataTail->matmulTiling);                  \
            REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), opTile.mm, mmTilingTile, opTail.mm, mmTilingTail);         \
            matmul.Init(aGM, bGM, biasGM, addGM, dequantGM, pertokenGM, commQuantScale1GM, commQuantScale2GM, cGM,     \
                        userWS, &tilingData, &tPipe);                                                                  \
            matmul.Process(opTile, opTail);                                                                            \
            tPipe.Destroy();                                                                                           \
        }                                                                                                              \
    } while (0)
#endif

namespace MatmulAllReduceImpl {
}

using namespace AscendC;
using namespace Mc2Tiling;
using namespace MatmulAllReduceImpl;

template <int MM_TYPE, TPL_PARAMS_COMM, TPL_PARAMS_SHARE_MM, TPL_PARAMS_FP_MM>
__global__ __aicore__ void fp_matmul_all_reduce(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR addGM,
                                                GM_ADDR antiquantScaleGM, GM_ADDR antiquantOffsetGM, GM_ADDR dequantGM,
                                                GM_ADDR pertokenGM, GM_ADDR commQuantScale1GM,
                                                GM_ADDR commQuantScale2GM, GM_ADDR cGM, GM_ADDR workspaceGM,
                                                GM_ADDR tilingGM, TPipe &tPipe, GM_ADDR userWS)
{
#if defined(MC2_QUANT_FP16)
#elif defined(MC2_QUANT_BF16)
#elif defined(MC2_WEIGHT_QUANT)
#else
    if constexpr (MM_TYPE == MATMUL_ALLREDUCE_MM_TYPE_FP_MM_CUBE_ONLY) {
        INVOKE_MC2_910_OP_IMPL(Mc2MatmulBaseKernel, Mc2CoreType::ON_CUBE);
    } else if constexpr (MM_TYPE == MATMUL_ALLREDUCE_MM_TYPE_FP_MM && EMPTY_INPUT == MATMUL_ALLREDUCE_EMPTY_INPUT_T) {
        INVOKE_MC2_EMPTY_TENSOR_OP_IMPL();
    } else if constexpr (MM_TYPE == MATMUL_ALLREDUCE_MM_TYPE_FP_MM && MIXND2NZ == MAT_MUL_V3_MIXND2NZ_FALSE) {
        INVOKE_MC2_910_OP_IMPL(Mc2MatmulBaseKernel, Mc2CoreType::ON_CUBE_AND_VECTOR);
    } else if constexpr (MM_TYPE == MATMUL_ALLREDUCE_MM_TYPE_FP_MM && MIXND2NZ == MAT_MUL_V3_MIXND2NZ_TRUE) {
        INVOKE_MC2_910_OP_IMPL(Mc2MatmulBaseUnAlignedKernel, Mc2CoreType::ON_CUBE_AND_VECTOR);
    }
#endif
}

template <int MM_TYPE, TPL_PARAMS_COMM, TPL_PARAMS_SHARE_MM, TPL_PARAMS_QUANT_MM>
__global__ __aicore__ void quant_matmul_allreduce(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR addGM,
                                                  GM_ADDR antiquantScaleGM, GM_ADDR antiquantOffsetGM,
                                                  GM_ADDR dequantGM, GM_ADDR pertokenGM, GM_ADDR commQuantScale1GM,
                                                  GM_ADDR commQuantScale2GM, GM_ADDR cGM, GM_ADDR workspaceGM,
                                                  GM_ADDR tilingGM, TPipe &tPipe, GM_ADDR userWS)
{
#if defined(MC2_QUANT_FP16)
    if constexpr (INT8_COMM == MATMUL_ALLREDUCE_INT8_COMM_F && TRANS == QUANT_BATCH_MATMUL_V3_NOT_TRANS &&
                  PERTOKEN == QUANT_BATCH_MATMUl_V3_NOT_PERTOKEN) {
        // 0
        INVOKE_MC2_QUANT_910_OP_IMPL(BmmDequant, Mc2CoreType::ON_CUBE_AND_VECTOR, REG_NO_MM_OBJ, false, int32_t,
                                     uint64_t, DTYPE_Y, false, false);
    } else if constexpr (INT8_COMM == MATMUL_ALLREDUCE_INT8_COMM_F && TRANS == QUANT_BATCH_MATMUL_V3_B_TRANS &&
                         PERTOKEN == QUANT_BATCH_MATMUl_V3_NOT_PERTOKEN) {
        // 1
        INVOKE_MC2_QUANT_910_OP_IMPL(BmmDequant, Mc2CoreType::ON_CUBE_AND_VECTOR, REG_NO_MM_OBJ, false, int32_t,
                                     uint64_t, DTYPE_Y, false, true);
    } else if constexpr (INT8_COMM == MATMUL_ALLREDUCE_INT8_COMM_F && TRANS == QUANT_BATCH_MATMUL_V3_NOT_TRANS &&
                         PERTOKEN == QUANT_BATCH_MATMUl_V3_IS_PERTOKEN) {
        // 16
        INVOKE_MC2_QUANT_910_OP_IMPL(BmmDequantPertoken, Mc2CoreType::ON_VECTOR, REG_MM_OBJ, true, float, DTYPE_Y,
                                     false, false);
    } else if constexpr (INT8_COMM == MATMUL_ALLREDUCE_INT8_COMM_F && TRANS == QUANT_BATCH_MATMUL_V3_B_TRANS &&
                         PERTOKEN == QUANT_BATCH_MATMUl_V3_IS_PERTOKEN) {
        // 17
        INVOKE_MC2_QUANT_910_OP_IMPL(BmmDequantPertoken, Mc2CoreType::ON_VECTOR, REG_MM_OBJ, true, float, DTYPE_Y,
                                     false, true);
    } else if constexpr (INT8_COMM == MATMUL_ALLREDUCE_INT8_COMM_T && TRANS == QUANT_BATCH_MATMUL_V3_NOT_TRANS &&
                         PERTOKEN == QUANT_BATCH_MATMUl_V3_NOT_PERTOKEN) {
        // 10
        INVOKE_QUANT_BMM_DEQUANT_FP16_IMPL_COMM_INT8(MatmulAllReduceQuantFP16CommInt8, false, false);
    } else if constexpr (INT8_COMM == MATMUL_ALLREDUCE_INT8_COMM_T && TRANS == QUANT_BATCH_MATMUL_V3_B_TRANS &&
                         PERTOKEN == QUANT_BATCH_MATMUl_V3_NOT_PERTOKEN) {
        // 11
        INVOKE_QUANT_BMM_DEQUANT_FP16_IMPL_COMM_INT8(MatmulAllReduceQuantFP16CommInt8, false, true);
    } else if constexpr (INT8_COMM == MATMUL_ALLREDUCE_INT8_COMM_T && TRANS == QUANT_BATCH_MATMUL_V3_NOT_TRANS &&
                         PERTOKEN == QUANT_BATCH_MATMUl_V3_IS_PERTOKEN) {
        // 26 // pertoken 适配 int8 通信
        BmmDequantPertoken<DTYPE_X1, DTYPE_X2, FORMAT_X1, FORMAT_X2, float, DTYPE_Y, false, false, true> opTile;
        BmmDequantPertoken<DTYPE_X1, DTYPE_X2, FORMAT_X1, FORMAT_X2, float, DTYPE_Y, false, false, true> opTail;
        INVOKE_QUANT_BATCH_MATMUL_DEQUANT_PERTOKEN_COMM_INT8_IMPL(MatmulAllReduceQuantPertokenInt8, float, opTile,
                                                                  opTail, false, false);
    } else if constexpr (INT8_COMM == MATMUL_ALLREDUCE_INT8_COMM_T && TRANS == QUANT_BATCH_MATMUL_V3_B_TRANS &&
                         PERTOKEN == QUANT_BATCH_MATMUl_V3_IS_PERTOKEN) {
        // 27// pertoken 适配 int8 通信
        BmmDequantPertoken<DTYPE_X1, DTYPE_X2, FORMAT_X1, FORMAT_X2, float, DTYPE_Y, false, true, true> opTile;
        BmmDequantPertoken<DTYPE_X1, DTYPE_X2, FORMAT_X1, FORMAT_X2, float, DTYPE_Y, false, true, true> opTail;
        INVOKE_QUANT_BATCH_MATMUL_DEQUANT_PERTOKEN_COMM_INT8_IMPL(MatmulAllReduceQuantPertokenInt8, float, opTile,
                                                                  opTail, false, true);
    }
#elif defined(MC2_QUANT_BF16)
    if constexpr (INT8_COMM == MATMUL_ALLREDUCE_INT8_COMM_F && TRANS == QUANT_BATCH_MATMUL_V3_NOT_TRANS &&
                  PERTOKEN == QUANT_BATCH_MATMUl_V3_NOT_PERTOKEN) {
        // 0
        INVOKE_MC2_QUANT_910_OP_IMPL(BmmDequantBf16, Mc2CoreType::ON_VECTOR, REG_MM_OBJ, false, DTYPE_Y, DTYPE_Y, false,
                                     false);
    } else if constexpr (INT8_COMM == MATMUL_ALLREDUCE_INT8_COMM_F && TRANS == QUANT_BATCH_MATMUL_V3_B_TRANS &&
                         PERTOKEN == QUANT_BATCH_MATMUl_V3_NOT_PERTOKEN) {
        // 1
        INVOKE_MC2_QUANT_910_OP_IMPL(BmmDequantBf16, Mc2CoreType::ON_VECTOR, REG_MM_OBJ, false, DTYPE_Y, DTYPE_Y, false,
                                     true);
    } else if constexpr (INT8_COMM == MATMUL_ALLREDUCE_INT8_COMM_F && TRANS == QUANT_BATCH_MATMUL_V3_NOT_TRANS &&
                         PERTOKEN == QUANT_BATCH_MATMUl_V3_IS_PERTOKEN) {
        // 16
        INVOKE_MC2_QUANT_910_OP_IMPL(BmmDequantPertoken, Mc2CoreType::ON_VECTOR, REG_MM_OBJ, true, DTYPE_Y, DTYPE_Y,
                                     false, false);
    } else if constexpr (INT8_COMM == MATMUL_ALLREDUCE_INT8_COMM_F && TRANS == QUANT_BATCH_MATMUL_V3_B_TRANS &&
                         PERTOKEN == QUANT_BATCH_MATMUl_V3_IS_PERTOKEN) {
        // 17
        INVOKE_MC2_QUANT_910_OP_IMPL(BmmDequantPertoken, Mc2CoreType::ON_VECTOR, REG_MM_OBJ, true, DTYPE_Y, DTYPE_Y,
                                     false, true);
    } else if constexpr (INT8_COMM == MATMUL_ALLREDUCE_INT8_COMM_T && TRANS == QUANT_BATCH_MATMUL_V3_NOT_TRANS &&
                         PERTOKEN == QUANT_BATCH_MATMUl_V3_NOT_PERTOKEN) {
        // 10
        BmmDequantBf16<DTYPE_X1, DTYPE_X2, FORMAT_X1, FORMAT_X2, DTYPE_Y, DTYPE_Y, false, false, true> opTile;
        BmmDequantBf16<DTYPE_X1, DTYPE_X2, FORMAT_X1, FORMAT_X2, DTYPE_Y, DTYPE_Y, false, false, true> opTail;
        INVOKE_QUANT_BATCH_MATMUL_DEQUANT_BF16_IMPL_COMM_INT8(MatmulAllReduceQuantBF16CommInt8, opTile, opTail, false,
                                                              false);
    } else if constexpr (INT8_COMM == MATMUL_ALLREDUCE_INT8_COMM_T && TRANS == QUANT_BATCH_MATMUL_V3_B_TRANS &&
                         PERTOKEN == QUANT_BATCH_MATMUl_V3_NOT_PERTOKEN) {
        // 11
        BmmDequantBf16<DTYPE_X1, DTYPE_X2, FORMAT_X1, FORMAT_X2, DTYPE_Y, DTYPE_Y, false, true, true> opTile;
        BmmDequantBf16<DTYPE_X1, DTYPE_X2, FORMAT_X1, FORMAT_X2, DTYPE_Y, DTYPE_Y, false, true, true> opTail;
        INVOKE_QUANT_BATCH_MATMUL_DEQUANT_BF16_IMPL_COMM_INT8(MatmulAllReduceQuantBF16CommInt8, opTile, opTail, false,
                                                              true);
    } else if constexpr (INT8_COMM == MATMUL_ALLREDUCE_INT8_COMM_T && TRANS == QUANT_BATCH_MATMUL_V3_NOT_TRANS &&
                         PERTOKEN == QUANT_BATCH_MATMUl_V3_IS_PERTOKEN) {
        // 26
        BmmDequantPertoken<DTYPE_X1, DTYPE_X2, FORMAT_X1, FORMAT_X2, DTYPE_Y, DTYPE_Y, false, false, true> opTile;
        BmmDequantPertoken<DTYPE_X1, DTYPE_X2, FORMAT_X1, FORMAT_X2, DTYPE_Y, DTYPE_Y, false, false, true> opTail;
        INVOKE_QUANT_BATCH_MATMUL_DEQUANT_PERTOKEN_COMM_INT8_IMPL(MatmulAllReduceQuantPertokenInt8, DTYPE_Y, opTile,
                                                                  opTail, false, false);
    } else if constexpr (INT8_COMM == MATMUL_ALLREDUCE_INT8_COMM_T && TRANS == QUANT_BATCH_MATMUL_V3_B_TRANS &&
                         PERTOKEN == QUANT_BATCH_MATMUl_V3_IS_PERTOKEN) {
        // 27
        BmmDequantPertoken<DTYPE_X1, DTYPE_X2, FORMAT_X1, FORMAT_X2, DTYPE_Y, DTYPE_Y, false, true, true> opTile;
        BmmDequantPertoken<DTYPE_X1, DTYPE_X2, FORMAT_X1, FORMAT_X2, DTYPE_Y, DTYPE_Y, false, true, true> opTail;
        INVOKE_QUANT_BATCH_MATMUL_DEQUANT_PERTOKEN_COMM_INT8_IMPL(MatmulAllReduceQuantPertokenInt8, DTYPE_Y, opTile,
                                                                  opTail, false, true);
    }
#endif
}

template <int MM_TYPE, TPL_PARAMS_COMM, TPL_PARAMS_SHARE_MM, TPL_PARAMS_WEIGHT_QUANT_MM>
__global__ __aicore__ void
weight_quant_matmul_allreduce(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR addGM, GM_ADDR antiquantScaleGM,
                              GM_ADDR antiquantOffsetGM, GM_ADDR dequantGM, GM_ADDR pertokenGM,
                              GM_ADDR commQuantScale1GM, GM_ADDR commQuantScale2GM, GM_ADDR cGM, GM_ADDR workspaceGM,
                              GM_ADDR tilingGM, TPipe &tPipe, GM_ADDR userWS)
{
#if defined(MC2_WEIGHT_QUANT)
    if constexpr (EMPTY_INPUT == MATMUL_ALLREDUCE_EMPTY_INPUT_T &&
                  MM_TYPE == MATMUL_ALLREDUCE_MM_TYPE_WEIGHT_QUANT_MATMUL) {
        // empty tensor
        INVOKE_MC2_EMPTY_TENSOR_OP_IMPL();
    } else {
        if constexpr (SUB_ALGORITHM_CUSTOM == WQBMMV2_SUB_ALGO_CUSTOM_ANTIQUANT && HAS_ANTIQUANT_OFFSET == 0 &&
                      ANTIQUANT_TYPE == WQBMMV2_ANTIQUANT_TYPE_PER_TENSOR && TRANS_B == 0) {
            // 310100
            INVOKE_MC2_WEIGHT_QUANT_910_OP_IMPL(false, Mc2QuantType::PER_TENSOR, false);
        } else if constexpr (SUB_ALGORITHM_CUSTOM == WQBMMV2_SUB_ALGO_CUSTOM_ANTIQUANT && HAS_ANTIQUANT_OFFSET == 0 &&
                             ANTIQUANT_TYPE == WQBMMV2_ANTIQUANT_TYPE_PER_TENSOR && TRANS_B == 1) {
            // 310110
            INVOKE_MC2_WEIGHT_QUANT_910_OP_IMPL(true, Mc2QuantType::PER_TENSOR, false);
        } else if constexpr (SUB_ALGORITHM_CUSTOM == WQBMMV2_SUB_ALGO_CUSTOM_ANTIQUANT && HAS_ANTIQUANT_OFFSET == 0 &&
                             ANTIQUANT_TYPE == WQBMMV2_ANTIQUANT_TYPE_PER_CHANNEL && TRANS_B == 0) {
            // 310200
            INVOKE_MC2_WEIGHT_QUANT_910_OP_IMPL(false, Mc2QuantType::PER_CHANNEL, false);
        } else if constexpr (SUB_ALGORITHM_CUSTOM == WQBMMV2_SUB_ALGO_CUSTOM_ANTIQUANT && HAS_ANTIQUANT_OFFSET == 0 &&
                             ANTIQUANT_TYPE == WQBMMV2_ANTIQUANT_TYPE_PER_CHANNEL && TRANS_B == 1) {
            // 310210
            INVOKE_MC2_WEIGHT_QUANT_910_OP_IMPL(true, Mc2QuantType::PER_CHANNEL, false);
        } else if constexpr (SUB_ALGORITHM_CUSTOM == WQBMMV2_SUB_ALGO_CUSTOM_ANTIQUANT && HAS_ANTIQUANT_OFFSET == 0 &&
                             ANTIQUANT_TYPE == WQBMMV2_ANTIQUANT_TYPE_PER_GROUP && TRANS_B == 0) {
            // 310300
            INVOKE_MC2_WEIGHT_QUANT_910_OP_IMPL(false, Mc2QuantType::PER_GROUP, false);
        } else if constexpr (SUB_ALGORITHM_CUSTOM == WQBMMV2_SUB_ALGO_CUSTOM_ANTIQUANT && HAS_ANTIQUANT_OFFSET == 0 &&
                             ANTIQUANT_TYPE == WQBMMV2_ANTIQUANT_TYPE_PER_GROUP && TRANS_B == 1) {
            // 310310
            INVOKE_MC2_WEIGHT_QUANT_910_OP_IMPL(true, Mc2QuantType::PER_GROUP, false);
        } else if constexpr (SUB_ALGORITHM_CUSTOM == WQBMMV2_SUB_ALGO_CUSTOM_ANTIQUANT && HAS_ANTIQUANT_OFFSET == 1 &&
                             ANTIQUANT_TYPE == WQBMMV2_ANTIQUANT_TYPE_PER_TENSOR && TRANS_B == 0) {
            // 311100
            INVOKE_MC2_WEIGHT_QUANT_910_OP_IMPL(false, Mc2QuantType::PER_TENSOR, true);
        } else if constexpr (SUB_ALGORITHM_CUSTOM == WQBMMV2_SUB_ALGO_CUSTOM_ANTIQUANT && HAS_ANTIQUANT_OFFSET == 1 &&
                             ANTIQUANT_TYPE == WQBMMV2_ANTIQUANT_TYPE_PER_TENSOR && TRANS_B == 1) {
            // 311110
            INVOKE_MC2_WEIGHT_QUANT_910_OP_IMPL(true, Mc2QuantType::PER_TENSOR, true);
        } else if constexpr (SUB_ALGORITHM_CUSTOM == WQBMMV2_SUB_ALGO_CUSTOM_ANTIQUANT && HAS_ANTIQUANT_OFFSET == 1 &&
                             ANTIQUANT_TYPE == WQBMMV2_ANTIQUANT_TYPE_PER_CHANNEL && TRANS_B == 0) {
            // 311200
            INVOKE_MC2_WEIGHT_QUANT_910_OP_IMPL(false, Mc2QuantType::PER_CHANNEL, true);
        } else if constexpr (SUB_ALGORITHM_CUSTOM == WQBMMV2_SUB_ALGO_CUSTOM_ANTIQUANT && HAS_ANTIQUANT_OFFSET == 1 &&
                             ANTIQUANT_TYPE == WQBMMV2_ANTIQUANT_TYPE_PER_CHANNEL && TRANS_B == 1) {
            // 311210
            INVOKE_MC2_WEIGHT_QUANT_910_OP_IMPL(true, Mc2QuantType::PER_CHANNEL, true);
        } else if constexpr (SUB_ALGORITHM_CUSTOM == WQBMMV2_SUB_ALGO_CUSTOM_ANTIQUANT && HAS_ANTIQUANT_OFFSET == 1 &&
                             ANTIQUANT_TYPE == WQBMMV2_ANTIQUANT_TYPE_PER_GROUP && TRANS_B == 0) {
            // 311300
            INVOKE_MC2_WEIGHT_QUANT_910_OP_IMPL(false, Mc2QuantType::PER_GROUP, true);
        } else if constexpr (SUB_ALGORITHM_CUSTOM == WQBMMV2_SUB_ALGO_CUSTOM_ANTIQUANT && HAS_ANTIQUANT_OFFSET == 1 &&
                             ANTIQUANT_TYPE == WQBMMV2_ANTIQUANT_TYPE_PER_GROUP && TRANS_B == 1) {
            // 311310
            INVOKE_MC2_WEIGHT_QUANT_910_OP_IMPL(true, Mc2QuantType::PER_GROUP, true);
        } else if constexpr (SUB_ALGORITHM_CUSTOM == WQBMMV2_SUB_ALGO_WEIGHT_NZ && HAS_ANTIQUANT_OFFSET == 0 &&
                             TRANS_B == 0) {
            // 810200
            INVOKE_MC2_WEIGHT_QUANT_910_OP_IMPL(false, Mc2QuantType::PER_CHANNEL, false);
        } else if constexpr (SUB_ALGORITHM_CUSTOM == WQBMMV2_SUB_ALGO_WEIGHT_NZ && HAS_ANTIQUANT_OFFSET == 0 &&
                             TRANS_B == 1) {
            // 810210
            INVOKE_MC2_WEIGHT_QUANT_910_OP_IMPL(true, Mc2QuantType::PER_CHANNEL, false);
        } else if constexpr (SUB_ALGORITHM_CUSTOM == WQBMMV2_SUB_ALGO_WEIGHT_NZ && HAS_ANTIQUANT_OFFSET == 1 &&
                             TRANS_B == 0) {
            // 811200
            INVOKE_MC2_WEIGHT_QUANT_910_OP_IMPL(false, Mc2QuantType::PER_CHANNEL, true);
        } else if constexpr (SUB_ALGORITHM_CUSTOM == WQBMMV2_SUB_ALGO_WEIGHT_NZ && HAS_ANTIQUANT_OFFSET == 1 &&
                             TRANS_B == 1) {
            // 811210
            INVOKE_MC2_WEIGHT_QUANT_910_OP_IMPL(true, Mc2QuantType::PER_CHANNEL, true);
        }
    }
#endif // defined(MC2_WEIGHT_QUANT)
}

template <int AICORE_TYPE, int MM_TYPE, TPL_PARAMS_COMM, TPL_PARAMS_SHARE_MM, TPL_PARAMS_FP_MM, TPL_PARAMS_QUANT_MM,
          TPL_PARAMS_WEIGHT_QUANT_MM>
__global__ __aicore__ void matmul_all_reduce(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR addGM,
                                             GM_ADDR antiquantScaleGM, GM_ADDR antiquantOffsetGM, GM_ADDR dequantGM,
                                             GM_ADDR pertokenGM, GM_ADDR commQuantScale1GM, GM_ADDR commQuantScale2GM,
                                             GM_ADDR cGM, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
#ifdef __CCE_KT_TEST__
    REGISTER_TILING_DEFAULT(Mc2Tiling::WeightQuantMatmulAllReduceTilingData);
#else
    REGISTER_TILING_DEFAULT(Mc2Tiling::WeightQuantMatmulAllReduceTilingData);
    REGISTER_TILING_FOR_TILINGKEY("MM_TYPE == MATMUL_ALLREDUCE_MM_TYPE_QUANT_MATMUL",
                                  Mc2Tiling::QuantMatmulAllReduceTilingData);
    REGISTER_TILING_FOR_TILINGKEY("(MM_TYPE == MATMUL_ALLREDUCE_MM_TYPE_WEIGHT_QUANT_MATMUL) && \
        (FORMAT_B == FORMAT_B_ND)",
                                  Mc2Tiling::WeightQuantMatmulAllReduceTilingData);
    REGISTER_TILING_FOR_TILINGKEY("(MM_TYPE == MATMUL_ALLREDUCE_MM_TYPE_WEIGHT_QUANT_MATMUL) && \
        (FORMAT_B == FORMAT_B_NZ)",
                                  Mc2Tiling::WeightQuantMatmulAllReduceNzTilingData);
    REGISTER_TILING_FOR_TILINGKEY("(MM_TYPE == MATMUL_ALLREDUCE_MM_TYPE_FP_MM) && \
        (AICORE_TYPE == ASCEND_910B)",
                                  Mc2Tiling::MatmulAllReduce910TilingData);
#endif
    if (workspaceGM == nullptr) {
        return;
    }

    GM_ADDR userWS = GetUserWorkspace(workspaceGM);
    if (userWS == nullptr) {
        return;
    }

    TPipe tPipe;
    if constexpr (MM_TYPE == MATMUL_ALLREDUCE_MM_TYPE_FP_MM || MM_TYPE == MATMUL_ALLREDUCE_MM_TYPE_FP_MM_CUBE_ONLY) {
        fp_matmul_all_reduce<MM_TYPE, PARAMS_COMM, PARAMS_SHARE_MM, PARAMS_FP_MM>(
            aGM, bGM, biasGM, addGM, antiquantScaleGM, antiquantOffsetGM, dequantGM, pertokenGM, commQuantScale1GM,
            commQuantScale2GM, cGM, workspaceGM, tilingGM, tPipe, userWS);
    } else if constexpr (MM_TYPE == MATMUL_ALLREDUCE_MM_TYPE_QUANT_MATMUL) {
        quant_matmul_allreduce<MM_TYPE, PARAMS_COMM, PARAMS_SHARE_MM, PARAMS_QUANT_MM>(
            aGM, bGM, biasGM, addGM, antiquantScaleGM, antiquantOffsetGM, dequantGM, pertokenGM, commQuantScale1GM,
            commQuantScale2GM, cGM, workspaceGM, tilingGM, tPipe, userWS);
    } else if constexpr (MM_TYPE == MATMUL_ALLREDUCE_MM_TYPE_WEIGHT_QUANT_MATMUL) {
        weight_quant_matmul_allreduce<MM_TYPE, PARAMS_COMM, PARAMS_SHARE_MM, PARAMS_WEIGHT_QUANT_MM>(
            aGM, bGM, biasGM, addGM, antiquantScaleGM, antiquantOffsetGM, dequantGM, pertokenGM, commQuantScale1GM,
            commQuantScale2GM, cGM, workspaceGM, tilingGM, tPipe, userWS);
    }
}
