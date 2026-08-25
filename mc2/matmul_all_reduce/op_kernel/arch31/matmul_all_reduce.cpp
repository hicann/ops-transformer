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
 * \brief arch31/310p
 */

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "lib/matmul_intf.h"
#include "../matmul_all_reduce_common.h"
#include "../matmul_all_reduce_tiling_key.h"

#ifdef __CCE_KT_TEST__
#include "../../tests/ut/op_kernel/rac_server_stub.h"
#else
#include "rac_server.h"
#endif
#if (ORIG_DTYPE_X1 == DT_INT8 && ORIG_DTYPE_X2 == DT_INT8)
#include "matmul_all_reduce_quant_bmm.h"
// 310p归一化weightNZ非量化
#elif (ORIG_DTYPE_X1 != DT_INT8 && ORIG_DTYPE_X2 != DT_INT8 && FORMAT_X2 != FORMAT_ND)
#include "../unquant_matmul_all_reduce_tiling_data.h" // 300I DUO复用A2 tilingData
#include "matmul_all_reduce_unquant_310.h"
// 310p weightNZ伪量化
#elif (ORIG_DTYPE_X1 != DT_INT8 && (ORIG_DTYPE_X2 == DT_INT8) && FORMAT_X2 != FORMAT_ND)
#include "../weight_quant_matmul_all_reduce_tiling_data.h" // 300I DUO复用A2 tilingData
#include "matmul_all_reduce_weight_quant_310.h"
#else
#include "../quant_matmul_all_reduce_tiling_data.h" // 300I DUO复用A2 tilingData
#include "matmul_all_reduce_310_general.h"
#endif // ORIG_DTYPE_X1 == DT_INT8 && ORIG_DTYPE_X2 == DT_INT8

#if ((ORIG_DTYPE_X1 == DT_INT8) && (ORIG_DTYPE_Y == DT_BF16))
#define INVOKE_QUANT_BATCH_MATMUL_DEQUANT_BF16_IMPL_COMM_INT8(templateClass, opTile, opTail, ...)                      \
    do {                                                                                                               \
        GET_TILING_DATA_MEMBER(Mc2Tiling::QuantMatmulAllReduceTilingData, msg, msg, tilingGM);                         \
        if (msg.debugMode != static_cast<uint8_t>(DebugMode::MC2_DEBUG_ONLY_AICPU)) {                                  \
            GET_TILING_DATA_WITH_STRUCT(Mc2Tiling::QuantMatmulAllReduceTilingData, tilingDataLocal, tilingGM);         \
            templateClass<DTYPE_X1, DTYPE_X2, FORMAT_X1, FORMAT_X2, DTYPE_Y, DTYPE_Y, int8_t, __VA_ARGS__> matmulOp;   \
            const Mc2Tiling::QuantMatmulAllReduceTilingData *quantMmrTilingPtr = &tilingDataLocal;                     \
            const Mc2QuantBatchMatmulV3TilingData *qBmmV3TilingPtr = &(quantMmrTilingPtr->tilematmulTiling);           \
            const AscendC::tiling::TCubeTiling *mmTilingTilePtr = &(qBmmV3TilingPtr->matmulTiling);                    \
            const Mc2QuantBatchMatmulV3TilingData *qBmmV3TilingTailPtr = &(quantMmrTilingPtr->tailmatmulTiling);       \
            const AscendC::tiling::TCubeTiling *mmTilingTailPtr = &(qBmmV3TilingTailPtr->matmulTiling);                \
            REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), opTile.mm, mmTilingTilePtr, opTail.mm, mmTilingTailPtr);   \
            matmulOp.Init(aGM, bGM, biasGM, addGM, dequantGM, commQuantScale1GM, commQuantScale2GM, cGM, userWS,       \
                          &tilingDataLocal, &tPipe);                                                                   \
            matmulOp.Process(opTile, opTail);                                                                          \
            tPipe.Destroy();                                                                                           \
        }                                                                                                              \
    } while (0)
#endif // (ORIG_DTYPE_X1 == DT_INT8) && (ORIG_DTYPE_Y == DT_BF16)

#if ((ORIG_DTYPE_X1 == DT_INT8) && (ORIG_DTYPE_Y == DT_FLOAT16))
#define INVOKE_QUANT_BMM_DEQUANT_FP16_IMPL_COMM_INT8(templateClass, ...)                                               \
    do {                                                                                                               \
        GET_TILING_DATA_WITH_STRUCT(Mc2Tiling::QuantMatmulAllReduceTilingData, tilingDataLocal, tilingGM);             \
        templateClass<DTYPE_X1, DTYPE_X2, int32_t, DTYPE_Y, int8_t, __VA_ARGS__> op;                                   \
        op.Init(aGM, bGM, dequantGM, biasGM, addGM, cGM, workspaceGM, &tilingDataLocal, &tPipe);                       \
        op.InitScale(commQuantScale1GM, commQuantScale2GM);                                                            \
        op.Process();                                                                                                  \
    } while (0)
#endif

#if ((ORIG_DTYPE_X1 == DT_INT8) && (ORIG_DTYPE_X2 == DT_INT8))
#define INVOKE_QUANT_BATCH_MATMUL_DEQUANT_PERTOKEN_COMM_INT8_IMPL(templateClass, scaleType, opTile, opTail, ...)       \
    do {                                                                                                               \
        GET_TILING_DATA_MEMBER(Mc2Tiling::QuantMatmulAllReduceTilingData, msg, msg, tilingGM);                         \
        if (msg.debugMode != static_cast<uint8_t>(DebugMode::MC2_DEBUG_ONLY_AICPU)) {                                  \
            GET_TILING_DATA_WITH_STRUCT(Mc2Tiling::QuantMatmulAllReduceTilingData, tilingDataLocal, tilingGM);         \
            templateClass<DTYPE_X1, DTYPE_X2, FORMAT_X1, FORMAT_X2, scaleType, DTYPE_Y, int8_t, __VA_ARGS__> matmulOp; \
            const Mc2Tiling::QuantMatmulAllReduceTilingData *quantMmrTilingPtr = &tilingDataLocal;                     \
            const Mc2QuantBatchMatmulV3TilingData *qBmmV3TilingPtr = &(quantMmrTilingPtr->tilematmulTiling);           \
            const AscendC::tiling::TCubeTiling *mmTilingTilePtr = &(qBmmV3TilingPtr->matmulTiling);                    \
            const Mc2QuantBatchMatmulV3TilingData *qBmmV3TilingTailPtr = &(quantMmrTilingPtr->tailmatmulTiling);       \
            const AscendC::tiling::TCubeTiling *mmTilingTailPtr = &(qBmmV3TilingTailPtr->matmulTiling);                \
            REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), opTile.mm, mmTilingTilePtr, opTail.mm, mmTilingTailPtr);   \
            matmulOp.Init(aGM, bGM, biasGM, addGM, dequantGM, pertokenGM, commQuantScale1GM, commQuantScale2GM, cGM,   \
                          userWS, &tilingDataLocal, &tPipe);                                                           \
            matmulOp.Process(opTile, opTail);                                                                          \
            tPipe.Destroy();                                                                                           \
        }                                                                                                              \
    } while (0)
#endif

namespace MatmulAllReduceImpl {
}

using namespace AscendC;
using namespace Mc2Tiling;
using namespace MatmulAllReduceImpl;

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
    REGISTER_TILING_FOR_TILINGKEY("(MM_TYPE == MATMUL_ALLREDUCE_MM_TYPE_FP_MM) && \
        (AICORE_TYPE == ASCEND_310P) && (FORMAT_B == FORMAT_B_ND)",
                                  Mc2Tiling::UnQuantMatmulAllReduceTilingData);
    REGISTER_TILING_FOR_TILINGKEY("(MM_TYPE == MATMUL_ALLREDUCE_MM_TYPE_FP_MM) && \
        (AICORE_TYPE == ASCEND_310P) && (FORMAT_B == FORMAT_B_NZ)",
                                  Mc2Tiling::MatmulAllReduceTilingData);
#endif
    if (workspaceGM == nullptr) {
        return;
    }

    GM_ADDR userWS = GetUserWorkspace(workspaceGM);
    if (userWS == nullptr) {
        return;
    }

    TPipe tPipe;
    // 310P
    HcclServer hcclServer;
#if (ORIG_DTYPE_X1 == DT_INT8 && ORIG_DTYPE_X2 == DT_INT8) // 归一化310p weightNZ全量化
    if constexpr (MM_TYPE == MATMUL_ALLREDUCE_MM_TYPE_QUANT_MATMUL) {
        INVOKE_QUANT_BMM_OP_IMPL(MatmulAllReduceQuantBmm, false, true);
    }
#elif (ORIG_DTYPE_X1 == DT_FLOAT16 && ORIG_DTYPE_X2 == DT_INT8 && FORMAT_X2 != FORMAT_ND) // 归一化310p weightNZ伪量化
    if constexpr (TRANS_A == 0 && ANTIQUANT_TYPE == WQBMMV2_ANTIQUANT_TYPE_PER_TENSOR &&
                  HAS_ANTIQUANT_OFFSET == 0) { // 80010
        INVOKE_WEIGHT_QUANT_BMM_OP_IMPL_310(MatmulAllReduceWeightQuant310, false, true, Mc2QuantType::PER_TENSOR,
                                            false);
    } else if constexpr (TRANS_A == 1 && ANTIQUANT_TYPE == WQBMMV2_ANTIQUANT_TYPE_PER_TENSOR &&
                         HAS_ANTIQUANT_OFFSET == 0) { // 80011
        INVOKE_WEIGHT_QUANT_BMM_OP_IMPL_310(MatmulAllReduceWeightQuant310, true, true, Mc2QuantType::PER_TENSOR, false);
    } else if constexpr (TRANS_A == 0 && ANTIQUANT_TYPE == WQBMMV2_ANTIQUANT_TYPE_PER_CHANNEL &&
                         HAS_ANTIQUANT_OFFSET == 0) { // 80020
        INVOKE_WEIGHT_QUANT_BMM_OP_IMPL_310(MatmulAllReduceWeightQuant310, false, true, Mc2QuantType::PER_CHANNEL,
                                            false);
    } else if constexpr (TRANS_A == 1 && ANTIQUANT_TYPE == WQBMMV2_ANTIQUANT_TYPE_PER_CHANNEL &&
                         HAS_ANTIQUANT_OFFSET == 0) { // 80021
        INVOKE_WEIGHT_QUANT_BMM_OP_IMPL_310(MatmulAllReduceWeightQuant310, true, true, Mc2QuantType::PER_CHANNEL,
                                            false);
    } else if constexpr (TRANS_A == 0 && ANTIQUANT_TYPE == WQBMMV2_ANTIQUANT_TYPE_PER_TENSOR &&
                         HAS_ANTIQUANT_OFFSET == 1) { // 80110
        INVOKE_WEIGHT_QUANT_BMM_OP_IMPL_310(MatmulAllReduceWeightQuant310, false, true, Mc2QuantType::PER_TENSOR, true);
    } else if constexpr (TRANS_A == 1 && ANTIQUANT_TYPE == WQBMMV2_ANTIQUANT_TYPE_PER_TENSOR &&
                         HAS_ANTIQUANT_OFFSET == 1) { // 80111
        INVOKE_WEIGHT_QUANT_BMM_OP_IMPL_310(MatmulAllReduceWeightQuant310, true, true, Mc2QuantType::PER_TENSOR, true);
    } else if constexpr (TRANS_A == 0 && ANTIQUANT_TYPE == WQBMMV2_ANTIQUANT_TYPE_PER_CHANNEL &&
                         HAS_ANTIQUANT_OFFSET == 1) { // 80120
        INVOKE_WEIGHT_QUANT_BMM_OP_IMPL_310(MatmulAllReduceWeightQuant310, false, true, Mc2QuantType::PER_CHANNEL,
                                            true);
    } else if constexpr (TRANS_A == 1 && ANTIQUANT_TYPE == WQBMMV2_ANTIQUANT_TYPE_PER_CHANNEL &&
                         HAS_ANTIQUANT_OFFSET == 1) { // 80121
        INVOKE_WEIGHT_QUANT_BMM_OP_IMPL_310(MatmulAllReduceWeightQuant310, true, true, Mc2QuantType::PER_CHANNEL, true);
    } else if constexpr (EMPTY_INPUT == MATMUL_ALLREDUCE_EMPTY_INPUT_T && FORMAT_B == FORMAT_B_NZ) { // 空kernel
        GET_TILING_DATA_WITH_STRUCT(Mc2Tiling::WeightQuantMatmulAllReduceNzTilingData, tilingData, tilingGM);
        WeightQuantEmptyTensorKernel(biasGM, cGM, workspaceGM, &tilingData, &hcclServer);
    }
#elif (ORIG_DTYPE_X1 != DT_INT8 && ORIG_DTYPE_X2 != DT_INT8 && FORMAT_X2 != FORMAT_ND) // 310p归一化weightNZ非量化
    if constexpr (MIXND2NZ == MAT_MUL_V3_MIXND2NZ_TRUE) {
        INVOKE_UNQUANT_BMM_OP_IMPL_310(MatmulAllReduceUnquant310);
    } else if constexpr (MIXND2NZ == MAT_MUL_V3_MIXND2NZ_FALSE) {
        INVOKE_UNQUANT_BMM_OP_IMPL_310(MatmulAllReduceUnquant310);
    } else if constexpr (EMPTY_INPUT == MATMUL_ALLREDUCE_EMPTY_INPUT_T && FORMAT_B == FORMAT_B_NZ) {
        // k==0 伪量化NZ
        GET_TILING_DATA_WITH_STRUCT(Mc2Tiling::UnQuantMatmulAllReduceTilingData, tilingData, tilingGM);
        MatMulEmptyTensorKernelUnquantNz(biasGM, cGM, workspaceGM, &tilingData, &hcclServer);
    }
#else  // 310p weightND(非归一化 非量化 伪量化)
    // L2cache, transB, isWeightQuant, AntiQuantType, hasAntiQuantOffset, 310Version
    if constexpr (ENABLE_L2_CACHE == 1 && MM_TYPE == MATMUL_ALLREDUCE_MM_TYPE_FP_MM &&
                  ANTIQUANT_TYPE == WQBMMV2_ANTIQUANT_TYPE_NONE && HAS_ANTIQUANT_OFFSET == 0 && TRANS_A == 1 &&
                  TRANS_B == 0) {
        // 开启L2cache, 非量化ND，非量化-不涉及antiQuantType，非量化-不涉及antiQuantOffset
        using aType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X1, true>;
        using cType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, C_DTYPE>;
        using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half>;
        using bType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X2, true>;
        INVOKE_MATMUL_ALL_REDUCE_OP_IMPL(MatmulAllReduce310General, true, false, AntiQuantType::NONE, false);
    } else if constexpr (ENABLE_L2_CACHE == 0 && MM_TYPE == MATMUL_ALLREDUCE_MM_TYPE_FP_MM &&
                         ANTIQUANT_TYPE == WQBMMV2_ANTIQUANT_TYPE_NONE && HAS_ANTIQUANT_OFFSET == 0 && TRANS_A == 1 &&
                         TRANS_B == 0) {
        // 关闭L2cache, 非量化ND，非量化-不涉及antiQuantType，非量化-不涉及antiQuantOffset，weight转置
        using aType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X1, true>;
        using cType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, C_DTYPE>;
        using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half>;
        using bType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X2, true>;
        INVOKE_MATMUL_ALL_REDUCE_OP_IMPL(MatmulAllReduce310General, false, false, AntiQuantType::NONE, false);
    } else if constexpr (ENABLE_L2_CACHE == 0 && MM_TYPE == MATMUL_ALLREDUCE_MM_TYPE_WEIGHT_QUANT_MATMUL &&
                         ANTIQUANT_TYPE == WQBMMV2_ANTIQUANT_TYPE_PER_TENSOR && HAS_ANTIQUANT_OFFSET == 0 &&
                         TRANS_A == 1 && TRANS_B == 1) {
        // 关闭L2cache, 伪量化ND，perTensor，无antiQuantOffset，weight转置
        using aType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X1, true>;
        using bType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X2, true>;
        using cType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, C_DTYPE>;
        using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half>;
        INVOKE_MATMUL_ALL_REDUCE_OP_IMPL(MatmulAllReduce310General, false, true, AntiQuantType::PER_TENSOR, false);
    } else if constexpr (ENABLE_L2_CACHE == 1 && MM_TYPE == MATMUL_ALLREDUCE_MM_TYPE_WEIGHT_QUANT_MATMUL &&
                         ANTIQUANT_TYPE == WQBMMV2_ANTIQUANT_TYPE_PER_TENSOR && HAS_ANTIQUANT_OFFSET == 0 &&
                         TRANS_A == 1 && TRANS_B == 1) {
        // 打开L2cache, 伪量化ND，perTensor，无antiQuantOffset，weight转置
        using aType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X1, true>;
        using bType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X2, true>;
        using cType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, C_DTYPE>;
        using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half>;
        INVOKE_MATMUL_ALL_REDUCE_OP_IMPL(MatmulAllReduce310General, true, true, AntiQuantType::PER_TENSOR, false);
    } else if constexpr (ENABLE_L2_CACHE == 0 && MM_TYPE == MATMUL_ALLREDUCE_MM_TYPE_WEIGHT_QUANT_MATMUL &&
                         ANTIQUANT_TYPE == WQBMMV2_ANTIQUANT_TYPE_PER_CHANNEL && HAS_ANTIQUANT_OFFSET == 0 &&
                         TRANS_A == 1 && TRANS_B == 1) {
        // 关闭L2cache, 伪量化ND，perChannel，无antiQuantOffset，weight转置
        using aType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X1, true>;
        using bType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X2, true>;
        using cType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, C_DTYPE>;
        using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half>;
        INVOKE_MATMUL_ALL_REDUCE_OP_IMPL(MatmulAllReduce310General, false, true, AntiQuantType::PER_CHANNEL, false);
    } else if constexpr (ENABLE_L2_CACHE == 1 && MM_TYPE == MATMUL_ALLREDUCE_MM_TYPE_WEIGHT_QUANT_MATMUL &&
                         ANTIQUANT_TYPE == WQBMMV2_ANTIQUANT_TYPE_PER_CHANNEL && HAS_ANTIQUANT_OFFSET == 0 &&
                         TRANS_A == 1 && TRANS_B == 1) {
        // 打开L2cache, 伪量化ND，perChannel，无antiQuantOffset，weight转置
        using aType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X1, true>;
        using bType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X2, true>;
        using cType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, C_DTYPE>;
        using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half>;
        INVOKE_MATMUL_ALL_REDUCE_OP_IMPL(MatmulAllReduce310General, true, true, AntiQuantType::PER_CHANNEL, false);
    } else if constexpr (ENABLE_L2_CACHE == 0 && MM_TYPE == MATMUL_ALLREDUCE_MM_TYPE_WEIGHT_QUANT_MATMUL &&
                         ANTIQUANT_TYPE == WQBMMV2_ANTIQUANT_TYPE_PER_TENSOR && HAS_ANTIQUANT_OFFSET == 1 &&
                         TRANS_A == 1 && TRANS_B == 1) {
        // 关闭L2cache, 伪量化ND，perTensor，带antiQuantOffset，weight转置
        using aType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X1, true>;
        using bType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X2, true>;
        using cType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, C_DTYPE>;
        using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half>;
        INVOKE_MATMUL_ALL_REDUCE_OP_IMPL(MatmulAllReduce310General, false, true, AntiQuantType::PER_TENSOR, true);
    } else if constexpr (ENABLE_L2_CACHE == 1 && MM_TYPE == MATMUL_ALLREDUCE_MM_TYPE_WEIGHT_QUANT_MATMUL &&
                         ANTIQUANT_TYPE == WQBMMV2_ANTIQUANT_TYPE_PER_TENSOR && HAS_ANTIQUANT_OFFSET == 1 &&
                         TRANS_A == 1 && TRANS_B == 1) {
        // 打开L2cache, 伪量化ND，perTensor，带antiQuantOffset，weight转置
        using aType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X1, true>;
        using bType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X2, true>;
        using cType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, C_DTYPE>;
        using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half>;
        INVOKE_MATMUL_ALL_REDUCE_OP_IMPL(MatmulAllReduce310General, true, true, AntiQuantType::PER_TENSOR, true);
    } else if constexpr (ENABLE_L2_CACHE == 0 && MM_TYPE == MATMUL_ALLREDUCE_MM_TYPE_WEIGHT_QUANT_MATMUL &&
                         ANTIQUANT_TYPE == WQBMMV2_ANTIQUANT_TYPE_PER_CHANNEL && HAS_ANTIQUANT_OFFSET == 1 &&
                         TRANS_A == 1 && TRANS_B == 1) {
        // 关闭L2cache, 伪量化ND，perChannel，带antiQuantOffset，weight转置
        using aType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X1, true>;
        using bType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X2, true>;
        using cType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, C_DTYPE>;
        using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half>;
        INVOKE_MATMUL_ALL_REDUCE_OP_IMPL(MatmulAllReduce310General, false, true, AntiQuantType::PER_CHANNEL, true);
    } else if constexpr (ENABLE_L2_CACHE == 1 && MM_TYPE == MATMUL_ALLREDUCE_MM_TYPE_WEIGHT_QUANT_MATMUL &&
                         ANTIQUANT_TYPE == WQBMMV2_ANTIQUANT_TYPE_PER_CHANNEL && HAS_ANTIQUANT_OFFSET == 1 &&
                         TRANS_A == 1 && TRANS_B == 1) {
        // 打开L2cache, 伪量化ND，perChannel，带antiQuantOffset，weight转置
        using aType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X1, true>;
        using bType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X2, true>;
        using cType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, C_DTYPE>;
        using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half>;
        INVOKE_MATMUL_ALL_REDUCE_OP_IMPL(MatmulAllReduce310General, true, true, AntiQuantType::PER_CHANNEL, true);
    } else if constexpr (ENABLE_L2_CACHE == 0 && MM_TYPE == MATMUL_ALLREDUCE_MM_TYPE_WEIGHT_QUANT_MATMUL &&
                         ANTIQUANT_TYPE == WQBMMV2_ANTIQUANT_TYPE_PER_TENSOR && HAS_ANTIQUANT_OFFSET == 0 &&
                         TRANS_A == 1 && TRANS_B == 0) {
        // 关闭L2cache, 伪量化ND，perTensor，无antiQuantOffset，weight非转置
        using aType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X1, true>;
        using bType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X2, false>;
        using cType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, C_DTYPE>;
        using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half>;
        INVOKE_MATMUL_ALL_REDUCE_OP_IMPL(MatmulAllReduce310General, false, true, AntiQuantType::PER_TENSOR, false);
    } else if constexpr (ENABLE_L2_CACHE == 1 && MM_TYPE == MATMUL_ALLREDUCE_MM_TYPE_WEIGHT_QUANT_MATMUL &&
                         ANTIQUANT_TYPE == WQBMMV2_ANTIQUANT_TYPE_PER_TENSOR && HAS_ANTIQUANT_OFFSET == 0 &&
                         TRANS_A == 1 && TRANS_B == 0) {
        // 打开L2cache, 伪量化ND，perTensor，无antiQuantOffset，weight非转置
        using aType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X1, true>;
        using bType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X2, false>;
        using cType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, C_DTYPE>;
        using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half>;
        INVOKE_MATMUL_ALL_REDUCE_OP_IMPL(MatmulAllReduce310General, true, true, AntiQuantType::PER_TENSOR, false);
    } else if constexpr (ENABLE_L2_CACHE == 0 && MM_TYPE == MATMUL_ALLREDUCE_MM_TYPE_WEIGHT_QUANT_MATMUL &&
                         ANTIQUANT_TYPE == WQBMMV2_ANTIQUANT_TYPE_PER_CHANNEL && HAS_ANTIQUANT_OFFSET == 0 &&
                         TRANS_A == 1 && TRANS_B == 0) {
        // 关闭L2cache, 伪量化ND，perChannel，无antiQuantOffset，weight非转置
        using aType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X1, true>;
        using bType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X2, false>;
        using cType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, C_DTYPE>;
        using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half>;
        INVOKE_MATMUL_ALL_REDUCE_OP_IMPL(MatmulAllReduce310General, false, true, AntiQuantType::PER_CHANNEL, false);
    } else if constexpr (ENABLE_L2_CACHE == 1 && MM_TYPE == MATMUL_ALLREDUCE_MM_TYPE_WEIGHT_QUANT_MATMUL &&
                         ANTIQUANT_TYPE == WQBMMV2_ANTIQUANT_TYPE_PER_CHANNEL && HAS_ANTIQUANT_OFFSET == 0 &&
                         TRANS_A == 1 && TRANS_B == 0) {
        // 打开L2cache, 伪量化ND，perChannel，无antiQuantOffset，weight非转置
        using aType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X1, true>;
        using bType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X2, false>;
        using cType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, C_DTYPE>;
        using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half>;
        INVOKE_MATMUL_ALL_REDUCE_OP_IMPL(MatmulAllReduce310General, true, true, AntiQuantType::PER_CHANNEL, false);
    } else if constexpr (ENABLE_L2_CACHE == 0 && MM_TYPE == MATMUL_ALLREDUCE_MM_TYPE_WEIGHT_QUANT_MATMUL &&
                         ANTIQUANT_TYPE == WQBMMV2_ANTIQUANT_TYPE_PER_TENSOR && HAS_ANTIQUANT_OFFSET == 1 &&
                         TRANS_A == 1 && TRANS_B == 0) {
        // 关闭L2cache, 伪量化ND，perTensor，带antiQuantOffset，weight非转置
        using aType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X1, true>;
        using bType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X2, false>;
        using cType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, C_DTYPE>;
        using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half>;
        INVOKE_MATMUL_ALL_REDUCE_OP_IMPL(MatmulAllReduce310General, false, true, AntiQuantType::PER_TENSOR, true);
    } else if constexpr (ENABLE_L2_CACHE == 1 && MM_TYPE == MATMUL_ALLREDUCE_MM_TYPE_WEIGHT_QUANT_MATMUL &&
                         ANTIQUANT_TYPE == WQBMMV2_ANTIQUANT_TYPE_PER_TENSOR && HAS_ANTIQUANT_OFFSET == 1 &&
                         TRANS_A == 1 && TRANS_B == 0) {
        // 打开L2cache, 伪量化ND，perTensor，带antiQuantOffset，weight非转置
        using aType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X1, true>;
        using bType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X2, false>;
        using cType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, C_DTYPE>;
        using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half>;
        INVOKE_MATMUL_ALL_REDUCE_OP_IMPL(MatmulAllReduce310General, true, true, AntiQuantType::PER_TENSOR, true);
    } else if constexpr (ENABLE_L2_CACHE == 0 && MM_TYPE == MATMUL_ALLREDUCE_MM_TYPE_WEIGHT_QUANT_MATMUL &&
                         ANTIQUANT_TYPE == WQBMMV2_ANTIQUANT_TYPE_PER_CHANNEL && HAS_ANTIQUANT_OFFSET == 1 &&
                         TRANS_A == 1 && TRANS_B == 0) {
        // 关闭L2cache, 伪量化ND，perChannel，带antiQuantOffset，weight非转置
        using aType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X1, true>;
        using bType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X2, false>;
        using cType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, C_DTYPE>;
        using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half>;
        INVOKE_MATMUL_ALL_REDUCE_OP_IMPL(MatmulAllReduce310General, false, true, AntiQuantType::PER_CHANNEL, true);
    } else if constexpr (ENABLE_L2_CACHE == 1 && MM_TYPE == MATMUL_ALLREDUCE_MM_TYPE_WEIGHT_QUANT_MATMUL &&
                         ANTIQUANT_TYPE == WQBMMV2_ANTIQUANT_TYPE_PER_CHANNEL && HAS_ANTIQUANT_OFFSET == 1 &&
                         TRANS_A == 1 && TRANS_B == 0) {
        // 打开L2cache, 伪量化ND，perChannel，带antiQuantOffset，weight非转置
        using aType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X1, true>;
        using bType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X2, false>;
        using cType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, C_DTYPE>;
        using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half>;
        INVOKE_MATMUL_ALL_REDUCE_OP_IMPL(MatmulAllReduce310General, true, true, AntiQuantType::PER_CHANNEL, true);
    } else if constexpr (EMPTY_INPUT == MATMUL_ALLREDUCE_EMPTY_INPUT_T && MM_TYPE == MATMUL_ALLREDUCE_MM_TYPE_FP_MM &&
                         FORMAT_B == FORMAT_B_ND) {
        // k==0 伪量化ND
        GET_TILING_DATA_WITH_STRUCT(Mc2Tiling::MatmulAllReduceTilingData, tilingData, tilingGM);
        MatMulEmptyTensorKernel(biasGM, cGM, workspaceGM, &tilingData, &hcclServer);
    }
#endif // ORIG_DTYPE_X1 == DT_INT8
}
