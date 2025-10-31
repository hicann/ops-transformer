/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mla_prolog_v3.cpp
 * \brief
 */

#if __has_include("../../mla_prolog/op_kernel/kernel_mla_prolog_split_n.h")
#include "../../mla_prolog/op_kernel/kernel_mla_prolog_split_n.h"
#else
#include "../mla_prolog/kernel_mla_prolog_split_n.h"
#endif

using namespace MlaProlog;

template<uint8_t CacheMode, uint8_t Scenario, uint8_t QuantMode,
         bool EnableDequantOpt, bool EnableGroupComputeOpt, uint8_t EmptyTensorMode>
__global__ __aicore__ void mla_prolog_v3(
    __gm__ uint8_t *tokenX,
    __gm__ uint8_t *weightDq,
    __gm__ uint8_t *weightUqQr,
    __gm__ uint8_t *weightUk,
    __gm__ uint8_t *weightDkvKr,
    __gm__ uint8_t *rmsnormGammaCq,
    __gm__ uint8_t *rmsnormGammaCkv,
    __gm__ uint8_t *ropeSin,
    __gm__ uint8_t *ropeCos,
    __gm__ uint8_t *kvCache,
    __gm__ uint8_t *krCache,
    __gm__ uint8_t *cacheIndex,
    __gm__ uint8_t *dequantScaleX,
    __gm__ uint8_t *dequantScaleWDq,
    __gm__ uint8_t *dequantScaleWUqQr,
    __gm__ uint8_t *dequantScaleWDkvKr,
    __gm__ uint8_t *quantScaleCkv,
    __gm__ uint8_t *quantScaleCkr,
    __gm__ uint8_t *smoothScalesCq,
    __gm__ uint8_t *actualSeqLen,
    __gm__ uint8_t *queryOut,
    __gm__ uint8_t *queryRopeOut,
    __gm__ uint8_t *kvCacheOut,
    __gm__ uint8_t *krCacheOut,
    __gm__ uint8_t *dequantScaleQNopeOut,
    __gm__ uint8_t *queryNormOut,
    __gm__ uint8_t *dequantScaleQNormOut,
    __gm__ uint8_t *workspace,
    __gm__ uint8_t *tiling)
{
    REGISTER_TILING_DEFAULT(optiling::MlaPrologTilingData);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    constexpr auto emptyMode = static_cast<EMPTY_TENSOR_MODE>(EmptyTensorMode);
    if constexpr (emptyMode == EMPTY_TENSOR_MODE::EMPTY_QUERY) {
        return;
    }
    constexpr auto cacheMode = static_cast<CACHE_MODE>(CacheMode);

    GET_TILING_DATA_WITH_STRUCT(optiling::MlaPrologTilingData, tilingDataIn, tiling);
    const optiling::MlaPrologTilingData *__restrict tilingData = nullptr;
    const optiling::MlaPrologBaseParams *__restrict tilingDataBaseParams = &tilingDataIn.baseParams;

    TPipe pipe;
    if constexpr (static_cast<SCENARIO>(Scenario) == SCENARIO::NO_QUANT) {
        MlaPrologVecS1CubS2<MLAPType<bfloat16_t, bfloat16_t, bfloat16_t, cacheMode,
            EnableDequantOpt, EnableGroupComputeOpt, emptyMode>> op(&pipe, tilingData, tilingDataBaseParams);
        op.Init(tokenX, weightDq, weightUqQr, weightUk, weightDkvKr, rmsnormGammaCq, rmsnormGammaCkv, ropeSin,
                ropeCos, cacheIndex, kvCacheOut, krCacheOut, dequantScaleX, dequantScaleWDq, dequantScaleWUqQr,
                dequantScaleWDkvKr, quantScaleCkv, quantScaleCkr, smoothScalesCq, queryOut, queryRopeOut,
                dequantScaleQNopeOut, workspace);
        op.Process();

    } else if constexpr (static_cast<SCENARIO>(Scenario) == SCENARIO::QUANT && static_cast<QUANT_MODE>(QuantMode) == QUANT_MODE::PARTIAL_QUANT_KV_NO_QUANT) {
        MlaPrologVecS1CubS2<MLAPType<bfloat16_t, int8_t, bfloat16_t, cacheMode,
            EnableDequantOpt, EnableGroupComputeOpt, emptyMode>> op(&pipe, tilingData, tilingDataBaseParams);
        op.Init(tokenX, weightDq, weightUqQr, weightUk, weightDkvKr, rmsnormGammaCq, rmsnormGammaCkv, ropeSin,
                ropeCos, cacheIndex, kvCacheOut, krCacheOut, dequantScaleX, dequantScaleWDq, dequantScaleWUqQr,
                dequantScaleWDkvKr, quantScaleCkv, quantScaleCkr, smoothScalesCq, queryOut, queryRopeOut,
                dequantScaleQNopeOut, workspace);
        op.Process();

    } else if constexpr (static_cast<SCENARIO>(Scenario) == SCENARIO::QUANT && static_cast<QUANT_MODE>(QuantMode) == QUANT_MODE::PARTIAL_QUANT_KV_QUANT) {
        MlaPrologVecS1CubS2<MLAPType<bfloat16_t, int8_t, int8_t, cacheMode,
            EnableDequantOpt, EnableGroupComputeOpt, emptyMode>> op(&pipe, tilingData, tilingDataBaseParams);
        op.Init(tokenX, weightDq, weightUqQr, weightUk, weightDkvKr, rmsnormGammaCq, rmsnormGammaCkv, ropeSin,
                ropeCos, cacheIndex, kvCacheOut, krCacheOut, dequantScaleX, dequantScaleWDq, dequantScaleWUqQr,
                dequantScaleWDkvKr, quantScaleCkv, quantScaleCkr, smoothScalesCq, queryOut, queryRopeOut,
                dequantScaleQNopeOut, workspace);
        op.Process();

    } else if constexpr (static_cast<SCENARIO>(Scenario) == SCENARIO::QUANT && static_cast<QUANT_MODE>(QuantMode) == QUANT_MODE::FULL_QUANT_KV_NO_QUANT) {
        MlaPrologVecS1CubS2<MLAPType<int8_t, int8_t, bfloat16_t, cacheMode,
            EnableDequantOpt, EnableGroupComputeOpt, emptyMode>> op(&pipe, tilingData, tilingDataBaseParams);
        op.Init(tokenX, weightDq, weightUqQr, weightUk, weightDkvKr, rmsnormGammaCq, rmsnormGammaCkv, ropeSin,
                ropeCos, cacheIndex, kvCacheOut, krCacheOut, dequantScaleX, dequantScaleWDq, dequantScaleWUqQr,
                dequantScaleWDkvKr, quantScaleCkv, quantScaleCkr, smoothScalesCq, queryOut, queryRopeOut,
                dequantScaleQNopeOut, workspace);
        op.Process();

    } else if constexpr (static_cast<SCENARIO>(Scenario) == SCENARIO::QUANT && static_cast<QUANT_MODE>(QuantMode) == QUANT_MODE::FULL_QUANT_KV_QUANT) {
        MlaPrologVecS1CubS2<MLAPType<int8_t, int8_t, int8_t, cacheMode,
            EnableDequantOpt, EnableGroupComputeOpt, emptyMode>> op(&pipe, tilingData, tilingDataBaseParams);
        op.Init(tokenX, weightDq, weightUqQr, weightUk, weightDkvKr, rmsnormGammaCq, rmsnormGammaCkv, ropeSin,
                ropeCos, cacheIndex, kvCacheOut, krCacheOut, dequantScaleX, dequantScaleWDq, dequantScaleWUqQr,
                dequantScaleWDkvKr, quantScaleCkv, quantScaleCkr, smoothScalesCq, queryOut, queryRopeOut,
                dequantScaleQNopeOut, workspace);
        op.Process();
    }
}