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
 * \file quant_compressor.cpp
 * \brief
 */

#include "arch35/quant_compressor_kernel.h"
#include "arch35/quant_compressor_kernel_full_load.h"

using namespace QuantCompressor;

#define INVOKE_QUANT_COMPRESSOR_GENERAL_OP_IMPL(templateClass, ...) \
    do { \
        templateClass<COMPType<__VA_ARGS__>> op(&pipe, tilingData); \
        op.Init(x, wKv, wGate, stateCache, ape, xDescale, wKvDescale, wGateDescale, stateBlockTable, cuSeqlens, \
                seqUsed, startPos, cmpKvOut, workspace); \
        op.Process(); \
    } while (0)

template <uint8_t XLayout, uint8_t XDType, uint8_t Coff, uint8_t CacheMode, uint8_t QuantMode, uint8_t TemplateId>
__global__ __aicore__ void quant_compressor(__gm__ uint8_t *x, __gm__ uint8_t *wKv, __gm__ uint8_t *wGate,
                                            __gm__ uint8_t *stateCache, __gm__ uint8_t *ape, __gm__ uint8_t *xDescale,
                                            __gm__ uint8_t *wKvDescale, __gm__ uint8_t *wGateDescale,
                                            __gm__ uint8_t *stateBlockTable, __gm__ uint8_t *cuSeqlens,
                                            __gm__ uint8_t *seqUsed, __gm__ uint8_t *startPos, __gm__ uint8_t *cmpKvOut,
                                            __gm__ uint8_t *stateCacheOut, __gm__ uint8_t *workspace,
                                            __gm__ uint8_t *tiling)
{
    REGISTER_TILING_DEFAULT(optiling::QuantCompressorTilingData);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    GET_TILING_DATA_WITH_STRUCT(optiling::QuantCompressorTilingData, tilingDataIn, tiling);
    if constexpr (static_cast<TEMPLATE_ID>(TemplateId) == TEMPLATE_ID::EMPTY_X) {
        return;
    }
    const optiling::QuantCompressorTilingData *__restrict tilingData = &tilingDataIn;
    TPipe pipe;
    constexpr auto xLayout = static_cast<X_LAYOUT>(XLayout);
    constexpr auto xDtype = static_cast<X_DTYPE>(XDType);
    constexpr auto coff = static_cast<COFF>(Coff);
    constexpr auto cacheMode = static_cast<CACHE_MODE>(CacheMode);
    constexpr auto quantMode = static_cast<QUANT_MODE>(QuantMode);
    if constexpr (static_cast<TEMPLATE_ID>(TemplateId) == TEMPLATE_ID::FULL_LOAD) {
        INVOKE_QUANT_COMPRESSOR_GENERAL_OP_IMPL(QuantCompressorKernelFullLoad, xLayout, xDtype, coff, cacheMode,
                                                quantMode);
    } else {
        INVOKE_QUANT_COMPRESSOR_GENERAL_OP_IMPL(QuantCompressorKernel, xLayout, xDtype, coff, cacheMode, quantMode);
    }
}
