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
 * \file basic_block_config.h
 * \brief
 */
#ifndef GMMFR_BASIC_BLOCK_CONFIG_H
#define GMMFR_BASIC_BLOCK_CONFIG_H

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif
#include "lib/matmul_intf.h"
#include "weight_quant_tool.h"

namespace WeightQuantBatchMatmulV2::Arch35 {

constexpr static uint16_t WEIGHT_F16_UB_NZ_STRIDE = 65;
constexpr int16_t SHIFT_FOR_BF16 = 1;

struct WqmmConfig {
    bool aTrans;
    bool bTrans;
    QuantType antiQuantType;
    bool hasAntiQuantOffset;
    QuantType quantType;
    CubeFormat weightFormat;
};

static constexpr WqmmConfig MXA8W4_NZNK = {false, true, QuantType::MX, false, QuantType::NONE, CubeFormat::NZ};

struct BasicBlockControlParam {
    uint64_t processId;
    uint64_t mSize;
    uint64_t mL1Size;
    uint64_t curBasicBlockId;
    uint64_t basicBlockLimit;
    uint64_t mOffset;
    uint64_t nOffset;
};

struct BasicBlockOffsetParam {
    uint64_t mL1Size;
    uint64_t kaL1Size;
    uint64_t kbL1Size;
    uint64_t nL1Size;

    uint64_t mOffset;
    uint64_t nOffset;

    uint64_t mSize;
    uint64_t kSize;
    uint64_t nSize;
    uint64_t kAlign;
    uint64_t nAlign;

    GM_ADDR yGmAddr;
};

struct VecAntiQuantConfig {
    uint64_t ubMte2BufferNum = 2;
    uint64_t ubMte2InnerSize = 512;
};
static constexpr VecAntiQuantConfig VEC_ANTIQUANT_CONFIG_DYNAMIC = {4, 0};

struct UbConsumeConfig {
    uint64_t ubVfBufferNum;
    uint64_t l1RequireVfComputeRealK;
    uint64_t l1RequireVfComputeRealN;
    uint64_t kWeightLowBitUbOffset;
    uint64_t nWeightLowBitUbOffset;
    uint64_t ubMxBiasNSize;
    bool calcMxBias = false;
    bool isBiasSingleVector = false;
};

struct L1ConsumeConfig {
    uint64_t l1SplitTwoVecExternalOffset;
    uint64_t l1RealExternalLen;
    uint64_t l1MxBiasSplitNOffset;
};

struct UbBufferInfo {
    uint64_t ubWeightOutputHighBitBufferNum;
    uint64_t weightInputLowbitUbTotalSize;
    uint64_t highBitDataUbTotalSize;
    uint64_t biasUbTotalSize;
    uint64_t biasReducedUbTotalSize;
    uint64_t antiQuantScaleUbTotalSize;
    uint64_t antiQuantScaleAfterCastUbTotalSize;
    uint64_t antiQuantOffsetUbTotalSize;
    uint64_t weightInputLowBitUbSingleBufferSize;
    uint64_t antiQuantScaleUbSingleBufferSize;
    uint64_t antiQuantScaleAfterCastUbSingleBufferSize;
    uint64_t biasUbSingleBufferSize;
    uint64_t biasReducedSingleBufferSize;
    uint64_t antiQuantOffsetUbSingleBufferSize;
    uint64_t highBitDataUbSingleBufferSize;
    uint32_t antiQuantScaleMaskBufferSize;
};

template <const VecAntiQuantConfig& vecConfig>
__aicore__ constexpr UbBufferInfo GetMxA8W4NzBufferInfo()
{
    return {.ubWeightOutputHighBitBufferNum = QUADRUPLE_BUFFER_NUM,
            .weightInputLowbitUbTotalSize = 64 * GetKBUnit<int8_t>(), // 64KB
            .highBitDataUbTotalSize = 128 * GetKBUnit<int8_t>(),      // 128KB
            .biasUbTotalSize = 2 * GetKBUnit<half>(),                 // 2KB
            .biasReducedUbTotalSize = 2 * GetKBUnit<half>(),          // 2KB
            .antiQuantScaleUbTotalSize = 0,
            .antiQuantScaleAfterCastUbTotalSize = 0,
            .antiQuantOffsetUbTotalSize = 0,
            .weightInputLowBitUbSingleBufferSize = 64 * GetKBUnit<int8_t>() / vecConfig.ubMte2BufferNum,
            .antiQuantScaleUbSingleBufferSize = 0,
            .antiQuantScaleAfterCastUbSingleBufferSize = 0,
            .biasUbSingleBufferSize = 2 * GetKBUnit<half>() / vecConfig.ubMte2BufferNum,
            .biasReducedSingleBufferSize = 2 * GetKBUnit<half>() / vecConfig.ubMte2BufferNum,
            .antiQuantOffsetUbSingleBufferSize = 0,
            .highBitDataUbSingleBufferSize = 128 * GetKBUnit<int8_t>() / QUADRUPLE_BUFFER_NUM,
            .antiQuantScaleMaskBufferSize = 0};
}

template <typename xType, const WqmmConfig& wqmmConfig, const VecAntiQuantConfig& vecConfig>
__aicore__ constexpr UbBufferInfo GetBufferConfig()
{
    return GetMxA8W4NzBufferInfo<vecConfig>();
}

struct VfConfig {
    uint64_t vfNStandardLen;
    uint64_t vfKStandardLen;
};

template <typename xType, const WqmmConfig& wqmmConfig, const VecAntiQuantConfig& vecConfig>
__aicore__ constexpr VfConfig GetVfConfig()
{
    return {.vfNStandardLen = 256, .vfKStandardLen = 64};
}
} // namespace WeightQuantBatchMatmulV2::Arch35
#endif // GMMFR_BASIC_BLOCK_CONFIG_H
