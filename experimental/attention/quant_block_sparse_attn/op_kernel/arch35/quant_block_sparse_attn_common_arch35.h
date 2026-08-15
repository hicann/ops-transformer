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
 * \file quant_block_sparse_attn_common_arch35.h
 * \brief
 */
#ifndef QUANT_BLOCK_SPARSE_ATTN_COMMON_ARCH35_H
#define QUANT_BLOCK_SPARSE_ATTN_COMMON_ARCH35_H
#include <type_traits>
#include "kernel_tiling/kernel_tiling.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "lib/matmul_intf.h"
#include "stdarg.h"
#include "common/util_regbase.h"
#include "quant_block_sparse_attn_metadata.h"

using matmul::MatmulType;
using namespace AscendC;
using namespace regbaseutil;

constexpr static int64_t SPARSE_MODE_INT_DEFAULT = 2147483647;

constexpr uint32_t NEGATIVE_MIN_VALUE_FP32 = 0xFF7FFFFF;
constexpr uint32_t NEGATIVE_MIN_VALUE_FP16 = 0xFBFF;
constexpr uint32_t POSITIVE_MAX_VALUE_FP32 = 0x7F7FFFFF;
constexpr uint32_t POSITIVE_MAX_VALUE_FP16 = 0x7BFF;
constexpr int64_t FP8_QUANT_BLOCK_SIZE = 128;

constexpr int64_t attenMaskBN2GS1S2 = 0;
constexpr int64_t attenMaskBS1S2 = 1;

constexpr uint16_t SHIFT_NUM_6 = 6;
constexpr uint16_t ADD_NUM_63 = 63;

constexpr uint32_t CV_RATIO = 2;
constexpr uint64_t SYNC_MODE = 4;
constexpr uint64_t KB_TO_BYTES = 1024;
constexpr uint64_t L0C_SIZE = 256;
constexpr uint64_t FLOAT_BYTES = 4;

namespace BaseApi {
struct CubeCoordInfo {
    uint32_t curBIdx;
    uint32_t s1Coord;
    uint32_t s2Coord;
    int32_t sparseBlockIdx[2]; // QBSA: 当前迭代的 2 个 sparse block index
};

__aicore__ constexpr uint16_t Align64Func(uint16_t data)
{
    return (data + ADD_NUM_63) >> SHIFT_NUM_6 << SHIFT_NUM_6;
}

__aicore__ constexpr bool IsDn()
{
    return true;
}

__aicore__ constexpr TPosition GetC2Position()
{
    return TPosition::VECCALC;
}
} // namespace BaseApi

#define CHILD_SPEC_TEMPLATE_ARGS \
    INPUT_T, T, layout, kvLayout, s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType, hasAtten, OUTPUT_T, \
        hasLse, isPa, useDn

#define TEMPLATE_INTF \
    template <typename INPUT_T, typename T, QBSALayout layout, QBSALayout kvLayout, S1TemplateType s1TemplateType, \
              S2TemplateType s2TemplateType, DTemplateType dTemplateType, DTemplateType dVTemplateType, bool hasAtten, \
              typename OUTPUT_T, bool hasLse, bool isPa, bool useDn = true>

#define TEMPLATE_INTF_ARGS \
    INPUT_T, T, layout, kvLayout, s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType, hasAtten, OUTPUT_T, \
        hasLse, isPa, useDn

#define CUBE_BLOCK_TRAITS_TYPE_FIELDS(X) \
    X(INPUT_T) \
    X(T) \
    X(OUTPUT_T)

#define CUBE_BLOCK_TRAITS_CONST_FIELDS(X) \
    X(layout, QBSALayout, QBSALayout::TND) \
    X(kvLayout, QBSALayout, QBSALayout::PA_BNBD) \
    X(s1TemplateType, S1TemplateType, S1TemplateType::Aligned128) \
    X(s2TemplateType, S2TemplateType, S2TemplateType::Aligned128) \
    X(dTemplateType, DTemplateType, DTemplateType::Aligned128) \
    X(dVTemplateType, DTemplateType, DTemplateType::Aligned128) \
    X(hasAtten, bool, false) \
    X(hasLse, bool, false) \
    X(isPa, bool, false) \
    X(useDn, bool, false)

/* 1. 生成带默认值的模版Template */
#define GEN_TYPE_PARAM(name) typename name,
#define GEN_CONST_PARAM(name, type, default_val) type name = default_val,

#define TEMPLATES_DEF \
    template <CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_TYPE_PARAM) CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_CONST_PARAM) bool end = \
                  true>

/* 2. 生成不带带默认值的模版Template */
#define GEN_TEMPLATE_TYPE_NODEF(name) typename name,
#define GEN_TEMPLATE_CONST_NODEF(name, type, default_val) type name,

#define TEMPLATES_DEF_NO_DEFAULT \
    template <CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_TEMPLATE_TYPE_NODEF) \
                  CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_TEMPLATE_CONST_NODEF) bool end>

/* 3. 生成有默认值, 不带ChildClass的Args */
#define GEN_ARG_NAME(name, ...) name,
#define TEMPLATE_ARGS \
    CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_ARG_NAME) \
    CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_ARG_NAME) \
    end

/* 4. 生成BASE的有默认值的Template, BASE带ChildClass */

#define TEMPLATES_DEF_BASE \
    template <typename ChildClass, CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_TYPE_PARAM) \
                                       CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_CONST_PARAM) bool end = true>

/* 5. 生成BASE的没有默认值的Template, BASE带ChildClass */

#define TEMPLATES_DEF_BASE_NO_DEFAULT \
    template <typename ChildClass, CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_TEMPLATE_TYPE_NODEF) \
                                       CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_TEMPLATE_CONST_NODEF) bool end>

/* 6. 生成BASE的BaseArgs, BASE带ChildClass */
#define TEMPLATE_BASE_ARGS \
    ChildClass, CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_ARG_NAME) CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_ARG_NAME) end

#endif
