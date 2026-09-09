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
 * \file quant_sparse_flash_mla_common_arch35.h
 * \brief
 */
#ifndef QUANT_SPARSE_FLASH_MLA_COMMON_ARCH35_H
#define QUANT_SPARSE_FLASH_MLA_COMMON_ARCH35_H
#include <type_traits>
#include "kernel_tiling/kernel_tiling.h"
#include "../quant_sparse_flash_mla_common.h"
#include "../../../sparse_flash_mla/op_kernel/arch35/common/smla_common_defs.h"

namespace BaseApi {
using AttentionCommon::Align2Func;
using AttentionCommon::Align8Func;
using AttentionCommon::Align16Func;
using AttentionCommon::Align64Func;

__aicore__ constexpr uint64_t Align32Func(uint64_t data)
{
    return (data + 31UL) >> 5UL << 5UL; // 向上16对齐, +15移位4
}
} // namespace BaseApi

#define TEMPLATE_INTF \
    template <typename Q_T, typename KV_T, typename T, typename OUTPUT_T, bool isFd, bool isPa, QSMLA_LAYOUT LAYOUT_T, \
              QSMLA_LAYOUT KV_LAYOUT_T, QSMLATemplateMode TEMPLATE_MODE, bool IS_SPLIT_G, bool IS_VEC_S2PHYADDR>

#define TEMPLATE_INTF_ARGS \
    Q_T, KV_T, T, OUTPUT_T, isFd, isPa, LAYOUT_T, KV_LAYOUT_T, TEMPLATE_MODE, IS_SPLIT_G, IS_VEC_S2PHYADDR

#define CUBE_BLOCK_TRAITS_TYPE_FIELDS(X) \
    X(Q_T) \
    X(KV_T) \
    X(T) \
    X(OUTPUT_T)

#define CUBE_BLOCK_TRAITS_CONST_FIELDS(X) \
    X(isFd, bool, false) \
    X(isPa, bool, true) \
    X(LAYOUT_T, QSMLA_LAYOUT, QSMLA_LAYOUT::BSND) \
    X(KV_LAYOUT_T, QSMLA_LAYOUT, QSMLA_LAYOUT::PA_BBND) \
    X(TEMPLATE_MODE, QSMLATemplateMode, QSMLATemplateMode::CSA_TEMPLATE_MODE) \
    X(IS_SPLIT_G, bool, false) \
    X(IS_VEC_S2PHYADDR, bool, false)

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

/* 3. 生成有默认值的Args */
#define GEN_ARG_NAME(name, ...) name,
#define TEMPLATE_ARGS \
    CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_ARG_NAME) \
    CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_ARG_NAME) end

#endif
