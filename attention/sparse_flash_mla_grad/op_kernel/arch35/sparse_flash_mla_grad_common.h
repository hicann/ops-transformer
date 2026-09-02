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
 * \file sparse_flash_mla_grad_common.h
 * \brief
 */

#ifndef SPARSE_FLASH_MLA_GRAD_COMMON_H
#define SPARSE_FLASH_MLA_GRAD_COMMON_H

#include "sparse_flash_mla_grad_arch35_common.h"
#include "../../../common/op_kernel/buffers_policy.h"
using namespace fa_base_matmul;

constexpr int32_t PROCESS_KV_SIZE = 4096;
constexpr int32_t PROCESS_KV_SIZE_NO_DETER = 128;
constexpr uint8_t SYNC_C3_TO_V0_FLAG[2] = {7, 9};
constexpr uint8_t SYNC_V0_TO_C1_FLAG[2] = {0, 1};
constexpr uint8_t SYNC_C1_TO_V2_FLAG[2] = {0, 1};
constexpr uint8_t SYNC_C2_TO_V2_FLAG[2] = {2, 3};
constexpr uint8_t SYNC_V3_TO_C3_FLAG = 4;
constexpr uint8_t SYNC_V5_TO_C4_FLAG[2] = {7, 9};
constexpr uint8_t SYNC_V4_TO_C5_FLAG = 5;
constexpr uint8_t SYNC_C5_TO_V5_FLAG = 6;
constexpr uint8_t SYNC_C4_TO_V3_FLAG = 8;
constexpr uint8_t SYNC_C5_TO_V4_FLAG = 10;
constexpr uint8_t SCATTER_CUBE_SYNC_FLAG = 11;
constexpr uint8_t SCATTER_VEC_SYNC_FLAG = 12;
constexpr uint8_t SCATTER_SYNC_FLAG = 13;
constexpr uint8_t SCATTER_TO_FIX_SYNC_FLAG = 14;

// MM_IDX
constexpr uint8_t DQ_IDX = 0;
constexpr uint8_t DK_IDX = 1;
constexpr uint8_t DV_IDX = 2;

template <typename T, bool IS_WRITE_UB>
struct DqkvResPos {
    using PosType = typename std::conditional<IS_WRITE_UB, LocalTensor<T> &, GlobalTensor<T> &>::type;
};

// max(mm1, mm2, mm3) + mm4 + mm5
#define IS_DKV_RESIDENT_L0C(CUBE_BASEM, CUBE_BASEN, HEAD_DIM_ALIGN) \
    (((CUBE_BASEM) * (HEAD_DIM_ALIGN) * sizeof(float)) + ((CUBE_BASEN) * (HEAD_DIM_ALIGN) * sizeof(float)) + \
     ((CUBE_BASEN) > (HEAD_DIM_ALIGN) ? (CUBE_BASEM) * (CUBE_BASEN) * sizeof(float) : \
                                        (CUBE_BASEM) * (HEAD_DIM_ALIGN) * sizeof(float))) <= L0C_MAX_SIZE

#define SFagTilingType const optiling::smlag::SparseFlashMlaGradTilingDataRegbase *__restrict

#define CUBE_BLOCK_TRAITS_TYPE_FIELDS(X) \
    X(INPUT_TYPE) \
    X(CALC_TYPE) \
    X(OUTDTYPE)

#define CUBE_BLOCK_TRAITS_CONST_FIELDS(X) \
    X(IS_TND, bool, false) \
    X(s2TemplateType, S2TemplateType, S2TemplateType::Aligned128) \
    X(dTemplateType, DTemplateType, DTemplateType::Aligned128) \
    X(IsOriKVExist, bool, false) \
    X(IsCmpKVExist, bool, false) \
    X(IsOriKVSparse, bool, false) \
    X(IsCmpKVSparse, bool, false) \
    X(IsDETER, bool, false)

/* 1. 生成带默认值的模版Template */
#define GEN_TYPE_PARAM(name) typename name,
#define GEN_CONST_PARAM(name, type, default_val) type(name) = (default_val),
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
    CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_ARG_NAME) end

/* 4. 生成BASE的有默认值的Template, BASE带ChildClass*/
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

#endif // SPARSE_FLASH_MLA_GRAD_COMMON_H
