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
 * \file mc2_templates.h
 * \brief
 */
#ifndef MC2_TEMPLATES_H
#define MC2_TEMPLATES_H

// 嵌套 TilingData 成员地址获取宏与 Tiling 指针类型，供本模板库接口与各 kernel 实现共用；
// Tiling 指针类型带 MC2_ 前缀，避免与 SDK matmul_constant_tiling_struct.h 中的 using TILING_TYPE 冲突
#if defined(CONST_TILING)
#define GET_NESTED_TILING_DATA_MEMBER_ADDR(outerType, innerType, outerMember, innerMember, var, tiling) \
    const outerType *outerPtr##var = (const outerType *)(tiling); \
    const innerType *innerPtr##var = &(outerPtr##var->outerPtr##var); \
    const int32_t *(var) = (const int32_t)((const uint8_t *)&(innerPtr##var->innerMember))
#else
#define GET_NESTED_TILING_DATA_MEMBER_ADDR(outerType, innerType, outerMember, innerMember, var, tiling) \
    size_t outerOffset##var = (size_t)(&((outerType *)0)->outerMember); \
    size_t innerOffset##var = (size_t)(&((innerType *)0)->innerMember); \
    __gm__ int32_t *(var) = (__gm__ int32_t *)((__gm__ uint8_t *)(tiling) + outerOffset##var + innerOffset##var)
#endif

#if defined(CONST_TILING)
#define MC2_TILING_TYPE const int32_t
#else
#define MC2_TILING_TYPE __gm__ int32_t
#endif

#include "scheduler/a2av_gmm_scheduler.h"
#include "scheduler/gmm_a2av_scheduler.h"
#include "communication/hccl_a2av_op.h"
#include "compute/quant_grouped_matmul.h"
#include "common/a2av_common_tiling.h"

#endif // MC2_TEMPLATES_H
