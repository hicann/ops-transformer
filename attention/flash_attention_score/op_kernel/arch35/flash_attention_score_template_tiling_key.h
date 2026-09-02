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
 * \file flash_attention_score_template_tiling_key.h
 * \brief
 */

#ifndef TEMPLATE_TILING_KEY_FA_H_
#define TEMPLATE_TILING_KEY_FA_H_
#include "ascendc/host_api/tiling/template_argument.h"

// kernel通过宏定义隔离dtype编译tilingkey，降低耗时。tiling侧没有相关宏
#ifndef ORIG_DTYPE_QUERY
#define ORIG_DTYPE_QUERY (-1)
#endif

#define ASCENDC_TPL_1_BW 1
#define ASCENDC_TPL_3_BW 3
#define ASCENDC_TPL_10_BW 10
#define ASCENDC_TPL_12_BW 12

// 可表示的tilingkey范围为64bit，注意不能超过限制
ASCENDC_TPL_ARGS_DECL(FlashAttentionScore, // 算子唯一标识，可以opType保持一致
                                           // bit:1-0 1: 空tensor场景
                      ASCENDC_TPL_UINT_DECL(KernelTypeKey, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST, 0, 1),
                      // bit:3-2 ImplMode 0: AA_HIGH_PRECISION
                      //          1: AA_HIGH_PERFORMANCE
                      //          2: AA_INVALID_LINE_HIGH_PRECISION
                      ASCENDC_TPL_UINT_DECL(ImplMode, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST, 0, 1, 2),
                      // bit:7-4 Layout 0: BSND --> LayoutTypeEnum::None
                      //        1: SBND
                      //        2: BNSD
                      //        3: TND  --> LayoutTypeEnum::LAYOUT_TND
                      ASCENDC_TPL_UINT_DECL(Layout, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3, 4),
                      // bit:17-8
                      ASCENDC_TPL_UINT_DECL(S1TemplateType, ASCENDC_TPL_10_BW, ASCENDC_TPL_UI_LIST, 0, 16, 64, 128,
                                            256),
                      // bit:27-18
                      ASCENDC_TPL_UINT_DECL(S2TemplateType, ASCENDC_TPL_10_BW, ASCENDC_TPL_UI_LIST, 0, 16, 32, 64, 128,
                                            256, 512),
                      // bit:39-28
                      ASCENDC_TPL_UINT_DECL(DTemplateType, ASCENDC_TPL_12_BW, ASCENDC_TPL_UI_LIST, 0, 16, 32, 48, 64,
                                            80, 96, 128, 160, 192, 256, 768),
                      // bit:51-40
                      ASCENDC_TPL_UINT_DECL(DvTemplateType, ASCENDC_TPL_12_BW, ASCENDC_TPL_UI_LIST, 0, 16, 32, 48, 64,
                                            80, 96, 128, 160, 192, 256, 768),
                      // bit:55-52 PseMode 0: PSE_OUTER_MUL_ADD_TYPE
                      //         1: PSE_OUTER_ADD_MUL_TYPE
                      //         2: PSE_INNER_MUL_ADD_TYPE
                      //         3: PSE_INNER_MUL_ADD_SQRT_TYPE
                      //         4: PSE_INVALID_TYPE
                      //         9: PSE_NONE_TYPE
                      ASCENDC_TPL_UINT_DECL(PseMode, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3, 4, 9),
                      // bit:56
                      ASCENDC_TPL_UINT_DECL(HasAtten, ASCENDC_TPL_1_BW, ASCENDC_TPL_UI_LIST, 0, 1),
                      // bit:57
                      ASCENDC_TPL_UINT_DECL(HasDrop, ASCENDC_TPL_1_BW, ASCENDC_TPL_UI_LIST, 0, 1),
                      // bit:58
                      ASCENDC_TPL_UINT_DECL(HasRope, ASCENDC_TPL_1_BW, ASCENDC_TPL_UI_LIST, 0, 1),
                      // bit:60-59
                      ASCENDC_TPL_UINT_DECL(OutDtype, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST, 0, 1, 2),
                      // bit:61
                      ASCENDC_TPL_UINT_DECL(Regbase, ASCENDC_TPL_1_BW, ASCENDC_TPL_UI_LIST, 0, 1),
                      // bit:62
                      ASCENDC_TPL_UINT_DECL(optionalDn, ASCENDC_TPL_1_BW, ASCENDC_TPL_UI_LIST, 0, 1), );

// DvTemplateType设置为0，表示DvTemplateType = DTemplateType
ASCENDC_TPL_SEL(
// BF16/FP16    400 = 2 * (20+40+60+80) AA_HIGH_PRECISION LayoutTypeEnum::None 64_128
#if (ORIG_DTYPE_QUERY == -1) || (ORIG_DTYPE_QUERY == DT_BF16) || (ORIG_DTYPE_QUERY == DT_FLOAT16)
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(KernelTypeKey, ASCENDC_TPL_UI_LIST, 0),
                         ASCENDC_TPL_UINT_SEL(ImplMode, ASCENDC_TPL_UI_LIST, 0),
                         ASCENDC_TPL_UINT_SEL(Layout, ASCENDC_TPL_UI_LIST, 1, 2, 3),
                         ASCENDC_TPL_UINT_SEL(S1TemplateType, ASCENDC_TPL_UI_LIST, 64),
                         ASCENDC_TPL_UINT_SEL(S2TemplateType, ASCENDC_TPL_UI_LIST, 128),
                         ASCENDC_TPL_UINT_SEL(DTemplateType, ASCENDC_TPL_UI_LIST, 192),
                         ASCENDC_TPL_UINT_SEL(DvTemplateType, ASCENDC_TPL_UI_LIST, 128),
                         ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3, 9),
                         ASCENDC_TPL_UINT_SEL(HasAtten, ASCENDC_TPL_UI_LIST, 0, 1),
                         ASCENDC_TPL_UINT_SEL(HasDrop, ASCENDC_TPL_UI_LIST, 0, 1),
                         ASCENDC_TPL_UINT_SEL(HasRope, ASCENDC_TPL_UI_LIST, 0, 1),
                         ASCENDC_TPL_UINT_SEL(OutDtype, ASCENDC_TPL_UI_LIST, 0),
                         ASCENDC_TPL_UINT_SEL(Regbase, ASCENDC_TPL_UI_LIST, 1),
                         ASCENDC_TPL_UINT_SEL(optionalDn, ASCENDC_TPL_UI_LIST, 0),
                         ASCENDC_TPL_TILING_STRUCT_SEL(optiling::FlashAttentionScoreSimplifiedTilingData)),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(KernelTypeKey, ASCENDC_TPL_UI_LIST, 0),
                         ASCENDC_TPL_UINT_SEL(ImplMode, ASCENDC_TPL_UI_LIST, 0),
                         ASCENDC_TPL_UINT_SEL(Layout, ASCENDC_TPL_UI_LIST, 1, 2, 3),
                         ASCENDC_TPL_UINT_SEL(S1TemplateType, ASCENDC_TPL_UI_LIST, 64),
                         ASCENDC_TPL_UINT_SEL(S2TemplateType, ASCENDC_TPL_UI_LIST, 128),
                         ASCENDC_TPL_UINT_SEL(DTemplateType, ASCENDC_TPL_UI_LIST, 64, 128, 192, 256),
                         ASCENDC_TPL_UINT_SEL(DvTemplateType, ASCENDC_TPL_UI_LIST, 0),
                         ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3, 9),
                         ASCENDC_TPL_UINT_SEL(HasAtten, ASCENDC_TPL_UI_LIST, 0, 1),
                         ASCENDC_TPL_UINT_SEL(HasDrop, ASCENDC_TPL_UI_LIST, 0, 1),
                         ASCENDC_TPL_UINT_SEL(HasRope, ASCENDC_TPL_UI_LIST, 0),
                         ASCENDC_TPL_UINT_SEL(OutDtype, ASCENDC_TPL_UI_LIST, 0),
                         ASCENDC_TPL_UINT_SEL(Regbase, ASCENDC_TPL_UI_LIST, 1),
                         ASCENDC_TPL_UINT_SEL(optionalDn, ASCENDC_TPL_UI_LIST, 0),
                         ASCENDC_TPL_TILING_STRUCT_SEL(optiling::FlashAttentionScoreSimplifiedTilingData)),

    // BF16/FP16    20 = 2 * (1+2+3+4) AA_HIGH_PRECISION LayoutTypeEnum::None 64_256
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(KernelTypeKey, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(ImplMode, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(Layout, ASCENDC_TPL_UI_LIST, 1, 2, 3),
        ASCENDC_TPL_UINT_SEL(S1TemplateType, ASCENDC_TPL_UI_LIST, 64),
        ASCENDC_TPL_UINT_SEL(S2TemplateType, ASCENDC_TPL_UI_LIST, 256),
        ASCENDC_TPL_UINT_SEL(DTemplateType, ASCENDC_TPL_UI_LIST, 192),
        ASCENDC_TPL_UINT_SEL(DvTemplateType, ASCENDC_TPL_UI_LIST, 128),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, 9), ASCENDC_TPL_UINT_SEL(HasAtten, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(HasDrop, ASCENDC_TPL_UI_LIST, 0), ASCENDC_TPL_UINT_SEL(HasRope, ASCENDC_TPL_UI_LIST, 0, 1),
        ASCENDC_TPL_UINT_SEL(OutDtype, ASCENDC_TPL_UI_LIST, 0), ASCENDC_TPL_UINT_SEL(Regbase, ASCENDC_TPL_UI_LIST, 1),
        ASCENDC_TPL_UINT_SEL(optionalDn, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_TILING_STRUCT_SEL(optiling::FlashAttentionScoreSimplifiedTilingData)),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(KernelTypeKey, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(ImplMode, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(Layout, ASCENDC_TPL_UI_LIST, 1, 2, 3),
        ASCENDC_TPL_UINT_SEL(S1TemplateType, ASCENDC_TPL_UI_LIST, 64),
        ASCENDC_TPL_UINT_SEL(S2TemplateType, ASCENDC_TPL_UI_LIST, 256),
        ASCENDC_TPL_UINT_SEL(DTemplateType, ASCENDC_TPL_UI_LIST, 64, 128, 192, 256),
        ASCENDC_TPL_UINT_SEL(DvTemplateType, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, 9), ASCENDC_TPL_UINT_SEL(HasAtten, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(HasDrop, ASCENDC_TPL_UI_LIST, 0), ASCENDC_TPL_UINT_SEL(HasRope, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(OutDtype, ASCENDC_TPL_UI_LIST, 0), ASCENDC_TPL_UINT_SEL(Regbase, ASCENDC_TPL_UI_LIST, 1),
        ASCENDC_TPL_UINT_SEL(optionalDn, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_TILING_STRUCT_SEL(optiling::FlashAttentionScoreSimplifiedTilingData)),

    // BF16/FP16    1002 = 2*(40+80+120+160+100+1) AA_HIGH_PRECISION LayoutTypeEnum::None   128_128/256
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(KernelTypeKey, ASCENDC_TPL_UI_LIST, 0),
                         ASCENDC_TPL_UINT_SEL(ImplMode, ASCENDC_TPL_UI_LIST, 0),
                         ASCENDC_TPL_UINT_SEL(Layout, ASCENDC_TPL_UI_LIST, 1, 2, 3),
                         ASCENDC_TPL_UINT_SEL(S1TemplateType, ASCENDC_TPL_UI_LIST, 128),
                         ASCENDC_TPL_UINT_SEL(S2TemplateType, ASCENDC_TPL_UI_LIST, 128),
                         ASCENDC_TPL_UINT_SEL(DTemplateType, ASCENDC_TPL_UI_LIST, 192),
                         ASCENDC_TPL_UINT_SEL(DvTemplateType, ASCENDC_TPL_UI_LIST, 128),
                         ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3, 9),
                         ASCENDC_TPL_UINT_SEL(HasAtten, ASCENDC_TPL_UI_LIST, 0, 1),
                         ASCENDC_TPL_UINT_SEL(HasDrop, ASCENDC_TPL_UI_LIST, 0, 1),
                         ASCENDC_TPL_UINT_SEL(HasRope, ASCENDC_TPL_UI_LIST, 0, 1),
                         ASCENDC_TPL_UINT_SEL(OutDtype, ASCENDC_TPL_UI_LIST, 0),
                         ASCENDC_TPL_UINT_SEL(Regbase, ASCENDC_TPL_UI_LIST, 1),
                         ASCENDC_TPL_UINT_SEL(optionalDn, ASCENDC_TPL_UI_LIST, 0),
                         ASCENDC_TPL_TILING_STRUCT_SEL(optiling::FlashAttentionScoreSimplifiedTilingData)),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(KernelTypeKey, ASCENDC_TPL_UI_LIST, 0),
                         ASCENDC_TPL_UINT_SEL(ImplMode, ASCENDC_TPL_UI_LIST, 0),
                         ASCENDC_TPL_UINT_SEL(Layout, ASCENDC_TPL_UI_LIST, 1, 2, 3),
                         ASCENDC_TPL_UINT_SEL(S1TemplateType, ASCENDC_TPL_UI_LIST, 128),
                         ASCENDC_TPL_UINT_SEL(S2TemplateType, ASCENDC_TPL_UI_LIST, 128),
                         ASCENDC_TPL_UINT_SEL(DTemplateType, ASCENDC_TPL_UI_LIST, 64, 128, 192, 256, 768),
                         ASCENDC_TPL_UINT_SEL(DvTemplateType, ASCENDC_TPL_UI_LIST, 0),
                         ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3, 9),
                         ASCENDC_TPL_UINT_SEL(HasAtten, ASCENDC_TPL_UI_LIST, 0, 1),
                         ASCENDC_TPL_UINT_SEL(HasDrop, ASCENDC_TPL_UI_LIST, 0, 1),
                         ASCENDC_TPL_UINT_SEL(HasRope, ASCENDC_TPL_UI_LIST, 0),
                         ASCENDC_TPL_UINT_SEL(OutDtype, ASCENDC_TPL_UI_LIST, 0),
                         ASCENDC_TPL_UINT_SEL(Regbase, ASCENDC_TPL_UI_LIST, 1),
                         ASCENDC_TPL_UINT_SEL(optionalDn, ASCENDC_TPL_UI_LIST, 0),
                         ASCENDC_TPL_TILING_STRUCT_SEL(optiling::FlashAttentionScoreSimplifiedTilingData)),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(KernelTypeKey, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(ImplMode, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(Layout, ASCENDC_TPL_UI_LIST, 1, 2, 3),
        ASCENDC_TPL_UINT_SEL(S1TemplateType, ASCENDC_TPL_UI_LIST, 128),
        ASCENDC_TPL_UINT_SEL(S2TemplateType, ASCENDC_TPL_UI_LIST, 256),
        ASCENDC_TPL_UINT_SEL(DTemplateType, ASCENDC_TPL_UI_LIST, 64),
        ASCENDC_TPL_UINT_SEL(DvTemplateType, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, 9), ASCENDC_TPL_UINT_SEL(HasAtten, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(HasDrop, ASCENDC_TPL_UI_LIST, 0), ASCENDC_TPL_UINT_SEL(HasRope, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(OutDtype, ASCENDC_TPL_UI_LIST, 0), ASCENDC_TPL_UINT_SEL(Regbase, ASCENDC_TPL_UI_LIST, 1),
        ASCENDC_TPL_UINT_SEL(optionalDn, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_TILING_STRUCT_SEL(optiling::FlashAttentionScoreSimplifiedTilingData)),

    // BF16/FP16    1420 = 2 * (20+40+60+80+1+2+3+4+(40+80+120+160+100)) AA_HIGH_PRECISION LayoutTypeEnum::LAYOUT_TND
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(KernelTypeKey, ASCENDC_TPL_UI_LIST, 0),
                         ASCENDC_TPL_UINT_SEL(ImplMode, ASCENDC_TPL_UI_LIST, 0),
                         ASCENDC_TPL_UINT_SEL(Layout, ASCENDC_TPL_UI_LIST, 4),
                         ASCENDC_TPL_UINT_SEL(S1TemplateType, ASCENDC_TPL_UI_LIST, 64),
                         ASCENDC_TPL_UINT_SEL(S2TemplateType, ASCENDC_TPL_UI_LIST, 128),
                         ASCENDC_TPL_UINT_SEL(DTemplateType, ASCENDC_TPL_UI_LIST, 192),
                         ASCENDC_TPL_UINT_SEL(DvTemplateType, ASCENDC_TPL_UI_LIST, 128),
                         ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3, 9),
                         ASCENDC_TPL_UINT_SEL(HasAtten, ASCENDC_TPL_UI_LIST, 0, 1),
                         ASCENDC_TPL_UINT_SEL(HasDrop, ASCENDC_TPL_UI_LIST, 0, 1),
                         ASCENDC_TPL_UINT_SEL(HasRope, ASCENDC_TPL_UI_LIST, 0, 1),
                         ASCENDC_TPL_UINT_SEL(OutDtype, ASCENDC_TPL_UI_LIST, 0),
                         ASCENDC_TPL_UINT_SEL(Regbase, ASCENDC_TPL_UI_LIST, 1),
                         ASCENDC_TPL_UINT_SEL(optionalDn, ASCENDC_TPL_UI_LIST, 0),
                         ASCENDC_TPL_TILING_STRUCT_SEL(optiling::FlashAttentionScoreSimplifiedTilingData)),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(KernelTypeKey, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(ImplMode, ASCENDC_TPL_UI_LIST, 0), ASCENDC_TPL_UINT_SEL(Layout, ASCENDC_TPL_UI_LIST, 4),
        ASCENDC_TPL_UINT_SEL(S1TemplateType, ASCENDC_TPL_UI_LIST, 64),
        ASCENDC_TPL_UINT_SEL(S2TemplateType, ASCENDC_TPL_UI_LIST, 128),
        ASCENDC_TPL_UINT_SEL(DTemplateType, ASCENDC_TPL_UI_LIST, 64, 128, 192, 256),
        ASCENDC_TPL_UINT_SEL(DvTemplateType, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3, 9),
        ASCENDC_TPL_UINT_SEL(HasAtten, ASCENDC_TPL_UI_LIST, 0, 1),
        ASCENDC_TPL_UINT_SEL(HasDrop, ASCENDC_TPL_UI_LIST, 0, 1), ASCENDC_TPL_UINT_SEL(HasRope, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(OutDtype, ASCENDC_TPL_UI_LIST, 0), ASCENDC_TPL_UINT_SEL(Regbase, ASCENDC_TPL_UI_LIST, 1),
        ASCENDC_TPL_UINT_SEL(optionalDn, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_TILING_STRUCT_SEL(optiling::FlashAttentionScoreSimplifiedTilingData)),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(KernelTypeKey, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(ImplMode, ASCENDC_TPL_UI_LIST, 0), ASCENDC_TPL_UINT_SEL(Layout, ASCENDC_TPL_UI_LIST, 4),
        ASCENDC_TPL_UINT_SEL(S1TemplateType, ASCENDC_TPL_UI_LIST, 64),
        ASCENDC_TPL_UINT_SEL(S2TemplateType, ASCENDC_TPL_UI_LIST, 256),
        ASCENDC_TPL_UINT_SEL(DTemplateType, ASCENDC_TPL_UI_LIST, 192),
        ASCENDC_TPL_UINT_SEL(DvTemplateType, ASCENDC_TPL_UI_LIST, 128),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, 9), ASCENDC_TPL_UINT_SEL(HasAtten, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(HasDrop, ASCENDC_TPL_UI_LIST, 0), ASCENDC_TPL_UINT_SEL(HasRope, ASCENDC_TPL_UI_LIST, 0, 1),
        ASCENDC_TPL_UINT_SEL(OutDtype, ASCENDC_TPL_UI_LIST, 0), ASCENDC_TPL_UINT_SEL(Regbase, ASCENDC_TPL_UI_LIST, 1),
        ASCENDC_TPL_UINT_SEL(optionalDn, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_TILING_STRUCT_SEL(optiling::FlashAttentionScoreSimplifiedTilingData)),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(KernelTypeKey, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(ImplMode, ASCENDC_TPL_UI_LIST, 0), ASCENDC_TPL_UINT_SEL(Layout, ASCENDC_TPL_UI_LIST, 4),
        ASCENDC_TPL_UINT_SEL(S1TemplateType, ASCENDC_TPL_UI_LIST, 64),
        ASCENDC_TPL_UINT_SEL(S2TemplateType, ASCENDC_TPL_UI_LIST, 256),
        ASCENDC_TPL_UINT_SEL(DTemplateType, ASCENDC_TPL_UI_LIST, 64, 128, 192, 256),
        ASCENDC_TPL_UINT_SEL(DvTemplateType, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, 9), ASCENDC_TPL_UINT_SEL(HasAtten, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(HasDrop, ASCENDC_TPL_UI_LIST, 0), ASCENDC_TPL_UINT_SEL(HasRope, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(OutDtype, ASCENDC_TPL_UI_LIST, 0), ASCENDC_TPL_UINT_SEL(Regbase, ASCENDC_TPL_UI_LIST, 1),
        ASCENDC_TPL_UINT_SEL(optionalDn, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_TILING_STRUCT_SEL(optiling::FlashAttentionScoreSimplifiedTilingData)),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(KernelTypeKey, ASCENDC_TPL_UI_LIST, 0),
                         ASCENDC_TPL_UINT_SEL(ImplMode, ASCENDC_TPL_UI_LIST, 0),
                         ASCENDC_TPL_UINT_SEL(Layout, ASCENDC_TPL_UI_LIST, 4),
                         ASCENDC_TPL_UINT_SEL(S1TemplateType, ASCENDC_TPL_UI_LIST, 128),
                         ASCENDC_TPL_UINT_SEL(S2TemplateType, ASCENDC_TPL_UI_LIST, 128),
                         ASCENDC_TPL_UINT_SEL(DTemplateType, ASCENDC_TPL_UI_LIST, 192),
                         ASCENDC_TPL_UINT_SEL(DvTemplateType, ASCENDC_TPL_UI_LIST, 128),
                         ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3, 9),
                         ASCENDC_TPL_UINT_SEL(HasAtten, ASCENDC_TPL_UI_LIST, 0, 1),
                         ASCENDC_TPL_UINT_SEL(HasDrop, ASCENDC_TPL_UI_LIST, 0, 1),
                         ASCENDC_TPL_UINT_SEL(HasRope, ASCENDC_TPL_UI_LIST, 0, 1),
                         ASCENDC_TPL_UINT_SEL(OutDtype, ASCENDC_TPL_UI_LIST, 0),
                         ASCENDC_TPL_UINT_SEL(Regbase, ASCENDC_TPL_UI_LIST, 1),
                         ASCENDC_TPL_UINT_SEL(optionalDn, ASCENDC_TPL_UI_LIST, 0),
                         ASCENDC_TPL_TILING_STRUCT_SEL(optiling::FlashAttentionScoreSimplifiedTilingData)),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(KernelTypeKey, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(ImplMode, ASCENDC_TPL_UI_LIST, 0), ASCENDC_TPL_UINT_SEL(Layout, ASCENDC_TPL_UI_LIST, 4),
        ASCENDC_TPL_UINT_SEL(S1TemplateType, ASCENDC_TPL_UI_LIST, 128),
        ASCENDC_TPL_UINT_SEL(S2TemplateType, ASCENDC_TPL_UI_LIST, 128),
        ASCENDC_TPL_UINT_SEL(DTemplateType, ASCENDC_TPL_UI_LIST, 64, 128, 192, 256, 768),
        ASCENDC_TPL_UINT_SEL(DvTemplateType, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3, 9),
        ASCENDC_TPL_UINT_SEL(HasAtten, ASCENDC_TPL_UI_LIST, 0, 1),
        ASCENDC_TPL_UINT_SEL(HasDrop, ASCENDC_TPL_UI_LIST, 0, 1), ASCENDC_TPL_UINT_SEL(HasRope, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(OutDtype, ASCENDC_TPL_UI_LIST, 0), ASCENDC_TPL_UINT_SEL(Regbase, ASCENDC_TPL_UI_LIST, 1),
        ASCENDC_TPL_UINT_SEL(optionalDn, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_TILING_STRUCT_SEL(optiling::FlashAttentionScoreSimplifiedTilingData)),

    // BF16/FP16    500 = 2 * (20+40+60+80+50) AA_INVALID_LINE_HIGH_PRECISION LayoutTypeEnum::None
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(KernelTypeKey, ASCENDC_TPL_UI_LIST, 0),
                         ASCENDC_TPL_UINT_SEL(ImplMode, ASCENDC_TPL_UI_LIST, 2),
                         ASCENDC_TPL_UINT_SEL(Layout, ASCENDC_TPL_UI_LIST, 1, 2, 3),
                         ASCENDC_TPL_UINT_SEL(S1TemplateType, ASCENDC_TPL_UI_LIST, 128),
                         ASCENDC_TPL_UINT_SEL(S2TemplateType, ASCENDC_TPL_UI_LIST, 128),
                         ASCENDC_TPL_UINT_SEL(DTemplateType, ASCENDC_TPL_UI_LIST, 192),
                         ASCENDC_TPL_UINT_SEL(DvTemplateType, ASCENDC_TPL_UI_LIST, 128),
                         ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3, 9),
                         ASCENDC_TPL_UINT_SEL(HasAtten, ASCENDC_TPL_UI_LIST, 1),
                         ASCENDC_TPL_UINT_SEL(HasDrop, ASCENDC_TPL_UI_LIST, 0, 1),
                         ASCENDC_TPL_UINT_SEL(HasRope, ASCENDC_TPL_UI_LIST, 0, 1),
                         ASCENDC_TPL_UINT_SEL(OutDtype, ASCENDC_TPL_UI_LIST, 0),
                         ASCENDC_TPL_UINT_SEL(Regbase, ASCENDC_TPL_UI_LIST, 1),
                         ASCENDC_TPL_UINT_SEL(optionalDn, ASCENDC_TPL_UI_LIST, 0),
                         ASCENDC_TPL_TILING_STRUCT_SEL(optiling::FlashAttentionScoreSimplifiedTilingData)),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(KernelTypeKey, ASCENDC_TPL_UI_LIST, 0),
                         ASCENDC_TPL_UINT_SEL(ImplMode, ASCENDC_TPL_UI_LIST, 2),
                         ASCENDC_TPL_UINT_SEL(Layout, ASCENDC_TPL_UI_LIST, 1, 2, 3),
                         ASCENDC_TPL_UINT_SEL(S1TemplateType, ASCENDC_TPL_UI_LIST, 128),
                         ASCENDC_TPL_UINT_SEL(S2TemplateType, ASCENDC_TPL_UI_LIST, 128),
                         ASCENDC_TPL_UINT_SEL(DTemplateType, ASCENDC_TPL_UI_LIST, 64, 128, 192, 256, 768),
                         ASCENDC_TPL_UINT_SEL(DvTemplateType, ASCENDC_TPL_UI_LIST, 0),
                         ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3, 9),
                         ASCENDC_TPL_UINT_SEL(HasAtten, ASCENDC_TPL_UI_LIST, 1),
                         ASCENDC_TPL_UINT_SEL(HasDrop, ASCENDC_TPL_UI_LIST, 0, 1),
                         ASCENDC_TPL_UINT_SEL(HasRope, ASCENDC_TPL_UI_LIST, 0),
                         ASCENDC_TPL_UINT_SEL(OutDtype, ASCENDC_TPL_UI_LIST, 0),
                         ASCENDC_TPL_UINT_SEL(Regbase, ASCENDC_TPL_UI_LIST, 1),
                         ASCENDC_TPL_UINT_SEL(optionalDn, ASCENDC_TPL_UI_LIST, 0),
                         ASCENDC_TPL_TILING_STRUCT_SEL(optiling::FlashAttentionScoreSimplifiedTilingData)),

    // BF16/FP16    500 = 2 * (20+40+60+80+50) AA_INVALID_LINE_HIGH_PRECISION LayoutTypeEnum::LAYOUT_TND
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(KernelTypeKey, ASCENDC_TPL_UI_LIST, 0),
                         ASCENDC_TPL_UINT_SEL(ImplMode, ASCENDC_TPL_UI_LIST, 2),
                         ASCENDC_TPL_UINT_SEL(Layout, ASCENDC_TPL_UI_LIST, 4),
                         ASCENDC_TPL_UINT_SEL(S1TemplateType, ASCENDC_TPL_UI_LIST, 128),
                         ASCENDC_TPL_UINT_SEL(S2TemplateType, ASCENDC_TPL_UI_LIST, 128),
                         ASCENDC_TPL_UINT_SEL(DTemplateType, ASCENDC_TPL_UI_LIST, 192),
                         ASCENDC_TPL_UINT_SEL(DvTemplateType, ASCENDC_TPL_UI_LIST, 128),
                         ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3, 9),
                         ASCENDC_TPL_UINT_SEL(HasAtten, ASCENDC_TPL_UI_LIST, 1),
                         ASCENDC_TPL_UINT_SEL(HasDrop, ASCENDC_TPL_UI_LIST, 0, 1),
                         ASCENDC_TPL_UINT_SEL(HasRope, ASCENDC_TPL_UI_LIST, 0, 1),
                         ASCENDC_TPL_UINT_SEL(OutDtype, ASCENDC_TPL_UI_LIST, 0),
                         ASCENDC_TPL_UINT_SEL(Regbase, ASCENDC_TPL_UI_LIST, 1),
                         ASCENDC_TPL_UINT_SEL(optionalDn, ASCENDC_TPL_UI_LIST, 0),
                         ASCENDC_TPL_TILING_STRUCT_SEL(optiling::FlashAttentionScoreSimplifiedTilingData)),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(KernelTypeKey, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(ImplMode, ASCENDC_TPL_UI_LIST, 2), ASCENDC_TPL_UINT_SEL(Layout, ASCENDC_TPL_UI_LIST, 4),
        ASCENDC_TPL_UINT_SEL(S1TemplateType, ASCENDC_TPL_UI_LIST, 128),
        ASCENDC_TPL_UINT_SEL(S2TemplateType, ASCENDC_TPL_UI_LIST, 128),
        ASCENDC_TPL_UINT_SEL(DTemplateType, ASCENDC_TPL_UI_LIST, 64, 128, 192, 256, 768),
        ASCENDC_TPL_UINT_SEL(DvTemplateType, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3, 9),
        ASCENDC_TPL_UINT_SEL(HasAtten, ASCENDC_TPL_UI_LIST, 1),
        ASCENDC_TPL_UINT_SEL(HasDrop, ASCENDC_TPL_UI_LIST, 0, 1), ASCENDC_TPL_UINT_SEL(HasRope, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(OutDtype, ASCENDC_TPL_UI_LIST, 0), ASCENDC_TPL_UINT_SEL(Regbase, ASCENDC_TPL_UI_LIST, 1),
        ASCENDC_TPL_UINT_SEL(optionalDn, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_TILING_STRUCT_SEL(optiling::FlashAttentionScoreSimplifiedTilingData)),

    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(KernelTypeKey, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(ImplMode, ASCENDC_TPL_UI_LIST, 0, 2),
        ASCENDC_TPL_UINT_SEL(Layout, ASCENDC_TPL_UI_LIST, 1, 2, 3, 4),
        ASCENDC_TPL_UINT_SEL(S1TemplateType, ASCENDC_TPL_UI_LIST, 128),
        ASCENDC_TPL_UINT_SEL(S2TemplateType, ASCENDC_TPL_UI_LIST, 128),
        ASCENDC_TPL_UINT_SEL(DTemplateType, ASCENDC_TPL_UI_LIST, 16, 32, 48, 64, 80, 96, 128, 160, 192, 256),
        ASCENDC_TPL_UINT_SEL(DvTemplateType, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, 9), ASCENDC_TPL_UINT_SEL(HasAtten, ASCENDC_TPL_UI_LIST, 1),
        ASCENDC_TPL_UINT_SEL(HasDrop, ASCENDC_TPL_UI_LIST, 0), ASCENDC_TPL_UINT_SEL(HasRope, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(OutDtype, ASCENDC_TPL_UI_LIST, 0), ASCENDC_TPL_UINT_SEL(Regbase, ASCENDC_TPL_UI_LIST, 1),
        ASCENDC_TPL_UINT_SEL(optionalDn, ASCENDC_TPL_UI_LIST, 1),
        ASCENDC_TPL_TILING_STRUCT_SEL(optiling::FlashAttentionScoreSimplifiedTilingData)),
#endif

// FP32 DT_FLOAT     300 = 100+20+40+60+80 AA_HIGH_PRECISION LayoutTypeEnum::None
#if (ORIG_DTYPE_QUERY == -1) || (ORIG_DTYPE_QUERY == DT_FLOAT)
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(KernelTypeKey, ASCENDC_TPL_UI_LIST, 0),
                         ASCENDC_TPL_UINT_SEL(ImplMode, ASCENDC_TPL_UI_LIST, 0),
                         ASCENDC_TPL_UINT_SEL(Layout, ASCENDC_TPL_UI_LIST, 1, 2, 3),
                         ASCENDC_TPL_UINT_SEL(S1TemplateType, ASCENDC_TPL_UI_LIST, 64),
                         ASCENDC_TPL_UINT_SEL(S2TemplateType, ASCENDC_TPL_UI_LIST, 128),
                         ASCENDC_TPL_UINT_SEL(DTemplateType, ASCENDC_TPL_UI_LIST, 768),
                         ASCENDC_TPL_UINT_SEL(DvTemplateType, ASCENDC_TPL_UI_LIST, 0),
                         ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3, 9),
                         ASCENDC_TPL_UINT_SEL(HasAtten, ASCENDC_TPL_UI_LIST, 0, 1),
                         ASCENDC_TPL_UINT_SEL(HasDrop, ASCENDC_TPL_UI_LIST, 0, 1),
                         ASCENDC_TPL_UINT_SEL(HasRope, ASCENDC_TPL_UI_LIST, 0),
                         ASCENDC_TPL_UINT_SEL(OutDtype, ASCENDC_TPL_UI_LIST, 0),
                         ASCENDC_TPL_UINT_SEL(Regbase, ASCENDC_TPL_UI_LIST, 1),
                         ASCENDC_TPL_UINT_SEL(optionalDn, ASCENDC_TPL_UI_LIST, 0),
                         ASCENDC_TPL_TILING_STRUCT_SEL(optiling::FlashAttentionScoreSimplifiedTilingData)),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(KernelTypeKey, ASCENDC_TPL_UI_LIST, 0),
                         ASCENDC_TPL_UINT_SEL(ImplMode, ASCENDC_TPL_UI_LIST, 0),
                         ASCENDC_TPL_UINT_SEL(Layout, ASCENDC_TPL_UI_LIST, 1, 2, 3),
                         ASCENDC_TPL_UINT_SEL(S1TemplateType, ASCENDC_TPL_UI_LIST, 128),
                         ASCENDC_TPL_UINT_SEL(S2TemplateType, ASCENDC_TPL_UI_LIST, 128),
                         ASCENDC_TPL_UINT_SEL(DTemplateType, ASCENDC_TPL_UI_LIST, 192),
                         ASCENDC_TPL_UINT_SEL(DvTemplateType, ASCENDC_TPL_UI_LIST, 128),
                         ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3, 9),
                         ASCENDC_TPL_UINT_SEL(HasAtten, ASCENDC_TPL_UI_LIST, 0, 1),
                         ASCENDC_TPL_UINT_SEL(HasDrop, ASCENDC_TPL_UI_LIST, 0, 1),
                         ASCENDC_TPL_UINT_SEL(HasRope, ASCENDC_TPL_UI_LIST, 0, 1),
                         ASCENDC_TPL_UINT_SEL(OutDtype, ASCENDC_TPL_UI_LIST, 0),
                         ASCENDC_TPL_UINT_SEL(Regbase, ASCENDC_TPL_UI_LIST, 1),
                         ASCENDC_TPL_UINT_SEL(optionalDn, ASCENDC_TPL_UI_LIST, 0),
                         ASCENDC_TPL_TILING_STRUCT_SEL(optiling::FlashAttentionScoreSimplifiedTilingData)),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(KernelTypeKey, ASCENDC_TPL_UI_LIST, 0),
                         ASCENDC_TPL_UINT_SEL(ImplMode, ASCENDC_TPL_UI_LIST, 0),
                         ASCENDC_TPL_UINT_SEL(Layout, ASCENDC_TPL_UI_LIST, 1, 2, 3),
                         ASCENDC_TPL_UINT_SEL(S1TemplateType, ASCENDC_TPL_UI_LIST, 128),
                         ASCENDC_TPL_UINT_SEL(S2TemplateType, ASCENDC_TPL_UI_LIST, 128),
                         ASCENDC_TPL_UINT_SEL(DTemplateType, ASCENDC_TPL_UI_LIST, 64, 128, 192, 256),
                         ASCENDC_TPL_UINT_SEL(DvTemplateType, ASCENDC_TPL_UI_LIST, 0),
                         ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3, 9),
                         ASCENDC_TPL_UINT_SEL(HasAtten, ASCENDC_TPL_UI_LIST, 0, 1),
                         ASCENDC_TPL_UINT_SEL(HasDrop, ASCENDC_TPL_UI_LIST, 0, 1),
                         ASCENDC_TPL_UINT_SEL(HasRope, ASCENDC_TPL_UI_LIST, 0),
                         ASCENDC_TPL_UINT_SEL(OutDtype, ASCENDC_TPL_UI_LIST, 0),
                         ASCENDC_TPL_UINT_SEL(Regbase, ASCENDC_TPL_UI_LIST, 1),
                         ASCENDC_TPL_UINT_SEL(optionalDn, ASCENDC_TPL_UI_LIST, 0),
                         ASCENDC_TPL_TILING_STRUCT_SEL(optiling::FlashAttentionScoreSimplifiedTilingData)),

    // FP32     300 = 100+20+40+60+80 AA_HIGH_PRECISION LayoutTypeEnum::LAYOUT_TND
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(KernelTypeKey, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(ImplMode, ASCENDC_TPL_UI_LIST, 0), ASCENDC_TPL_UINT_SEL(Layout, ASCENDC_TPL_UI_LIST, 4),
        ASCENDC_TPL_UINT_SEL(S1TemplateType, ASCENDC_TPL_UI_LIST, 64),
        ASCENDC_TPL_UINT_SEL(S2TemplateType, ASCENDC_TPL_UI_LIST, 128),
        ASCENDC_TPL_UINT_SEL(DTemplateType, ASCENDC_TPL_UI_LIST, 768),
        ASCENDC_TPL_UINT_SEL(DvTemplateType, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3, 9),
        ASCENDC_TPL_UINT_SEL(HasAtten, ASCENDC_TPL_UI_LIST, 0, 1),
        ASCENDC_TPL_UINT_SEL(HasDrop, ASCENDC_TPL_UI_LIST, 0, 1), ASCENDC_TPL_UINT_SEL(HasRope, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(OutDtype, ASCENDC_TPL_UI_LIST, 0), ASCENDC_TPL_UINT_SEL(Regbase, ASCENDC_TPL_UI_LIST, 1),
        ASCENDC_TPL_UINT_SEL(optionalDn, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_TILING_STRUCT_SEL(optiling::FlashAttentionScoreSimplifiedTilingData)),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(KernelTypeKey, ASCENDC_TPL_UI_LIST, 0),
                         ASCENDC_TPL_UINT_SEL(ImplMode, ASCENDC_TPL_UI_LIST, 0),
                         ASCENDC_TPL_UINT_SEL(Layout, ASCENDC_TPL_UI_LIST, 4),
                         ASCENDC_TPL_UINT_SEL(S1TemplateType, ASCENDC_TPL_UI_LIST, 128),
                         ASCENDC_TPL_UINT_SEL(S2TemplateType, ASCENDC_TPL_UI_LIST, 128),
                         ASCENDC_TPL_UINT_SEL(DTemplateType, ASCENDC_TPL_UI_LIST, 192),
                         ASCENDC_TPL_UINT_SEL(DvTemplateType, ASCENDC_TPL_UI_LIST, 128),
                         ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3, 9),
                         ASCENDC_TPL_UINT_SEL(HasAtten, ASCENDC_TPL_UI_LIST, 0, 1),
                         ASCENDC_TPL_UINT_SEL(HasDrop, ASCENDC_TPL_UI_LIST, 0, 1),
                         ASCENDC_TPL_UINT_SEL(HasRope, ASCENDC_TPL_UI_LIST, 0, 1),
                         ASCENDC_TPL_UINT_SEL(OutDtype, ASCENDC_TPL_UI_LIST, 0),
                         ASCENDC_TPL_UINT_SEL(Regbase, ASCENDC_TPL_UI_LIST, 1),
                         ASCENDC_TPL_UINT_SEL(optionalDn, ASCENDC_TPL_UI_LIST, 0),
                         ASCENDC_TPL_TILING_STRUCT_SEL(optiling::FlashAttentionScoreSimplifiedTilingData)),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(KernelTypeKey, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(ImplMode, ASCENDC_TPL_UI_LIST, 0), ASCENDC_TPL_UINT_SEL(Layout, ASCENDC_TPL_UI_LIST, 4),
        ASCENDC_TPL_UINT_SEL(S1TemplateType, ASCENDC_TPL_UI_LIST, 128),
        ASCENDC_TPL_UINT_SEL(S2TemplateType, ASCENDC_TPL_UI_LIST, 128),
        ASCENDC_TPL_UINT_SEL(DTemplateType, ASCENDC_TPL_UI_LIST, 64, 128, 192, 256),
        ASCENDC_TPL_UINT_SEL(DvTemplateType, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3, 9),
        ASCENDC_TPL_UINT_SEL(HasAtten, ASCENDC_TPL_UI_LIST, 0, 1),
        ASCENDC_TPL_UINT_SEL(HasDrop, ASCENDC_TPL_UI_LIST, 0, 1), ASCENDC_TPL_UINT_SEL(HasRope, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(OutDtype, ASCENDC_TPL_UI_LIST, 0), ASCENDC_TPL_UINT_SEL(Regbase, ASCENDC_TPL_UI_LIST, 1),
        ASCENDC_TPL_UINT_SEL(optionalDn, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_TILING_STRUCT_SEL(optiling::FlashAttentionScoreSimplifiedTilingData)),
#endif

// FP8     48 = 3 * 16
#if (ORIG_DTYPE_QUERY == -1) || (ORIG_DTYPE_QUERY == DT_FLOAT8_E5M2) || (ORIG_DTYPE_QUERY == DT_FLOAT8_E4M3FN) || \
    (ORIG_DTYPE_QUERY == DT_HIFLOAT8)
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(KernelTypeKey, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(ImplMode, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(Layout, ASCENDC_TPL_UI_LIST, 1, 2, 3),
        ASCENDC_TPL_UINT_SEL(S1TemplateType, ASCENDC_TPL_UI_LIST, 128),
        ASCENDC_TPL_UINT_SEL(S2TemplateType, ASCENDC_TPL_UI_LIST, 512),
        ASCENDC_TPL_UINT_SEL(DTemplateType, ASCENDC_TPL_UI_LIST, 64, 128),
        ASCENDC_TPL_UINT_SEL(DvTemplateType, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, 9), ASCENDC_TPL_UINT_SEL(HasAtten, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(HasDrop, ASCENDC_TPL_UI_LIST, 0), ASCENDC_TPL_UINT_SEL(HasRope, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(OutDtype, ASCENDC_TPL_UI_LIST, 2), ASCENDC_TPL_UINT_SEL(Regbase, ASCENDC_TPL_UI_LIST, 1),
        ASCENDC_TPL_UINT_SEL(optionalDn, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_TILING_STRUCT_SEL(optiling::FlashAttentionScoreSimplifiedTilingData)),
    // invalid line in fp8
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(KernelTypeKey, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(ImplMode, ASCENDC_TPL_UI_LIST, 2),
        ASCENDC_TPL_UINT_SEL(Layout, ASCENDC_TPL_UI_LIST, 1, 2, 3),
        ASCENDC_TPL_UINT_SEL(S1TemplateType, ASCENDC_TPL_UI_LIST, 128),
        ASCENDC_TPL_UINT_SEL(S2TemplateType, ASCENDC_TPL_UI_LIST, 512),
        ASCENDC_TPL_UINT_SEL(DTemplateType, ASCENDC_TPL_UI_LIST, 64, 128),
        ASCENDC_TPL_UINT_SEL(DvTemplateType, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, 9), ASCENDC_TPL_UINT_SEL(HasAtten, ASCENDC_TPL_UI_LIST, 1),
        ASCENDC_TPL_UINT_SEL(HasDrop, ASCENDC_TPL_UI_LIST, 0), ASCENDC_TPL_UINT_SEL(HasRope, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(OutDtype, ASCENDC_TPL_UI_LIST, 2), ASCENDC_TPL_UINT_SEL(Regbase, ASCENDC_TPL_UI_LIST, 1),
        ASCENDC_TPL_UINT_SEL(optionalDn, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_TILING_STRUCT_SEL(optiling::FlashAttentionScoreSimplifiedTilingData)),
#endif

    // 空tensor
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(KernelTypeKey, ASCENDC_TPL_UI_LIST, 1),
        ASCENDC_TPL_UINT_SEL(ImplMode, ASCENDC_TPL_UI_LIST, 0), ASCENDC_TPL_UINT_SEL(Layout, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(S1TemplateType, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(S2TemplateType, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(DTemplateType, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(DvTemplateType, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, 0), ASCENDC_TPL_UINT_SEL(HasAtten, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(HasDrop, ASCENDC_TPL_UI_LIST, 0), ASCENDC_TPL_UINT_SEL(HasRope, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(OutDtype, ASCENDC_TPL_UI_LIST, 0), ASCENDC_TPL_UINT_SEL(Regbase, ASCENDC_TPL_UI_LIST, 1),
        ASCENDC_TPL_UINT_SEL(optionalDn, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_TILING_STRUCT_SEL(optiling::FlashAttentionScoreEmptyInputTilingDataRegbase)),

);
#endif
