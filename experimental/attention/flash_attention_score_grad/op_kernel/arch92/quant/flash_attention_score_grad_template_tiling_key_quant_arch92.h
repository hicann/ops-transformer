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
 * \file flash_attention_score_grad_template_tiling_key_quant_arch92.h
 * \brief
 */

#ifndef TEMPLATE_TILING_KEY_FAG_QUANT_ARCH92_H_
#define TEMPLATE_TILING_KEY_FAG_QUANT_ARCH92_H_
#include "ascendc/host_api/tiling/template_argument.h"
#include "flash_attention_score_grad_tiling_data_regbase_arch92.h"

// kernel通过宏定义隔离dtype编译tilingkey，降低耗时。tiling侧没有相关宏
#ifndef ORIG_DTYPE_QUERY
#define ORIG_DTYPE_QUERY (-1)
#endif

#define ASCENDC_TPL_3_BW 3
#define ASCENDC_TPL_4_BW 4
#define ASCENDC_TPL_8_BW 8
#define ASCENDC_TPL_10_BW 10
#define ASCENDC_TPL_12_BW 12

// Match A5 naming: suffix flags are isDeter, isNewDeter, isTnd, isTndSwizzle.
// Keep all schema aliases used by the shared Host InitTilingData branches.
// Only FFFF and TFFF are registered below for the supported quant kernels.
using FagTilingWithTemplateQuantFFFF =
    optiling::fag::quant::FlashAttentionScoreGradTilingDataUs1s2Bbn2gs1s2Regbase<false, false, false, false>;
using FagTilingWithTemplateQuantFFTT =
    optiling::fag::quant::FlashAttentionScoreGradTilingDataUs1s2Bbn2gs1s2Regbase<false, false, true, true>;
using FagTilingWithTemplateQuantFFTF =
    optiling::fag::quant::FlashAttentionScoreGradTilingDataUs1s2Bbn2gs1s2Regbase<false, false, true, false>;

using FagTilingWithTemplateQuantTFFF =
    optiling::fag::quant::FlashAttentionScoreGradTilingDataUs1s2Bbn2gs1s2Regbase<true, false, false, false>;
using FagTilingWithTemplateQuantTFTF =
    optiling::fag::quant::FlashAttentionScoreGradTilingDataUs1s2Bbn2gs1s2Regbase<true, false, true, false>;
using FagTilingWithTemplateQuantTTTF =
    optiling::fag::quant::FlashAttentionScoreGradTilingDataUs1s2Bbn2gs1s2Regbase<true, true, true, false>;
using FagTilingWithTemplateQuantTTTT =
    optiling::fag::quant::FlashAttentionScoreGradTilingDataUs1s2Bbn2gs1s2Regbase<true, true, true, true>;

// 可表示的tilingkey范围为64bit，注意不能超过限制
ASCENDC_TPL_ARGS_DECL(FlashAttentionScoreGrad, // 算子唯一标识，可以opType保持一致
                                               // bit: 0 IsEmptyTensor
                      ASCENDC_TPL_BOOL_DECL(IsEmptyTensor, 0, 1),
                      // bit: 3-1 SplitAxis
                      //      0: BN2GS1S2
                      //      1: BN2
                      ASCENDC_TPL_UINT_DECL(SplitAxis, ASCENDC_TPL_3_BW, ASCENDC_TPL_UI_LIST, 0, 1),
                      // bit: 6-4 InputDType
                      //      1: FLOAT32
                      //      2: BFLOAT16
                      //      3: FLOAT16_PRECISION
                      //      4: FP8_E5M2
                      //      5: FP8_E4M3FN
                      //      6: HIFP8
                      ASCENDC_TPL_UINT_DECL(InputDType, ASCENDC_TPL_3_BW, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3, 6),
                      // bit: 7 IsTnd
                      //      0: DISABLE
                      //      1: ENABLE
                      ASCENDC_TPL_BOOL_DECL(IsTnd, 0, 1),
                      // bit: 8 DropValue
                      //      0: DISABLE
                      //      1: ENABLE
                      ASCENDC_TPL_BOOL_DECL(IsDrop, 0, 1),
                      // bit: 9 PseValue
                      //      0: DISABLE
                      //      1: ENABLE
                      ASCENDC_TPL_BOOL_DECL(IsPse, 0, 1),
                      // bit: 10 AttenMaskCfg
                      //      0: DISABLE
                      //      1: ENABLE
                      ASCENDC_TPL_BOOL_DECL(IsAttenMask, 0, 1),
                      // bit: 18-11 S1TemplateType
                      ASCENDC_TPL_UINT_DECL(S1TemplateNum, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_LIST, 0, 64, 256, 512),
                      // bit: 28-19 S2TemplateType
                      ASCENDC_TPL_UINT_DECL(S2TemplateNum, ASCENDC_TPL_10_BW, ASCENDC_TPL_UI_LIST, 0, 128, 512),
                      // bit: 40-29 DTemplateType
                      ASCENDC_TPL_UINT_DECL(DTemplateNum, ASCENDC_TPL_12_BW, ASCENDC_TPL_UI_LIST, 0, 64, 128, 192, 256,
                                            768),
                      // bit: 44-41 DeterType
                      //     0: NO_DETER
                      //     1: DETER_OLD
                      //     2: DETER_DENSE
                      //     3: DETER_CAUSAL
                      //     4: DETER_BAND
                      ASCENDC_TPL_UINT_DECL(DeterType, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3, 4),
                      // bit: 45 is N1 N2 equal
                      // 0: DISABLE
                      // 1: ENABLE
                      ASCENDC_TPL_BOOL_DECL(IsNEqual, 0, 1),
                      // bit: 46 IS_BN2_MULTIBLK
                      //     0: DISABLE
                      //     1: ENABLE
                      ASCENDC_TPL_BOOL_DECL(IS_BN2_MULTIBLK, 0, 1),
                      // bit: 47 IsDNoEqual
                      //      0: DISABLE
                      //      1: ENABLE
                      ASCENDC_TPL_BOOL_DECL(IsDNoEqual, 0, 1),
                      // bit: 50-48 OutDType
                      //      1: FLOAT32
                      //      2: BFLOAT16
                      //      3: FLOAT16_PRECISION
                      ASCENDC_TPL_UINT_DECL(OutDType, ASCENDC_TPL_3_BW, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3),
                      // bit: 51 IsSwizzle
                      //      0: DISABLE
                      //      1: ENABLE
                      ASCENDC_TPL_BOOL_DECL(IsSwizzle, 0, 1),
                      // bit: 52 HasRope
                      //      0: DISABLE
                      //      1: ENABLE
                      ASCENDC_TPL_BOOL_DECL(HasRope, 0, 1),
                      // bit: 53 IsRegbasePlatformValue
                      ASCENDC_TPL_BOOL_DECL(IsDavid, 0, 1));

ASCENDC_TPL_SEL(

#if (ORIG_DTYPE_QUERY == -1) || (ORIG_DTYPE_QUERY == DT_HIFLOAT8)
    // Arch92 dense HIFLOAT8 quant: separate ABI from the David kernels.
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_BOOL_SEL(IsEmptyTensor, 0),
                         ASCENDC_TPL_UINT_SEL(SplitAxis, ASCENDC_TPL_UI_LIST, 0),
                         ASCENDC_TPL_UINT_SEL(InputDType, ASCENDC_TPL_UI_LIST, 6), ASCENDC_TPL_BOOL_SEL(IsTnd, 0),
                         ASCENDC_TPL_BOOL_SEL(IsDrop, 0), ASCENDC_TPL_BOOL_SEL(IsPse, 0),
                         ASCENDC_TPL_BOOL_SEL(IsAttenMask, 0),
                         ASCENDC_TPL_UINT_SEL(S1TemplateNum, ASCENDC_TPL_UI_LIST, 512),
                         ASCENDC_TPL_UINT_SEL(S2TemplateNum, ASCENDC_TPL_UI_LIST, 512),
                         ASCENDC_TPL_UINT_SEL(DTemplateNum, ASCENDC_TPL_UI_LIST, 128),
                         ASCENDC_TPL_UINT_SEL(DeterType, ASCENDC_TPL_UI_LIST, 0), ASCENDC_TPL_BOOL_SEL(IsNEqual, 0),
                         ASCENDC_TPL_BOOL_SEL(IS_BN2_MULTIBLK, 0), ASCENDC_TPL_BOOL_SEL(IsDNoEqual, 0),
                         ASCENDC_TPL_UINT_SEL(OutDType, ASCENDC_TPL_UI_LIST, 2), ASCENDC_TPL_BOOL_SEL(IsSwizzle, 0),
                         ASCENDC_TPL_BOOL_SEL(HasRope, 0), ASCENDC_TPL_BOOL_SEL(IsDavid, 1),
                         ASCENDC_TPL_TILING_STRUCT_SEL(FagTilingWithTemplateQuantFFFF)),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_BOOL_SEL(IsEmptyTensor, 0),
                         ASCENDC_TPL_UINT_SEL(SplitAxis, ASCENDC_TPL_UI_LIST, 0),
                         ASCENDC_TPL_UINT_SEL(InputDType, ASCENDC_TPL_UI_LIST, 6), ASCENDC_TPL_BOOL_SEL(IsTnd, 0),
                         ASCENDC_TPL_BOOL_SEL(IsDrop, 0), ASCENDC_TPL_BOOL_SEL(IsPse, 0),
                         ASCENDC_TPL_BOOL_SEL(IsAttenMask, 0),
                         ASCENDC_TPL_UINT_SEL(S1TemplateNum, ASCENDC_TPL_UI_LIST, 512),
                         ASCENDC_TPL_UINT_SEL(S2TemplateNum, ASCENDC_TPL_UI_LIST, 512),
                         ASCENDC_TPL_UINT_SEL(DTemplateNum, ASCENDC_TPL_UI_LIST, 128),
                         ASCENDC_TPL_UINT_SEL(DeterType, ASCENDC_TPL_UI_LIST, 2), ASCENDC_TPL_BOOL_SEL(IsNEqual, 1),
                         ASCENDC_TPL_BOOL_SEL(IS_BN2_MULTIBLK, 0), ASCENDC_TPL_BOOL_SEL(IsDNoEqual, 0),
                         ASCENDC_TPL_UINT_SEL(OutDType, ASCENDC_TPL_UI_LIST, 2), ASCENDC_TPL_BOOL_SEL(IsSwizzle, 0),
                         ASCENDC_TPL_BOOL_SEL(HasRope, 0), ASCENDC_TPL_BOOL_SEL(IsDavid, 1),
                         ASCENDC_TPL_TILING_STRUCT_SEL(FagTilingWithTemplateQuantTFFF)),
#endif

);
#endif
