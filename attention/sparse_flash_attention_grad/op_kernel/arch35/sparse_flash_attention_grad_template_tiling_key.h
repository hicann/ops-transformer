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
 * \file sparse_flash_attention_grad_template_tiling_key.h
 * \brief
 */

#ifndef TEMPLATE_TILING_KEY_SFAG_H_
#define TEMPLATE_TILING_KEY_SFAG_H_
#include "ascendc/host_api/tiling/template_argument.h"
#include "sparse_flash_attention_grad_tiling_data_regbase.h"

// kernel通过宏定义隔离dtype编译tilingkey，降低耗时。tiling侧没有相关宏
#ifndef ORIG_DTYPE_QUERY
#define ORIG_DTYPE_QUERY (-1)
#endif

#define ASCENDC_TPL_3_BW 3
#define ASCENDC_TPL_4_BW 4
#define ASCENDC_TPL_8_BW 8
#define ASCENDC_TPL_12_BW 12

using sfagTilingWithTemplate = optiling::sfag::SparseFlashAttentionGradTilingDataRegbase;

// 可表示的tilingkey范围为64bit，注意不能超过限制
ASCENDC_TPL_ARGS_DECL(SparseFlashAttentionGrad, // 算子唯一标识，可以opType保持一致
                                                // bit: 3-0 InputDType
                                                //      0: FLOAT16
                                                //      1: BF16
                      ASCENDC_TPL_UINT_DECL(InputDType, ASCENDC_TPL_3_BW, ASCENDC_TPL_UI_LIST, 0, 1),
                      // bit: 4 IsTnd
                      //      0: DISABLE
                      //      1: ENABLE
                      ASCENDC_TPL_BOOL_DECL(IsTnd, 0, 1),
                      // bit: 12-5 GTemplateType
                      ASCENDC_TPL_UINT_DECL(GTemplateNum, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_LIST, 128),
                      // bit: 20-13 S2TemplateType
                      ASCENDC_TPL_UINT_DECL(S2TemplateNum, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_LIST, 128),
                      // bit: 32-21 DTemplateType
                      ASCENDC_TPL_UINT_DECL(DTemplateNum, ASCENDC_TPL_12_BW, ASCENDC_TPL_UI_LIST, 512, 576),
                      // bit: 33 IsRope
                      //      0: DISABLE
                      //      1: ENABLE
                      ASCENDC_TPL_BOOL_DECL(IsRope, 0, 1),
                      // bit: 34 Deterministic
                      //      0: DISABLE
                      //      1: ENABLE
                      ASCENDC_TPL_BOOL_DECL(Deterministic, 0, 1),
                      // bit: 35 KvMerge
                      //      0: DISABLE
                      //      1: ENABLE
                      ASCENDC_TPL_BOOL_DECL(KvMerge, 0, 1),
                      // bit: 36 HasSinks
                      //      0: DISABLE
                      //      1: ENABLE
                      ASCENDC_TPL_BOOL_DECL(HasSinks, 0, 1));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(InputDType, ASCENDC_TPL_UI_LIST, 0, 1),
                                     ASCENDC_TPL_BOOL_SEL(IsTnd, 0, 1),
                                     ASCENDC_TPL_UINT_SEL(GTemplateNum, ASCENDC_TPL_UI_LIST, 128),
                                     ASCENDC_TPL_UINT_SEL(S2TemplateNum, ASCENDC_TPL_UI_LIST, 128),
                                     ASCENDC_TPL_UINT_SEL(DTemplateNum, ASCENDC_TPL_UI_LIST, 512, 576),
                                     ASCENDC_TPL_BOOL_SEL(IsRope, 0, 1), ASCENDC_TPL_BOOL_SEL(Deterministic, 0, 1),
                                     ASCENDC_TPL_BOOL_SEL(KvMerge, 0, 1), ASCENDC_TPL_BOOL_SEL(HasSinks, 0, 1),
                                     ASCENDC_TPL_TILING_STRUCT_SEL(sfagTilingWithTemplate)));
#endif
