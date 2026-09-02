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
 * \file sparse_flash_mla_grad_template_tiling_key.h
 * \brief
 */

#ifndef TEMPLATE_TILING_KEY_SASG_H_
#define TEMPLATE_TILING_KEY_SASG_H_
#include "ascendc/host_api/tiling/template_argument.h"
#include "sparse_flash_mla_grad_tiling_data_regbase.h"

// kernel通过宏定义隔离dtype编译tilingkey，降低耗时。tiling侧没有相关宏
#ifndef ORIG_DTYPE_QUERY
#define ORIG_DTYPE_QUERY (-1)
#endif

#define ASCENDC_TPL_3_BW 3
#define ASCENDC_TPL_4_BW 4
#define ASCENDC_TPL_8_BW 8
#define ASCENDC_TPL_12_BW 12

using sfmgTilingWithTemplate = optiling::smlag::SparseFlashMlaGradTilingDataRegbase;

// 可表示的tilingkey范围为64bit，注意不能超过限制
ASCENDC_TPL_ARGS_DECL(SparseFlashMlaGrad, // 算子唯一标识，可以opType保持一致
                                          // bit: 2-0 InputDType
                                          //      0: FLOAT16
                                          //      1: BF16
                      ASCENDC_TPL_UINT_DECL(InputDType, ASCENDC_TPL_3_BW, ASCENDC_TPL_UI_LIST, 0, 1),
                      // bit: 3 IsTnd
                      //      0: DISABLE
                      //      1: ENABLE
                      ASCENDC_TPL_BOOL_DECL(IsTnd, 0, 1),
                      // bit: 11-4 GTemplateType
                      ASCENDC_TPL_UINT_DECL(GTemplateNum, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_LIST, 128),
                      // bit: 19-12 S2TemplateType
                      ASCENDC_TPL_UINT_DECL(S2TemplateNum, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_LIST, 128),
                      // bit: 31-20 DTemplateType
                      ASCENDC_TPL_UINT_DECL(DTemplateNum, ASCENDC_TPL_12_BW, ASCENDC_TPL_UI_LIST, 512),
                      // bit: 32
                      ASCENDC_TPL_BOOL_DECL(IsOriKVExist, 0, 1),
                      // bit: 33
                      ASCENDC_TPL_BOOL_DECL(IsCmpKVExist, 0, 1),
                      // bit: 34
                      ASCENDC_TPL_BOOL_DECL(IsOriKVSparse, 0, 1),
                      // bit: 35
                      ASCENDC_TPL_BOOL_DECL(IsCmpKVSparse, 0, 1),
                      // bit: 36
                      ASCENDC_TPL_BOOL_DECL(Deterministic, 0, 1));

ASCENDC_TPL_SEL(
#if (ORIG_DTYPE_QUERY == -1) || (ORIG_DTYPE_QUERY == DT_FLOAT16)
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(InputDType, ASCENDC_TPL_UI_LIST, 0), ASCENDC_TPL_BOOL_SEL(IsTnd, 0, 1),
                         ASCENDC_TPL_UINT_SEL(GTemplateNum, ASCENDC_TPL_UI_LIST, 128),
                         ASCENDC_TPL_UINT_SEL(S2TemplateNum, ASCENDC_TPL_UI_LIST, 128),
                         ASCENDC_TPL_UINT_SEL(DTemplateNum, ASCENDC_TPL_UI_LIST, 512),
                         ASCENDC_TPL_BOOL_SEL(IsOriKVExist, 1), ASCENDC_TPL_BOOL_SEL(IsCmpKVExist, 0),
                         ASCENDC_TPL_BOOL_SEL(IsOriKVSparse, 0), ASCENDC_TPL_BOOL_SEL(IsCmpKVSparse, 0),
                         ASCENDC_TPL_BOOL_SEL(Deterministic, 0, 1),
                         ASCENDC_TPL_TILING_STRUCT_SEL(sfmgTilingWithTemplate)),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(InputDType, ASCENDC_TPL_UI_LIST, 0), ASCENDC_TPL_BOOL_SEL(IsTnd, 0, 1),
                         ASCENDC_TPL_UINT_SEL(GTemplateNum, ASCENDC_TPL_UI_LIST, 128),
                         ASCENDC_TPL_UINT_SEL(S2TemplateNum, ASCENDC_TPL_UI_LIST, 128),
                         ASCENDC_TPL_UINT_SEL(DTemplateNum, ASCENDC_TPL_UI_LIST, 512),
                         ASCENDC_TPL_BOOL_SEL(IsOriKVExist, 0, 1), ASCENDC_TPL_BOOL_SEL(IsCmpKVExist, 1),
                         ASCENDC_TPL_BOOL_SEL(IsOriKVSparse, 0), ASCENDC_TPL_BOOL_SEL(IsCmpKVSparse, 0),
                         ASCENDC_TPL_BOOL_SEL(Deterministic, 0, 1),
                         ASCENDC_TPL_TILING_STRUCT_SEL(sfmgTilingWithTemplate)),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(InputDType, ASCENDC_TPL_UI_LIST, 0), ASCENDC_TPL_BOOL_SEL(IsTnd, 0, 1),
                         ASCENDC_TPL_UINT_SEL(GTemplateNum, ASCENDC_TPL_UI_LIST, 128),
                         ASCENDC_TPL_UINT_SEL(S2TemplateNum, ASCENDC_TPL_UI_LIST, 128),
                         ASCENDC_TPL_UINT_SEL(DTemplateNum, ASCENDC_TPL_UI_LIST, 512),
                         ASCENDC_TPL_BOOL_SEL(IsOriKVExist, 0, 1), ASCENDC_TPL_BOOL_SEL(IsCmpKVExist, 1),
                         ASCENDC_TPL_BOOL_SEL(IsOriKVSparse, 0), ASCENDC_TPL_BOOL_SEL(IsCmpKVSparse, 1),
                         ASCENDC_TPL_BOOL_SEL(Deterministic, 0, 1),
                         ASCENDC_TPL_TILING_STRUCT_SEL(sfmgTilingWithTemplate)),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(InputDType, ASCENDC_TPL_UI_LIST, 0), ASCENDC_TPL_BOOL_SEL(IsTnd, 0, 1),
                         ASCENDC_TPL_UINT_SEL(GTemplateNum, ASCENDC_TPL_UI_LIST, 128),
                         ASCENDC_TPL_UINT_SEL(S2TemplateNum, ASCENDC_TPL_UI_LIST, 128),
                         ASCENDC_TPL_UINT_SEL(DTemplateNum, ASCENDC_TPL_UI_LIST, 512),
                         ASCENDC_TPL_BOOL_SEL(IsOriKVExist, 1), ASCENDC_TPL_BOOL_SEL(IsCmpKVExist, 0),
                         ASCENDC_TPL_BOOL_SEL(IsOriKVSparse, 1), ASCENDC_TPL_BOOL_SEL(IsCmpKVSparse, 0),
                         ASCENDC_TPL_BOOL_SEL(Deterministic, 0, 1),
                         ASCENDC_TPL_TILING_STRUCT_SEL(sfmgTilingWithTemplate)),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(InputDType, ASCENDC_TPL_UI_LIST, 0), ASCENDC_TPL_BOOL_SEL(IsTnd, 0, 1),
                         ASCENDC_TPL_UINT_SEL(GTemplateNum, ASCENDC_TPL_UI_LIST, 128),
                         ASCENDC_TPL_UINT_SEL(S2TemplateNum, ASCENDC_TPL_UI_LIST, 128),
                         ASCENDC_TPL_UINT_SEL(DTemplateNum, ASCENDC_TPL_UI_LIST, 512),
                         ASCENDC_TPL_BOOL_SEL(IsOriKVExist, 1), ASCENDC_TPL_BOOL_SEL(IsCmpKVExist, 1),
                         ASCENDC_TPL_BOOL_SEL(IsOriKVSparse, 1), ASCENDC_TPL_BOOL_SEL(IsCmpKVSparse, 1),
                         ASCENDC_TPL_BOOL_SEL(Deterministic, 0, 1),
                         ASCENDC_TPL_TILING_STRUCT_SEL(sfmgTilingWithTemplate)),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(InputDType, ASCENDC_TPL_UI_LIST, 0), ASCENDC_TPL_BOOL_SEL(IsTnd, 0, 1),
                         ASCENDC_TPL_UINT_SEL(GTemplateNum, ASCENDC_TPL_UI_LIST, 128),
                         ASCENDC_TPL_UINT_SEL(S2TemplateNum, ASCENDC_TPL_UI_LIST, 128),
                         ASCENDC_TPL_UINT_SEL(DTemplateNum, ASCENDC_TPL_UI_LIST, 512),
                         ASCENDC_TPL_BOOL_SEL(IsOriKVExist, 1), ASCENDC_TPL_BOOL_SEL(IsCmpKVExist, 1),
                         ASCENDC_TPL_BOOL_SEL(IsOriKVSparse, 1), ASCENDC_TPL_BOOL_SEL(IsCmpKVSparse, 0),
                         ASCENDC_TPL_BOOL_SEL(Deterministic, 0, 1),
                         ASCENDC_TPL_TILING_STRUCT_SEL(sfmgTilingWithTemplate)),
#endif
#if (ORIG_DTYPE_QUERY == -1) || (ORIG_DTYPE_QUERY == DT_BF16)
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(InputDType, ASCENDC_TPL_UI_LIST, 1), ASCENDC_TPL_BOOL_SEL(IsTnd, 0, 1),
                         ASCENDC_TPL_UINT_SEL(GTemplateNum, ASCENDC_TPL_UI_LIST, 128),
                         ASCENDC_TPL_UINT_SEL(S2TemplateNum, ASCENDC_TPL_UI_LIST, 128),
                         ASCENDC_TPL_UINT_SEL(DTemplateNum, ASCENDC_TPL_UI_LIST, 512),
                         ASCENDC_TPL_BOOL_SEL(IsOriKVExist, 1), ASCENDC_TPL_BOOL_SEL(IsCmpKVExist, 0),
                         ASCENDC_TPL_BOOL_SEL(IsOriKVSparse, 0), ASCENDC_TPL_BOOL_SEL(IsCmpKVSparse, 0),
                         ASCENDC_TPL_BOOL_SEL(Deterministic, 0, 1),
                         ASCENDC_TPL_TILING_STRUCT_SEL(sfmgTilingWithTemplate)),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(InputDType, ASCENDC_TPL_UI_LIST, 1), ASCENDC_TPL_BOOL_SEL(IsTnd, 0, 1),
                         ASCENDC_TPL_UINT_SEL(GTemplateNum, ASCENDC_TPL_UI_LIST, 128),
                         ASCENDC_TPL_UINT_SEL(S2TemplateNum, ASCENDC_TPL_UI_LIST, 128),
                         ASCENDC_TPL_UINT_SEL(DTemplateNum, ASCENDC_TPL_UI_LIST, 512),
                         ASCENDC_TPL_BOOL_SEL(IsOriKVExist, 0, 1), ASCENDC_TPL_BOOL_SEL(IsCmpKVExist, 1),
                         ASCENDC_TPL_BOOL_SEL(IsOriKVSparse, 0), ASCENDC_TPL_BOOL_SEL(IsCmpKVSparse, 0),
                         ASCENDC_TPL_BOOL_SEL(Deterministic, 0, 1),
                         ASCENDC_TPL_TILING_STRUCT_SEL(sfmgTilingWithTemplate)),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(InputDType, ASCENDC_TPL_UI_LIST, 1), ASCENDC_TPL_BOOL_SEL(IsTnd, 0, 1),
                         ASCENDC_TPL_UINT_SEL(GTemplateNum, ASCENDC_TPL_UI_LIST, 128),
                         ASCENDC_TPL_UINT_SEL(S2TemplateNum, ASCENDC_TPL_UI_LIST, 128),
                         ASCENDC_TPL_UINT_SEL(DTemplateNum, ASCENDC_TPL_UI_LIST, 512),
                         ASCENDC_TPL_BOOL_SEL(IsOriKVExist, 0, 1), ASCENDC_TPL_BOOL_SEL(IsCmpKVExist, 1),
                         ASCENDC_TPL_BOOL_SEL(IsOriKVSparse, 0), ASCENDC_TPL_BOOL_SEL(IsCmpKVSparse, 1),
                         ASCENDC_TPL_BOOL_SEL(Deterministic, 0, 1),
                         ASCENDC_TPL_TILING_STRUCT_SEL(sfmgTilingWithTemplate)),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(InputDType, ASCENDC_TPL_UI_LIST, 1), ASCENDC_TPL_BOOL_SEL(IsTnd, 0, 1),
                         ASCENDC_TPL_UINT_SEL(GTemplateNum, ASCENDC_TPL_UI_LIST, 128),
                         ASCENDC_TPL_UINT_SEL(S2TemplateNum, ASCENDC_TPL_UI_LIST, 128),
                         ASCENDC_TPL_UINT_SEL(DTemplateNum, ASCENDC_TPL_UI_LIST, 512),
                         ASCENDC_TPL_BOOL_SEL(IsOriKVExist, 1), ASCENDC_TPL_BOOL_SEL(IsCmpKVExist, 0),
                         ASCENDC_TPL_BOOL_SEL(IsOriKVSparse, 1), ASCENDC_TPL_BOOL_SEL(IsCmpKVSparse, 0),
                         ASCENDC_TPL_BOOL_SEL(Deterministic, 0, 1),
                         ASCENDC_TPL_TILING_STRUCT_SEL(sfmgTilingWithTemplate)),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(InputDType, ASCENDC_TPL_UI_LIST, 1), ASCENDC_TPL_BOOL_SEL(IsTnd, 0, 1),
                         ASCENDC_TPL_UINT_SEL(GTemplateNum, ASCENDC_TPL_UI_LIST, 128),
                         ASCENDC_TPL_UINT_SEL(S2TemplateNum, ASCENDC_TPL_UI_LIST, 128),
                         ASCENDC_TPL_UINT_SEL(DTemplateNum, ASCENDC_TPL_UI_LIST, 512),
                         ASCENDC_TPL_BOOL_SEL(IsOriKVExist, 1), ASCENDC_TPL_BOOL_SEL(IsCmpKVExist, 1),
                         ASCENDC_TPL_BOOL_SEL(IsOriKVSparse, 1), ASCENDC_TPL_BOOL_SEL(IsCmpKVSparse, 1),
                         ASCENDC_TPL_BOOL_SEL(Deterministic, 0, 1),
                         ASCENDC_TPL_TILING_STRUCT_SEL(sfmgTilingWithTemplate)),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(InputDType, ASCENDC_TPL_UI_LIST, 1), ASCENDC_TPL_BOOL_SEL(IsTnd, 0, 1),
                         ASCENDC_TPL_UINT_SEL(GTemplateNum, ASCENDC_TPL_UI_LIST, 128),
                         ASCENDC_TPL_UINT_SEL(S2TemplateNum, ASCENDC_TPL_UI_LIST, 128),
                         ASCENDC_TPL_UINT_SEL(DTemplateNum, ASCENDC_TPL_UI_LIST, 512),
                         ASCENDC_TPL_BOOL_SEL(IsOriKVExist, 1), ASCENDC_TPL_BOOL_SEL(IsCmpKVExist, 1),
                         ASCENDC_TPL_BOOL_SEL(IsOriKVSparse, 1), ASCENDC_TPL_BOOL_SEL(IsCmpKVSparse, 0),
                         ASCENDC_TPL_BOOL_SEL(Deterministic, 0, 1),
                         ASCENDC_TPL_TILING_STRUCT_SEL(sfmgTilingWithTemplate)),
#endif
);
#endif
