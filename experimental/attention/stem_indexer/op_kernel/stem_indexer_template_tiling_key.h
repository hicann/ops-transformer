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
 * \file stem_indexer_template_tiling_key.h
 * \brief
 */

#ifndef STEM_INDEXER_TEMPLATE_TILING_KEY_H
#define STEM_INDEXER_TEMPLATE_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"

#define SI_TPL_BF16 27
#define SI_TPL_INT32 3

// 模板参数支持的范围定义
#if !defined(__CCE_AICORE__) || (__CCE_AICORE__ == 310) || defined(__DAV_310R6__)
// clang-format off
ASCENDC_TPL_ARGS_DECL(StemIndexer, // 算子OpType
                      ASCENDC_TPL_DTYPE_DECL(DT_Q, SI_TPL_BF16), ASCENDC_TPL_DTYPE_DECL(DT_K, SI_TPL_BF16),
                      ASCENDC_TPL_DTYPE_DECL(DT_OUT, SI_TPL_INT32), ASCENDC_TPL_BOOL_DECL(CAUSAL, 1, 0),
                      ASCENDC_TPL_UINT_DECL(TOPK_SCORE_PRECISION, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST, 1, 2),);
// 支持的模板参数组合
// 用于调用GET_TPL_TILING_KEY获取TilingKey时，接口内部校验TilingKey是否合法
ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DTYPE_SEL(DT_Q, SI_TPL_BF16), ASCENDC_TPL_DTYPE_SEL(DT_K, SI_TPL_BF16),
                                     ASCENDC_TPL_DTYPE_SEL(DT_OUT, SI_TPL_INT32), ASCENDC_TPL_BOOL_SEL(CAUSAL, 0),
                                     ASCENDC_TPL_UINT_SEL(TOPK_SCORE_PRECISION, ASCENDC_TPL_UI_LIST, 1, 2),),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DTYPE_SEL(DT_Q, SI_TPL_BF16), ASCENDC_TPL_DTYPE_SEL(DT_K, SI_TPL_BF16),
                                     ASCENDC_TPL_DTYPE_SEL(DT_OUT, SI_TPL_INT32), ASCENDC_TPL_BOOL_SEL(CAUSAL, 1),
                                     ASCENDC_TPL_UINT_SEL(TOPK_SCORE_PRECISION, ASCENDC_TPL_UI_LIST, 1, 2),),);
// clang-format on

#endif

#endif
