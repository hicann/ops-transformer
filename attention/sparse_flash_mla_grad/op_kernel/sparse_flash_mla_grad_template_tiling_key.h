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

#ifndef SPARSE_FLASH_MLA_GRAD_TEMPLATE_TILING_KEY_H
#define SPARSE_FLASH_MLA_GRAD_TEMPLATE_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"

#define SMLAG_LAYOUT_BSND 0
#define SMLAG_LAYOUT_TND 1
#define SMLAG_SWA_MODE 0
#define SMLAG_CFA_MODE 1
#define SMLAG_SCFA_MODE 2
#define ASCENDC_TPL_4_BW 4

// 模板参数支持的范围定义
ASCENDC_TPL_ARGS_DECL(SparseFlashMlaGrad, // 算子OpType
                      ASCENDC_TPL_UINT_DECL(LAYOUT, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST, SMLAG_LAYOUT_BSND,
                                            SMLAG_LAYOUT_TND),
                      ASCENDC_TPL_UINT_DECL(MODE, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST, SMLAG_SWA_MODE, SMLAG_CFA_MODE,
                                            SMLAG_SCFA_MODE), );

// 支持的模板参数组合
// 用于调用GET_TPL_TILING_KEY获取TilingKey时，接口内部校验TilingKey是否合法
ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(LAYOUT, ASCENDC_TPL_UI_LIST, SMLAG_LAYOUT_BSND,
                                                          SMLAG_LAYOUT_TND),
                                     ASCENDC_TPL_UINT_SEL(MODE, ASCENDC_TPL_UI_LIST, SMLAG_SWA_MODE, SMLAG_CFA_MODE,
                                                          SMLAG_SCFA_MODE), ), );

#endif // TEMPLATE_TILING_KEY
