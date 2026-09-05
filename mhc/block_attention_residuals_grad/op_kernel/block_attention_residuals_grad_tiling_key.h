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
 * \file block_attention_residuals_grad_tiling_key.h
 * \brief block_attention_residuals_grad tiling key declare
 */

#ifndef __BLOCK_ATTENTION_RESIDUALS_GRAD_TILING_KEY_H__
#define __BLOCK_ATTENTION_RESIDUALS_GRAD_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

#define TPL_H_MODE_FULL 0
#define TPL_H_MODE_SPLIT 1

ASCENDC_TPL_ARGS_DECL(BlockAttentionResidualsGrad,
                      ASCENDC_TPL_UINT_DECL(hMode, 2, ASCENDC_TPL_UI_LIST, TPL_H_MODE_FULL, TPL_H_MODE_SPLIT));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(hMode, ASCENDC_TPL_UI_LIST, TPL_H_MODE_FULL,
                                                          TPL_H_MODE_SPLIT)));

#endif
