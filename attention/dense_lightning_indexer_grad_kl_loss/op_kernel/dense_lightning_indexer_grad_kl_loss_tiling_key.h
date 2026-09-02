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
 * \file dense_lightning_indexer_grad_kl_loss_tiling_key.h
 * \brief dense_lightning_indexer_grad_kl_loss tiling key declare
 */

#ifndef __DENSE_LIGHTNING_INDEXER_GRAD_KL_LOSS_TILING_KEY_H__
#define __DENSE_LIGHTNING_INDEXER_GRAD_KL_LOSS_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

enum LayoutType : uint8_t {
    LAYOUT_BSND = 0,
    LAYOUT_TND = 1,
    LAYOUT_NONE
};

#define MQA_SPARSE 3 // 当前仅支持MQAsparseMode3
#define ASCENDC_TPL_4_BW 4

// 模板参数支持的范围定义
ASCENDC_TPL_ARGS_DECL(DenseLightningIndexerGradKLLoss, // 算子OpType
                      ASCENDC_TPL_BOOL_DECL(HASROPE, 0, 1),
                      ASCENDC_TPL_UINT_DECL(LAYOUT_QT, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST, 0, 1), // 0为BSND 1为TND
                      ASCENDC_TPL_UINT_DECL(LAYOUT_KT, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST, 0, 1),
                      ASCENDC_TPL_UINT_DECL(SPARSE_MODE, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST, MQA_SPARSE),
                      ASCENDC_TPL_BOOL_DECL(DETERMINISTIC, 0, 1), );

// 支持的模板参数组合
// 用于调用GET_TPL_TILING_KEY获取TilingKey时，接口内部校验TilingKey是否合法
ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(
    ASCENDC_TPL_BOOL_SEL(HASROPE, 0, 1), ASCENDC_TPL_UINT_SEL(LAYOUT_QT, ASCENDC_TPL_UI_LIST, 0, 1),
    ASCENDC_TPL_UINT_SEL(LAYOUT_KT, ASCENDC_TPL_UI_LIST, 0, 1),
    ASCENDC_TPL_UINT_SEL(SPARSE_MODE, ASCENDC_TPL_UI_LIST, MQA_SPARSE), ASCENDC_TPL_BOOL_SEL(DETERMINISTIC, 0, 1),
    ASCENDC_TPL_TILING_STRUCT_SEL(optiling::DenseLightningIndexerGradKLLossTilingData)));

#endif
