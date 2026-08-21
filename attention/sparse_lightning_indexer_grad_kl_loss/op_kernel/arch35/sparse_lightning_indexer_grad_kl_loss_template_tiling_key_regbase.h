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
 * \file sparse_lightning_indexer_grad_kl_loss_template_tiling_key_regbase.h
 * \brief
 */

#ifndef SPARSE_LIGHTNING_INDEXER_GRAD_KL_LOSS_TEMPLATE_TILING_KEY_REGBASE_H
#define SPARSE_LIGHTNING_INDEXER_GRAD_KL_LOSS_TEMPLATE_TILING_KEY_REGBASE_H

#include "ascendc/host_api/tiling/template_argument.h"

enum class LayoutTypeRegbase : uint8_t {
    LAYOUT_BSND = 0,
    LAYOUT_TND = 1,
    LAYOUT_NONE
};

enum class TopKRangeRegbase  : uint8_t {
    RANGE_0_2K = 0,
    RANGE_2K_8K = 1,
    RANGE_NONE
};

#define MQA_SPARSE 3        // 当前仅支持MQAsparseMode3
#define ASCENDC_TPL_4_BW 4

// 模板参数支持的范围定义
ASCENDC_TPL_ARGS_DECL(SparseLightningIndexerGradKLLoss, // 算子OpType
    ASCENDC_TPL_BOOL_DECL(HASROPE, 0, 1),
    ASCENDC_TPL_UINT_DECL(TOPK_RANGE, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST, 0, 1),
    ASCENDC_TPL_UINT_DECL(LAYOUT_QT, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST, 0, 1), // 0为BSND 1为TND
    ASCENDC_TPL_UINT_DECL(LAYOUT_KT, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST, 0, 1),
    ASCENDC_TPL_UINT_DECL(SPARSE_MODE, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST, 0, 3),
    ASCENDC_TPL_BOOL_DECL(DETERMINISTIC, 0, 1),
);

// 支持的模板参数组合
// 用于调用GET_TPL_TILING_KEY获取TilingKey时，接口内部校验TilingKey是否合法
ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_BOOL_SEL(HASROPE, 0, 1),
        ASCENDC_TPL_UINT_SEL(TOPK_RANGE, ASCENDC_TPL_UI_LIST, 0, 1),
        ASCENDC_TPL_UINT_SEL(LAYOUT_QT, ASCENDC_TPL_UI_LIST, 0, 1),
        ASCENDC_TPL_UINT_SEL(LAYOUT_KT, ASCENDC_TPL_UI_LIST, 0, 1),
        ASCENDC_TPL_UINT_SEL(SPARSE_MODE, ASCENDC_TPL_UI_LIST, 3),
        ASCENDC_TPL_BOOL_SEL(DETERMINISTIC, 0, 1),
        ASCENDC_TPL_TILING_STRUCT_SEL(optiling::SparseLightningIndexerGradKLLossRegBaseTilingData)
    )
);
#endif // TEMPLATE_TILING_KEY