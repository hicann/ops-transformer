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
 * \file grouped_mat_mul_allto_allv_tiling_key.h
 * \brief Quant Grouped MatMul AlltoAllV TilingKey 定义
 */
#ifndef QUANT_GROUPED_MAT_MUL_ALLTO_ALLV_TILING_KEY_H__
#define QUANT_GROUPED_MAT_MUL_ALLTO_ALLV_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

// 通信模式
#define TILINGKEY_TPL_CCU 0
#define TILINGKEY_TPL_AICPU 1

// 量化模式宏定义
#define QUANT_MODE_NONE 0 // 无量化
#define QUANT_MODE_TT 1   // TT
#define QUANT_MODE_MX 6   // MX量化

// 通信量化模式宏定义
#define COMM_QUANT_MODE_NONE 0 // 不量化

// 是否有共享专家 MM 计算
#define NO_COMPUTE_MATMUL 0  // 无共享专家
#define HAS_COMPUTE_MATMUL 1 // 有共享专家

// 权重转置标志
#define WEIGHT_NO_TRANS 0 // 不转置
#define WEIGHT_TRANS 1    // 转置

/**
 * TilingKey 定义
 * 使用 ASCENDC_TPL_ARGS_DECL 宏实现 TilingKey 到模板参数的自动转换
 *
 * 参数说明：
 * - TILINGKEY_GMM_WEIGHT_TRANS: GMM 计算权重转置场景 (0=不转置, 1=转置)
 * - TILINGKEY_SHARED_MM_WEIGHT_TRANS: 共享专家 MM 计算权重转置场景 (0=不转置, 1=转置)
 * - TILINGKEY_GMM_QUANT_MODE: GMM 量化模式 (0-6, 当前仅支持1)
 * - TILINGKEY_SHARED_MM_QUANT_MODE: 共享专家 MM 量化模式 (0-6, 当前仅支持1)
 */
ASCENDC_TPL_ARGS_DECL(QuantGroupedMatMulAlltoAllv,

                      // 是否进行MM计算
                      ASCENDC_TPL_BOOL_DECL(TILINGKEY_COMPUTE_MATMUL, 0, 1),

                      // GMM 计算转置场景
                      ASCENDC_TPL_BOOL_DECL(TILINGKEY_GMM_WEIGHT_TRANS, 0, 1),

                      // 共享专家 MM 计算转置场景
                      ASCENDC_TPL_BOOL_DECL(TILINGKEY_SHARED_MM_WEIGHT_TRANS, 0, 1),

                      // 通信引擎
                      ASCENDC_TPL_UINT_DECL(TILINGKEY_COMM_MODE, ASCENDC_TPL_1_BW, ASCENDC_TPL_UI_LIST,
                                            TILINGKEY_TPL_CCU, TILINGKEY_TPL_AICPU), );

/**
 * TilingKey 选择器定义
 * 定义所有有效的 TilingKey 组合
 */
ASCENDC_TPL_SEL(
    // 场景1: 无共享专家，GMM不转置，MM不转置
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_BOOL_SEL(TILINGKEY_COMPUTE_MATMUL, NO_COMPUTE_MATMUL),
                         ASCENDC_TPL_BOOL_SEL(TILINGKEY_GMM_WEIGHT_TRANS, WEIGHT_NO_TRANS),
                         ASCENDC_TPL_BOOL_SEL(TILINGKEY_SHARED_MM_WEIGHT_TRANS, WEIGHT_NO_TRANS),
                         ASCENDC_TPL_UINT_SEL(TILINGKEY_COMM_MODE, ASCENDC_TPL_UI_LIST, TILINGKEY_TPL_CCU,
                                              TILINGKEY_TPL_AICPU)),

    // 场景2: 无共享专家，GMM不转置，MM转置
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_BOOL_SEL(TILINGKEY_COMPUTE_MATMUL, NO_COMPUTE_MATMUL),
                         ASCENDC_TPL_BOOL_SEL(TILINGKEY_GMM_WEIGHT_TRANS, WEIGHT_NO_TRANS),
                         ASCENDC_TPL_BOOL_SEL(TILINGKEY_SHARED_MM_WEIGHT_TRANS, WEIGHT_TRANS),
                         ASCENDC_TPL_UINT_SEL(TILINGKEY_COMM_MODE, ASCENDC_TPL_UI_LIST, TILINGKEY_TPL_CCU,
                                              TILINGKEY_TPL_AICPU)),

    // 场景3: 无共享专家，GMM转置，MM不转置
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_BOOL_SEL(TILINGKEY_COMPUTE_MATMUL, NO_COMPUTE_MATMUL),
                         ASCENDC_TPL_BOOL_SEL(TILINGKEY_GMM_WEIGHT_TRANS, WEIGHT_TRANS),
                         ASCENDC_TPL_BOOL_SEL(TILINGKEY_SHARED_MM_WEIGHT_TRANS, WEIGHT_NO_TRANS),
                         ASCENDC_TPL_UINT_SEL(TILINGKEY_COMM_MODE, ASCENDC_TPL_UI_LIST, TILINGKEY_TPL_CCU,
                                              TILINGKEY_TPL_AICPU)),

    // 场景4: 无共享专家，GMM转置，MM转置
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_BOOL_SEL(TILINGKEY_COMPUTE_MATMUL, NO_COMPUTE_MATMUL),
                         ASCENDC_TPL_BOOL_SEL(TILINGKEY_GMM_WEIGHT_TRANS, WEIGHT_TRANS),
                         ASCENDC_TPL_BOOL_SEL(TILINGKEY_SHARED_MM_WEIGHT_TRANS, WEIGHT_TRANS),
                         ASCENDC_TPL_UINT_SEL(TILINGKEY_COMM_MODE, ASCENDC_TPL_UI_LIST, TILINGKEY_TPL_CCU,
                                              TILINGKEY_TPL_AICPU)),

    // 场景5: 有共享专家，GMM不转置，MM不转置
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_BOOL_SEL(TILINGKEY_COMPUTE_MATMUL, HAS_COMPUTE_MATMUL),
                         ASCENDC_TPL_BOOL_SEL(TILINGKEY_GMM_WEIGHT_TRANS, WEIGHT_NO_TRANS),
                         ASCENDC_TPL_BOOL_SEL(TILINGKEY_SHARED_MM_WEIGHT_TRANS, WEIGHT_NO_TRANS),
                         ASCENDC_TPL_UINT_SEL(TILINGKEY_COMM_MODE, ASCENDC_TPL_UI_LIST, TILINGKEY_TPL_CCU,
                                              TILINGKEY_TPL_AICPU)),

    // 场景6: 有共享专家，GMM不转置，MM转置
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_BOOL_SEL(TILINGKEY_COMPUTE_MATMUL, HAS_COMPUTE_MATMUL),
                         ASCENDC_TPL_BOOL_SEL(TILINGKEY_GMM_WEIGHT_TRANS, WEIGHT_NO_TRANS),
                         ASCENDC_TPL_BOOL_SEL(TILINGKEY_SHARED_MM_WEIGHT_TRANS, WEIGHT_TRANS),
                         ASCENDC_TPL_UINT_SEL(TILINGKEY_COMM_MODE, ASCENDC_TPL_UI_LIST, TILINGKEY_TPL_CCU,
                                              TILINGKEY_TPL_AICPU)),

    // 场景8: 有共享专家，GMM转置，MM不转置
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_BOOL_SEL(TILINGKEY_COMPUTE_MATMUL, HAS_COMPUTE_MATMUL),
                         ASCENDC_TPL_BOOL_SEL(TILINGKEY_GMM_WEIGHT_TRANS, WEIGHT_TRANS),
                         ASCENDC_TPL_BOOL_SEL(TILINGKEY_SHARED_MM_WEIGHT_TRANS, WEIGHT_NO_TRANS),
                         ASCENDC_TPL_UINT_SEL(TILINGKEY_COMM_MODE, ASCENDC_TPL_UI_LIST, TILINGKEY_TPL_CCU,
                                              TILINGKEY_TPL_AICPU)),

    // 场景9: 有共享专家，GMM转置，MM转置
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_BOOL_SEL(TILINGKEY_COMPUTE_MATMUL, HAS_COMPUTE_MATMUL),
                         ASCENDC_TPL_BOOL_SEL(TILINGKEY_GMM_WEIGHT_TRANS, WEIGHT_TRANS),
                         ASCENDC_TPL_BOOL_SEL(TILINGKEY_SHARED_MM_WEIGHT_TRANS, WEIGHT_TRANS),
                         ASCENDC_TPL_UINT_SEL(TILINGKEY_COMM_MODE, ASCENDC_TPL_UI_LIST, TILINGKEY_TPL_CCU,
                                              TILINGKEY_TPL_AICPU)),

);

#endif // QUANT_GROUPED_MAT_MUL_ALLTO_ALLV_TILING_KEY_H__
