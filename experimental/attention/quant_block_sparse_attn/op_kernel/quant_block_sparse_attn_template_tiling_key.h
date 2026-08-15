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
 * \file quant_block_sparse_attn_template_tiling_key.h
 * \brief QuantBlockSparseAttn template tiling key declarations.
 */

#ifndef QUANT_BLOCK_SPARSE_ATTN_TEMPLATE_TILING_KEY_H
#define QUANT_BLOCK_SPARSE_ATTN_TEMPLATE_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"

// Template tiling key macros: values match QBSALayout enum (in quant_block_sparse_attn_const.h)
// for ASCENDC_TPL_ARGS compile-time dispatch.
#define QBSA_LAYOUT_TND 2 // == QBSALayout::TND
#define QBSA_LAYOUT_NTD 5 // == QBSALayout::NTD

#define QBSA_KV_LAYOUT_CONTIGUOUS 0
#define QBSA_KV_LAYOUT_PA_ND 1
#define QBSA_KV_LAYOUT_PA_BNSD 3 // == QBSALayout::PA_BNBD

#define QBSA_MASK_NONE 0
#define QBSA_MASK_CAUSAL 3

#define QBSA_DTYPE_FP8_E4M3FN 0

#define FP8QuantMode 1
#define MXFullQuantMode 2

// Config 数值是 tiling-key 枚举值，不是运行期 block size。
// FP8 保持原有 S2=256 模板；MXFullQuantMode 选择独立 S2=512 模板，
// 使 dispatch 能进入隔离后的 MX kernel 与 MX tiling-data 路径。
#define Config_S1Aligned128_S2Aligned256_DAligned128_DVAligned128 0
#define Config_S1Aligned128_S2Aligned512_DAligned128_DVAligned128 1

#define QBSA_TPL_4_BW 4

ASCENDC_TPL_ARGS_DECL(
    QuantBlockSparseAttn, ASCENDC_TPL_UINT_DECL(QKV_DTYPE, QBSA_TPL_4_BW, ASCENDC_TPL_UI_LIST, QBSA_DTYPE_FP8_E4M3FN),
    ASCENDC_TPL_UINT_DECL(LAYOUT_T, QBSA_TPL_4_BW, ASCENDC_TPL_UI_LIST, QBSA_LAYOUT_TND, QBSA_LAYOUT_NTD),
    ASCENDC_TPL_UINT_DECL(KV_LAYOUT_T, QBSA_TPL_4_BW, ASCENDC_TPL_UI_LIST, QBSA_KV_LAYOUT_CONTIGUOUS,
                          QBSA_KV_LAYOUT_PA_BNSD),
    ASCENDC_TPL_UINT_DECL(MASK_MODE, QBSA_TPL_4_BW, ASCENDC_TPL_UI_LIST, QBSA_MASK_NONE, QBSA_MASK_CAUSAL),
    ASCENDC_TPL_BOOL_DECL(RETURN_SOFTMAX_LSE, 0, 1),
    ASCENDC_TPL_UINT_DECL(Config, QBSA_TPL_4_BW, ASCENDC_TPL_UI_LIST,
                          Config_S1Aligned128_S2Aligned256_DAligned128_DVAligned128,
                          Config_S1Aligned128_S2Aligned512_DAligned128_DVAligned128),
    ASCENDC_TPL_UINT_DECL(QUANT_MODE, QBSA_TPL_4_BW, ASCENDC_TPL_UI_LIST, FP8QuantMode, MXFullQuantMode), );

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(QKV_DTYPE, ASCENDC_TPL_UI_LIST, QBSA_DTYPE_FP8_E4M3FN),
        ASCENDC_TPL_UINT_SEL(LAYOUT_T, ASCENDC_TPL_UI_LIST, QBSA_LAYOUT_TND, QBSA_LAYOUT_NTD),
        ASCENDC_TPL_UINT_SEL(KV_LAYOUT_T, ASCENDC_TPL_UI_LIST, QBSA_KV_LAYOUT_CONTIGUOUS, QBSA_KV_LAYOUT_PA_BNSD),
        ASCENDC_TPL_UINT_SEL(MASK_MODE, ASCENDC_TPL_UI_LIST, QBSA_MASK_NONE, QBSA_MASK_CAUSAL),
        ASCENDC_TPL_BOOL_SEL(RETURN_SOFTMAX_LSE, 0, 1),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned128_S2Aligned256_DAligned128_DVAligned128),
        ASCENDC_TPL_UINT_SEL(QUANT_MODE, ASCENDC_TPL_UI_LIST, FP8QuantMode)),
    // MXFP8 全量化有意收敛到 host check 要求的特性约束：
    // TND + PA BNBD + mask none/causal + S2=512 + quant_mode=2。
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(QKV_DTYPE, ASCENDC_TPL_UI_LIST, QBSA_DTYPE_FP8_E4M3FN),
                         ASCENDC_TPL_UINT_SEL(LAYOUT_T, ASCENDC_TPL_UI_LIST, QBSA_LAYOUT_TND),
                         ASCENDC_TPL_UINT_SEL(KV_LAYOUT_T, ASCENDC_TPL_UI_LIST, QBSA_KV_LAYOUT_PA_BNSD),
                         ASCENDC_TPL_UINT_SEL(MASK_MODE, ASCENDC_TPL_UI_LIST, QBSA_MASK_NONE, QBSA_MASK_CAUSAL),
                         ASCENDC_TPL_BOOL_SEL(RETURN_SOFTMAX_LSE, 0, 1),
                         ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST,
                                              Config_S1Aligned128_S2Aligned512_DAligned128_DVAligned128),
                         ASCENDC_TPL_UINT_SEL(QUANT_MODE, ASCENDC_TPL_UI_LIST, MXFullQuantMode)), );

#endif // QUANT_BLOCK_SPARSE_ATTN_TEMPLATE_TILING_KEY_H
