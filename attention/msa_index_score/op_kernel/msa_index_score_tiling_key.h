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
 * \file msa_index_score_tiling_key.h
 * \brief Compile-time dispatch modes with unchanged Host key encoding.
 */
#ifndef MSA_INDEX_SCORE_TILING_KEY_H
#define MSA_INDEX_SCORE_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"
#include "msa_index_score_common.h"

// UINT parameters encode their declaration index. A zero-based dense range keeps
// the existing sparse mode numbers; the selection below compiles only supported modes.
ASCENDC_TPL_ARGS_DECL(MsaIndexScore, ASCENDC_TPL_UINT_DECL(schMode, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_RANGE, 1, 0,
                                                           MSA_TILING_KEY_MXFP4_LATENCY));

// Keep the original entry order to preserve the tested code layout.
ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(
    schMode, ASCENDC_TPL_UI_LIST, MSA_TILING_KEY_BF16, MSA_TILING_KEY_FP16, MSA_TILING_KEY_FP16_INT8,
    MSA_TILING_KEY_FP8_E4M3FN, MSA_TILING_KEY_FP8_E5M2, MSA_TILING_KEY_HIFLOAT8, MSA_TILING_KEY_BF16_BLK16,
    MSA_TILING_KEY_FP16_BLK16, MSA_TILING_KEY_BF16_TOPK, MSA_TILING_KEY_FP16_TOPK, MSA_TILING_KEY_FP8_E5M2_TOPK,
    MSA_TILING_KEY_FP8_E4M3FN_TOPK, MSA_TILING_KEY_MXFP4_TOPK, MSA_TILING_KEY_MXFP4, MSA_TILING_KEY_MXFP4_BLK16,
    MSA_TILING_KEY_MXFP4_PREFILL, MSA_TILING_KEY_MXFP4_LATENCY)));

#endif // MSA_INDEX_SCORE_TILING_KEY_H
