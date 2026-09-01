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
 * \file attention_to_ffn_tiling_key.h
 * \brief
 */
#ifndef ATTENTION_TO_FFN_V2_TILING_KEY_H
#define ATTENTION_TO_FFN_V2_TILING_KEY_H
#include "ascendc/host_api/tiling/template_argument.h"

#define TILINGKEY_TPL_A3 1
#define TILINGKEY_TPL_A5 2

// User-facing quant_mode attr values:
//   0: NO_QUANT
//   2: PERTOKEN + INT8  (legacy)
//   3: MX + FP8_E5M2
//   4: MX + FP8_E4M3
//   5: MX + FP4_E2M1
//   6: MX_CLIP + FP8_E5M2
//   7: MX_CLIP + FP8_E4M3
//
// Internally the tiling key splits quant_mode into two fields:
//   ATTN_FFN_QUANT_MODE (algorithm):  0=NO_QUANT, 2=PERTOKEN, 3=MX, 4=MX_CLIP
//   ATTN_FFN_OUT_DTYPE (output type): 0=INT8, 1=E5M2, 2=E4M3, 3=E2M1
// The tiling layer maps: 3→(MX, E5M2), 4→(MX, E4M3), 5→(MX, E2M1),
//                                   6→(MX_CLIP, E5M2), 7→(MX_CLIP, E4M3).
#define ATTN_FFN_TILINGKEY_NO_QUANT 0
#define ATTN_FFN_TILINGKEY_PERTOKEN_INT8 2
#define ATTN_FFN_TILINGKEY_MX 3
#define ATTN_FFN_TILINGKEY_MX_CLIP 4

// outDtype values (only meaningful for MX/MX_CLIP):
//   0: INT8 (used for PERTOKEN, ignored for NO_QUANT)
//   1: FP8_E5M2
//   2: FP8_E4M3
//   3: FP4_E2M1
#define ATTN_FFN_TILINGKEY_OUT_INT8 0
#define ATTN_FFN_TILINGKEY_OUT_E5M2 1
#define ATTN_FFN_TILINGKEY_OUT_E4M3 2
#define ATTN_FFN_TILINGKEY_OUT_E2M1 3

namespace Mc2Tiling {
ASCENDC_TPL_ARGS_DECL(AttentionToFfnV2,
                      ASCENDC_TPL_UINT_DECL(ATTN_FFN_QUANT_MODE, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST,
                                            ATTN_FFN_TILINGKEY_NO_QUANT, ATTN_FFN_TILINGKEY_PERTOKEN_INT8,
                                            ATTN_FFN_TILINGKEY_MX, ATTN_FFN_TILINGKEY_MX_CLIP),
                      ASCENDC_TPL_UINT_DECL(ATTN_FFN_OUT_DTYPE, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST,
                                            ATTN_FFN_TILINGKEY_OUT_INT8, ATTN_FFN_TILINGKEY_OUT_E5M2,
                                            ATTN_FFN_TILINGKEY_OUT_E4M3, ATTN_FFN_TILINGKEY_OUT_E2M1),
                      ASCENDC_TPL_BOOL_DECL(ATTN_FFN_SCALE, 0, 1), ASCENDC_TPL_BOOL_DECL(TILINGKEY_SYNC, 0, 1),
                      ASCENDC_TPL_BOOL_DECL(TILINGKEY_ACTIVE_MASK, 0, 1),
                      ASCENDC_TPL_UINT_DECL(ARCH_TAG, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST, TILINGKEY_TPL_A5), );
ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(ATTN_FFN_QUANT_MODE, ASCENDC_TPL_UI_LIST,
                                                          ATTN_FFN_TILINGKEY_NO_QUANT, ATTN_FFN_TILINGKEY_PERTOKEN_INT8,
                                                          ATTN_FFN_TILINGKEY_MX, ATTN_FFN_TILINGKEY_MX_CLIP),
                                     ASCENDC_TPL_UINT_SEL(ATTN_FFN_OUT_DTYPE, ASCENDC_TPL_UI_LIST,
                                                          ATTN_FFN_TILINGKEY_OUT_INT8, ATTN_FFN_TILINGKEY_OUT_E5M2,
                                                          ATTN_FFN_TILINGKEY_OUT_E4M3, ATTN_FFN_TILINGKEY_OUT_E2M1),
                                     ASCENDC_TPL_BOOL_SEL(ATTN_FFN_SCALE, 0, 1),
                                     ASCENDC_TPL_BOOL_SEL(TILINGKEY_SYNC, 0, 1),
                                     ASCENDC_TPL_BOOL_SEL(TILINGKEY_ACTIVE_MASK, 0, 1),
                                     ASCENDC_TPL_UINT_SEL(ARCH_TAG, ASCENDC_TPL_UI_LIST, TILINGKEY_TPL_A5), ), );
} // namespace Mc2Tiling
#endif
