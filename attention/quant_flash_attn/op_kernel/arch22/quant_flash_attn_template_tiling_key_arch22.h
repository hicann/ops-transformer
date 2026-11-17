/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QUANT_FLASH_ATTN_TEMPLATE_TILING_KEY_ARCH22_H_
#define QUANT_FLASH_ATTN_TEMPLATE_TILING_KEY_ARCH22_H_

#include "ascendc/host_api/tiling/template_argument.h"
#include "quant_flash_attn_tiling_data_int8.h"

// Preserve the seven-field ABI. Config uses 10 bits (pipeline, window and LSE), quant mode 5 bits,
// and KV layout 2 bits. Each unsigned field declares one inclusive range from zero to its bit-width limit.
ASCENDC_TPL_ARGS_DECL(QuantFlashAttn,
                      ASCENDC_TPL_UINT_DECL(InOutLayoutType, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_RANGE, 1, 0, 255),
                      ASCENDC_TPL_UINT_DECL(Config, 10, ASCENDC_TPL_UI_RANGE, 1, 0, 1023),
                      ASCENDC_TPL_UINT_DECL(QuantMode, 5, ASCENDC_TPL_UI_RANGE, 1, 0, 31),
                      ASCENDC_TPL_BOOL_DECL(HasAttenMask, 0, 1),
                      ASCENDC_TPL_UINT_DECL(KvLayoutType, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_RANGE, 1, 0, 3),
                      ASCENDC_TPL_BOOL_DECL(IsFd, 0, 1), ASCENDC_TPL_BOOL_DECL(HasVTail, 0, 1));

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, QFA_INT8_TPL_BSND_OR_MIXED),
                         ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, QFA_INT8_CFG_D96_M512_FIXPIPE,
                                              QFA_INT8_CFG_D96_M256_CAUSAL_MHA, QFA_INT8_CFG_D96_M512_FIXPIPE_LSE,
                                              QFA_INT8_CFG_D96_M256_CAUSAL_MHA_LSE),
                         ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, QFA_INT8_QUANT_MODE),
                         ASCENDC_TPL_BOOL_SEL(HasAttenMask, 0),
                         ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST, QFA_INT8_TPL_KV_LAYOUT),
                         ASCENDC_TPL_BOOL_SEL(IsFd, 0), ASCENDC_TPL_BOOL_SEL(HasVTail, 0),
                         ASCENDC_TPL_TILING_STRUCT_SEL(QuantFlashAttnTilingQBlockPayloadInt8)),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, QFA_INT8_TPL_BNSD, QFA_INT8_TPL_TND,
                             QFA_INT8_TPL_BSND_OR_MIXED),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, QFA_INT8_CFG_SHARED_QP, QFA_INT8_CFG_RESIDENT_Q,
                             QFA_INT8_CFG_D64_M512, QFA_INT8_CFG_D128_M512, QFA_INT8_CFG_GENERIC_M512,
                             QFA_INT8_CFG_DYNAMIC_MHA, QFA_INT8_CFG_WINDOW_FLAG, QFA_INT8_CFG_SHARED_QP_LSE,
                             QFA_INT8_CFG_RESIDENT_Q_LSE, QFA_INT8_CFG_D64_M512_LSE, QFA_INT8_CFG_D128_M512_LSE,
                             QFA_INT8_CFG_GENERIC_M512_LSE, QFA_INT8_CFG_DYNAMIC_MHA_LSE, QFA_INT8_CFG_WINDOW_LSE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, QFA_INT8_QUANT_MODE),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, 0),
        ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST, QFA_INT8_TPL_KV_LAYOUT), ASCENDC_TPL_BOOL_SEL(IsFd, 0),
        ASCENDC_TPL_BOOL_SEL(HasVTail, 0), ASCENDC_TPL_TILING_STRUCT_SEL(QuantFlashAttnTilingPayloadInt8)),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, QFA_INT8_TPL_BNSD),
                         ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, QFA_INT8_CFG_ISOLATED_SHARED_QP,
                                              QFA_INT8_CFG_D96_M512_FIXPIPE, QFA_INT8_CFG_ISOLATED_SHARED_QP_LSE,
                                              QFA_INT8_CFG_D96_M512_FIXPIPE_LSE),
                         ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, QFA_INT8_QUANT_MODE),
                         ASCENDC_TPL_BOOL_SEL(HasAttenMask, 0),
                         ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST, QFA_INT8_TPL_KV_LAYOUT),
                         ASCENDC_TPL_BOOL_SEL(IsFd, 0), ASCENDC_TPL_BOOL_SEL(HasVTail, 0),
                         ASCENDC_TPL_TILING_STRUCT_SEL(QuantFlashAttnTilingPayloadInt8)),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, QFA_INT8_TPL_BNSD, QFA_INT8_TPL_TND,
                                              QFA_INT8_TPL_BSND_OR_MIXED),
                         ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, QFA_INT8_CFG_D64_M512_TASK_MAP,
                                              QFA_INT8_CFG_D64_M512_TASK_MAP_LSE),
                         ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, QFA_INT8_QUANT_MODE),
                         ASCENDC_TPL_BOOL_SEL(HasAttenMask, 0),
                         ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST, QFA_INT8_TPL_KV_LAYOUT),
                         ASCENDC_TPL_BOOL_SEL(IsFd, 0), ASCENDC_TPL_BOOL_SEL(HasVTail, 0),
                         ASCENDC_TPL_TILING_STRUCT_SEL(QuantFlashAttnTilingQBlockPayloadInt8)));
#endif
