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
 * \file flash_attn_template_tiling_key_arch92.h
 * \brief FlashAttn arch92 tiling-key declarations and template selections.
 */

#ifndef FLASH_ATTN_TEMPLATE_TILING_KEY_ARCH92_H_
#define FLASH_ATTN_TEMPLATE_TILING_KEY_ARCH92_H_

#include "ascendc/host_api/tiling/template_argument.h"
#include "flash_attn_config_def_arch92.h"
#include "../flash_attn_tiling_data.h"

#ifndef ORIG_DTYPE_QUERY
#define ORIG_DTYPE_QUERY (DT_BF16)
#endif

#ifndef ORIG_DTYPE_KEY
#define ORIG_DTYPE_KEY (DT_BF16)
#endif

#ifndef ORIG_DTYPE_ATTENTION_OUT
#define ORIG_DTYPE_ATTENTION_OUT (DT_BF16)
#endif

#define KV_STORAGE_MODE_CONTINUE 0
#define KV_STORAGE_MODE_PA_BSND 1
#define KV_STORAGE_MODE_PA_BNSD 2
#define KV_STORAGE_MODE_PA_NZ 3

ASCENDC_TPL_ARGS_DECL(FlashAttn,
                      // q_out layout (8-bit)
                      //    0: BSND
                      //    1: BNSD
                      //    2: TND
                      //    3: BNSD_BSND
                      ASCENDC_TPL_UINT_DECL(InOutLayoutType, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_RANGE, 1, 0, 3),

                      // kv存储类型 (8-bit)
                      //    0: 连续（不开pageattention）
                      //    1: PA_BBND
                      //    2: PA_BNBD
                      //    3: PA_NZ
                      ASCENDC_TPL_UINT_DECL(KvLayoutType, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_RANGE, 1, 0, 3),

                      // hasAttenMask (1-bit)
                      //    0: false
                      //    1: true
                      ASCENDC_TPL_BOOL_DECL(HasAttenMask, false, true),

                      // Dropout is not implemented by the arch92 paths.
                      ASCENDC_TPL_BOOL_DECL(HasDrop, false, true),

                      // templateId (8-bit)
                      //    0: ND模板
                      //    1: DN模板(合轴)
                      //    2: DN模板(不合轴)
                      ASCENDC_TPL_UINT_DECL(TemplateId, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_RANGE, 1, 0, 2),

                      // Config (4-bit): dimensions, buffer counts and selections are defined per architecture.
                      ASCENDC_TPL_UINT_DECL(Config, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_RANGE, 1, 0, 15), );

ASCENDC_TPL_SEL(
    // ND: D=64/128/256; BAND with D=256 uses config=5.
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BSND,
                                              InOutLayoutType_BNSD, InOutLayoutType_TND, InOutLayoutType_BNSD_BSND),
                         ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3),
                         ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true), ASCENDC_TPL_BOOL_SEL(HasDrop, false),
                         ASCENDC_TPL_UINT_SEL(TemplateId, ASCENDC_TPL_UI_LIST, 0),
                         ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 1, 2, 4, 5),
                         ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttnTilingData)),
    // Unmerged DN: D=64/128.
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BSND,
                                              InOutLayoutType_BNSD, InOutLayoutType_TND, InOutLayoutType_BNSD_BSND),
                         ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3),
                         ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true), ASCENDC_TPL_BOOL_SEL(HasDrop, false),
                         ASCENDC_TPL_UINT_SEL(TemplateId, ASCENDC_TPL_UI_LIST, 2),
                         ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 1, 3),
                         ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttnTilingData)));

#endif // FLASH_ATTN_TEMPLATE_TILING_KEY_ARCH92_H_
