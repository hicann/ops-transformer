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
 * \file flash_attn_template_tiling_key.h
 * \brief FlashAttn TilingKey定义（非量化，仅FP16/BF16）
 */

#ifndef TEMPLATE_TILING_KEY_FLASH_ATTN_H_
#define TEMPLATE_TILING_KEY_FLASH_ATTN_H_

#include "ascendc/host_api/tiling/template_argument.h"
#include "../utils/flash_attn_common_def.h"
#include "flash_attn_tiling_data.h"

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

                      // config (3-bit), support D=64/128/256
                      //    config=0: sOuter=64, sInner=128 → D=64,  DV=64
                      //    config=1: sOuter=32, sInner=256 → D=64,  DV=64
                      //    config=2: sOuter=64, sInner=128 → D=128, DV=128
                      //    config=3: sOuter=32, sInner=256 → D=128, DV=128
                      //    config=4: sOuter=64, sInner=128 → D=256, DV=256
                      //    config=5: sOuter=32, sInner=256 → D=256, DV=256
                      ASCENDC_TPL_UINT_DECL(Config, ASCENDC_TPL_3_BW, ASCENDC_TPL_UI_RANGE, 1, 0, 5), );

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BSND,
                                                          InOutLayoutType_BNSD, InOutLayoutType_TND,
                                                          InOutLayoutType_BNSD_BSND),
                                     ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3),
                                     ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
                                     ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3, 4, 5),
                                     ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttnTilingData)), );

#endif // TEMPLATE_TILING_KEY_FLASH_ATTN_H_
