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
 * \file quant_flash_mla_with_kvcache_template_tiling_key.h
 * \brief QuantFlashMlaWithKvcache TilingKey定义（MLA FP8/HIF8 全量化）
 */

#ifndef TEMPLATE_TILING_KEY_QUANT_FLASH_MLA_WITH_KVCACHE_H_
#define TEMPLATE_TILING_KEY_QUANT_FLASH_MLA_WITH_KVCACHE_H_

#include "ascendc/host_api/tiling/template_argument.h"
#include "quant_flash_mla_with_kvcache_common_def.h"
#include "quant_flash_mla_with_kvcache_tiling_data.h"

using namespace optiling;

ASCENDC_TPL_ARGS_DECL(QuantFlashMlaWithKvcache,
                      //    InOutLayoutType (8-bit)
                      //    0: InOutLayoutType_TND_TND（输入TND, 输出TND）
                      //    1: InOutLayoutType_TND_BSND（输入TND, 输出BSND）
                      //    2: InOutLayoutType_TND_BNSD（输入TND, 输出BNSD）
                      //    3: InOutLayoutType_TND_NTD（输入TND, 输出NTD）
                      ASCENDC_TPL_UINT_DECL(InOutLayoutType, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_RANGE, 1, 0, 255),
                      //    Config (10-bit)
                      //    0: Config_S1Aligned64_S2Aligned256_DAligned576_DVAligned512
                      ASCENDC_TPL_UINT_DECL(Config, ASCENDC_TPL_10_BW, ASCENDC_TPL_UI_RANGE, 1, 0, 1023),
                      //    QuantMode (5-bit)
                      //    1: QMLA_MLA_FP8_E4M3_FULLQUANT
                      ASCENDC_TPL_UINT_DECL(QuantMode, ASCENDC_TPL_5_BW, ASCENDC_TPL_UI_RANGE, 1, 0, 31),
                      //    HasAttenMask
                      //    false / true
                      ASCENDC_TPL_BOOL_DECL(HasAttenMask, false, true),
                      //    KvLayoutType (2-bit)
                      //    0: KvLayoutType_PA_BBND
                      //    1: KvLayoutType_PA_BNBD
                      //    2: KvLayoutType_PA_NZ
                      ASCENDC_TPL_UINT_DECL(KvLayoutType, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_RANGE, 1, 0, 3),
                      //    IsFd
                      //    false / true
                      ASCENDC_TPL_BOOL_DECL(IsFd, false, true));

ASCENDC_TPL_SEL(
    // FP8
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_TND_TND, InOutLayoutType_TND_BSND,
                             InOutLayoutType_TND_BNSD, InOutLayoutType_TND_NTD),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned64_S2Aligned256_DAligned576_DVAligned512),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, QMLA_MLA_FP8_E4M3_FULLQUANT),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST, KvLayoutType_PA_BBND, KvLayoutType_PA_BNBD,
                             KvLayoutType_PA_NZ),
        ASCENDC_TPL_BOOL_SEL(IsFd, false, true), ASCENDC_TPL_TILING_STRUCT_SEL(QuantFlashMlaWithKvcacheTilingData)));

#endif // TEMPLATE_TILING_KEY_QUANT_FLASH_MLA_WITH_KVCACHE_H_
