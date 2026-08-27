/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_flash_attn_template_tiling_key.h
 * \brief QuantFlashAttn TilingKey定义（量化，MXFP8_FP32_PREFILL/DECODE）
 */

#ifndef TEMPLATE_TILING_KEY_QUANT_FLASH_ATTN_H_
#define TEMPLATE_TILING_KEY_QUANT_FLASH_ATTN_H_

#include "ascendc/host_api/tiling/template_argument.h"
#include "quant_flash_attn_common_def.h"
#include "quant_flash_attn_tiling_data.h"

using namespace optiling;

ASCENDC_TPL_ARGS_DECL(QuantFlashAttn,
                      //    InOutLayoutType (8-bit)
                      //    0: InOutLayoutType_BSND_BSND
                      //    1: InOutLayoutType_BNSD_BNSD
                      //    2: InOutLayoutType_TND_TND
                      ASCENDC_TPL_UINT_DECL(InOutLayoutType, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_RANGE, 1, 0, 255),
                      //    Config (10-bit)
                      //    0: Config_S1Aligned128_S2Aligned512_DAligned64_DVAligned64
                      //    1: Config_S1Aligned128_S2Aligned512_DAligned128_DVAligned128
                      //    2: Config_S1Aligned128_S2Aligned256_DAligned128_DVAligned128
                      //    3: Config_S1Aligned128_S2Aligned256_DAligned256_DVAligned256
                      //    4: Config_S1Aligned128_S2Aligned512_DAligned72_DVAligned72
                      ASCENDC_TPL_UINT_DECL(Config, ASCENDC_TPL_10_BW, ASCENDC_TPL_UI_RANGE, 1, 0, 1023),
                      //    QuantMode (5-bit)
                      //    1: QFA_MXFP8_FP32_PREFILL
                      //    2: QFA_MXFP8_FP32_DECODE
                      //    6: QFA_GQA_FP8_FULLQUANT
                      //    0: QFA_HIF8_FP32
                      ASCENDC_TPL_UINT_DECL(QuantMode, ASCENDC_TPL_5_BW, ASCENDC_TPL_UI_RANGE, 1, 0, 31),
                      //    HasAttenMask
                      //    false / true
                      ASCENDC_TPL_BOOL_DECL(HasAttenMask, false, true),
                      //    KvLayoutType (2-bit)
                      //    0: KvLayoutType_NO_PA
                      //    1: KvLayoutType_PA_BBND
                      //    2: KvLayoutType_PA_BNBD
                      //    3: KvLayoutType_PA_NZ
                      ASCENDC_TPL_UINT_DECL(KvLayoutType, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_RANGE, 1, 0, 3),
                      //    IsFd
                      //    false / true
                      ASCENDC_TPL_BOOL_DECL(IsFd, false, true));

ASCENDC_TPL_SEL(
    // MXFP8
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned128_S2Aligned512_DAligned64_DVAligned64,
                             Config_S1Aligned128_S2Aligned512_DAligned128_DVAligned128,
                             Config_S1Aligned128_S2Aligned256_DAligned256_DVAligned256,
                             Config_S1Aligned128_S2Aligned512_DAligned72_DVAligned72),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, QFA_MXFP8_FP32_PREFILL, QFA_MXFP8_FP32_DECODE),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST, KvLayoutType_NO_PA, KvLayoutType_PA_BBND,
                             KvLayoutType_PA_BNBD, KvLayoutType_PA_NZ),
        ASCENDC_TPL_BOOL_SEL(IsFd, false), ASCENDC_TPL_TILING_STRUCT_SEL(QuantFlashAttnTilingData)),
    // FP8
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_NTD_TND),
                         ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST,
                                              Config_S1Aligned128_S2Aligned256_DAligned128_DVAligned128),
                         ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, QFA_GQA_FP8_FULLQUANT),
                         ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
                         ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST, KvLayoutType_PA_BNBD),
                         ASCENDC_TPL_BOOL_SEL(IsFd, false), ASCENDC_TPL_TILING_STRUCT_SEL(QuantFlashAttnTilingData)),
    // HIF8
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BSND_BSND,
                                              InOutLayoutType_BNSD_BNSD, InOutLayoutType_TND_TND),
                         ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST,
                                              Config_S1Aligned128_S2Aligned256_DAligned128_DVAligned128),
                         ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, QFA_HIF8_FP32),
                         ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
                         ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST, KvLayoutType_NO_PA),
                         ASCENDC_TPL_BOOL_SEL(IsFd, false), ASCENDC_TPL_TILING_STRUCT_SEL(QuantFlashAttnTilingData)));

#endif // TEMPLATE_TILING_KEY_QUANT_FLASH_ATTN_H_
