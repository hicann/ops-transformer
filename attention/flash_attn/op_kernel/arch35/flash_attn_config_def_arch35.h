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
 * \file flash_attn_config_def_arch35.h
 * \brief FlashAttn arch35 configuration definitions.
 */

#ifndef FLASH_ATTN_CONFIG_DEF_ARCH35_H_
#define FLASH_ATTN_CONFIG_DEF_ARCH35_H_

#include <cstdint>
#include "../utils/flash_attn_common_def.h"

// A5: S1=sOuter*2, S2=sInner; D/DV are template widths, originalDV is the actual V/output width.
// ND and merged DN share the BMM result slots: ubMm1ResBufcnt is the shared count, ubMm2ResBufcnt=0.
// Fields: S1/S2/D/DV/originalDV, UB mm1/mm2/vec1/mask, L1 P/Q/KV.
static constexpr ConfigParams ConfigValue[] = {
    {inferS1TemplateType::Aligned128, inferS2TemplateType::Aligned128, inferDTemplateType::Aligned64,
     inferDTemplateType::Aligned64, inferDTemplateType::Aligned64, 4, 0, 2, 2, 3, 2, 4}, // config=0
    {inferS1TemplateType::Aligned64, inferS2TemplateType::Aligned256, inferDTemplateType::Aligned64,
     inferDTemplateType::Aligned64, inferDTemplateType::Aligned64, 4, 0, 2, 2, 3, 2, 4}, // config=1
    {inferS1TemplateType::Aligned128, inferS2TemplateType::Aligned128, inferDTemplateType::Aligned128,
     inferDTemplateType::Aligned128, inferDTemplateType::Aligned128, 4, 0, 2, 2, 3, 2, 4}, // config=2
    {inferS1TemplateType::Aligned64, inferS2TemplateType::Aligned256, inferDTemplateType::Aligned128,
     inferDTemplateType::Aligned128, inferDTemplateType::Aligned128, 4, 0, 2, 2, 3, 2, 4}, // config=3
    {inferS1TemplateType::Aligned128, inferS2TemplateType::Aligned128, inferDTemplateType::Aligned256,
     inferDTemplateType::Aligned256, inferDTemplateType::Aligned256, 2, 0, 2, 2, 3, 2, 4}, // config=4
    {inferS1TemplateType::Aligned64, inferS2TemplateType::Aligned256, inferDTemplateType::Aligned256,
     inferDTemplateType::Aligned256, inferDTemplateType::Aligned256, 2, 0, 2, 2, 3, 2, 2}, // config=5
    {inferS1TemplateType::Aligned128, inferS2TemplateType::Aligned128, inferDTemplateType::Aligned192,
     inferDTemplateType::Aligned128, inferDTemplateType::Aligned128, 2, 0, 2, 2, 3, 2, 4}, // config=6
    {inferS1TemplateType::Aligned64, inferS2TemplateType::Aligned256, inferDTemplateType::Aligned192,
     inferDTemplateType::Aligned128, inferDTemplateType::Aligned128, 2, 0, 2, 2, 3, 2, 2}, // config=7
    {inferS1TemplateType::Aligned128, inferS2TemplateType::Aligned128, inferDTemplateType::Aligned128,
     inferDTemplateType::Aligned128, inferDTemplateType::Aligned72, 4, 0, 2, 2, 3, 2, 4}, // config=8
    {inferS1TemplateType::Aligned64, inferS2TemplateType::Aligned256, inferDTemplateType::Aligned128,
     inferDTemplateType::Aligned128, inferDTemplateType::Aligned72, 4, 0, 2, 2, 3, 2, 4}, // config=9
    {inferS1TemplateType::Aligned128, inferS2TemplateType::Aligned256, inferDTemplateType::Aligned64,
     inferDTemplateType::Aligned64, inferDTemplateType::Aligned64, 2, 1, 2, 0, 3, 2, 4}, // config=10
    {inferS1TemplateType::Aligned128, inferS2TemplateType::Aligned256, inferDTemplateType::Aligned128,
     inferDTemplateType::Aligned128, inferDTemplateType::Aligned128, 2, 0, 2, 0, 3, 2, 4}, // config=11
    {inferS1TemplateType::Aligned128, inferS2TemplateType::Aligned256, inferDTemplateType::Aligned192,
     inferDTemplateType::Aligned128, inferDTemplateType::Aligned128, 2, 0, 2, 0, 3, 2, 4}, // config=12
    {inferS1TemplateType::Aligned128, inferS2TemplateType::Aligned256, inferDTemplateType::Aligned128,
     inferDTemplateType::Aligned128, inferDTemplateType::Aligned72, 2, 0, 2, 0, 3, 2, 4}, // config=13
};

// Config macro definitions for D=64 (原始DV=64)
#define Config_S1Aligned128_S2Aligned128_DAligned64_DVAligned64_OriginalDV64 0
#define Config_S1Aligned64_S2Aligned256_DAligned64_DVAligned64_OriginalDV64 1

// Config macro definitions for D=128 (原始DV=128, 与模板档一致)
#define Config_S1Aligned128_S2Aligned128_DAligned128_DVAligned128_OriginalDV128 2
#define Config_S1Aligned64_S2Aligned256_DAligned128_DVAligned128_OriginalDV128 3

// Config macro definitions for D=128, 原始DV=72 (DV=72 复用 128 档; 除 originalDV 外与 OriginalDV128 档位相同)
#define Config_S1Aligned128_S2Aligned128_DAligned128_DVAligned128_OriginalDV72 8
#define Config_S1Aligned64_S2Aligned256_DAligned128_DVAligned128_OriginalDV72 9

// Config macro definitions for D=256 (原始DV=256)
#define Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned256_OriginalDV256 4
#define Config_S1Aligned64_S2Aligned256_DAligned256_DVAligned256_OriginalDV256 5

// Config macro definitions for D=192, DV=128 (MLA qk != v, 原始DV=128)
#define Config_S1Aligned128_S2Aligned128_DAligned192_DVAligned128_OriginalDV128 6
#define Config_S1Aligned64_S2Aligned256_DAligned192_DVAligned128_OriginalDV128 7

// Config macro definitions for 不合轴 DN(templateId=2), S2=Aligned256
#define Config_S1Aligned128_S2Aligned256_DAligned64_DVAligned64_OriginalDV64 10
#define Config_S1Aligned128_S2Aligned256_DAligned128_DVAligned128_OriginalDV128 11
#define Config_S1Aligned128_S2Aligned256_DAligned192_DVAligned128_OriginalDV128 12
#define Config_S1Aligned128_S2Aligned256_DAligned128_DVAligned128_OriginalDV72 13

#endif // FLASH_ATTN_CONFIG_DEF_ARCH35_H_
