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
 * \file flash_attn_config_def_arch92.h
 * \brief FlashAttn arch92 configuration definitions.
 */

#ifndef FLASH_ATTN_CONFIG_DEF_ARCH92_H_
#define FLASH_ATTN_CONFIG_DEF_ARCH92_H_

#include <cstdint>
#include "../utils/flash_attn_common_def.h"

// A6: keep host config indices and instantiate only combinations used by this architecture.
// config=0 is reserved; ND uses 1/2/4/5, unmerged DN uses 1/3.
// Fields: S1/S2/D/DV/originalDV, UB mm1/mm2/vec1/mask, L1 P/Q/KV.
static constexpr ConfigParams ConfigValue[] = {
    {inferS1TemplateType::Aligned128, inferS2TemplateType::Aligned128, inferDTemplateType::Aligned64,
     inferDTemplateType::Aligned64, inferDTemplateType::Aligned64, 2, 1, 2, 1, 3, 2, 4}, // config=0
    {inferS1TemplateType::Aligned64, inferS2TemplateType::Aligned256, inferDTemplateType::Aligned64,
     inferDTemplateType::Aligned64, inferDTemplateType::Aligned64, 2, 2, 2, 2, 3, 2, 4}, // config=1
    {inferS1TemplateType::Aligned128, inferS2TemplateType::Aligned128, inferDTemplateType::Aligned128,
     inferDTemplateType::Aligned128, inferDTemplateType::Aligned128, 2, 1, 2, 2, 3, 2, 4}, // config=2
    {inferS1TemplateType::Aligned128, inferS2TemplateType::Aligned256, inferDTemplateType::Aligned128,
     inferDTemplateType::Aligned128, inferDTemplateType::Aligned128, 1, 1, 1, 1, 3, 2, 4}, // config=3
    {inferS1TemplateType::Aligned96, inferS2TemplateType::Aligned256, inferDTemplateType::Aligned256,
     inferDTemplateType::Aligned256, inferDTemplateType::Aligned256, 1, 1, 1, 1, 3, 2, 2}, // config=4
    {inferS1TemplateType::Aligned64, inferS2TemplateType::Aligned256, inferDTemplateType::Aligned256,
     inferDTemplateType::Aligned256, inferDTemplateType::Aligned256, 2, 1, 2, 2, 3, 2, 2}, // config=5
};

#endif // FLASH_ATTN_CONFIG_DEF_ARCH92_H_
