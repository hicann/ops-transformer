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
 * \file fused_gdn_gating_tiling_key.h
 * \brief fused_gdn_gating tiling key definitions
 */

#ifndef FUSED_GDN_GATING_TILING_KEY_H
#define FUSED_GDN_GATING_TILING_KEY_H

// 910/910B tiling keys (dtype combinations)
#define TILING_KEY_FGG_BF16_FLOAT 1UL
#define TILING_KEY_FGG_FP16_FLOAT 2UL
#define TILING_KEY_FGG_BF16_BF16 3UL
#define TILING_KEY_FGG_FP16_BF16 4UL
#define TILING_KEY_FGG_BF16_FP16 5UL
#define TILING_KEY_FGG_FP16_FP16 6UL

// 310P tiling key
#define TILING_KEY_FGG_310P_DEFAULT 200000UL

#endif // FUSED_GDN_GATING_TILING_KEY_H
