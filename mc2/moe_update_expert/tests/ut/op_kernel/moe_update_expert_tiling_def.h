/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef MOE_UPDATE_EXPERT_TILING_DEF_H
#define MOE_UPDATE_EXPERT_TILING_DEF_H

#include <cstdint>
#include <cstring>

#include "kernel_tiling/kernel_tiling.h"
#include "../../../op_kernel/moe_update_expert_tiling.h"

using namespace MoeUpdateExpertNamespace;

inline void InitMoeUpdateExpertTilingData(uint8_t *tiling, MoeUpdateExpertTilingData *constData)
{
    memcpy(constData, tiling, sizeof(MoeUpdateExpertTilingData));
}

#define GET_TILING_DATA(tilingData, tilingArg) \
    MoeUpdateExpertTilingData tilingData; \
    InitMoeUpdateExpertTilingData(tilingArg, &tilingData)
#endif // MOE_UPDATE_EXPERT_TILING_DEF_H
