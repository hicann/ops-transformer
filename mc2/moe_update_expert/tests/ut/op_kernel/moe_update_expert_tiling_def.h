/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
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

#define __aicore__

using namespace MoeUpdateExpertNamespace;

inline void InitMoeUpdateExpertTilingData(uint8_t* tiling, MoeUpdateExpertTilingData* const_data)
{
    memcpy(const_data, tiling, sizeof(MoeUpdateExpertTilingData));
}

#define GET_TILING_DATA(tiling_data, tiling_arg) \
    MoeUpdateExpertTilingData tiling_data;       \
    InitMoeUpdateExpertTilingData(tiling_arg, &tiling_data)
#endif // MOE_UPDATE_EXPERT_TILING_DEF_H