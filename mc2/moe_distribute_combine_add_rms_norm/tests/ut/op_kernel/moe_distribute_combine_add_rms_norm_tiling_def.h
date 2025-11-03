/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
#ifndef MOE_DISTRIBUTE_COMBINE_ADD_RMS_NORM_TILING_DEF_H
#define MOE_DISTRIBUTE_COMBINE_ADD_RMS_NORM_TILING_DEF_H

#include "kernel_tiling/kernel_tiling.h"
#include "../../../../../mc2/moe_distribute_combine_v2/op_kernel/moe_distribute_combine_v2_tiling.h"

inline void InitMoeDistributeCombineAddRmsNormTilingData(uint8_t* tiling,
                                                         MoeDistributeCombineV2TilingData* const_data)
{
    memcpy(const_data, tiling, sizeof(MoeDistributeCombineV2TilingData));
}

#define GET_TILING_DATA(tiling_data, tiling_arg)                                                      \
        MoeDistributeCombineV2TilingData tiling_data;                                                 \
        InitMoeDistributeCombineAddRmsNormTilingData(tiling_arg, &tiling_data)

#define GET_TILING_DATA_WITH_STRUCT(MoeDistributeCombineV2TilingData, tiling_data, tiling_arg)        \
        MoeDistributeCombineV2TilingData tiling_data;                                                 \
        InitMoeDistributeCombineAddRmsNormTilingData(tiling_arg, &tiling_data)

struct float16_t {
    uint16_t raw;

    constexpr float16_t(uint16_t, bool) : raw(raw) {}

    float16_t() = default;
    float16_t(float f) { (*this) = f; }

    float16_t &operator=(float f);

    operator float() const;
    float f() { return (float)(*this); }

    float16_t &operator+=(float16_t a) {
        (*this) = float(f() + a.f());
        return *this;
    }
};

#define DTYPE_X int64_t


#endif