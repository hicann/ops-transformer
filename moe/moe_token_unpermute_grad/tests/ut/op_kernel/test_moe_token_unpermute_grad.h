/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_moe_token_unpermute_grad.h
 * \brief
 */
#ifndef _MOE_TOKEN_UNPERMUTE_GRAD_TILING_H_
#define _MOE_TOKEN_UNPERMUTE_GRAD_TILING_H_

#include "kernel_tiling/kernel_tiling.h"

#include <cstdint>
#include <cstring>

#define DT_BF16 bfloat16_t
#define ORIG_DTYPE_START DT_BF16
#define __CCE_UT_TEST__

inline void InitMoeTokenUnpermuteGradTilingData(uint8_t *tiling, MoeTokenUnpermuteGradTilingData *const_data)
{
    memcpy(const_data, tiling, sizeof(MoeTokenUnpermuteGradTilingData));
}
#define DTYPE_PERMUTED_TOKENS bfloat16_t
#define DTYPE_PROBS float
#define GET_TILING_DATA(tilingData, tilingPointer) \
    MoeTokenUnpermuteGradTilingData tilingData; \
    InitMoeTokenUnpermuteGradTilingData(tilingPointer, &tilingData)
#endif
