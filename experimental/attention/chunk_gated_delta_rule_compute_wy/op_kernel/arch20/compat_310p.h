/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CHUNK_GATED_DELTA_RULE_COMPUTE_WY_ARCH20_COMPAT_310P_H
#define CHUNK_GATED_DELTA_RULE_COMPUTE_WY_ARCH20_COMPAT_310P_H

#ifndef __CCE_KT_TEST__
#include "kernel_operator.h"
#endif

// Dummy bfloat16_t only needed on 310P (dav_m200) where the compiler
// doesn't provide a native bf16 type. On 910B/910C the compiler's
// __clang_cce_types.h already typedefs bfloat16_t from __bf16.
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 200) && !defined(__bfloat16_t_defined)
#define __bfloat16_t_defined
#define __COMPAT_310P_ACTIVE__
struct bfloat16_t {
    uint16_t val;
    bfloat16_t() = default;
    bfloat16_t(float v)
        : val(0)
    {
        (void)v;
    }
    operator float() const
    {
        return 0.f;
    }
};
#endif

// 310P has no fixpipe unit; post-matmul stores go through MTE3
#ifndef PIPE_FIX
#define PIPE_FIX PIPE_MTE3
#endif

// 310P renames LoadDataWithSparse → LoadDataWithSparseCal
#ifdef __COMPAT_310P_ACTIVE__
#define LoadDataWithSparse LoadDataWithSparseCal
#endif

// 310P has no AscendC::ToFloat — dummy bfloat16_t already has operator float()
#ifdef __COMPAT_310P_ACTIVE__
namespace AscendC {
inline float ToFloat(bfloat16_t v)
{
    return (float)v;
}
} // namespace AscendC
#endif

#endif // CHUNK_GATED_DELTA_RULE_COMPUTE_WY_ARCH20_COMPAT_310P_H
