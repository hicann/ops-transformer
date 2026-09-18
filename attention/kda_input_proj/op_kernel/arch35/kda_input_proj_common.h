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
 * \file kda_input_proj_common.h
 * \brief
 */

#ifndef KDA_INPUT_PROJ_COMMON_H
#define KDA_INPUT_PROJ_COMMON_H

#include "kernel_operator.h"

#ifndef DTYPE_X
#define DTYPE_X bfloat16_t
#endif

#ifndef DTYPE_QKV
#define DTYPE_QKV bfloat16_t
#endif

#ifndef DTYPE_BETA
#define DTYPE_BETA float
#endif

#ifndef DTYPE_GATE
#define DTYPE_GATE bfloat16_t
#endif

#ifndef DTYPE_G
#define DTYPE_G bfloat16_t
#endif

namespace KdaInputProj {
template <bool TransWeightQkv = true, bool TransWeightBeta = true, bool TransWeightGate = true,
          bool TransWeightG = true>
struct KdaInputProjType {
    using DtypeX = DTYPE_X;
    using DtypeQkv = DTYPE_QKV;
    using DtypeBeta = DTYPE_BETA;
    using DtypeGate = DTYPE_GATE;
    using DtypeG = DTYPE_G;

    static constexpr bool TRANS_WEIGHT_QKV = TransWeightQkv;
    static constexpr bool TRANS_WEIGHT_BETA = TransWeightBeta;
    static constexpr bool TRANS_WEIGHT_GATE = TransWeightGate;
    static constexpr bool TRANS_WEIGHT_G = TransWeightG;
};
} // namespace KdaInputProj

#endif // KDA_INPUT_PROJ_COMMON_H
