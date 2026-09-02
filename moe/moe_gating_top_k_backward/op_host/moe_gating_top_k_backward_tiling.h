/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file moe_gating_top_k_backward_tiling.h
 * \brief
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_MOE_GATING_TOP_K_BACKWARD_H
#define AIR_CXX_RUNTIME_V2_OP_IMPL_MOE_GATING_TOP_K_BACKWARD_H

#include <cmath>
#include <cstdint>
#include <vector>
#include <algorithm>

#include "tiling/tiling_api.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"
#include "register/op_def_registry.h"
#include "log/log.h"

#include "platform/platform_infos_def.h"
#include "util/math_util.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MoeGatingTopKBackwardTilingData)
TILING_DATA_FIELD_DEF(int64_t, needCoreNum);  // 所需要的总核数
TILING_DATA_FIELD_DEF(int64_t, perCoreRows);  // 头核需要处理的 token 数
TILING_DATA_FIELD_DEF(int64_t, lastCoreRows); // 尾核需要处理的 token 数
TILING_DATA_FIELD_DEF(int64_t,
                      baseRows); // 头核、尾核的头 loop 一次搬运的 token 数，同时也是一个 UB 可以处理的最多 token 数
TILING_DATA_FIELD_DEF(int64_t, perLoopTimes);   // 头核需要 loop 的次数
TILING_DATA_FIELD_DEF(int64_t, perTailRows);    // 头核尾 loop 一次搬运的 token 数
TILING_DATA_FIELD_DEF(int64_t, lastLoopTimes);  // 尾核需要 loop 的次数
TILING_DATA_FIELD_DEF(int64_t, lastTailRows);   // 尾核尾 loop 一次搬运的 token 数
TILING_DATA_FIELD_DEF(int64_t, tokenCount);     // x_norm shape [M,N] 中的 M, 所有的 token 数量
TILING_DATA_FIELD_DEF(int64_t, expertCount);    // x_norm shape [M,N] 中的 N, 所有的 expert 数量
TILING_DATA_FIELD_DEF(int64_t, k);              // grad_y shape [M,K] 中的 K，选择的前 K 个 expert 数量
TILING_DATA_FIELD_DEF(int64_t, gradYDtypeSize); // grad_y的数据类型
TILING_DATA_FIELD_DEF(int64_t, renorm);
TILING_DATA_FIELD_DEF(int64_t, normType);
TILING_DATA_FIELD_DEF(float, routedScalingFactor);
TILING_DATA_FIELD_DEF(float, eps);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(MoeGatingTopKBackward, MoeGatingTopKBackwardTilingData)

struct MoeGatingTopKBackwardCompileInfo {
    uint64_t ubSize = 0;
    uint64_t numBlocks = 0;
};
} // namespace optiling
#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_MOE_GATING_TOP_K_BACKWARD_H
