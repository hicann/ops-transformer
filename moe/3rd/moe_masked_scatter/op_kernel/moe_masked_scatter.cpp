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
 * \file masked_scatter_apt.cpp
 * \brief kernel file of masked_scatter
 */

#include "kernel_operator.h"
#include "arch35/moe_masked_scatter_arch35.h"

using namespace MoeMaskedScatter;

#define TILING_KEY_BIT_WIDTH_1_PREFIX_SUM_INT32 10
#define TILING_KEY_BIT_WIDTH_2_PREFIX_SUM_INT32 20
#define TILING_KEY_BIT_WIDTH_4_PREFIX_SUM_INT32 40
#define TILING_KEY_BIT_WIDTH_8_PREFIX_SUM_INT32 80
#define TILING_KEY_BIT_WIDTH_1_PREFIX_SUM_INT64 11
#define TILING_KEY_BIT_WIDTH_2_PREFIX_SUM_INT64 21
#define TILING_KEY_BIT_WIDTH_4_PREFIX_SUM_INT64 41
#define TILING_KEY_BIT_WIDTH_8_PREFIX_SUM_INT64 81

extern "C" __global__ __aicore__ void moe_masked_scatter(
    GM_ADDR x, GM_ADDR mask, GM_ADDR updates, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
  if (workspace == nullptr) {
    return;
  }
  SetSysWorkspace(workspace);
  GM_ADDR userWs = GetUserWorkspace(workspace);
  if (userWs == nullptr) {
    return;
  }
  GET_TILING_DATA(tilingData, tiling);
  TPipe pipe;
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
  if (TILING_KEY_IS(TILING_KEY_BIT_WIDTH_1_PREFIX_SUM_INT32)) {
    MoeMaskedScatter::MoeMaskedScatterImpl<int8_t, int32_t> op(tilingData, pipe);
    op.Init(x, mask, updates, y, userWs);
    op.Process();
  } else if (TILING_KEY_IS(TILING_KEY_BIT_WIDTH_2_PREFIX_SUM_INT32)) {
    MoeMaskedScatter::MoeMaskedScatterImpl<int16_t, int32_t> op(tilingData, pipe);
    op.Init(x, mask, updates, y, userWs);
    op.Process();
  } else if (TILING_KEY_IS(TILING_KEY_BIT_WIDTH_4_PREFIX_SUM_INT32)) {
    MoeMaskedScatter::MoeMaskedScatterImpl<int32_t, int32_t> op(tilingData, pipe);
    op.Init(x, mask, updates, y, userWs);
    op.Process();
  } else if (TILING_KEY_IS(TILING_KEY_BIT_WIDTH_8_PREFIX_SUM_INT32)) {
    MoeMaskedScatter::MoeMaskedScatterImpl<int64_t, int32_t> op(tilingData, pipe);
    op.Init(x, mask, updates, y, userWs);
    op.Process();
  } else if (TILING_KEY_IS(TILING_KEY_BIT_WIDTH_1_PREFIX_SUM_INT64)) {
    MoeMaskedScatter::MoeMaskedScatterImpl<int8_t, int64_t> op(tilingData, pipe);
    op.Init(x, mask, updates, y, userWs);
    op.Process();
  } else if (TILING_KEY_IS(TILING_KEY_BIT_WIDTH_2_PREFIX_SUM_INT64)) {
    MoeMaskedScatter::MoeMaskedScatterImpl<int16_t, int64_t> op(tilingData, pipe);
    op.Init(x, mask, updates, y, userWs);
    op.Process();
  } else if (TILING_KEY_IS(TILING_KEY_BIT_WIDTH_4_PREFIX_SUM_INT64)) {
    MoeMaskedScatter::MoeMaskedScatterImpl<int32_t, int64_t> op(tilingData, pipe);
    op.Init(x, mask, updates, y, userWs);
    op.Process();
  } else if (TILING_KEY_IS(TILING_KEY_BIT_WIDTH_8_PREFIX_SUM_INT64)) {
    MoeMaskedScatter::MoeMaskedScatterImpl<int64_t, int64_t> op(tilingData, pipe);
    op.Init(x, mask, updates, y, userWs);
    op.Process();
  }
  return;
}
