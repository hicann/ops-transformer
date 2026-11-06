/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once

#include <cstdint>
#include <register/tilingdata_base.h>

namespace optiling {

BEGIN_TILING_DATA_DEF(TextGatedResidualTilingData)
TILING_DATA_FIELD_DEF(int64_t, B);
TILING_DATA_FIELD_DEF(int64_t, S);
TILING_DATA_FIELD_DEF(uint32_t, M);
TILING_DATA_FIELD_DEF(uint32_t, D);
TILING_DATA_FIELD_DEF(uint32_t, R);
TILING_DATA_FIELD_DEF(uint32_t, H);
TILING_DATA_FIELD_DEF(uint32_t, C);
TILING_DATA_FIELD_DEF(uint32_t, usedAic);
TILING_DATA_FIELD_DEF(uint32_t, mTask);
TILING_DATA_FIELD_DEF(uint32_t, taskNum);
TILING_DATA_FIELD_DEF(uint32_t, L1M0);
TILING_DATA_FIELD_DEF(uint32_t, L1N0);
TILING_DATA_FIELD_DEF(uint32_t, L1M1);
TILING_DATA_FIELD_DEF(uint32_t, L1N1);
TILING_DATA_FIELD_DEF(uint32_t, szT);
TILING_DATA_FIELD_DEF(uint32_t, useCombine);
TILING_DATA_FIELD_DEF(uint32_t, hasGamma);
TILING_DATA_FIELD_DEF(uint32_t, tilingKey);
TILING_DATA_FIELD_DEF(float, eps);
TILING_DATA_FIELD_DEF(uint32_t, smallN);
TILING_DATA_FIELD_DEF(uint32_t, splitkFactor);
TILING_DATA_FIELD_DEF(uint32_t, nConcat);
TILING_DATA_FIELD_DEF(uint32_t, nWs);
TILING_DATA_FIELD_DEF(uint32_t, nWsGate);
TILING_DATA_FIELD_DEF(uint32_t, needBackward);
TILING_DATA_FIELD_DEF(uint32_t, pad0);
TILING_DATA_FIELD_DEF(uint32_t, pad1);
TILING_DATA_FIELD_DEF(uint32_t, pad2);
TILING_DATA_FIELD_DEF(uint32_t, pad3);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(TextGatedResidual, TextGatedResidualTilingData)

struct TextGatedResidualCompileInfo {};
} // namespace optiling
