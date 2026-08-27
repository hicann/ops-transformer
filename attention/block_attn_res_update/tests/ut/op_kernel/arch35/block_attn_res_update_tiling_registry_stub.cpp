/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>

#include "register/tilingdata_base.h"

namespace optiling {

// AddOpTestCase always asks the CANN tiling registry to generate a kernel-side
// tiling header. The production operator intentionally uses a POD ABI, so this
// UT-only descriptor mirrors that ABI and keeps the generator path valid.
BEGIN_TILING_DATA_DEF(BlockAttnResUpdateTilingData)
TILING_DATA_FIELD_DEF(uint32_t, dSize);
TILING_DATA_FIELD_DEF(uint32_t, tPerCore);
TILING_DATA_FIELD_DEF(uint32_t, lastTPerCore);
TILING_DATA_FIELD_DEF(uint32_t, tileT);
TILING_DATA_FIELD_DEF(uint32_t, statsTStride);
TILING_DATA_FIELD_DEF(float, eps);
TILING_DATA_FIELD_DEF(float, invD);
TILING_DATA_FIELD_DEF(uint16_t, usedCoreNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(BlockAttnResUpdate, BlockAttnResUpdateTilingData)

} // namespace optiling
