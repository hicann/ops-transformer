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

BEGIN_TILING_DATA_DEF(BlockAttnResPrepareMixTilingData)
TILING_DATA_FIELD_DEF(uint64_t, qL1Elems);
TILING_DATA_FIELD_DEF(uint64_t, vL1Elems);
TILING_DATA_FIELD_DEF(uint64_t, eL1Elems);
TILING_DATA_FIELD_DEF(uint64_t, vUbElems);
TILING_DATA_FIELD_DEF(uint64_t, dotUbElems);
TILING_DATA_FIELD_DEF(uint64_t, reduceUbElems);
TILING_DATA_FIELD_DEF(uint64_t, softmaxUbElems);
TILING_DATA_FIELD_DEF(uint64_t, workspacePerCoreElems);
TILING_DATA_FIELD_DEF(uint32_t, totalT);
TILING_DATA_FIELD_DEF(uint32_t, totalS);
TILING_DATA_FIELD_DEF(uint32_t, totalWorkUnits);
TILING_DATA_FIELD_DEF(uint32_t, totalD);
TILING_DATA_FIELD_DEF(uint32_t, baseT);
TILING_DATA_FIELD_DEF(uint32_t, baseS);
TILING_DATA_FIELD_DEF(uint32_t, baseD);
TILING_DATA_FIELD_DEF(uint32_t, baseDAlign);
TILING_DATA_FIELD_DEF(uint32_t, sTileNum);
TILING_DATA_FIELD_DEF(uint32_t, dTileNum);
TILING_DATA_FIELD_DEF(uint32_t, sAlign);
TILING_DATA_FIELD_DEF(uint32_t, nAlign);
TILING_DATA_FIELD_DEF(uint32_t, mm1NAlign);
TILING_DATA_FIELD_DEF(uint32_t, dAlign);
TILING_DATA_FIELD_DEF(float, eps);
TILING_DATA_FIELD_DEF(uint16_t, usedCoreNum);
TILING_DATA_FIELD_DEF(uint16_t, aicCoreNum);
TILING_DATA_FIELD_DEF(uint16_t, aivCoreNum);
TILING_DATA_FIELD_DEF(uint8_t, totalN);
TILING_DATA_FIELD_DEF(uint8_t, qL1BufferNum);
TILING_DATA_FIELD_DEF(uint8_t, vL1BufferNum);
TILING_DATA_FIELD_DEF(uint8_t, vUbBufferNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(BlockAttnResPrepare, BlockAttnResPrepareMixTilingData)

} // namespace optiling
