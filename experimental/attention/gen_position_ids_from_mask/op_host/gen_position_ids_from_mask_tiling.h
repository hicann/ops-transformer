/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/**
 * @file gen_position_ids_from_mask_tiling.h
 */
#ifndef GEN_POSITION_IDS_FROM_MASK_TILING_H_
#define GEN_POSITION_IDS_FROM_MASK_TILING_H_

#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(GenPositionIdsFromMaskTilingData)
TILING_DATA_FIELD_DEF(int64_t, b);                // batch
TILING_DATA_FIELD_DEF(int64_t, s);                // seq len
TILING_DATA_FIELD_DEF(int64_t, paddingFillValue); // padding 填充值
TILING_DATA_FIELD_DEF(uint32_t, coreNum);         // 实际使用核数
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GenPositionIdsFromMask, GenPositionIdsFromMaskTilingData)

} // namespace optiling

#endif // GEN_POSITION_IDS_FROM_MASK_TILING_H_
