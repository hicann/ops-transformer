/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fused_gdn_gating_tiling.h
 * \brief CPU simulation (tikicpulib) tiling stub for fused_gdn_gating kernel UT.
 *        Struct definition is shared from op_kernel/fused_gdn_gating_tiling_data.h.
 */

#ifndef TEST_FUSED_GDN_GATING_TILING_H
#define TEST_FUSED_GDN_GATING_TILING_H

#include <cstdint>
#include <cstring>

#include "fused_gdn_gating_tiling_data.h"

inline void InitFusedGdnGatingTilingData(uint8_t *tiling, FusedGdnGating::FusedGdnGatingTilingData *tilingData)
{
    std::memcpy(tilingData, tiling, sizeof(FusedGdnGating::FusedGdnGatingTilingData));
}

#ifdef GET_TILING_DATA
#undef GET_TILING_DATA
#endif

#define GET_TILING_DATA(tilingData, tilingArg) \
    FusedGdnGating::FusedGdnGatingTilingData tilingData; \
    InitFusedGdnGatingTilingData(reinterpret_cast<uint8_t *>(tilingArg), &tilingData)

#ifdef GET_TILING_DATA_WITH_STRUCT
#undef GET_TILING_DATA_WITH_STRUCT
#endif

#define GET_TILING_DATA_WITH_STRUCT(tilingStruct, tilingData, tilingArg) \
    tilingStruct tilingData; \
    InitFusedGdnGatingTilingData(reinterpret_cast<uint8_t *>(tilingArg), &tilingData)

#ifdef REGISTER_TILING_DEFAULT
#undef REGISTER_TILING_DEFAULT
#endif

#define REGISTER_TILING_DEFAULT(structName)

#endif // TEST_FUSED_GDN_GATING_TILING_H
