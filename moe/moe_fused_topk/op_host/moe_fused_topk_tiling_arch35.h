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
 * \file moe_fused_topk_tiling_arch35.h
 * \brief Ascend950 tiling data definitions for MoeFusedTopk.
 */

#ifndef ASCEND_OPS_MOE_FUSED_TOPK_TILING_ARCH35_H
#define ASCEND_OPS_MOE_FUSED_TOPK_TILING_ARCH35_H

#include <cstdint>

#include "kernel_tiling/kernel_tiling.h"
#include "moe_fused_topk_tiling.h"

namespace optiling {

// Reuse the registered tiling-data class so nested TopkTiling data is
// serialized into its device-side POD layout instead of copying the host-side
// TilingDef object (which contains pointers and bookkeeping fields).
using MoeFusedTopkArch35TilingData = MoeFusedTopkTilingData;

ge::graphStatus TilingMoeFusedTopkArch35(gert::TilingContext *context);

} // namespace optiling

#endif // ASCEND_OPS_MOE_FUSED_TOPK_TILING_ARCH35_H
