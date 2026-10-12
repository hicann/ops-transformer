/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_TILING_H_
#define QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_TILING_H_

#include "../../../op_kernel/arch35/qkv_rms_norm_rope_cache_with_k_scale_tiling_data.h"
#include "register/op_impl_registry.h"

namespace optiling {

using QkvRmsNormRopeCacheWithKScaleKernelTiling::QkvRmsNormRopeCacheWithKScaleTilingData;

struct QkvRmsNormRopeCacheWithKScaleCompileInfo {
    uint32_t aicNum = 0;
    uint32_t aivNum = 0;
    uint64_t ubSize = 0;
    uint64_t l1Size = 0;
    uint64_t l0cSize = 0;
    uint64_t opWorkspaceSize = 0;
};

ge::graphStatus Tiling4QkvRmsNormRopeCacheWithKScale(gert::TilingContext* context);
ge::graphStatus TilingPrepare4QkvRmsNormRopeCacheWithKScale(gert::TilingParseContext* context);

} // namespace optiling

#endif // QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_TILING_H_
