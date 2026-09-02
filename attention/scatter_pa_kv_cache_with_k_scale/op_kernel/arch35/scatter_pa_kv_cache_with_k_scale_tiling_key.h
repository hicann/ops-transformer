/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SCATTER_PA_KV_CACHE_WITH_K_SCALE_TILING_KEY_H
#define SCATTER_PA_KV_CACHE_WITH_K_SCALE_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"

// 场景模式：特化场景（k_head_size == v_head_size）
#define SCATTER_KV_CACHE_SCENE_SPECIALIZED 0
// 场景模式：泛化场景（k_head_size != v_head_size）
#define SCATTER_KV_CACHE_SCENE_GENERALIZED 1

ASCENDC_TPL_ARGS_DECL(ScatterPaKvCacheWithKScale,
                      ASCENDC_TPL_UINT_DECL(schMode, 1, ASCENDC_TPL_UI_LIST, SCATTER_KV_CACHE_SCENE_SPECIALIZED,
                                            SCATTER_KV_CACHE_SCENE_GENERALIZED));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST,
                                                          SCATTER_KV_CACHE_SCENE_SPECIALIZED,
                                                          SCATTER_KV_CACHE_SCENE_GENERALIZED)));

#endif // SCATTER_PA_KV_CACHE_WITH_K_SCALE_TILING_KEY_H
