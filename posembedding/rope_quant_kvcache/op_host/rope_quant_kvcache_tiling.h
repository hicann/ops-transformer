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
 * \file rope_quant_kvcache_tiling.h
 * \brief
 */
#ifndef ROPE_QUANT_KVCACHE_TILING_H
#define ROPE_QUANT_KVCACHE_TILING_H
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"
#include "util/math_util.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(RopeQuantKvcacheTilingData)
TILING_DATA_FIELD_DEF(uint64_t, qHeadNum);
TILING_DATA_FIELD_DEF(uint64_t, kvHeadNum);
TILING_DATA_FIELD_DEF(uint64_t, hiddenSize);
TILING_DATA_FIELD_DEF(uint64_t, cacheSeqlen);
TILING_DATA_FIELD_DEF(uint64_t, qHiddenSize);
TILING_DATA_FIELD_DEF(uint64_t, kHiddenSize);
TILING_DATA_FIELD_DEF(uint64_t, vHiddenSize);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(RopeQuantKvcache, RopeQuantKvcacheTilingData)
struct RopeQuantKvcacheCompileInfo {};
} // namespace optiling
#endif
