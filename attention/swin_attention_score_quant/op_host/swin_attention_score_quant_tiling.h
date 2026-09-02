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
 * \file swin_attention_score_quant_tiling.h
 * \brief SwinAttentionScoreQuant tiling define
 */
#ifndef SWIN_ATTENTION_SCORE_QUANT_TILING_H
#define SWIN_ATTENTION_SCORE_QUANT_TILING_H

#include "log/log.h"
#include "register/op_impl_registry.h"
#include "tiling/tiling_api.h"
#include "op_host/tiling_type.h"
#include "register/tilingdata_base.h"
#include "register/op_def_registry.h"
#include "util/math_util.h"
#include "platform/platform_info.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"
#include "op_common/op_host/util/platform_util.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(SwinAttentionScoreQuantTilingData)
TILING_DATA_FIELD_DEF(uint32_t, coreLoops);
TILING_DATA_FIELD_DEF(uint32_t, dimB);
TILING_DATA_FIELD_DEF(uint32_t, dimN);
TILING_DATA_FIELD_DEF(uint32_t, dimS);
TILING_DATA_FIELD_DEF(uint32_t, dimH);
TILING_DATA_FIELD_DEF(uint32_t, qSize);
TILING_DATA_FIELD_DEF(uint32_t, kSize);
TILING_DATA_FIELD_DEF(uint32_t, pSize);
TILING_DATA_FIELD_DEF(uint32_t, vSize);
TILING_DATA_FIELD_DEF(uint32_t, cubeSharedUbSize);
TILING_DATA_FIELD_DEF(uint32_t, vecSharedUbSize);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, qkBmmTilingData);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, pvBmmTilingData);
TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, softmaxTilingData);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(SwinAttentionScoreQuant, SwinAttentionScoreQuantTilingData)
} // namespace optiling
#endif
