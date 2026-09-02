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
 * \file swin_transformer_ln_qkv_tiling.h
 * \brief
 */
#ifndef SWIN_TRANSFORMER_LN_QKV_TILING_H_
#define SWIN_TRANSFORMER_LN_QKV_TILING_H_
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "util/shape_util.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "op_host/tiling_templates_registry.h"
#include "op_host/tiling_base.h"
#include "util/math_util.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(SwinTransformerLnQKVBaseInfo)
TILING_DATA_FIELD_DEF(uint32_t, inputsize);         // [B,S,H]
TILING_DATA_FIELD_DEF(uint32_t, hSize);             // layernorm [H]
TILING_DATA_FIELD_DEF(uint32_t, baseLoopNum);       // for every vec block
TILING_DATA_FIELD_DEF(uint32_t, remainderBlockNum); // remainder for some vec

END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(SwinTransformerLnQKVBaseInfoOp, SwinTransformerLnQKVBaseInfo)

BEGIN_TILING_DATA_DEF(SwinTransformerLnQKVLayernormTilingData)
TILING_DATA_FIELD_DEF(uint32_t, bLength);
TILING_DATA_FIELD_DEF(uint32_t, sLength);
TILING_DATA_FIELD_DEF(uint32_t, hLength);
TILING_DATA_FIELD_DEF(uint32_t, bsLength);
TILING_DATA_FIELD_DEF(uint32_t, shLength);
TILING_DATA_FIELD_DEF(uint32_t, loopSize);
TILING_DATA_FIELD_DEF(uint32_t, elementPerBlock);
TILING_DATA_FIELD_DEF(uint32_t, remainderElementPerBlock);
TILING_DATA_FIELD_DEF(uint32_t, innerLoopLength);
TILING_DATA_FIELD_DEF(uint32_t, innerLoopNum); //  8
TILING_DATA_FIELD_DEF(uint32_t, normalBlockElementOffset);
TILING_DATA_FIELD_DEF(uint32_t, rollOffset);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(SwinTransformerLnQKVLayernormTilingDataOp, SwinTransformerLnQKVLayernormTilingData)

BEGIN_TILING_DATA_DEF(SwinTransformerLnQKVMatmulTilingData)
TILING_DATA_FIELD_DEF(uint32_t, bLength);
TILING_DATA_FIELD_DEF(uint32_t, biasLength);

END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(SwinTransformerLnQKVMatmulTilingDataOp, SwinTransformerLnQKVMatmulTilingData)

BEGIN_TILING_DATA_DEF(SwinTransformerLnQKVTilingData)
TILING_DATA_FIELD_DEF(uint32_t, maxCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, useVectorNum);
TILING_DATA_FIELD_DEF(uint32_t, workspaceSize);
TILING_DATA_FIELD_DEF(uint32_t, inputSizeSum);
TILING_DATA_FIELD_DEF_STRUCT(SwinTransformerLnQKVBaseInfo, opBaseInfo);
TILING_DATA_FIELD_DEF_STRUCT(SwinTransformerLnQKVLayernormTilingData, layernormTilingParams);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mmTilingParams);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SwinTransformerLnQKV, SwinTransformerLnQKVTilingData)
} // namespace optiling

#endif // SWIN_TRANSFORMER_LN_QKV_TILING_H_
