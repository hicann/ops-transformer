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
 * \file swin_transformer_ln_qkv_quant_tiling.h
 * \brief
 */
#ifndef SWIN_TRANSFORMER_LN_QKV_QUANT_TILING_H
#define SWIN_TRANSFORMER_LN_QKV_QUANT_TILING_H
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "util/shape_util.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "op_host/tiling_templates_registry.h"
#include "op_host/tiling_base.h"
#include "util/math_util.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(SwinTransformerLnQkvQuantBaseInfo)
TILING_DATA_FIELD_DEF(uint32_t, bSize);
TILING_DATA_FIELD_DEF(uint32_t, sSize);
TILING_DATA_FIELD_DEF(uint32_t, hSize);
TILING_DATA_FIELD_DEF(uint32_t, headNum);
TILING_DATA_FIELD_DEF(uint32_t, hWinSize);
TILING_DATA_FIELD_DEF(uint32_t, wWinSize);
TILING_DATA_FIELD_DEF(uint32_t, sizePerHead);
TILING_DATA_FIELD_DEF(uint32_t, patchHeight);
TILING_DATA_FIELD_DEF(uint32_t, patchWeight);
TILING_DATA_FIELD_DEF(uint32_t, lnBaseM);
TILING_DATA_FIELD_DEF(uint32_t, lnBaseK);
TILING_DATA_FIELD_DEF(uint32_t, lnBufferM);
TILING_DATA_FIELD_DEF(uint32_t, lnBufferK);
TILING_DATA_FIELD_DEF(uint32_t, lnMSubLoop);
TILING_DATA_FIELD_DEF(uint32_t, lnKSubLoop);
TILING_DATA_FIELD_DEF(uint32_t, loopNum);
TILING_DATA_FIELD_DEF(uint32_t, loopSum);
TILING_DATA_FIELD_DEF(uint32_t, singleCoreLnBsSize);
TILING_DATA_FIELD_DEF(uint32_t, lnBufferNum);
TILING_DATA_FIELD_DEF(uint32_t, resverd1);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(SwinTransformerLnQkvQuantBaseInfoOp, SwinTransformerLnQkvQuantBaseInfo)

BEGIN_TILING_DATA_DEF(SwinTransformerLnQkvQuantMmInfo)
TILING_DATA_FIELD_DEF(uint32_t, mmSizeM);
TILING_DATA_FIELD_DEF(uint32_t, mmSizeK);
TILING_DATA_FIELD_DEF(uint32_t, mmSizeN);
TILING_DATA_FIELD_DEF(uint32_t, dimNum);
TILING_DATA_FIELD_DEF(uint32_t, mDim);
TILING_DATA_FIELD_DEF(uint32_t, nDim);
TILING_DATA_FIELD_DEF(uint32_t, shareUbForMm);
TILING_DATA_FIELD_DEF(uint32_t, mmLoopNum);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(SwinTransformerLnQkvQuantMmInfoOp, SwinTransformerLnQkvQuantMmInfo)

BEGIN_TILING_DATA_DEF(SwinTransformerLnQkvQuantTilingData)
TILING_DATA_FIELD_DEF(uint32_t, size);
TILING_DATA_FIELD_DEF(uint32_t, maxCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, lnBlockNum);
TILING_DATA_FIELD_DEF(uint32_t, workSpaceSize);
TILING_DATA_FIELD_DEF(uint32_t, inputSizeSum);
TILING_DATA_FIELD_DEF(uint32_t, tmpShareBufferForLn);
TILING_DATA_FIELD_DEF(uint32_t, tmpBufferForQuant);
TILING_DATA_FIELD_DEF(uint32_t, weightK);
TILING_DATA_FIELD_DEF(uint32_t, weightN);
TILING_DATA_FIELD_DEF(float, epsilon);
TILING_DATA_FIELD_DEF_STRUCT(SwinTransformerLnQkvQuantBaseInfo, opBaseInfo);
TILING_DATA_FIELD_DEF_STRUCT(SwinTransformerLnQkvQuantMmInfo, mmInfo);
TILING_DATA_FIELD_DEF_STRUCT(LayerNormTiling, layernromTilingData);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mmTilingParams);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SwinTransformerLnQkvQuant, SwinTransformerLnQkvQuantTilingData)
struct SwinTransformerLnQkvQuantCompileInfo {
    uint32_t coreNum = 0;
};
} // namespace optiling

#endif // SWIN_TRANSFORMER_LN_QKV_QUANT_TILING_H_
