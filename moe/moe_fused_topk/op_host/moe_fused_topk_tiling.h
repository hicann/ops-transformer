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
 * \file moe_fused_topk_tiling.h
 * \brief
 */

#ifndef ASCEND_OPS_MOE_FUSED_TOPK_TILING_H
#define ASCEND_OPS_MOE_FUSED_TOPK_TILING_H

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"

namespace optiling {

struct MoeFusedTopkCompileInfo {
    uint32_t coreNum = 0;
    uint64_t ubSize = 0;
};

BEGIN_TILING_DATA_DEF(MoeFusedTopkTilingData)
TILING_DATA_FIELD_DEF(uint32_t, firstDimSize);
TILING_DATA_FIELD_DEF(uint32_t, secondDimSize);
TILING_DATA_FIELD_DEF(uint32_t, addNumDimSize);
TILING_DATA_FIELD_DEF(uint32_t, groupNum);
TILING_DATA_FIELD_DEF(uint32_t, groupTopk);
TILING_DATA_FIELD_DEF(uint32_t, topN);
TILING_DATA_FIELD_DEF(uint32_t, topK);

TILING_DATA_FIELD_DEF(uint32_t, activateType);
TILING_DATA_FIELD_DEF(uint32_t, isNorm);
TILING_DATA_FIELD_DEF(float, scale);
TILING_DATA_FIELD_DEF(uint32_t, groupEles);
TILING_DATA_FIELD_DEF(uint32_t, blockNum);
TILING_DATA_FIELD_DEF(uint32_t, ubFactorElement);
TILING_DATA_FIELD_DEF(uint32_t, batchPerCore);
TILING_DATA_FIELD_DEF(uint32_t, tailBatch);

TILING_DATA_FIELD_DEF(uint32_t, expertNum);
TILING_DATA_FIELD_DEF(uint32_t, tableDim);
TILING_DATA_FIELD_DEF(uint32_t, topkMaxValue);
TILING_DATA_FIELD_DEF(uint32_t, topkMinValue);
TILING_DATA_FIELD_DEF(uint32_t, reserved);
TILING_DATA_FIELD_DEF(uint64_t, workspacePerCore);
TILING_DATA_FIELD_DEF_STRUCT(TopkTiling, topkTilingData);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MoeFusedTopk, MoeFusedTopkTilingData);

ge::graphStatus TilingMoeFusedTopk(gert::TilingContext *context);
} // namespace optiling

#endif // ASCEND_OPS_MOE_FUSED_TOPK_TILING_H
