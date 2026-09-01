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
 * \file gen_position_ids_from_mask_tiling.cpp
 * \brief
 */
#include "gen_position_ids_from_mask_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"

using namespace ge;

namespace optiling {

constexpr uint32_t DIMS_LIMIT = 2; // [B, S]
constexpr uint32_t DIM_B = 0;
constexpr uint32_t DIM_S = 1;
constexpr size_t ATTR_PADDING_FILL_VALUE_INDEX = 0;

constexpr uint64_t TILING_KEY_INT32 = 1;
constexpr uint64_t TILING_KEY_INT64 = 2;
constexpr uint64_t TILING_KEY_BOOL = 3;

static uint64_t GetTilingKeyByDtype(ge::DataType dt)
{
    switch (dt) {
        case ge::DT_INT32:
            return TILING_KEY_INT32;
        case ge::DT_INT64:
            return TILING_KEY_INT64;
        case ge::DT_BOOL:
            return TILING_KEY_BOOL;
        default:
            return TILING_KEY_INT32;
    }
}

static ge::graphStatus TilingGenPositionIdsFromMask(gert::TilingContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    GenPositionIdsFromMaskTilingData tiling;

    // ---- shape: 仅支持 [B, S] ----
    const gert::StorageShape *maskShape = context->GetInputShape(0);
    if (maskShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape &shape = maskShape->GetStorageShape();
    if (shape.GetDimNum() != DIMS_LIMIT) {
        return ge::GRAPH_FAILED;
    }
    const int64_t b = shape.GetDim(DIM_B);
    const int64_t s = shape.GetDim(DIM_S);
    if (b <= 0 || s <= 0) {
        return ge::GRAPH_FAILED;
    }

    // ---- 读属性 paddingFillValue ----
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const int64_t *pFill = attrs->GetAttrPointer<int64_t>(ATTR_PADDING_FILL_VALUE_INDEX);
    const int64_t fillValue = (pFill != nullptr) ? *pFill : 1;

    // 各行相互独立，按可用 Vector 核分配整行任务，保证 carry 不跨核。
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());

    uint32_t availableCoreNum = ascendcPlatform.GetCoreNum();
    if (availableCoreNum == 0) {
        availableCoreNum = 1;
    }

    const uint32_t blockDim =
        (b < static_cast<int64_t>(availableCoreNum)) ? static_cast<uint32_t>(b) : availableCoreNum;

    tiling.set_b(b);
    tiling.set_s(s);
    tiling.set_paddingFillValue(fillValue);
    tiling.set_coreNum(blockDim);

    context->SetBlockDim(blockDim);

    const gert::CompileTimeTensorDesc *maskDesc = context->GetInputDesc(0);
    if (maskDesc == nullptr) {
        return ge::GRAPH_FAILED;
    }
    context->SetTilingKey(GetTilingKeyByDtype(maskDesc->GetDataType()));

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t *workspaces = context->GetWorkspaceSizes(1);
    if (workspaces == nullptr) {
        return ge::GRAPH_FAILED;
    }
    workspaces[0] = 0;

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(GenPositionIdsFromMask).Tiling(TilingGenPositionIdsFromMask);

} // namespace optiling
