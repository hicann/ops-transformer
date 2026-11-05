/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "group_memcpy_tiling.h"
#include "group_memcpy_tiling_policy.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <algorithm>
#include <limits>

namespace optiling {
namespace {
// Runtime platform data is read by TilingGroupMemcpy. Register an explicit
// compile-info parser so aclnn does not fall back to the auto-tiling parser,
// which expects an unrelated `_pattern` field in the kernel compile info.
struct GroupMemcpyCompileInfo {};

ge::graphStatus ParseGroupMemcpy(gert::TilingParseContext* context)
{
    return context == nullptr ? ge::GRAPH_FAILED : ge::GRAPH_SUCCESS;
}

uint64_t ItemBytes(ge::DataType dtype)
{
    switch (dtype) {
        case ge::DT_INT8:
        case ge::DT_UINT8:
        case ge::DT_BOOL:
            return 1;
        case ge::DT_INT16:
        case ge::DT_FLOAT16:
        case ge::DT_BF16:
            return 2;
        case ge::DT_INT32:
        case ge::DT_FLOAT:
            return 4;
        case ge::DT_INT64:
            return 8;
        default:
            return 0;
    }
}
} // namespace

static ge::graphStatus TilingGroupMemcpy(gert::TilingContext* context)
{
    if (context == nullptr || context->GetPlatformInfo() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto* first = context->GetDynamicInputDesc(0, 0);
    if (first == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto dtype = first->GetDataType();
    const uint64_t itemBytes = ItemBytes(dtype);
    if (itemBytes == 0) {
        return ge::GRAPH_FAILED;
    }
    uint64_t count = 0;
    uint64_t width = 0;
    while (const auto* src = context->GetDynamicInputShape(0, count)) {
        const auto* srcDesc = context->GetDynamicInputDesc(0, count);
        const auto* dst = context->GetOutputShape(count);
        const auto* dstDesc = context->GetOutputDesc(count);
        if (count >= static_cast<uint64_t>(std::numeric_limits<int32_t>::max()) || srcDesc == nullptr ||
            dst == nullptr || dstDesc == nullptr || srcDesc->GetDataType() != dtype ||
            dstDesc->GetDataType() != dtype || srcDesc->GetStorageFormat() != ge::FORMAT_ND ||
            dstDesc->GetStorageFormat() != ge::FORMAT_ND || src->GetOriginShape() != dst->GetOriginShape()) {
            return ge::GRAPH_FAILED;
        }
        const int64_t elements = src->GetOriginShape().GetShapeSize();
        if (elements < 0 || static_cast<uint64_t>(elements) >
                                (static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) - 64) / itemBytes) {
            return ge::GRAPH_FAILED;
        }
        const uint64_t bytes = static_cast<uint64_t>(elements) * itemBytes;
        if (count != 0 && bytes != width) {
            return ge::GRAPH_FAILED;
        }
        width = bytes;
        ++count;
    }
    if (count == 0) {
        return ge::GRAPH_FAILED;
    }
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ubBytes = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
    const uint64_t tileBytes = group_memcpy::TileBytes(width, ubBytes);
    const uint64_t cores = platform.GetCoreNumAiv();
    if (cores == 0 || (width != 0 && tileBytes == 0)) {
        return ge::GRAPH_FAILED;
    }
    GroupMemcpyTilingData data;
    data.set_pairCount(count);
    data.set_bytesPerCopy(width);
    data.set_tileBytes(tileBytes);
    context->SetBlockDim(static_cast<uint32_t>(std::min(cores, count)));
    context->SetTilingKey(0);
    data.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(data.GetDataSize());
    context->GetWorkspaceSizes(1)[0] = platform.GetLibApiWorkSpaceSize();
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_OPTILING(GroupMemcpy).Tiling(TilingGroupMemcpy).TilingParse<GroupMemcpyCompileInfo>(ParseGroupMemcpy);
} // namespace optiling
