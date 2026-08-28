/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

#include "log/log.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "scaled_cosine_attention_score_tiling.h"

namespace optiling {
namespace {
constexpr size_t IN_QUERY = 0;
constexpr size_t IN_KEY = 1;
constexpr size_t IN_SCALE = 2;
constexpr size_t ATTR_CLAMP_MAX = 0;
constexpr size_t ATTR_EPS = 1;
constexpr uint32_t MAX_KEY_TILE_ROWS = 128;
constexpr uint64_t ALIGN_BYTES = 32;
constexpr uint64_t SCALAR_WORK_BYTES = 12U * ALIGN_BYTES;

uint64_t AlignUp(uint64_t value, uint64_t alignment)
{
    return (value + alignment - 1U) / alignment * alignment;
}

ge::DataType GetInputDtype(gert::TilingContext *context, size_t index)
{
    auto desc = context->GetInputDesc(index);
    return desc == nullptr ? ge::DT_UNDEFINED : desc->GetDataType();
}

bool IsSupportedType(ge::DataType type)
{
    return type == ge::DT_FLOAT16 || type == ge::DT_BF16 || type == ge::DT_FLOAT;
}

uint32_t TypeBytes(ge::DataType type)
{
    return type == ge::DT_FLOAT ? sizeof(float) : sizeof(uint16_t);
}

uint64_t TilingKeyFor(ge::DataType type)
{
    if (type == ge::DT_BF16) {
        return SCAS_TILING_KEY_BF16;
    }
    if (type == ge::DT_FLOAT) {
        return SCAS_TILING_KEY_FP32;
    }
    return SCAS_TILING_KEY_FP16;
}

bool ScaleShapeMatches(const gert::Shape &scale, int64_t heads)
{
    if (scale.GetDimNum() == 1) {
        return scale.GetDim(0) == heads;
    }
    return scale.GetDimNum() == 3 && scale.GetDim(0) == heads && scale.GetDim(1) == 1 && scale.GetDim(2) == 1;
}

struct RuntimeParams {
    uint32_t batch = 0;
    uint32_t heads = 0;
    uint32_t seqLen = 0;
    uint32_t headDim = 0;
    uint32_t alignedHeadDim = 0;
    uint32_t keyTileRows = 0;
    uint32_t usedCoreNum = 0;
    uint64_t totalQueryRows = 0;
    float clampMax = 0.0F;
    float eps = 0.0F;
    ge::DataType dtype = ge::DT_UNDEFINED;
};

ge::graphStatus ParseRuntimeParams(gert::TilingContext *context, platform_ascendc::PlatformAscendC &platform,
                                   RuntimeParams &params)
{
    const auto queryShape = context->GetInputShape(IN_QUERY);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryShape);
    const auto keyShape = context->GetInputShape(IN_KEY);
    OP_CHECK_NULL_WITH_CONTEXT(context, keyShape);
    const auto scaleShape = context->GetInputShape(IN_SCALE);
    OP_CHECK_NULL_WITH_CONTEXT(context, scaleShape);

    const auto &query = queryShape->GetStorageShape();
    const auto &key = keyShape->GetStorageShape();
    const auto &scale = scaleShape->GetStorageShape();
    OP_CHECK_IF(query.GetDimNum() != 4 || key.GetDimNum() != 4, OP_LOGE(context, "query/key rank must be 4"),
                return ge::GRAPH_FAILED);
    for (size_t i = 0; i < 4; ++i) {
        OP_CHECK_IF(query.GetDim(i) <= 0 || query.GetDim(i) != key.GetDim(i),
                    OP_LOGE(context, "invalid or mismatched query/key dimension %zu: %ld/%ld", i, query.GetDim(i),
                            key.GetDim(i)),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(static_cast<uint64_t>(query.GetDim(i)) > std::numeric_limits<uint32_t>::max(),
                    OP_LOGE(context, "dimension %zu exceeds uint32", i), return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF(!ScaleShapeMatches(scale, query.GetDim(1)), OP_LOGE(context, "scale must have shape [H] or [H,1,1]"),
                return ge::GRAPH_FAILED);

    params.dtype = GetInputDtype(context, IN_QUERY);
    OP_CHECK_IF(!IsSupportedType(params.dtype) || GetInputDtype(context, IN_KEY) != params.dtype ||
                    GetInputDtype(context, IN_SCALE) != ge::DT_FLOAT,
                OP_LOGE(context, "query/key dtype must match and scale must be float32"), return ge::GRAPH_FAILED);

    const auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const float *clampMax = attrs->GetAttrPointer<float>(ATTR_CLAMP_MAX);
    const float *eps = attrs->GetAttrPointer<float>(ATTR_EPS);
    OP_CHECK_IF(
        clampMax == nullptr || eps == nullptr || !std::isfinite(*clampMax) || !std::isfinite(*eps) || *eps <= 0.0F,
        OP_LOGE(context, "clamp_max must be finite and eps must be finite and positive"), return ge::GRAPH_FAILED);

    params.batch = static_cast<uint32_t>(query.GetDim(0));
    params.heads = static_cast<uint32_t>(query.GetDim(1));
    params.seqLen = static_cast<uint32_t>(query.GetDim(2));
    params.headDim = static_cast<uint32_t>(query.GetDim(3));
    params.clampMax = *clampMax;
    params.eps = *eps;

    const uint32_t inputBytes = TypeBytes(params.dtype);
    OP_CHECK_IF(params.headDim > std::numeric_limits<uint32_t>::max() / inputBytes,
                OP_LOGE(context, "headDim byte length exceeds DataCopy limit"), return ge::GRAPH_FAILED);
    params.alignedHeadDim =
        static_cast<uint32_t>(AlignUp(static_cast<uint64_t>(params.headDim) * inputBytes, ALIGN_BYTES) / inputBytes);

    OP_CHECK_IF(params.batch > std::numeric_limits<uint64_t>::max() / params.heads ||
                    static_cast<uint64_t>(params.batch) * params.heads >
                        std::numeric_limits<uint64_t>::max() / params.seqLen,
                OP_LOGE(context, "B*H*N overflows uint64"), return ge::GRAPH_FAILED);
    params.totalQueryRows = static_cast<uint64_t>(params.batch) * params.heads * params.seqLen;
    OP_CHECK_IF(params.totalQueryRows > std::numeric_limits<uint64_t>::max() / params.seqLen,
                OP_LOGE(context, "output element count overflows uint64"), return ge::GRAPH_FAILED);

    uint64_t ubBytes = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
    const uint64_t budget = ubBytes * 4U / 5U;
    // Fixed: typed q, q-fp32, tmp0, tmp1, reduction work and scalar/scale state.
    const uint64_t fixedBytes = AlignUp(params.alignedHeadDim * inputBytes, ALIGN_BYTES) +
                                5U * AlignUp(params.alignedHeadDim * sizeof(float), ALIGN_BYTES) +
                                AlignUp(sizeof(float), ALIGN_BYTES) + SCALAR_WORK_BYTES;
    // Per key row: typed/fp32 key plus fp32/typed output slots.
    const uint64_t perKeyRow = AlignUp(params.alignedHeadDim * inputBytes, ALIGN_BYTES) +
                               AlignUp(params.alignedHeadDim * sizeof(float), ALIGN_BYTES) + sizeof(float) + inputBytes;
    OP_CHECK_IF(ubBytes == 0 || budget <= fixedBytes + perKeyRow,
                OP_LOGE(context, "UB size %lu is insufficient", ubBytes), return ge::GRAPH_FAILED);
    params.keyTileRows = static_cast<uint32_t>(
        std::min<uint64_t>(std::min<uint64_t>((budget - fixedBytes) / perKeyRow, MAX_KEY_TILE_ROWS), params.seqLen));
    OP_CHECK_IF(params.keyTileRows == 0, OP_LOGE(context, "calculated key tile is zero"), return ge::GRAPH_FAILED);

    uint32_t coreNum = platform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "AIV core num is zero"), return ge::GRAPH_FAILED);
    params.usedCoreNum = static_cast<uint32_t>(std::min<uint64_t>(params.totalQueryRows, coreNum));
    return ge::GRAPH_SUCCESS;
}
} // namespace

static ge::graphStatus ScaledCosineAttentionScoreTilingFunc(gert::TilingContext *context)
{
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto platform = platform_ascendc::PlatformAscendC(platformInfo);

    RuntimeParams params;
    OP_CHECK_IF(ParseRuntimeParams(context, platform, params) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "failed to parse ScaledCosineAttentionScore tiling parameters"),
                return ge::GRAPH_FAILED);

    ScaledCosineAttentionScoreTilingData tiling;
    tiling.set_batch(params.batch);
    tiling.set_heads(params.heads);
    tiling.set_seqLen(params.seqLen);
    tiling.set_headDim(params.headDim);
    tiling.set_alignedHeadDim(params.alignedHeadDim);
    tiling.set_keyTileRows(params.keyTileRows);
    tiling.set_usedCoreNum(params.usedCoreNum);
    tiling.set_reserved(0);
    tiling.set_totalQueryRows(params.totalQueryRows);
    tiling.set_clampMax(params.clampMax);
    tiling.set_eps(params.eps);

    context->SetBlockDim(params.usedCoreNum);
    context->SetTilingKey(TilingKeyFor(params.dtype));
    size_t *workspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspace);
    workspace[0] = 0;

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepareForScaledCosineAttentionScore(gert::TilingParseContext *context)
{
    auto compileInfo = context->GetCompiledInfo<ScaledCosineAttentionScoreCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto platform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->aivCoreNum = platform.GetCoreNumAiv();
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->ubSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ScaledCosineAttentionScore)
    .Tiling(ScaledCosineAttentionScoreTilingFunc)
    .TilingParse<ScaledCosineAttentionScoreCompileInfo>(TilingPrepareForScaledCosineAttentionScore);
} // namespace optiling
