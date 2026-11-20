/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include "engram_gate_pure_layout_arch35.h"
#include "../op_kernel/arch35/engram_gate_tiling_data.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "log/log.h"

namespace optiling {
namespace {
constexpr int64_t SUPPORTED_HEADS = 4;
constexpr uint64_t VECTOR_LANES = 64;

uint64_t AlignDownToVectorLanes(uint64_t value)
{
    return value / VECTOR_LANES * VECTOR_LANES;
}

bool CheckShape(gert::TilingContext* ctx, const char* name, const gert::Shape& got, const gert::Shape& expected)
{
    if (got != expected) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(ctx->GetNodeName(), name, "mismatch",
                                              "shape differs from EngramGate contract");
        return false;
    }
    return true;
}

bool CheckDtype(gert::TilingContext* ctx, const char* name, ge::DataType got, ge::DataType expected)
{
    if (got != expected) {
        OP_LOGE_FOR_INVALID_DTYPE(ctx->GetNodeName(), name, std::to_string(got).c_str(),
                                  std::to_string(expected).c_str());
        return false;
    }
    return true;
}

ge::graphStatus CheckDtypesAndStats(gert::TilingContext* ctx, EngramGateTilingData& data, int64_t tokens, int64_t heads,
                                    ge::DataType dtype)
{
    auto kvDesc = ctx->GetInputDesc(1);
    auto wDesc = ctx->GetInputDesc(2);
    auto yDesc = ctx->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(ctx, kvDesc);
    OP_CHECK_NULL_WITH_CONTEXT(ctx, wDesc);
    OP_CHECK_NULL_WITH_CONTEXT(ctx, yDesc);
    if (!CheckDtype(ctx, "kv", kvDesc->GetDataType(), dtype) ||
        !CheckDtype(ctx, "weight_fused", wDesc->GetDataType(), ge::DT_FLOAT) ||
        !CheckDtype(ctx, "output", yDesc->GetDataType(), dtype)) {
        return ge::GRAPH_FAILED;
    }
    for (size_t i = 1; i < 5; ++i) {
        auto stat = ctx->GetOutputShape(i);
        auto desc = ctx->GetOutputDesc(i);
        if (data.saveStats == 0 && stat == nullptr) {
            continue;
        }
        OP_CHECK_NULL_WITH_CONTEXT(ctx, stat);
        OP_CHECK_NULL_WITH_CONTEXT(ctx, desc);
        const gert::Shape expected = data.saveStats != 0 ? gert::Shape{tokens, heads} : gert::Shape{0};
        if (!CheckShape(ctx, "statistics", stat->GetStorageShape(), expected) ||
            !CheckDtype(ctx, "statistics", desc->GetDataType(), ge::DT_FLOAT)) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckDimensions(gert::TilingContext* ctx, int64_t tokens, int64_t heads, int64_t dim)
{
    OP_CHECK_IF(tokens < 0,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(ctx->GetNodeName(), "hidden_states", "dimensions",
                                                      "T must be non-negative"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(heads != SUPPORTED_HEADS,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(ctx->GetNodeName(), "hidden_states", "H", "H must be 4"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(dim <= 0 || dim % static_cast<int64_t>(VECTOR_LANES) != 0,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(ctx->GetNodeName(), "hidden_states", "D",
                                                      "D must be a positive multiple of 64"),
                return ge::GRAPH_FAILED);
    constexpr int64_t maxBytes = std::numeric_limits<int64_t>::max();
    // Check factors separately before forming any products, including H+1.
    OP_CHECK_IF(heads >= maxBytes / sizeof(float) || dim > maxBytes / sizeof(float) / (heads + 1) ||
                    tokens > maxBytes / sizeof(float) / ((heads + 1) * dim),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(ctx->GetNodeName(), "inputs", "byte size",
                                                      "int64 address range exceeded"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckInputs(gert::TilingContext* ctx, EngramGateTilingData& data)
{
    auto x = ctx->GetInputShape(0);
    auto kv = ctx->GetInputShape(1);
    auto w = ctx->GetInputShape(2);
    auto y = ctx->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(ctx, x);
    OP_CHECK_NULL_WITH_CONTEXT(ctx, kv);
    OP_CHECK_NULL_WITH_CONTEXT(ctx, w);
    OP_CHECK_NULL_WITH_CONTEXT(ctx, y);
    const auto& shape = x->GetStorageShape();
    OP_CHECK_IF(shape.GetDimNum() != 3,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(ctx->GetNodeName(), "hidden_states", "rank", "expected [T,H,D]"),
                return ge::GRAPH_FAILED);
    const int64_t tokens = shape.GetDim(0), heads = shape.GetDim(1), dim = shape.GetDim(2);
    if (CheckDimensions(ctx, tokens, heads, dim) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (!CheckShape(ctx, "kv", kv->GetStorageShape(), {tokens, (heads + 1) * dim}) ||
        !CheckShape(ctx, "weight_fused", w->GetStorageShape(), {heads, dim}) ||
        !CheckShape(ctx, "output", y->GetStorageShape(), shape)) {
        return ge::GRAPH_FAILED;
    }
    auto xDesc = ctx->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(ctx, xDesc);
    const auto dtype = xDesc->GetDataType();
    OP_CHECK_IF(dtype != ge::DT_BF16 && dtype != ge::DT_FLOAT16,
                OP_LOGE_FOR_INVALID_DTYPE(ctx->GetNodeName(), "hidden_states", std::to_string(dtype).c_str(),
                                          "BF16 or FLOAT16"),
                return ge::GRAPH_FAILED);
    if (CheckDtypesAndStats(ctx, data, tokens, heads, dtype) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    data.heads = heads;
    data.dim = dim;
    data.rows = tokens * heads;
    data.invDim = 1.0F / static_cast<float>(dim);
    data.invSqrtDim = 1.0F / std::sqrt(static_cast<float>(dim));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckAttrs(gert::TilingContext* ctx, EngramGateTilingData& data)
{
    auto attrs = ctx->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(ctx, attrs);
    auto eps = attrs->GetAttrPointer<float>(0);
    auto clamp = attrs->GetAttrPointer<float>(1);
    auto save = attrs->GetAttrPointer<bool>(2);
    const float epsValue = eps == nullptr ? 1e-6F : *eps;
    const float clampValue = clamp == nullptr ? 1e-6F : *clamp;
    OP_CHECK_IF(!std::isfinite(epsValue) || epsValue <= 0.0F,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ctx->GetNodeName(), "eps", std::to_string(epsValue),
                                                      "must be finite and positive"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!std::isfinite(clampValue) || clampValue <= 0.0F,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ctx->GetNodeName(), "clamp_value", std::to_string(clampValue),
                                                      "must be finite and positive"),
                return ge::GRAPH_FAILED);
    data.eps = epsValue;
    data.clampValue = clampValue;
    data.saveStats = save == nullptr || *save;
    return ge::GRAPH_SUCCESS;
}

void ConfigurePureTiling(EngramGateTilingData& data, uint64_t ub, uint64_t l2Size, uint32_t coreCount,
                         bool allowSingleRow, uint32_t& launchCores)
{
    EngramGatePure::Layout layout;
    if (data.rows == 0 || data.heads != 4 ||
        !EngramGatePure::ComputeLayout(data.dim, data.saveStats != 0, ub, layout, allowSingleRow && coreCount >= 4)) {
        return;
    }
    // For short rows, a large K leaves most AIVs idle.  Shrink K so the
    // available cores each receive one compact batch; large D still keeps its
    // stream-friendly K because the UB solver naturally selects a small value.
    if (data.dim <= 2048 && layout.rowsPerSlot > 4) {
        uint64_t targetRows = (static_cast<uint64_t>(data.rows) + coreCount - 1U) / coreCount;
        uint32_t targetK = static_cast<uint32_t>(
            std::min<uint64_t>(layout.rowsPerSlot, std::max<uint64_t>(4, (targetRows + 3U) & ~3U)));
        for (uint32_t K = targetK; K >= 4; K -= 4) {
            EngramGatePure::Layout tuned;
            if (EngramGatePure::TryLayout(data.dim, data.saveStats != 0, ub, K, 2, tuned)) {
                layout = tuned;
                break;
            }
        }
    }
    const uint64_t batches = (static_cast<uint64_t>(data.rows) + layout.rowsPerSlot - 1) / layout.rowsPerSlot;
    uint64_t used = std::min<uint64_t>(coreCount, batches);
    // Each K=1 core retains one weight row; advancing by usedCores must not
    // change its head. K=2 retains the existing head-pair scheduling.
    if (layout.rowsPerSlot == 1) {
        used -= used % static_cast<uint64_t>(data.heads);
    } else if (layout.rowsPerSlot == 2) {
        used -= used % 2U;
    }
    if (used == 0) {
        return;
    }
    launchCores = static_cast<uint32_t>(used);
    data.usedCores = launchCores;
    data.pureUbBytes = layout.total;
    data.pureSlots = layout.slots;
    data.pureRowsPerSlot = layout.rowsPerSlot;
    data.pureXOff = layout.xOff;
    data.pureKvOff = layout.kvOff;
    data.pureWOff = layout.wOff;
    data.pureOutOff = layout.outOff;
    data.pureStatsOff = layout.statsOff;
    auto saturatingMul = [](uint64_t a, uint64_t b) {
        return b != 0 && a > std::numeric_limits<uint64_t>::max() / b ? std::numeric_limits<uint64_t>::max() : a * b;
    };
    auto saturatingAdd = [](uint64_t a, uint64_t b) {
        return a > std::numeric_limits<uint64_t>::max() - b ? std::numeric_limits<uint64_t>::max() : a + b;
    };
    const uint64_t xBytes = saturatingMul(
        saturatingMul(static_cast<uint64_t>(data.rows), static_cast<uint64_t>(data.dim)), sizeof(uint16_t));
    const uint64_t kvBytes =
        saturatingMul(saturatingMul(static_cast<uint64_t>(data.rows) / static_cast<uint64_t>(data.heads),
                                    static_cast<uint64_t>(data.heads + 1)),
                      saturatingMul(static_cast<uint64_t>(data.dim), sizeof(uint16_t)));
    const uint64_t wBytes =
        saturatingMul(saturatingMul(static_cast<uint64_t>(data.heads), static_cast<uint64_t>(data.dim)), sizeof(float));
    const uint64_t workingSet = saturatingAdd(saturatingAdd(xBytes, kvBytes), wBytes);
    data.pureBypass = workingSet > l2Size;
}

void ConfigureChunkedTiling(EngramGateTilingData& data, uint64_t ub, uint32_t coreCount, uint32_t& launchCores)
{
    const uint64_t streamCapacity = ub > EngramGateTiling::CHUNK_FIXED_BYTES ?
                                        AlignDownToVectorLanes((ub - EngramGateTiling::CHUNK_FIXED_BYTES) /
                                                               EngramGateTiling::CHUNK_STREAM_BYTES_PER_ELEMENT) :
                                        0;
    if (data.rows == 0 || data.pureSlots != 0 || data.dim % VECTOR_LANES != 0 || streamCapacity < VECTOR_LANES ||
        ub > std::numeric_limits<uint32_t>::max()) {
        return;
    }
    launchCores = static_cast<uint32_t>(std::min<int64_t>(coreCount, data.rows));
    data.usedCores = launchCores;
    data.chunkedEnabled = 1;
    data.ubLayoutBytes = static_cast<uint32_t>(ub);
    const uint64_t wBytes = static_cast<uint64_t>(data.dim) * sizeof(float);
    const bool residentCandidate = launchCores % data.heads == 0 &&
                                   wBytes <= ub / EngramGateTiling::WEIGHT_RESIDENT_UB_DIVISOR &&
                                   ub > EngramGateTiling::CHUNK_FIXED_BYTES + wBytes;
    const uint64_t residentCapacity = residentCandidate ?
                                          AlignDownToVectorLanes((ub - EngramGateTiling::CHUNK_FIXED_BYTES - wBytes) /
                                                                 EngramGateTiling::CHUNK_RESIDENT_BYTES_PER_ELEMENT) :
                                          0;
    data.weightResident = residentCapacity >= VECTOR_LANES;
    const uint64_t selectedCapacity = data.weightResident != 0 ? residentCapacity : streamCapacity;
    data.chunk = static_cast<uint32_t>(std::min<uint64_t>(static_cast<uint64_t>(data.dim), selectedCapacity));
}
} // namespace

ge::graphStatus TilingForEngramGate(gert::TilingContext* ctx)
{
    auto data = ctx->GetTilingData<EngramGateTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(ctx, data);
    *data = {};
    if (CheckAttrs(ctx, *data) != ge::GRAPH_SUCCESS || CheckInputs(ctx, *data) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_NULL_WITH_CONTEXT(ctx, ctx->GetPlatformInfo());
    platform_ascendc::PlatformAscendC platform(ctx->GetPlatformInfo());
    uint64_t ub = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
    const auto coreCount = platform.GetCoreNumAiv();
    OP_CHECK_IF(coreCount == 0 || ub < EngramGateTiling::CHUNK_FIXED_BYTES +
                                           VECTOR_LANES * EngramGateTiling::CHUNK_STREAM_BYTES_PER_ELEMENT,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ctx->GetNodeName(), "platform", "resources",
                                                      "insufficient AIV/UB resources"),
                return ge::GRAPH_FAILED);
    uint32_t launchCores = 1;
    uint64_t l2Size = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, l2Size);
    // The single-row pipeline is measured on Ascend950; keep the original
    // K>=2 selection on other SoCs sharing this host and kernel source.
    const bool allowSingleRow = platform.GetSocVersion() == platform_ascendc::SocVersion::ASCEND950;
    ConfigurePureTiling(*data, ub, l2Size, coreCount, allowSingleRow, launchCores);
    ConfigureChunkedTiling(*data, ub, coreCount, launchCores);
    OP_CHECK_IF(data->rows != 0 && data->pureSlots == 0 && data->chunkedEnabled == 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ctx->GetNodeName(), "platform", "resources",
                                                      "no valid full-load or chunked UB layout"),
                return ge::GRAPH_FAILED);
    ctx->SetBlockDim(launchCores);
    ctx->SetTilingKey(0);
    auto workspace = ctx->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(ctx, workspace);
    workspace[0] = platform.GetLibApiWorkSpaceSize();
    ctx->GetRawTilingData()->SetDataSize(sizeof(EngramGateTilingData));
    return ge::GRAPH_SUCCESS;
}
namespace {
struct EngramGateCompileInfo {};

ge::graphStatus ParseEngramGate(gert::TilingParseContext* context)
{
    return context == nullptr ? ge::GRAPH_FAILED : ge::GRAPH_SUCCESS;
}
} // namespace

IMPL_OP_OPTILING(EngramGate).Tiling(TilingForEngramGate).TilingParse<EngramGateCompileInfo>(ParseEngramGate);
} // namespace optiling
