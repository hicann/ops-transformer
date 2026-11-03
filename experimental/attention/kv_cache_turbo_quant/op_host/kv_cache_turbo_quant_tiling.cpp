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

#include "kv_cache_turbo_quant_tiling.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

namespace optiling {
constexpr uint64_t SYSTEM_WORKSPACE_BYTES = 16ULL * 1024 * 1024;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    const auto* inputShape = context->GetInputShape(0);
    const auto* rotationShape = context->GetInputShape(1);
    const auto* qjlShape = context->GetInputShape(2);
    const auto* attrs = context->GetAttrs();
    if (inputShape == nullptr || rotationShape == nullptr || qjlShape == nullptr || attrs == nullptr ||
        context->GetPlatformInfo() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto& input = inputShape->GetStorageShape();
    const auto& rotation = rotationShape->GetStorageShape();
    const auto& qjl = qjlShape->GetStorageShape();
    if (input.GetDimNum() != 3 || rotation.GetDimNum() != 2 || qjl.GetDimNum() != 2) {
        return ge::GRAPH_FAILED;
    }
    const int64_t headDim64 = input.GetDim(2);
    const int64_t qjlDim64 = qjl.GetDim(0);
    auto supportedDim = [](int64_t dim) { return dim == 64 || dim == 128 || dim == 256; };
    if (!supportedDim(headDim64) || !supportedDim(qjlDim64) || rotation.GetDim(0) != headDim64 ||
        rotation.GetDim(1) != headDim64 || qjl.GetDim(1) != headDim64) {
        return ge::GRAPH_FAILED;
    }
    const int64_t heads = input.GetDim(1);
    if (heads < 4 || heads > 32) {
        return ge::GRAPH_FAILED;
    }
    const auto* bitsPointer = attrs->GetAttrPointer<int64_t>(0);
    const auto* outputDtypePointer = attrs->GetAttrPointer<int64_t>(1);
    if (bitsPointer == nullptr || (*bitsPointer != 2 && *bitsPointer != 3 && *bitsPointer != 4)) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t outputDtype = outputDtypePointer == nullptr ? 0U : static_cast<uint32_t>(*outputDtypePointer);
    if (outputDtype > 1U) {
        return ge::GRAPH_FAILED;
    }
    const auto* rotationDesc = context->GetInputDesc(1);
    const auto* qjlDesc = context->GetInputDesc(2);
    if (rotationDesc == nullptr || qjlDesc == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto rotationType = rotationDesc->GetDataType();
    const auto qjlType = qjlDesc->GetDataType();
    if ((rotationType != ge::DT_FLOAT && rotationType != ge::DT_BF16) ||
        (qjlType != ge::DT_FLOAT && qjlType != ge::DT_BF16)) {
        return ge::GRAPH_FAILED;
    }
    const uint64_t vectors64 = static_cast<uint64_t>(input.GetDim(0)) * heads;
    if (vectors64 > UINT32_MAX) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t vectors = static_cast<uint32_t>(vectors64);
    const uint32_t bits = static_cast<uint32_t>(*bitsPointer);
    const uint32_t headDim = static_cast<uint32_t>(headDim64);
    const uint32_t qjlDim = static_cast<uint32_t>(qjlDim64);
    const uint32_t indexStride = (headDim * bits + 7) / 8;
    const uint32_t qjlStride = (qjlDim + 7) / 8;
    const bool fixed128 = headDim == 128 && qjlDim == 128;
    const bool dtypeFallback = rotationType == ge::DT_BF16 || qjlType == ge::DT_BF16;
    if (dtypeFallback && !fixed128) {
        return ge::GRAPH_FAILED;
    }

    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    const uint32_t aicNum = std::max(1U, platform.GetCoreNumAic());
    const uint32_t aivNum = std::max(1U, platform.GetCoreNumAiv());
    const uint32_t mode =
        dtypeFallback ? 4U : (!fixed128 ? 3U : (vectors >= 128 && vectors % 128 == 0 ? 2U : (vectors > 0 ? 1U : 0U)));
    uint32_t blocks = 1;
    uint32_t launchAicBlocks = 1;
    const uint32_t cubeRows = mode == 1 ? 8 : 128;
    if (mode == 1) {
        blocks = vectors <= 8 ? 1 : 2;
    } else if (mode == 2) {
        launchAicBlocks = std::max(1U, std::min(std::min(vectors / 128, aicNum), 8U));
        blocks = 2 * launchAicBlocks;
    } else {
        const uint32_t wantedAiv = std::max(1U, std::min(vectors, aivNum));
        launchAicBlocks = (wantedAiv + 1) / 2;
        blocks = 2 * launchAicBlocks;
    }

    auto* tiling = context->GetTilingData<KvCacheTurboQuantTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling->vectors = vectors;
    tiling->blocks = blocks;
    tiling->mseBits = bits;
    tiling->idxStride = indexStride;
    tiling->qjlStride = qjlStride;
    tiling->headDim = headDim;
    tiling->qjlDim = qjlDim;
    tiling->mode = mode;
    tiling->rotationDtype = rotationType == ge::DT_BF16 ? 1U : 0U;
    tiling->qjlDtype = qjlType == ge::DT_BF16 ? 1U : 0U;
    tiling->outputDtype = outputDtype;
    tiling->codebookScale = std::sqrt(128.0F / static_cast<float>(headDim));
    if (mode == 1 || mode == 2) {
        matmul_tiling::MatmulApiTiling matmul(platform);
        matmul.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT,
                        false);
        matmul.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT,
                        true);
        matmul.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
        matmul.SetShape(cubeRows, 128, 128);
        matmul.SetOrgShape(cubeRows, 128, 128);
        matmul.SetBufferSpace(-1, -1, -1);
        if (matmul.GetTiling(tiling->cubeTiling) != 0) {
            return ge::GRAPH_FAILED;
        }
    }
    const uint64_t userWorkspace = (mode == 1 || mode == 2) ? ((static_cast<uint64_t>(vectors) + 7) / 8 * 8) *
                                                                  (4ULL * 128ULL * sizeof(float) + sizeof(float)) :
                                                              0;
    auto* workspaces = context->GetWorkspaceSizes(1);
    if (workspaces == nullptr) {
        return ge::GRAPH_FAILED;
    }
    workspaces[0] = SYSTEM_WORKSPACE_BYTES + userWorkspace;
    context->SetBlockDim(launchAicBlocks);
    context->SetTilingKey(0);
    return ge::GRAPH_SUCCESS;
}

struct KvCacheTurboQuantCompileInfo {};

static ge::graphStatus TilingParseForKvCacheTurboQuant(gert::TilingParseContext*)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(KvCacheTurboQuant)
    .Tiling(TilingFunc)
    .TilingParse<KvCacheTurboQuantCompileInfo>(TilingParseForKvCacheTurboQuant);
} // namespace optiling
