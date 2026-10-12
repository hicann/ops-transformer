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
 * \file rotary_position_embedding_grad.cc
 * \brief
 */
#include "rotary_position_embedding_grad_tiling.h"
#include "rope_half_grad_tiling.h"
#include "rope_interleaved_grad_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "op_host/tiling_templates_registry.h"

namespace optiling {
using namespace Ops::Base;
constexpr uint32_t MODE_ATTR_IDX = 0;

ge::graphStatus RotaryPosEmbeddingGradMembaseTilingClass::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo != nullptr) {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        aicoreParams_.numBlocks = ascendcPlatform.GetCoreNumAiv();
        uint64_t ubSizePlatForm;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
        aicoreParams_.ubSize = ubSizePlatForm;
        socVersion_ = ascendcPlatform.GetSocVersion();
    } else {
        auto compileInfoPtr = context_->GetCompileInfo<RotaryPositionEmbeddingGradCompileInfo>();
        OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context_, "compile info is null"), return ge::GRAPH_FAILED);
        aicoreParams_.numBlocks = compileInfoPtr->numBlocks;
        aicoreParams_.ubSize = compileInfoPtr->ubSize;
        socVersion_ = compileInfoPtr->socVersion;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RotaryPosEmbeddingGradMembaseTilingClass::GetShapeAttrsInfo()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE(context_->GetNodeName(), "attrs is null."), return ge::GRAPH_FAILED);
    const uint32_t inputMode = *(attrs->GetAttrPointer<char>(MODE_ATTR_IDX));
    OP_LOGI("[RotaryPositionEmbedding]", "[mode]: %d", inputMode);

    auto platformInfo = context_->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    if (!Ops::Transformer::OpTiling::IsRegbaseSocVersion(context_) &&
        (inputMode != MODE_ROTATE_HALF && inputMode != MODE_ROTATE_INTERLEAVED)) {
        std::string modeStr = std::to_string(inputMode);
        OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "mode", modeStr.c_str(), "0 or 1");
        return ge::GRAPH_FAILED;
    }
    inputMode_ = inputMode;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Tiling4RotaryPositionEmbeddingGrad(gert::TilingContext* context)
{
    return Ops::Transformer::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

/*
 * \brief TilingPrepare for reduce template, if reduce operator has its own compile info, use this interface
 *
 * @tparam ContextT
 *  can be TilingParseContext or TilingContext
 */
template <typename ContextT>
ge::graphStatus TilingPrepare4ReduceOp(ContextT* context, Ops::Base::ReduceOpCompileInfo* compileInfo)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->vectorCoreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((compileInfo->vectorCoreNum == 0UL),
                OP_LOGE(context->GetNodeName(), "ReduceOp GetHardwareInfo Failed, vectorCoreNum:%lu",
                        compileInfo->vectorCoreNum),
                return ge::GRAPH_FAILED);

    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize <= Ops::Base::CACHE_BUF_SIZE,
                OP_LOGE(context->GetNodeName(), "ReduceOp GetHardwareInfo Failed, ubSize:%lu, at least:%lu.",
                        compileInfo->ubSize, Ops::Base::CACHE_BUF_SIZE),
                return ge::GRAPH_FAILED);
    compileInfo->ubSize = ubSize;

    compileInfo->cacheLineSize = Ops::Base::GetCacheLineSize(context);
    OP_CHECK_IF(compileInfo->cacheLineSize == 0UL,
                OP_LOGE(context->GetNodeName(), "ReduceOp GetHardwareInfo Failed, cacheLineSize:%lu.",
                        compileInfo->cacheLineSize),
                return ge::GRAPH_FAILED);

    compileInfo->ubBlockSize = Ops::Base::GetUbBlockSize(context);
    OP_CHECK_IF(
        compileInfo->ubBlockSize == 0UL,
        OP_LOGE(context->GetNodeName(), "ReduceOp GetHardwareInfo Failed, ubBlockSize:%lu.", compileInfo->ubBlockSize),
        return ge::GRAPH_FAILED);

    compileInfo->vRegSize = Ops::Base::GetVRegSize(context);
    OP_CHECK_IF(
        compileInfo->vRegSize == 0UL,
        OP_LOGE(context->GetNodeName(), "ReduceOp GetHardwareInfo Failed, vRegSize:%lu.", compileInfo->vRegSize),
        return ge::GRAPH_FAILED);

    OP_LOGD(context->GetNodeName(), "GetCoreNum:%lu, ubSize:%lu, cacheLineSize:%lu, ubBlockSize:%lu, vRegSize:%lu",
            compileInfo->vectorCoreNum, compileInfo->ubSize, compileInfo->cacheLineSize, compileInfo->ubBlockSize,
            compileInfo->vRegSize);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForRotaryPositionEmbeddingGrad(gert::TilingParseContext* context)
{
    auto platformInfo = context->GetPlatformInfo();
    auto compileInfoPtr = context->GetCompiledInfo<RotaryPositionEmbeddingGradCompileInfo>();
    OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context, "compile info is null"), return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    compileInfoPtr->numBlocks = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfoPtr->ubSize = ubSizePlatForm;
    compileInfoPtr->socVersion = ascendcPlatform.GetSocVersion();
    if (Ops::Transformer::OpTiling::IsRegbaseSocVersion(context)) {
        return TilingPrepare4ReduceOp(context, &compileInfoPtr->opInfo);
    }
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(RotaryPositionEmbeddingGrad, RopeInterLeavedGradTlingClass, 50000);
REGISTER_OPS_TILING_TEMPLATE(RotaryPositionEmbeddingGrad, RopeRotateHalfGradTlingClass, 60000);

IMPL_OP_OPTILING(RotaryPositionEmbeddingGrad)
    .Tiling(Tiling4RotaryPositionEmbeddingGrad)
    .TilingParse<RotaryPositionEmbeddingGradCompileInfo>(TilingPrepareForRotaryPositionEmbeddingGrad);
} // namespace optiling
