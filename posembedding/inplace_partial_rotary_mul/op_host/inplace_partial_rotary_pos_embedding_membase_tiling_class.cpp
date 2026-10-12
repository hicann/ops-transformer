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
 * \file inplace_partial_rotary_pos_embedding_membase_tiling_class.cpp
 * \brief
 */
#include "inplace_partial_rotary_mul_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include <vector>
namespace optiling {
constexpr uint32_t MODE_ATTR_IDX = 0;

ge::graphStatus InplacePartialRotaryPosEmbeddingMembaseTilingClass::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo != nullptr) {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        aicoreParams_.blockDim = ascendcPlatform.GetCoreNumAiv();
        uint64_t ubSizePlatForm;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
        aicoreParams_.ubSize = ubSizePlatForm;
    } else {
        auto compileInfoPtr =
            reinterpret_cast<const InplacePartialRotaryPositionEmbeddingCompileInfo*>(context_->GetCompileInfo());
        OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context_, "compile info is null"), return ge::GRAPH_FAILED);
        aicoreParams_.ubSize = compileInfoPtr->ubSize;
        aicoreParams_.blockDim = compileInfoPtr->blockDim;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InplacePartialRotaryPosEmbeddingMembaseTilingClass::GetShapeAttrsInfo()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    const uint32_t inputMode = *(attrs->GetAttrPointer<uint32_t>(MODE_ATTR_IDX));
    OP_LOGI(context_->GetNodeName(), "[mode]: %d", inputMode);
    inputMode_ = inputMode;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Tiling4InplacePartialRotaryMulDispatcher(gert::TilingContext* context)
{
    OP_LOGI(context, "Tiling4InplacePartialRotaryMulDispatcher start");
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "Tiling context is null"), return ge::GRAPH_FAILED);

    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OP_LOGE(context, "Tiling platformInfo is null"), return ge::GRAPH_FAILED);
    if (Ops::Transformer::OpTiling::IsRegbaseSocVersion(context)) {
        std::vector<std::unique_ptr<InplacePartialRopeRegBaseTilingClass>> regBaseTilingCases;
        regBaseTilingCases.push_back(std::unique_ptr<InplacePartialRopeRegBaseTilingClass>(
            new InplacePartialRopeRegBaseTilingClassAAndB(context)));
        regBaseTilingCases.push_back(
            std::unique_ptr<InplacePartialRopeRegBaseTilingClass>(new InplacePartialRopeRegBaseTilingClassAB(context)));
        regBaseTilingCases.push_back(std::unique_ptr<InplacePartialRopeRegBaseTilingClass>(
            new InplacePartialRopeRegBaseTilingClassABAAndBA(context)));
        regBaseTilingCases.push_back(std::unique_ptr<InplacePartialRopeRegBaseTilingClass>(
            new InplacePartialRopeRegBaseTilingClassBAB(context)));
        OP_LOGI(context, "Using arch35 tiling for regbase soc");

        for (const auto& ptr : regBaseTilingCases) {
            if (ptr) {
                ge::graphStatus status = ptr->DoTiling();
                if (status != ge::GRAPH_PARAM_INVALID) {
                    OP_LOGI(context, "Do general op tiling success priority");
                    return status;
                }
                OP_LOGI(context, "Ignore general op tiling priority");
            }
        }
        OP_LOGI(context, "Using tiling for ASCEND910_71");
        InplacePartialRotaryPosEmbeddingMembaseTilingClass rotaryPosEmbeddingMembaseTilingClass(context);
        return rotaryPosEmbeddingMembaseTilingClass.DoOpTiling();
    } else {
        return Tiling4InplacePartialRotaryMul(context);
    }
}

ge::graphStatus TilingPrepareForInplacePartialRotaryMul(gert::TilingParseContext* context)
{
    OP_LOGI(context, "TilingPrepareForInplacePartialRotaryMul context success");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(InplacePartialRotaryMul)
    .Tiling(Tiling4InplacePartialRotaryMulDispatcher)
    .TilingParse<InplacePartialRotaryPositionEmbeddingCompileInfo>(TilingPrepareForInplacePartialRotaryMul);
} // namespace optiling
