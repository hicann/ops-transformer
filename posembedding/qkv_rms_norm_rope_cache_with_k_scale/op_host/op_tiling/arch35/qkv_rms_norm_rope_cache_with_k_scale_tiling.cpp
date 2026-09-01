/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "qkv_rms_norm_rope_cache_with_k_scale_tiling.h"

#include "log/log.h"
#include "op_host/tiling_templates_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include <string>

namespace optiling {
namespace {
constexpr const char *OP_NAME = "QkvRmsNormRopeCacheWithKScale";
} // namespace

ge::graphStatus Tiling4QkvRmsNormRopeCacheWithKScale(gert::TilingContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(OP_NAME, "context is nullptr."), return ge::GRAPH_FAILED);
    OP_LOGD(context, "QkvRmsNormRopeCacheWithKScale tiling start.");
    return Ops::Transformer::OpTiling::TilingRegistryArch::GetInstance().DoTilingImpl(context);
}

ge::graphStatus TilingPrepare4QkvRmsNormRopeCacheWithKScale(gert::TilingParseContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(OP_NAME, "context is nullptr."), return ge::GRAPH_FAILED);
    auto compileInfo = context->GetCompiledInfo<QkvRmsNormRopeCacheWithKScaleCompileInfo>();
    OP_CHECK_IF(compileInfo == nullptr, OP_LOGE(context, "compileInfo is nullptr."), return ge::GRAPH_FAILED);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OP_LOGE(context, "platformInfo is nullptr."), return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->aicNum = static_cast<uint32_t>(ascendcPlatform.GetCoreNumAic());
    compileInfo->aivNum = static_cast<uint32_t>(ascendcPlatform.GetCoreNumAiv());
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfo->l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, compileInfo->l0cSize);
    compileInfo->opWorkspaceSize = static_cast<uint64_t>(ascendcPlatform.GetLibApiWorkSpaceSize());
    OP_LOGD(context,
            "QkvRmsNormRopeCacheWithKScale tiling prepare resources: aicNum=%u aivNum=%u ubSize=%llu "
            "l1Size=%llu l0cSize=%llu opWorkspaceSize=%llu.",
            compileInfo->aicNum, compileInfo->aivNum, compileInfo->ubSize, compileInfo->l1Size, compileInfo->l0cSize,
            compileInfo->opWorkspaceSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(QkvRmsNormRopeCacheWithKScale)
    .Tiling(Tiling4QkvRmsNormRopeCacheWithKScale)
    .TilingParse<QkvRmsNormRopeCacheWithKScaleCompileInfo>(TilingPrepare4QkvRmsNormRopeCacheWithKScale);

} // namespace optiling
