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
 * \file nsa_compress_with_cache_tiling.cpp
 * \brief
 */
#include "nsa_compress_with_cache_tiling.h"
#include "nsa_compress_with_cache_tiling_common.h"

#include <climits>
#include <graph/utils/type_utils.h>
#include <vector>
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "op_host/tiling_templates_registry.h"

using namespace ge;
using namespace AscendC;
using namespace Ops::Transformer::OpTiling;

namespace optiling {

constexpr size_t INPUT_INPUT_INDEX = 0;
constexpr size_t WEIGHT_INPUT_INDEX = 1;
constexpr size_t OUTPUT_CACHE_INPUT_INDEX = 3;

static void ReportTilingFailure(const std::string& reason)
{
    constexpr const char* opName = "NsaCompressWithCache";
    OP_LOGE_WITHOUT_REPORT(opName, "%s", reason.c_str());
    const std::vector<const char*> keys = {"opName", "reason"};
    const std::vector<const char*> values = {opName, reason.c_str()};
    REPORT_PREDEFINED_ERR_MSG("EZ1008", keys, values);
}

static ge::graphStatus CheckParams(const gert::TilingContext* context)
{
    if (context->GetInputShape(INPUT_INPUT_INDEX) != nullptr && context->GetInputShape(WEIGHT_INPUT_INDEX) != nullptr &&
        context->GetInputShape(OUTPUT_CACHE_INPUT_INDEX) != nullptr && context->GetAttrs() != nullptr) {
        return ge::SUCCESS;
    }
    OP_LOGW(context, "fail to get shape or attr from context");
    return ge::GRAPH_FAILED;
}

static bool IsEmptyInput(gert::TilingContext* context)
{
    auto inputShape = context->GetInputShape(INPUT_INPUT_INDEX);
    OP_CHECK_IF(inputShape == nullptr, ReportTilingFailure("Input shape is null."), return false);
    auto weightShape = context->GetInputShape(WEIGHT_INPUT_INDEX);
    OP_CHECK_IF(weightShape == nullptr, ReportTilingFailure("Weight shape is null."), return false);
    auto outputCacheShape = context->GetInputShape(OUTPUT_CACHE_INPUT_INDEX);

    OP_CHECK_IF(outputCacheShape == nullptr, ReportTilingFailure("Output cache shape is null."), return false);

    int64_t inputShapeSize = inputShape->GetStorageShape().GetShapeSize();
    int64_t weightShapeSize = weightShape->GetStorageShape().GetShapeSize();
    int64_t outputCacheShapeSize = outputCacheShape->GetStorageShape().GetShapeSize();
    if (inputShapeSize == 0 || weightShapeSize == 0 || outputCacheShapeSize == 0) {
        return true;
    }
    return false;
}

ASCENDC_EXTERN_C ge::graphStatus NsaCompressWithCache(gert::TilingContext* context)
{
    if (CheckParams(context) != ge::GRAPH_SUCCESS) {
        ReportTilingFailure("Failed to get input, weight or output_cache shape or attributes from the tiling context.");
        return ge::GRAPH_FAILED;
    }
    if (IsEmptyInput(context)) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("NsaCompressWithCache", "input, weight, output_cache",
                                                 "Empty tensors are unsupported.");
        return ge::GRAPH_FAILED;
    } else {
        auto resultCode = TilingRegistry::GetInstance().DoTilingImpl(context);
        return resultCode;
    }
    return ge::GRAPH_FAILED;
}

ASCENDC_EXTERN_C ge::graphStatus TilingPrepareForNsaCompressWithCache(gert::TilingParseContext* context)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfoPtr == nullptr, ReportTilingFailure("Tiling parse context platform info is null."),
                return ge::GRAPH_FAILED);
    auto compileInfoPtr = context->GetCompiledInfo<NsaCompressWithCacheCompileInfo>();
    OP_CHECK_IF(compileInfoPtr == nullptr, ReportTilingFailure("Tiling parse context compile info is null."),
                return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->aivNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);

    OP_CHECK_IF((compileInfoPtr->aivNum == 0 || compileInfoPtr->ubSize == 0),
                ReportTilingFailure("Invalid platform info: aivNum=" + std::to_string(compileInfoPtr->aivNum) +
                                    ", ubSize=" + std::to_string(compileInfoPtr->ubSize) + "."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(NsaCompressWithCache)
    .Tiling(NsaCompressWithCache)
    .TilingParse<NsaCompressWithCacheCompileInfo>(TilingPrepareForNsaCompressWithCache); // regist into the framework
} // namespace optiling
