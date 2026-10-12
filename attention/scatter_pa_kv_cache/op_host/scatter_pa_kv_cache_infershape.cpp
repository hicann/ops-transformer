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
 * \file scatter_pa_kv_cache_infershape.cpp
 * \brief
 */

#include <exe_graph/runtime/infer_shape_context.h>
#include <register/op_impl_registry.h>
#include "log/log.h"

using namespace ge;

namespace ops {

constexpr int64_t INPUT_KEY = 0;
constexpr int64_t INPUT_KEY_CACHE = 1;
constexpr int64_t INPUT_SLOT_MAPPING = 2;
constexpr int64_t INPUT_VALUE = 3;
constexpr int64_t INPUT_VALUE_CACHE = 4;
constexpr int64_t ATTR_CACHE_MODE_INDEX = 0;
constexpr int64_t INPUT_SCATTER_MODE_INDEX = 1;
constexpr int64_t OPTIONAL_INPUT_COMPRESS_LENS = 5;
constexpr int64_t OPTIONAL_INPUT_COMPRESS_SEQ_OFFSET = 6;
constexpr int64_t OPTIONAL_INPUT_SEQ_LENS = 7;

constexpr int64_t OUTPUT_KEY_CACHE = 0;
constexpr int64_t OUTPUT_VALUE_CACHE = 1;

ge::graphStatus InferShape4ScatterPaKvCache(gert::InferShapeContext* context)
{
    // inferShape key_cache
    OP_LOGD(context, "Begin to do InferShape4ScatterPaKvCache");
    auto inputKeyCacheShape = context->GetInputShape(INPUT_KEY_CACHE);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputKeyCacheShape);
    auto outputKeyCacheShape = context->GetOutputShape(OUTPUT_KEY_CACHE);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputKeyCacheShape);
    *outputKeyCacheShape = *inputKeyCacheShape;

    // inferShape value_cache
    auto inputValueCacheShape = context->GetInputShape(INPUT_VALUE_CACHE);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputValueCacheShape);
    auto outputValueCacheShape = context->GetOutputShape(OUTPUT_VALUE_CACHE);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputValueCacheShape);
    *outputValueCacheShape = *inputValueCacheShape;
    OP_LOGD(context, "End to do InferShape4ScatterPaKvCache");
    return GRAPH_SUCCESS;
}

ge::graphStatus InferDataType4ScatterPaKvCache(gert::InferDataTypeContext* context)
{
    OP_LOGD(context, "Begin to do InferDataType4ScatterPaKvCache");
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    auto cache_mode = attrs->GetAttrPointer<char>(ATTR_CACHE_MODE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, cache_mode);
    if (strcmp(cache_mode, "") != 0 && strcmp(cache_mode, "Norm") != 0 && strcmp(cache_mode, "PA_NZ") != 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "cache_mode", cache_mode,
                                              "The cache_mode must be one of '', 'Norm', or 'PA_NZ'.");
        return ge::GRAPH_FAILED;
    }
    auto scatter_mode = attrs->GetAttrPointer<char>(INPUT_SCATTER_MODE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, scatter_mode);
    if (strcmp(scatter_mode, "") != 0 && strcmp(scatter_mode, "None") != 0 && strcmp(scatter_mode, "Alibi") != 0 &&
        strcmp(scatter_mode, "Rope") != 0 && strcmp(scatter_mode, "Omni") != 0 && strcmp(scatter_mode, "Nct") != 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            context->GetNodeName(), "scatter_mode", scatter_mode,
            "The scatter_mode must be one of '', 'None', 'Alibi', 'Rope', 'Omni', or 'Nct'.");
        return ge::GRAPH_FAILED;
    }

    context->SetOutputDataType(OUTPUT_KEY_CACHE, context->GetInputDataType(INPUT_KEY));
    context->SetOutputDataType(OUTPUT_VALUE_CACHE, context->GetInputDataType(INPUT_VALUE));
    OP_LOGD(context, "End to do InferDataType4ScatterPaKvCache");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ScatterPaKvCache)
    .InferShape(InferShape4ScatterPaKvCache)
    .InferDataType(InferDataType4ScatterPaKvCache);
} // namespace ops
