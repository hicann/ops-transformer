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
 * \file scatter_pa_cache.cpp
 * \brief
 */

#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace ops {

constexpr int64_t INPUT_KEY = 0;
constexpr int64_t INPUT_KEY_CACHE = 1;
constexpr int64_t INPUT_SLOT_MAPPING = 2;
constexpr int64_t ATTR_CACHE_MODE_INDEX = 0;
constexpr int64_t OPTIONAL_INPUT_COMPRESS_LENS = 3;
constexpr int64_t OPTIONAL_INPUT_COMPRESS_SEQ_OFFSET = 4;
constexpr int64_t OPTIONAL_INPUT_SEQ_LENS = 5;

constexpr int64_t OUTPUT_KEY_CACHE = 0;

ge::graphStatus InferShape4ScatterPaCache(gert::InferShapeContext* context)
{
    // inferShape key_cache
    OP_LOGD(context, "Begin to do InferShape4ScatterPaCache");
    auto inputKeyCacheShape = context->GetRequiredInputShape(INPUT_KEY_CACHE);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputKeyCacheShape);
    auto outputKeyCacheShape = context->GetOutputShape(OUTPUT_KEY_CACHE);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputKeyCacheShape);
    *outputKeyCacheShape = *inputKeyCacheShape;

    return GRAPH_SUCCESS;
}

ge::graphStatus InferDataType4ScatterPaCache(gert::InferDataTypeContext* context)
{
    OP_LOGD(context, "Begin to do InferDataType4ScatterPaCache");
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    auto cache_mode = attrs->GetAttrPointer<char>(ATTR_CACHE_MODE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, cache_mode);
    OP_CHECK_IF(strcmp(cache_mode, "") != 0 && strcmp(cache_mode, "Norm") != 0 && strcmp(cache_mode, "PA_NZ") != 0,
                OP_LOGE(context, "invalid cache mode : %s", cache_mode), return ge::GRAPH_FAILED);

    context->SetOutputDataType(OUTPUT_KEY_CACHE, context->GetRequiredInputDataType(INPUT_KEY));
    OP_LOGD(context, "End to do InferDataType4ScatterPaCache");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ScatterPaCache).InferShape(InferShape4ScatterPaCache).InferDataType(InferDataType4ScatterPaCache);
} // namespace ops
