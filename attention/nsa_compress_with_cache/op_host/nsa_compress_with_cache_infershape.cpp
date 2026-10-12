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
 * \file nsa_compress_with_cache_proto.cc
 * \brief
 */

#include "register/op_impl_registry.h"
#include "log/log.h"
#include "platform/platform_info.h"

using namespace ge;
namespace ops {
static constexpr size_t INDEX_IN_OUTPUT_CACHE = 3;
static constexpr size_t INDEX_OUT_OUTPUT_CACHE = 0;

static ge::graphStatus InferShapeNsaCompressWithCache(gert::InferShapeContext* context)
{
    const gert::Shape* outputCacheShape = context->GetInputShape(INDEX_IN_OUTPUT_CACHE);
    gert::Shape* outputCacheRefShape = context->GetOutputShape(INDEX_OUT_OUTPUT_CACHE);
    *outputCacheRefShape = *outputCacheShape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeNsaCompressWithCache(gert::InferDataTypeContext* context)
{
    const auto outputCacheDataType = context->GetInputDataType(INDEX_IN_OUTPUT_CACHE);
    context->SetOutputDataType(INDEX_OUT_OUTPUT_CACHE, outputCacheDataType);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(NsaCompressWithCache)
    .InferShape(InferShapeNsaCompressWithCache)
    .InferDataType(InferDataTypeNsaCompressWithCache);
} // namespace ops
