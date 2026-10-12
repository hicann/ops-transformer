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
 * \file indexer_quant_cache_infershape.cpp
 * \brief
 */
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>

using namespace ge;
namespace ops {

graphStatus InferShape4IndexerQuantCache(gert::InferShapeContext* context)
{
    // 输出 cache / cache_scale 与各自输入（原地更新）shape 一致。图模式下 GE 依赖此处推导 NetOutput
    // shape，缺省（空实现）会导致 GE NetOutput shape 为 [] 与 FX 图不一致。单算子直调路径不经过此函数。
    const gert::Shape* cacheShape = context->GetInputShape(0);
    const gert::Shape* cacheScaleShape = context->GetInputShape(1);
    gert::Shape* outCacheShape = context->GetOutputShape(0);
    gert::Shape* outCacheScaleShape = context->GetOutputShape(1);
    if (cacheShape == nullptr || cacheScaleShape == nullptr || outCacheShape == nullptr ||
        outCacheScaleShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *outCacheShape = *cacheShape;
    *outCacheScaleShape = *cacheScaleShape;
    return ge::GRAPH_SUCCESS;
}

graphStatus InferDtype4IndexerQuantCache(gert::InferDataTypeContext* context)
{
    // 输出 cache / cache_scale 与各自输入 dtype 一致。
    context->SetOutputDataType(0, context->GetInputDataType(0));
    context->SetOutputDataType(1, context->GetInputDataType(1));
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(IndexerQuantCache)
    .InferShape(InferShape4IndexerQuantCache)
    .InferDataType(InferDtype4IndexerQuantCache);
} // namespace ops
