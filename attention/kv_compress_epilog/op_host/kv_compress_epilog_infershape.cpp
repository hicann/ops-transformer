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
 * \file kv_compress_epilog_infershape.cpp
 * \brief
 */
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>

using namespace ge;
namespace ops {

graphStatus InferShape4KvCompressEpilog(gert::InferShapeContext* context)
{
    // 输出 cache 与输入 cache（原地更新）shape 一致。图模式下 GE 依赖此处推导 NetOutput shape，
    // 缺省（空实现）会导致 GE NetOutput shape 为 [] 与 FX 图不一致。单算子直调路径不经过此函数。
    const gert::Shape* cacheShape = context->GetInputShape(0);
    gert::Shape* outCacheShape = context->GetOutputShape(0);
    if (cacheShape == nullptr || outCacheShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *outCacheShape = *cacheShape;
    return ge::GRAPH_SUCCESS;
}

graphStatus InferDtype4KvCompressEpilog(gert::InferDataTypeContext* context)
{
    // 输出 cache 与输入 cache dtype 一致（UINT8）。
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(KvCompressEpilog).InferShape(InferShape4KvCompressEpilog).InferDataType(InferDtype4KvCompressEpilog);
} // namespace ops
