/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MC2_COMBINE_INFERSHAPE_H
#define MC2_COMBINE_INFERSHAPE_H

#include <string>
#include "register/op_impl_registry.h"
#include "mc2_log.h"

namespace ops {
namespace Mc2InferShape {

inline ge::graphStatus InferCombineOutputShape(gert::InferShapeContext *context, size_t expandXIndex,
                                               size_t expertIdsIndex, size_t outputIndex)
{
    const gert::Shape *expandXShape = context->GetInputShape(expandXIndex);
    OPS_CHECK_NULL_WITH_CONTEXT(context, expandXShape);
    const gert::Shape *expertIdsShape = context->GetInputShape(expertIdsIndex);
    OPS_CHECK_NULL_WITH_CONTEXT(context, expertIdsShape);
    gert::Shape *xShape = context->GetOutputShape(outputIndex);
    OPS_CHECK_NULL_WITH_CONTEXT(context, xShape);

    int64_t bs = ((expertIdsShape->GetDimNum() == 1U) ? -1 : expertIdsShape->GetDim(0));
    int64_t h = ((expandXShape->GetDimNum() == 1U) ? -1 : expandXShape->GetDim(1));
    xShape->SetDimNum(2UL);
    xShape->SetDim(0U, bs);
    xShape->SetDim(1U, h);
    OP_LOGD(context->GetNodeName(), "x shape shape is :%s after infershape.", Ops::Base::ToString(*xShape).c_str());
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus InferCombineOutputXDataType(gert::InferDataTypeContext *context, size_t expandXIndex,
                                                   size_t outputIndex)
{
    auto xDtype = context->GetInputDataType(expandXIndex);
    context->SetOutputDataType(outputIndex, xDtype);
    return ge::GRAPH_SUCCESS;
}

} // namespace Mc2InferShape
} // namespace ops

#endif // MC2_COMBINE_INFERSHAPE_H
