/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "fag_input_checker.h"

#include "log/log.h"

namespace optiling {

bool IsOptionalTensorExist(gert::TilingContext *context, size_t index)
{
    auto shape = context->GetOptionalInputShape(index);
    return (shape != nullptr) && (shape->GetStorageShape().GetDimNum() > 0);
}

ge::graphStatus FagInputExistenceChecker::Check(FagCheckCtx &ctx)
{
    gert::TilingContext *context = ctx.context;
    OP_CHECK_IF(context->GetInputShape(Q_INDEX) == nullptr || context->GetInputShape(K_INDEX) == nullptr ||
                    context->GetInputShape(V_INDEX) == nullptr || context->GetInputShape(DOUT_INDEX) == nullptr ||
                    context->GetInputShape(ATTN_OUT_INDEX) == nullptr ||
                    context->GetInputShape(SOFTMAX_LSE_INDEX) == nullptr,
                OP_LOGE(ctx.opName, "required input (q/k/v/dout/attn_out/softmax_lse) shape is nullptr."),
                return ge::GRAPH_PARAM_INVALID);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FagDtypeChecker::Check(FagCheckCtx &ctx)
{
    gert::TilingContext *context = ctx.context;
    const char *opName = ctx.opName;

    struct DescRef {
        const char *name;
        const gert::CompileTimeTensorDesc *desc;
    };

    auto qDesc = context->GetInputDesc(Q_INDEX);
    const DescRef others[] = {
        {"k", context->GetInputDesc(K_INDEX)},        {"v", context->GetInputDesc(V_INDEX)},
        {"dout", context->GetInputDesc(DOUT_INDEX)},  {"attn_out", context->GetInputDesc(ATTN_OUT_INDEX)},
        {"dq", context->GetOutputDesc(DQ_OUT_INDEX)}, {"dk", context->GetOutputDesc(DK_OUT_INDEX)},
        {"dv", context->GetOutputDesc(DV_OUT_INDEX)},
    };

    OP_CHECK_IF(qDesc == nullptr, OP_LOGE(opName, "failed to get input/output desc for dtype check."),
                return ge::GRAPH_PARAM_INVALID);
    for (const auto &item : others) {
        OP_CHECK_IF(item.desc == nullptr, OP_LOGE(opName, "failed to get input/output desc for dtype check."),
                    return ge::GRAPH_PARAM_INVALID);
    }

    const ge::DataType qDtype = qDesc->GetDataType();
    for (const auto &item : others) {
        OP_CHECK_IF(item.desc->GetDataType() != qDtype,
                    OP_LOGE(opName, "dtype of %s must be the same as q(%d), but got %d.", item.name,
                            static_cast<int32_t>(qDtype), static_cast<int32_t>(item.desc->GetDataType())),
                    return ge::GRAPH_PARAM_INVALID);
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
