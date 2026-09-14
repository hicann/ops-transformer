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
 * \file quant_sparse_flash_mla_infershape.cpp
 * \brief
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "err/ops_err.h"
#include "quant_sparse_flash_mla_check.h"
#include "../../sparse_flash_mla/op_host/common/smla_infershape_common.h"

using namespace optiling;
using namespace ge;

namespace ops {
ge::graphStatus InferShapeQuantSparseFlashMla(gert::InferShapeContext *context)
{
    OP_CHECK_IF(context == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("QuantSparseFlashMla", "InferShapeContext",
                                                         "InferShapeContext is nullptr"),
                return ge::GRAPH_FAILED);
    return SMLAInferShape(context, "QuantSparseFlashMla", ATTR_RETURN_SOFTMAX_LSE_INDEX, ATTR_LAYOUT_Q_INDEX,
                          ATTR_LAYOUT_KV_INDEX);
}

ge::graphStatus InferDataTypeQuantSparseFlashMla(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(ATTN_OUT_INDEX, ge::DT_BF16); // 目前仅支持BF16
    context->SetOutputDataType(SOFTMAX_LSE_INDEX, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(QuantSparseFlashMla).InferShape(InferShapeQuantSparseFlashMla).InferDataType(InferDataTypeQuantSparseFlashMla);
} // namespace ops
