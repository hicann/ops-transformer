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
 * \file sparse_flash_mla_infershape.cpp
 * \brief
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "err/ops_err.h"
#include "common/smla_infershape_common.h"

using namespace ge;

namespace ops {
constexpr uint32_t RETURN_SOFTMAX_INDEX = 9;
constexpr uint32_t LAYOUT_Q_ATTR_INDEX = 6;
constexpr uint32_t LAYOUT_KV_ATTR_INDEX = 7;

ge::graphStatus InferShapeSparseFlashMla(gert::InferShapeContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("SparseFlashMla", "InferShapeContext is nullptr"), return ge::GRAPH_FAILED);
    return optiling::SMLAInferShape(context, "SparseFlashMla", RETURN_SOFTMAX_INDEX, LAYOUT_Q_ATTR_INDEX,
                                    LAYOUT_KV_ATTR_INDEX);
}

ge::graphStatus InferDataTypeSparseFlashMla(gert::InferDataTypeContext *context)
{
    return optiling::SMLAInferDataTypeByInput(context, "SparseFlashMla");
}

IMPL_OP_INFERSHAPE(SparseFlashMla).InferShape(InferShapeSparseFlashMla).InferDataType(InferDataTypeSparseFlashMla);
} // namespace ops
