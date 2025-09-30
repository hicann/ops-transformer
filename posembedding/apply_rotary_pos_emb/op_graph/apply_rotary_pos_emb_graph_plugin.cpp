/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file apply_rotary_pos_emb_graph_plugin.cpp
 * \brief
 */

#include "register/op_impl_registry.h"
#include "log/log.h"

namespace ops {
static ge::graphStatus ApplyRotaryPosEmbInferDtype(gert::InferDataTypeContext *context)
{
    OP_LOGD(context, "ApplyRotaryPosEmbInferDtype begin.");
    context->SetOutputDataType(OUTPUT_0_IDX, context->GetInputDataType(INPUT0));
    context->SetOutputDataType(OUTPUT_1_IDX, context->GetInputDataType(INPUT1));
    OP_LOGD(context, "ApplyRotaryPosEmbInferDtype end.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(ApplyRotaryPosEmb).InferDataType(ApplyRotaryPosEmbInferDtype);
}; // namespace ops
