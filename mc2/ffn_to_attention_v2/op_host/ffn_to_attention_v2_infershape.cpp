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
 * \file ffn_to_attention_v2_infershape.cpp
 * \brief
 */
#include "runtime_util.h"
#include "mc2_log.h"
using namespace ge;
namespace ops {
static ge::graphStatus InferShapeFFNToAttentionV2(gert::InferShapeContext *context)
{
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeFFNToAttentionV2(gert::InferDataTypeContext *context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(FFNToAttentionV2)
    .InferShape(InferShapeFFNToAttentionV2)
    .InferDataType(InferDataTypeFFNToAttentionV2);
} // namespace ops
