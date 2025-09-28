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
 * \file nsa_compress_grad.cc
 * \brief
 */
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "log/log.h"
using namespace ge;
namespace ops {
static ge::graphStatus InferShape4NsaCompressGrad(gert::InferShapeContext* context) {
  OP_LOGD(context->GetNodeName(), "Begin to do NsaCompressGradInfershape.");
  // 获取输入shape
  const gert::Shape* inputShape = context->GetInputShape(1);
  OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
  const gert::Shape* weightShape = context->GetInputShape(2);
  OP_CHECK_NULL_WITH_CONTEXT(context, weightShape);

  gert::Shape* inputGradShape = context->GetOutputShape(0);
  OP_CHECK_NULL_WITH_CONTEXT(context, inputGradShape);
  gert::Shape* weightGradShape = context->GetOutputShape(1);
  OP_CHECK_NULL_WITH_CONTEXT(context, weightGradShape);

  *inputGradShape = *inputShape;
  *weightGradShape = *weightShape;

  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4NsaCompressGrad(gert::InferDataTypeContext* context) {
  OP_LOGD(context->GetNodeName(), "Begin to do NsaCompressGradInferDataType.");
  auto inputDtype = context->GetInputDataType(1);
  context->SetOutputDataType(0, inputDtype);
  context->SetOutputDataType(1, inputDtype);
  OP_LOGD(context->GetNodeName(), "End to do NsaCompressGradInferDataType.");
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(NsaCompressGrad).InferShape(InferShape4NsaCompressGrad).InferDataType(InferDataType4NsaCompressGrad);
}  // namespace ops