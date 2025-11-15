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
 * \file nsa_selected_attention_infer_proto.cpp
 * \brief
 */

 #include <graph/utils/type_utils.h>
 #include <register/op_impl_registry.h>
 #include "log/log.h"
 
 using namespace ge;
 
 namespace ops {
 static const uint64_t DIM_NUM_3 = 3;
 static const uint64_t DIM_NUM_2 = 2;
 constexpr size_t QUERY_INPUT_INDEX = 0;
 constexpr size_t KEY_INPUT_INDEX = 1;
 constexpr size_t VALUE_INPUT_INDEX = 2;
 constexpr size_t INPUT_LAYOUT_ATTR_INDEX = 0;
 static const int64_t HEAD_DIM_V_MAX = 128;
 
 ge::graphStatus InferShapeNsaSelectAttentionInfer(gert::InferShapeContext *context)
 {
     OP_LOGI(context, "Enter NsaSelectedAttentionInfer runtime infershape impl.");
     if (context == nullptr) {
         return ge::GRAPH_FAILED;
     }
     const gert::Shape *queryShape = context->GetInputShape(QUERY_INPUT_INDEX);
     OP_CHECK_NULL_WITH_CONTEXT(context, queryShape);
     const gert::Shape *keyShape = context->GetInputShape(KEY_INPUT_INDEX);
     OP_CHECK_NULL_WITH_CONTEXT(context, keyShape);
     const gert::Shape *valueShape = context->GetInputShape(VALUE_INPUT_INDEX);
     OP_CHECK_NULL_WITH_CONTEXT(context, valueShape);
     auto attrs = context->GetAttrs();
     const auto *queryDesc = context->GetInputDesc(QUERY_INPUT_INDEX);
     OP_CHECK_NULL_WITH_CONTEXT(context, queryDesc);
     OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
 
     const char *inputLayout = attrs->GetAttrPointer<char>(INPUT_LAYOUT_ATTR_INDEX);
     OP_CHECK_NULL_WITH_CONTEXT(context, inputLayout);
     std::string inputLayoutStr = std::string(inputLayout);
     for (auto &c : inputLayoutStr) {
         c = toupper(c);
     }
    if (inputLayoutStr != "BSND" && inputLayoutStr != "BSH" && inputLayoutStr != "TND") {
         OP_LOGE(context, "The inputLayout should be BSND or BSH or TND, but got %s.",
                 inputLayoutStr.c_str());
         return GRAPH_FAILED;
     }

     gert::Shape *attentionOutShape = context->GetOutputShape(0);
     OP_CHECK_NULL_WITH_CONTEXT(context, attentionOutShape);
     *attentionOutShape = *queryShape;
     if (inputLayoutStr == "BSND") {
         auto shapeD2 = valueShape->GetDim(DIM_NUM_3);
         attentionOutShape->SetDim(DIM_NUM_3, shapeD2);
     } else if (inputLayoutStr == "BSH") {
         if (queryShape->GetDim(DIM_NUM_2) % keyShape->GetDim(DIM_NUM_2) != 0) {
             OP_LOGE(context, "The H of query shape cannot div  %s.",
                 inputLayoutStr.c_str());
             return GRAPH_FAILED;
         }
         auto group = queryShape->GetDim(DIM_NUM_2) / keyShape->GetDim(DIM_NUM_2);
         auto shapeD2 = valueShape->GetDim(DIM_NUM_2) * group;
         attentionOutShape->SetDim(DIM_NUM_2, shapeD2);
     } else if (inputLayoutStr == "TND") {
         attentionOutShape->SetDim(DIM_NUM_2, HEAD_DIM_V_MAX);
     }
 
     return GRAPH_SUCCESS;
 }    
 
 ge::graphStatus InferDataTypeNsaSelectAttentionInfer(gert::InferDataTypeContext *context)
 {   
     if (context == nullptr) {
         return ge::GRAPH_FAILED;
     }
     const auto inputDataType = context->GetInputDataType(0);
     context->SetOutputDataType(0, inputDataType);
     return ge::GRAPH_SUCCESS;
 }
 
 IMPL_OP(NsaSelectedAttentionInfer).InferShape(InferShapeNsaSelectAttentionInfer).InferDataType(InferDataTypeNsaSelectAttentionInfer);
 } // namespace ops
  