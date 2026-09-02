/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <register/op_impl_registry.h>
#include "log/log.h"
#include <cstring>

using namespace ge;

namespace ops {

// 算子输入/输出的索引常量，便于在推断逻辑中按索引获取形状与数据类型。
static constexpr uint32_t QUERY_INDEX = 0;         // query 输入索引
static constexpr uint32_t ATTENTION_OUT_INDEX = 0; // attentionOut 输出索引
static constexpr uint32_t SOFTMAX_LSE_INDEX = 1;   // softmaxLse 输出索引
static constexpr uint32_t TND_DIM_NUM = 3;         // TND 布局的维度数（T, N, D）
static constexpr uint32_t RANK4_DIM_NUM = 4;       // BNSD/BSND 布局的维度数
static constexpr uint32_t LSE_DIM_NUM = 3;         // TND softmaxLse 维度数
static constexpr uint32_t LSE_RANK4_DIM_NUM = 4;   // BNSD/BSND softmaxLse 维度数
static constexpr int32_t UNKNOWN_DIMS = -2;        // 未知维度标记，用于动态形状场景
static constexpr int64_t NUM_0 = 0;
static constexpr int64_t NUM_1 = 1;
// Matches op_def Attr order: numKeyValueHeads, scaleValue, blockSize, topK,
// innerPrecise, softmaxLseFlag, inputLayout.
static constexpr uint32_t ATTR_SOFTMAX_LSE_FLAG_INDEX = 5; // softmaxLseFlag 属性索引
static constexpr uint32_t ATTR_INPUT_LAYOUT_INDEX = 6;     // inputLayout 属性索引

static constexpr uint32_t LAYOUT_TND = 0;  // TND [T, N, D]
static constexpr uint32_t LAYOUT_BNSD = 1; // BNSD [B, N, S, D]
static constexpr uint32_t LAYOUT_BSND = 2; // BSND [B, S, N, D]

static bool ParseInputLayout(const char *layoutStr, uint32_t &layoutType)
{
    if (layoutStr == nullptr || layoutStr[0] == '\0') {
        layoutType = LAYOUT_TND;
        return true;
    }
    if (strcmp(layoutStr, "TND") == 0) {
        layoutType = LAYOUT_TND;
        return true;
    }
    if (strcmp(layoutStr, "BNSD") == 0) {
        layoutType = LAYOUT_BNSD;
        return true;
    }
    if (strcmp(layoutStr, "BSND") == 0) {
        layoutType = LAYOUT_BSND;
        return true;
    }
    return false;
}

// InferShapeMinimaxSparseAttentionSplitKv: 输出形状推断函数。
// 推断规则：输出 attentionOut 的形状等于输入 query 的形状（TND 布局: [total_q_tokens, num_q_heads, D]）。
// 特殊处理：当 query 为未知维度（UNKNOWN_DIMS, 即 -2）时，输出也设为单维未知，用于图编译期的动态形状场景。
// 校验：query 必须为 3 维（TND_DIM_NUM），否则返回 GRAPH_FAILED。
static ge::graphStatus InferShapeMinimaxSparseAttentionSplitKv(gert::InferShapeContext *context)
{
    if (context == nullptr) {
        OP_LOGE("MinimaxSparseAttentionSplitKv", "context is nullptr!");
        return ge::GRAPH_FAILED;
    }

    // 获取输入 query 的形状，作为输出形状推断的依据。
    const gert::Shape *queryShape = context->GetInputShape(QUERY_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryShape);
    // 获取输出 attentionOut 的形状对象，待填充。
    gert::Shape *attentionOutShape = context->GetOutputShape(ATTENTION_OUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, attentionOutShape);
    gert::Shape *softmaxLseShape = context->GetOutputShape(SOFTMAX_LSE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, softmaxLseShape);

    // 动态形状场景：query 整体未知时，输出同样标记为未知维度，直接返回成功。
    if (queryShape->GetDimNum() == 1 && queryShape->GetDim(0) == UNKNOWN_DIMS) {
        attentionOutShape->SetDimNum(1);
        (*attentionOutShape)[0] = UNKNOWN_DIMS;
        softmaxLseShape->SetDimNum(1);
        (*softmaxLseShape)[0] = UNKNOWN_DIMS;
        return ge::GRAPH_SUCCESS;
    }

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    uint32_t layoutType = LAYOUT_TND;
    if (!ParseInputLayout(attrs->GetAttrPointer<char>(ATTR_INPUT_LAYOUT_INDEX), layoutType)) {
        OP_LOGE(context->GetNodeName(), "inputLayout must be TND, BNSD or BSND.");
        return ge::GRAPH_FAILED;
    }

    const size_t qDimNum = queryShape->GetDimNum();
    if (layoutType == LAYOUT_TND) {
        if (qDimNum != TND_DIM_NUM) {
            OP_LOGE(context->GetNodeName(), "inputLayout TND requires query rank 3 [T, N, D], got %zu.", qDimNum);
            return ge::GRAPH_FAILED;
        }
    } else if (qDimNum != RANK4_DIM_NUM) {
        OP_LOGE(context->GetNodeName(), "inputLayout BNSD/BSND requires query rank 4, got %zu.", qDimNum);
        return ge::GRAPH_FAILED;
    }

    // 输出形状直接等于 query 形状（TND）。
    *attentionOutShape = *queryShape;

    const bool *softmaxLsePtr = attrs->GetAttrPointer<bool>(ATTR_SOFTMAX_LSE_FLAG_INDEX);
    bool softmaxLseFlag = (softmaxLsePtr != nullptr) ? *softmaxLsePtr : false;
    if (softmaxLseFlag) {
        if (layoutType == LAYOUT_BNSD) {
            // BNSD LSE [B, N, S, 1] fp32.
            softmaxLseShape->SetDimNum(LSE_RANK4_DIM_NUM);
            *softmaxLseShape = {queryShape->GetDim(0), queryShape->GetDim(1), queryShape->GetDim(2), NUM_1};
        } else if (layoutType == LAYOUT_BSND) {
            // BSND LSE [B, S, N, 1] fp32.
            softmaxLseShape->SetDimNum(LSE_RANK4_DIM_NUM);
            *softmaxLseShape = {queryShape->GetDim(0), queryShape->GetDim(1), queryShape->GetDim(2), NUM_1};
        } else {
            // TND LSE, same as FIA: [T, N, 1] fp32.
            softmaxLseShape->SetDimNum(LSE_DIM_NUM);
            *softmaxLseShape = {queryShape->GetDim(0), queryShape->GetDim(1), NUM_1};
        }
    } else {
        softmaxLseShape->SetDimNum(1);
        (*softmaxLseShape)[0] = NUM_0;
    }
    return ge::GRAPH_SUCCESS;
}

// InferDataTypeMinimaxSparseAttentionSplitKv: 输出数据类型推断函数。
// 推断规则：输出 attentionOut 的数据类型等于输入 query 的数据类型（bf16）。
static ge::graphStatus InferDataTypeMinimaxSparseAttentionSplitKv(gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    // 取 query 的数据类型并赋给输出 attentionOut。
    auto dtype = context->GetInputDataType(QUERY_INDEX);
    if (dtype == ge::DT_FLOAT8_E4M3FN) {
        context->SetOutputDataType(ATTENTION_OUT_INDEX, ge::DT_BF16);
    } else {
        context->SetOutputDataType(ATTENTION_OUT_INDEX, dtype);
    }
    context->SetOutputDataType(SOFTMAX_LSE_INDEX, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

// 注册算子的 InferShape 与 InferDataType 回调到 GE。
IMPL_OP_INFERSHAPE(MinimaxSparseAttentionSplitKv)
    .InferShape(InferShapeMinimaxSparseAttentionSplitKv)
    .InferDataType(InferDataTypeMinimaxSparseAttentionSplitKv);

} // namespace ops
