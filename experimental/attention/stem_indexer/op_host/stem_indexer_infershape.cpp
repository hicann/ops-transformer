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
 * \file stem_indexer_infershape.cpp
 * \brief
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>

#include <string>

#include "err/ops_err.h"

using namespace ge;

namespace ops {
constexpr uint32_t QFLAT_INDEX = 0;
constexpr uint32_t KFLAT_INDEX = 1;
constexpr uint32_t SPARSE_INDICES_INDEX = 0;
constexpr uint32_t SPARSE_SEQ_LEN_INDEX = 1;
constexpr uint32_t DIM_IDX_ZERO = 0;
constexpr uint32_t DIM_IDX_ONE = 1;
constexpr uint32_t DIM_IDX_TWO = 2;
constexpr uint32_t DIM_NUM_FOUR = 4;

static ge::graphStatus InferShapeStemIndexer(gert::InferShapeContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("StemIndexer", "InferShapeContext is nullptr!"), return ge::GRAPH_FAILED);

    const gert::Shape *qflatShape = context->GetInputShape(QFLAT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, qflatShape);
    const gert::Shape *kflatShape = context->GetInputShape(KFLAT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, kflatShape);

    OP_CHECK_IF(qflatShape->GetDimNum() != DIM_NUM_FOUR,
                OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "qflat",
                                             (std::to_string(qflatShape->GetDimNum()) + "D").c_str(), "4D"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(kflatShape->GetDimNum() != DIM_NUM_FOUR,
                OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "kflat",
                                             (std::to_string(kflatShape->GetDimNum()) + "D").c_str(), "4D"),
                return ge::GRAPH_FAILED);

    gert::Shape *sparseIndicesShape = context->GetOutputShape(SPARSE_INDICES_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, sparseIndicesShape);
    gert::Shape *sparseSeqLenShape = context->GetOutputShape(SPARSE_SEQ_LEN_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, sparseSeqLenShape);

    sparseIndicesShape->SetDimNum(DIM_NUM_FOUR);
    sparseIndicesShape->SetDim(0, qflatShape->GetDim(DIM_IDX_ZERO));
    sparseIndicesShape->SetDim(1, qflatShape->GetDim(DIM_IDX_ONE));
    sparseIndicesShape->SetDim(2, qflatShape->GetDim(DIM_IDX_TWO));
    sparseIndicesShape->SetDim(3, kflatShape->GetDim(DIM_IDX_TWO));

    sparseSeqLenShape->SetDimNum(3);
    sparseSeqLenShape->SetDim(0, qflatShape->GetDim(DIM_IDX_ZERO));
    sparseSeqLenShape->SetDim(1, qflatShape->GetDim(DIM_IDX_ONE));
    sparseSeqLenShape->SetDim(2, qflatShape->GetDim(DIM_IDX_TWO));

    OP_LOGI(context->GetNodeName(), "StemIndexer InferShape end.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeStemIndexer(gert::InferDataTypeContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("StemIndexer", "InferDataTypeContext is nullptr!"),
                return ge::GRAPH_FAILED);
    context->SetOutputDataType(SPARSE_INDICES_INDEX, ge::DT_INT32);
    context->SetOutputDataType(SPARSE_SEQ_LEN_INDEX, ge::DT_INT32);
    OP_LOGI(context->GetNodeName(), "StemIndexer InferDataType end.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(StemIndexer).InferShape(InferShapeStemIndexer).InferDataType(InferDataTypeStemIndexer);
} // namespace ops
