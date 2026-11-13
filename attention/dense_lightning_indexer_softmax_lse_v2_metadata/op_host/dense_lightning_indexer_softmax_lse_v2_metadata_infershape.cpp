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
 * \\file dense_lightning_indexer_softmax_lse_v2_metadata_infershape.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "../../dense_lightning_indexer_softmax_lse_v2/op_kernel/dense_lightning_indexer_softmax_lse_v2_metadata.h"

#include <cstring>

using namespace ge;

namespace ops {
static ge::graphStatus InferShapeDenseLightningIndexerSoftmaxLseV2Metadata(gert::InferShapeContext* context)
{
    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return GRAPH_FAILED;
    }
    const auto* batchSize = attrs->GetAttrPointer<int64_t>(0);
    const auto* maxSeqLenQ = attrs->GetAttrPointer<int64_t>(1);
    const char* const* layoutQPtr = attrs->GetAttrPointer<const char*>(6);
    const char* layoutQ = (layoutQPtr != nullptr) ? *layoutQPtr : nullptr;
    if (batchSize == nullptr || maxSeqLenQ == nullptr) {
        return GRAPH_FAILED;
    }
    int64_t itemBound = 0;
    if (layoutQ != nullptr && strcmp(layoutQ, "TND") == 0) {
        itemBound = (*maxSeqLenQ > 0) ? *maxSeqLenQ : static_cast<int64_t>(optiling::DLI_SCHED_ITEM_MAX_NUM);
    } else if (*batchSize > 0 && *maxSeqLenQ > 0) {
        itemBound = (*batchSize) * (*maxSeqLenQ);
    }
    gert::Shape* oShape = context->GetOutputShape(0);
    oShape->SetDimNum(1);
    oShape->SetDim(0, optiling::DliSchedMetadataSize(itemBound));
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDtypeDenseLightningIndexerSoftmaxLseV2Metadata(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, DT_INT32);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(DenseLightningIndexerSoftmaxLseV2Metadata)
    .InferShape(InferShapeDenseLightningIndexerSoftmaxLseV2Metadata)
    .InferDataType(InferDtypeDenseLightningIndexerSoftmaxLseV2Metadata);
} // namespace ops
