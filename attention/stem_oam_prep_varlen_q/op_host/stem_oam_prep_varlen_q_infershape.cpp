/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <exe_graph/runtime/infer_shape_context.h>
#include <register/op_impl_registry.h>
#include "log/log.h"
#include "util/shape_util.h"

namespace ops {

static constexpr int64_t DIM_QK = 128;
static constexpr int64_t Q_DIM_NUM = 3;
static constexpr int64_t Q_SEQ_LENS_DIM_NUM = 1;
static constexpr int64_t Q_HEAD_DIM_INDEX = 1;
static constexpr int64_t Q_LAST_DIM_INDEX = 2;
static constexpr int64_t INPUT_Q = 0;
static constexpr int64_t INPUT_Q_SEQ_LENS = 1;
static constexpr int64_t OUTPUT_Q_FLAT = 0;
static constexpr int64_t ATTR_STEM_BLOCK_SIZE = 0;
static constexpr int64_t ATTR_STEM_STRIDE = 1;
static constexpr int64_t UNKNOWN_DIM_VALUE = -1LL;
static constexpr int64_t UNKNOWN_RANK_DIM_VALUE = -2LL;
static constexpr int64_t SUPPORTED_STEM_BLOCK_SIZE = 128;
static constexpr int64_t SUPPORTED_STEM_STRIDE = 16;

static bool IsUnknownRank(const gert::Shape &shape)
{
    return shape.GetDimNum() == 1 && shape.GetDim(0) == UNKNOWN_RANK_DIM_VALUE;
}

static bool IsUnknownShape(const gert::Shape &shape)
{
    size_t dimNum = shape.GetDimNum();
    for (size_t i = 0; i < dimNum; i++) {
        if (shape.GetDim(i) == UNKNOWN_DIM_VALUE) {
            return true;
        }
    }
    return false;
}

static ge::graphStatus InferShapeForStemOamPrepVarlenQ(gert::InferShapeContext *context)
{
    OP_LOGI(context, "Enter StemOamPrepVarlenQ runtime infershape impl.");
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape *qShape = context->GetInputShape(INPUT_Q);
    OP_CHECK_NULL_WITH_CONTEXT(context, qShape);
    const gert::Shape *qSeqLensShape = context->GetInputShape(INPUT_Q_SEQ_LENS);
    OP_CHECK_NULL_WITH_CONTEXT(context, qSeqLensShape);

    gert::Shape *qFlatShape = context->GetOutputShape(OUTPUT_Q_FLAT);
    OP_CHECK_NULL_WITH_CONTEXT(context, qFlatShape);

    if (IsUnknownRank(*qShape) || IsUnknownRank(*qSeqLensShape)) {
        qFlatShape->SetDimNum(0);
        qFlatShape->AppendDim(UNKNOWN_RANK_DIM_VALUE);
        OP_LOGD(context, "StemOamPrepVarlenQ infershape handles unknown rank.");
        return ge::GRAPH_SUCCESS;
    }

    if (qShape->GetDimNum() != Q_DIM_NUM) {
        OP_LOGE(context, "q shape dim num must be %ld, but got %zu.", Q_DIM_NUM, qShape->GetDimNum());
        return ge::GRAPH_FAILED;
    }
    int64_t dimD = qShape->GetDim(Q_LAST_DIM_INDEX);
    if (dimD != UNKNOWN_DIM_VALUE && dimD != DIM_QK) {
        OP_LOGE(context, "q last dim must be %ld, but got %ld.", DIM_QK, dimD);
        return ge::GRAPH_FAILED;
    }
    if (qSeqLensShape->GetDimNum() != Q_SEQ_LENS_DIM_NUM) {
        OP_LOGE(context, "qSeqLens shape dim num must be %ld, but got %zu.", Q_SEQ_LENS_DIM_NUM,
                qSeqLensShape->GetDimNum());
        return ge::GRAPH_FAILED;
    }

    int64_t hQ = qShape->GetDim(Q_HEAD_DIM_INDEX);
    int64_t batch = qSeqLensShape->GetDim(0);

    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    auto stemBlockSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_STEM_BLOCK_SIZE);
    OP_CHECK_NULL_WITH_CONTEXT(context, stemBlockSizePtr);
    int64_t stemBlockSize = *stemBlockSizePtr;
    if (stemBlockSize != SUPPORTED_STEM_BLOCK_SIZE) {
        OP_LOGE(context, "stemBlockSize only supports %ld, but got %ld.", SUPPORTED_STEM_BLOCK_SIZE, stemBlockSize);
        return ge::GRAPH_FAILED;
    }

    auto stemStridePtr = attrs->GetAttrPointer<int64_t>(ATTR_STEM_STRIDE);
    OP_CHECK_NULL_WITH_CONTEXT(context, stemStridePtr);
    int64_t stemStride = *stemStridePtr;
    if (stemStride != SUPPORTED_STEM_STRIDE) {
        OP_LOGE(context, "stemStride only supports %ld, but got %ld.", SUPPORTED_STEM_STRIDE, stemStride);
        return ge::GRAPH_FAILED;
    }
    int64_t kflat_dim = stemStride * DIM_QK;

    bool hasUnknownDim = IsUnknownShape(*qShape) || IsUnknownShape(*qSeqLensShape);
    if (hasUnknownDim) {
        qFlatShape->SetDimNum(4);
        qFlatShape->SetDim(0, batch);
        qFlatShape->SetDim(1, hQ);
        qFlatShape->SetDim(2, UNKNOWN_DIM_VALUE);
        qFlatShape->SetDim(3, kflat_dim);
        OP_LOGD(context, "StemOamPrepVarlenQ infershape handles unknown dim.");
        return ge::GRAPH_SUCCESS;
    }

    const gert::Tensor *qSeqLensTensor = context->GetInputTensor(INPUT_Q_SEQ_LENS);
    OP_CHECK_NULL_WITH_CONTEXT(context, qSeqLensTensor);
    const int64_t *qSeqLensData = qSeqLensTensor->GetData<int64_t>();
    OP_CHECK_NULL_WITH_CONTEXT(context, qSeqLensData);

    int64_t maxQLen = 0;
    for (int64_t i = 0; i < batch; i++) {
        if (qSeqLensData[i] > maxQLen) {
            maxQLen = qSeqLensData[i];
        }
    }
    int64_t maxQPadded = ((maxQLen + stemBlockSize - 1) / stemBlockSize) * stemBlockSize;
    int64_t maxQb = maxQPadded / stemBlockSize;

    *qFlatShape = gert::Shape({batch, hQ, maxQb, kflat_dim});

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForStemOamPrepVarlenQ(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(OUTPUT_Q_FLAT, ge::DT_BF16);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(StemOamPrepVarlenQ)
    .InferShape(InferShapeForStemOamPrepVarlenQ)
    .InferDataType(InferDataTypeForStemOamPrepVarlenQ)
    .InputsDataDependency({INPUT_Q_SEQ_LENS});

} // namespace ops
